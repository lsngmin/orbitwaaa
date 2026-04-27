import math

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    BOARD_SIZE,
    CENTER,
    SUN_RADIUS,
    point_to_segment_distance,
)

CENTER_X = 50.0
CENTER_Y = 50.0
MAX_SPEED = 6.0


def orbital_radius(planet):
    dx = planet.x - CENTER_X
    dy = planet.y - CENTER_Y
    return math.hypot(dx, dy)


def is_orbiting(planet):
    return orbital_radius(planet) + planet.radius < 50.0


def initial_angle(planet):
    return math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)


def predict_position(planet, angular_velocity, turns):
    """N턴 후 행성 위치 반환. 정적 행성은 현재 위치 그대로."""
    if not is_orbiting(planet):
        return planet.x, planet.y

    r = orbital_radius(planet)
    angle = initial_angle(planet) + angular_velocity * turns
    x = CENTER_X + r * math.cos(angle)
    y = CENTER_Y + r * math.sin(angle)
    return x, y


class PositionCache:
    """Step-local 캐시: planet-level 상수 + (pid, turn)→(x, y) 메모이즈.

    같은 obs(같은 av)에서 mask/decode가 반복적으로 predict_position을 호출하는
    핫스팟을 제거. step 시작에 만들고 step 끝에 폐기 (av가 step마다 바뀔 수 있고
    planet.x/y도 step마다 obs로 새로 들어옴).

    `predict()` 시그니처는 `predict_position(planet, av, turns)`와 동일한 결과.
    """

    __slots__ = ("av", "_meta", "_pos")

    def __init__(self, planets, av):
        self.av    = av
        self._meta = {}   # pid -> (orbiting, r, init_angle, x, y)
        self._pos  = {}   # (pid, turns) -> (x, y), 공전 행성만 캐시
        for p in planets:
            r        = math.hypot(p.x - CENTER_X, p.y - CENTER_Y)
            orbiting = (r + p.radius < 50.0)
            ia       = math.atan2(p.y - CENTER_Y, p.x - CENTER_X)
            self._meta[p.id] = (orbiting, r, ia, p.x, p.y)

    def predict(self, planet, turns):
        meta = self._meta.get(planet.id)
        if meta is None:
            # planet not pre-registered (e.g. mid-step new entity) — fallback.
            return predict_position(planet, self.av, turns)
        orbiting, r, ia, px, py = meta
        if not orbiting:
            return px, py
        key    = (planet.id, turns)
        cached = self._pos.get(key)
        if cached is not None:
            return cached
        angle = ia + self.av * turns
        x = CENTER_X + r * math.cos(angle)
        y = CENTER_Y + r * math.sin(angle)
        self._pos[key] = (x, y)
        return x, y


def fleet_speed(num_ships):
    """ships 수에 따른 fleet 속도. 엔진은 min(speed, MAX_SPEED)로 캡."""
    if num_ships <= 1:
        return 1.0
    speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(num_ships) / math.log(1000)) ** 1.5
    return min(speed, MAX_SPEED)


def estimate_arrival_turn(distance, num_ships):
    """fleet이 도착하는 데 걸리는 턴 수."""
    speed = fleet_speed(num_ships)
    return math.ceil(distance / speed)


def crosses_sun(src_x, src_y, dst_x, dst_y, sun_radius=10.5):
    """fleet 직선 경로가 태양(중심 50,50)을 통과하는지 체크. 선분-원 교차 판정."""
    dx = dst_x - src_x
    dy = dst_y - src_y
    fx = src_x - CENTER_X
    fy = src_y - CENTER_Y

    a = dx * dx + dy * dy
    if a == 0:
        # 선분이 아닌 점 — 점과 태양 거리만 비교
        return fx * fx + fy * fy <= sun_radius * sun_radius
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - sun_radius * sun_radius

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    # t가 [0, 1] 범위 안에 있으면 선분이 원과 교차
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)


def sun_approach_distance(src_x, src_y, dst_x, dst_y):
    """src → dst 경로에서 태양 중심(50, 50)까지의 최소 거리 반환.
    선분과 점 사이의 최소 거리 (t를 [0,1] 클램핑)."""
    dx = dst_x - src_x
    dy = dst_y - src_y
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(src_x - CENTER_X, src_y - CENTER_Y)
    fx = src_x - CENTER_X
    fy = src_y - CENTER_Y
    t = -(fx * dx + fy * dy) / length_sq
    t = max(0.0, min(1.0, t))
    cx = src_x + t * dx
    cy = src_y + t * dy
    return math.hypot(cx - CENTER_X, cy - CENTER_Y)


def first_collision_on_path(src_planet, angle, num_ships, planets, av,
                            max_turns=120, pos_cache=None):
    """launch 전 경로 유효성: 그 angle/ships로 날릴 때 첫 충돌을 예측.

    엔진과 동일 순서로 체크: out → sun → planet direct.
    각 flight turn t에서 planet 위치는 (t-1)번 회전한 상태로 본다
    (엔진은 fleet movement 시점에 아직 그 step의 planet rotation이 적용 안 됨).
    Sweep은 여기선 안 본다(첫 컷). Source 자기자신과의 충돌은 무시.

    pos_cache: PositionCache. None이면 직접 predict_position 호출 (테스트/단발용).

    Returns:
        (cause, planet_id_or_None) — cause ∈ {"out", "sun", "planet", "none"}
    """
    speed = fleet_speed(num_ships)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    cur_x = src_planet.x + cos_a * (src_planet.radius + 0.1)
    cur_y = src_planet.y + sin_a * (src_planet.radius + 0.1)

    for t in range(1, max_turns + 1):
        new_x = cur_x + cos_a * speed
        new_y = cur_y + sin_a * speed

        if not (0.0 <= new_x <= BOARD_SIZE and 0.0 <= new_y <= BOARD_SIZE):
            return ("out", None)

        if point_to_segment_distance((CENTER, CENTER),
                                     (cur_x, cur_y), (new_x, new_y)) < SUN_RADIUS:
            return ("sun", None)

        for planet in planets:
            if planet.id == src_planet.id:
                continue
            if pos_cache is not None:
                px, py = pos_cache.predict(planet, t - 1)
            else:
                px, py = predict_position(planet, av, t - 1)
            if point_to_segment_distance((px, py),
                                         (cur_x, cur_y), (new_x, new_y)) < planet.radius:
                return ("planet", planet.id)

        cur_x, cur_y = new_x, new_y

    return ("none", None)


def aim(src_planet, dst_planet, angular_velocity, num_ships, pos_cache=None):
    """
    dst_planet에 fleet이 도착할 시점의 위치를 예측해서
    (angle, arrival_x, arrival_y, turns) 반환.
    정적 행성은 현재 위치 그대로 조준.

    pos_cache: PositionCache. None이면 직접 predict_position 호출.
    """
    if not is_orbiting(dst_planet):
        dx = dst_planet.x - src_planet.x
        dy = dst_planet.y - src_planet.y
        dist = math.hypot(dx, dy)
        turns = estimate_arrival_turn(dist, num_ships)
        return math.atan2(dy, dx), dst_planet.x, dst_planet.y, turns

    dist = math.hypot(dst_planet.x - src_planet.x, dst_planet.y - src_planet.y)
    turns = estimate_arrival_turn(dist, num_ships)

    if pos_cache is not None:
        _predict = pos_cache.predict
        for _ in range(10):
            tx, ty = _predict(dst_planet, turns)
            dist = math.hypot(tx - src_planet.x, ty - src_planet.y)
            new_turns = estimate_arrival_turn(dist, num_ships)
            if new_turns == turns:
                break
            turns = new_turns
        tx, ty = _predict(dst_planet, turns)
    else:
        for _ in range(10):
            tx, ty = predict_position(dst_planet, angular_velocity, turns)
            dist = math.hypot(tx - src_planet.x, ty - src_planet.y)
            new_turns = estimate_arrival_turn(dist, num_ships)
            if new_turns == turns:
                break
            turns = new_turns
        tx, ty = predict_position(dst_planet, angular_velocity, turns)
    return math.atan2(ty - src_planet.y, tx - src_planet.x), tx, ty, turns


def fleet_dst_and_eta(fleet, planets, radius_margin=1.5):
    """fleet 의 ray-cast 첫 충돌 행성 id 와 ETA(turns).

    encode_fleets 의 dst_idx 산출 logic 과 동일 (radius * 1.5 lenient margin).
    충돌 행성 없으면 (-1, math.inf) 반환.

    Returns: (dst_planet_id, eta)  — eta 는 ceil(distance / speed), 최소 1.
    """
    dx = math.cos(fleet.angle)
    dy = math.sin(fleet.angle)
    dst_pid = -1
    first_t = math.inf
    for p in planets:
        fx = fleet.x - p.x
        fy = fleet.y - p.y
        t  = -(fx * dx + fy * dy)
        if t <= 0:
            continue
        cx = fleet.x + t * dx
        cy = fleet.y + t * dy
        if math.hypot(cx - p.x, cy - p.y) > p.radius * radius_margin:
            continue
        if t < first_t:
            first_t = t
            dst_pid = p.id
    if dst_pid == -1:
        return -1, math.inf
    speed = fleet_speed(fleet.ships)
    eta = max(1, int(math.ceil(first_t / speed)))
    return dst_pid, eta


def project_target_at_eta(target, eta, planets, fleets):
    """target 행성을 eta 시점까지 forward simulate.

    in-flight fleet 들의 도착을 시간순으로 적용해서 (proj_owner, proj_ships) 반환.
    "freeze" 가정 외부 — 이 함수는 obs 에 보이는 사실만 사용 (적이 새로 안 쏠
    거라는 가정은 호출자 수준의 한계이지 이 함수의 한계가 아님).

    production 은 owner 무관하게 누적 (기존 required_ships 공식과 동일 동작).
    점령 swap 발생 시 sim_owner 갱신, defender 함선 0 미만 안 되도록 클램핑.

    Args:
        target:   Planet — 시뮬 대상
        eta:      int    — sim horizon (turns). target 의 함선 변화는 이 시점 기준.
        planets:  list[Planet] — 모든 행성 (ray-cast 용 좌표)
        fleets:   list[Fleet]  — 모든 in-flight fleet

    Returns:
        (proj_owner, proj_ships) — eta 시점 owner (-1/0/1) 와 함선 수
    """
    sim_owner = target.owner
    sim_ships = float(target.ships)

    arrivals = []
    for f in fleets:
        dst_pid, f_eta = fleet_dst_and_eta(f, planets)
        if dst_pid == target.id and f_eta <= eta:
            arrivals.append((f_eta, f))
    arrivals.sort(key=lambda x: x[0])

    last_t = 0
    for arrive_t, f in arrivals:
        sim_ships += target.production * (arrive_t - last_t)
        if f.owner == sim_owner:
            sim_ships += f.ships
        else:
            if f.ships > sim_ships:
                sim_owner = f.owner
                sim_ships = f.ships - sim_ships
            else:
                sim_ships -= f.ships
        last_t = arrive_t

    sim_ships += target.production * (eta - last_t)
    return sim_owner, sim_ships


def resolve_ships_for_capture(src, dst, angular_velocity, bin_value, src_ships,
                               pos_cache=None, max_iter=5, fleets=None,
                               planets=None, amount_mode="multiplier"):
    """
    Pair-wise ship-count decode (target=dst 점령 수학).

    관계식 (amount_mode="multiplier", Phase A default):
      required     = ETA forward sim 기반 도착 시 필요 함선 (in-flight 반영)
      ships_needed = clip(ceil(required × multiplier), 1, src_ships)
      bin_value = multiplier ∈ ships_multipliers (config.yaml).

    관계식 (amount_mode="surplus", legacy):
      surplus      = max(0, src_ships - required)
      ships_needed = clip(round(required + bin × surplus), 1, src_ships)
      bin_value = bin ∈ ships_surplus_bins.

    공통:
      src_ships < required (capacity short) 인 경우 ships_needed = 0
      (점령 수학적 불가 — dominated action. 호출자가 under_invested 로 집계 후
       launch 폐기.)

    고정점 반복: required 가 ships(=속도) 에 의존 → 한번에 안 풀림.
      ships↑ → 속도↑ → turns↓ → required↓ → ships 변동 → 반복.
      multiplier 모드는 ships=ceil(req·m) 라 monotone 수렴 더 빠름.
      oscillate 시 best_ships (conservative=가장 큰) 채택.

    returns: (ships_needed, angle, tx, ty, turns, required, converged)
    """
    src_ships = int(src_ships)
    if src_ships <= 0:
        angle, tx, ty, turns = aim(src, dst, angular_velocity, 1, pos_cache=pos_cache)
        return 0, angle, tx, ty, turns, dst.ships + 1, True

    use_dynamic = fleets is not None and planets is not None

    def _required_at(eff_turns):
        if use_dynamic:
            proj_owner, proj_ships = project_target_at_eta(dst, eff_turns, planets, fleets)
            if proj_owner == src.owner:
                # 도착 시점에 이미 내 거 — 점령 의미 없음 (호출자가 mask off)
                return 0
            return max(1, int(proj_ships) + 1)
        return dst.ships + dst.production * eff_turns + 1

    def _send_for(req):
        if req <= 0:
            return 0
        if src_ships < req:
            # capacity short: 점령 수학적 불가 — dominated action 으로 차단.
            # 0 반환 → 호출자가 under_invested 로 집계 후 launch 자체 폐기.
            return 0
        if amount_mode == "multiplier":
            # required 는 BASE — surplus 를 더하지 않는다.
            # multiplier ∈ [1, ~1.2]: 1.0 = floor, 1.2 = +20% buffer.
            raw = req * bin_value
            ships = int(math.ceil(raw))
        else:  # "surplus" — legacy
            surplus = src_ships - req
            raw    = req + bin_value * surplus
            ships  = int(round(raw))
        return min(src_ships, max(1, ships))

    ships_rep = src_ships
    angle = tx = ty = None
    turns = 0
    required = 0
    best_ships = 0
    converged = False
    for _ in range(max_iter):
        angle, tx, ty, turns = aim(src, dst, angular_velocity, ships_rep, pos_cache=pos_cache)
        eff_turns  = turns if turns else 1
        required   = _required_at(eff_turns)
        if required <= 0:
            return 0, angle, tx, ty, turns, 0, True
        new_needed = _send_for(required)
        if new_needed > best_ships:
            best_ships = new_needed
        if new_needed == ships_rep:
            converged = True
            break
        ships_rep = new_needed

    if not converged and best_ships != ships_rep:
        ships_rep = best_ships
        angle, tx, ty, turns = aim(src, dst, angular_velocity, ships_rep, pos_cache=pos_cache)
        eff_turns = turns if turns else 1
        required  = _required_at(eff_turns)

    return ships_rep, angle, tx, ty, turns, required, converged
