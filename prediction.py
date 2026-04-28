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
    """fleet 직선 경로가 태양(중심 50,50)을 통과하는지 체크. 선분-원 교차 판정.

    sun_radius 기본 10.5 = 엔진 SUN_RADIUS(10.0) + 0.5 보수 margin
    (mask 가 경계 fleet 도 sun-cross 로 차단). 엔진과 정확 일치는
    호출자가 sun_radius=SUN_RADIUS 명시.
    """
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

    # GAME_RULES "Fleet Launch": fleet 은 행성 radius 바로 바깥(+0.1)에서 스폰.
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


def fleet_dst_and_eta(fleet, planets, av=0.0, radius_margin=1.0, max_turns=200,
                       pos_cache=None):
    """fleet 의 미래 도착 행성 + ETA — 엔진 turn order 정확 모델.

    매 turn t (1..max_turns) 시뮬레이션 (GAME_RULES "Turn Order" 정확 미러):
      Step 5 (Fleet Movement):
        a. fleet 1턴 이동 → new_pos.
        b. OOB / sun cross / planet collision 순서 체크.
        c. planet collision 은 planet at (t-1) rotation 위치 기준
           (엔진은 fleet movement 시점에 그 step 의 planet rotation 미적용).
        d. 같은 turn 에 여러 행성이 후보면 segment 따라 가장 먼저 만나는 것.
      Step 6 (Planet Rotation + Sweep):
        - 공전 행성: position_{t-1} → position_t 호 (chord 근사).
        - 그 chord 가 fleet 의 post-Step-5 위치를 휩쓸면 sweep hit.
        - 정적 행성은 sweep 없음.

    Args:
        fleet:    Fleet — in-flight 함대 (현재 obs 위치/각도).
        planets:  list[Planet] — 모든 행성 (현재 obs 위치).
        av:       float — angular_velocity (radians/turn). obs 에서 가져와 전달.
                  0 이면 모두 정적 가정 (정적 행성만 있는 환경에서만 안전).
        radius_margin: float — 충돌 반경 배율. default 1.0 = engine 정확 (엔진은
                  point_to_segment_distance < planet.radius strict). lenient
                  필요 시 (encoder dst_idx 같은 ML 피처) 호출자가 1.5 명시.
        max_turns: int — 최대 시뮬 턴 (default 200, board 대각선 + 안전 여유).
        pos_cache: PositionCache — 있으면 predict_position 캐싱 (성능).

    Returns:
        (dst_planet_id, eta) — 충돌 없으면 (-1, math.inf). eta 는 정확한 첫 도착 turn.
    """
    speed = fleet_speed(fleet.ships)
    cos_a = math.cos(fleet.angle)
    sin_a = math.sin(fleet.angle)

    cur_x, cur_y = fleet.x, fleet.y

    def _ppos(planet, turns):
        if pos_cache is not None:
            return pos_cache.predict(planet, turns)
        return predict_position(planet, av, turns)

    for t in range(1, max_turns + 1):
        new_x = cur_x + cos_a * speed
        new_y = cur_y + sin_a * speed

        # Step 5a: out-of-bounds (GAME_RULES "Fleet Movement").
        if not (0.0 <= new_x <= BOARD_SIZE and 0.0 <= new_y <= BOARD_SIZE):
            return -1, math.inf

        # Step 5b: sun cross — fleet 소멸. SUN_RADIUS strict (engine 일치).
        if point_to_segment_distance((CENTER, CENTER),
                                     (cur_x, cur_y), (new_x, new_y)) < SUN_RADIUS:
            return -1, math.inf

        # Step 5c: direct planet collision (planet at (t-1) rotation).
        # 같은 turn 에 여러 후보면 segment 상에서 더 먼저 만나는 것 (smaller seg_t).
        seg_dx = new_x - cur_x
        seg_dy = new_y - cur_y
        seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
        best_dst  = -1
        best_segt = math.inf
        for planet in planets:
            px, py = _ppos(planet, t - 1)
            d = point_to_segment_distance((px, py), (cur_x, cur_y), (new_x, new_y))
            if d >= planet.radius * radius_margin:
                continue
            if seg_len_sq <= 0:
                seg_t = 0.0
            else:
                seg_t = ((px - cur_x) * seg_dx + (py - cur_y) * seg_dy) / seg_len_sq
                seg_t = max(0.0, min(1.0, seg_t))
            if seg_t < best_segt:
                best_segt = seg_t
                best_dst  = planet.id
        if best_dst != -1:
            return best_dst, t

        # Step 6: planet rotation + sweep (GAME_RULES "Planet rotation").
        # 공전 행성이 (t-1) → (t) 호를 그리며 fleet 의 post-Step-5 위치를 휩쓸 수 있음.
        # 정적 행성은 px_prev==px_now 라 자동 skip.
        for planet in planets:
            px_prev, py_prev = _ppos(planet, t - 1)
            px_now,  py_now  = _ppos(planet, t)
            if px_prev == px_now and py_prev == py_now:
                continue
            d = point_to_segment_distance((new_x, new_y),
                                          (px_prev, py_prev), (px_now, py_now))
            if d < planet.radius * radius_margin:
                return planet.id, t

        cur_x, cur_y = new_x, new_y

    return -1, math.inf


def project_target_at_eta(target, eta, planets, fleets, av=0.0, pos_cache=None):
    """target 행성을 eta 시점까지 forward simulate.

    in-flight fleet 들의 도착을 시간순으로 적용해서 (proj_owner, proj_ships) 반환.
    "freeze" 가정 외부 — 이 함수는 obs 에 보이는 사실만 사용 (적이 새로 안 쏠
    거라는 가정은 호출자 수준의 한계이지 이 함수의 한계가 아님).

    production 은 sim_owner != -1 구간에만 누적 (GAME_RULES "Production":
    owned planet 만 함선 생성). 점령 swap 발생 시 sim_owner 갱신, defender
    함선 0 미만 안 되도록 클램핑.

    ⚠️ GAME_RULES 와 다른 점 (남은 한계):
      Combat 처리: 같은 ETA 에 서로 다른 owner fleet 이 여럿 도착할 때
      GAME_RULES "Combat" 은 "largest vs second largest 먼저 → 잔존이
      garrison 과 전투". 이 함수는 시간순 1:1 처리 → 4P 게임 동시 도착
      부정확. (2P 게임에선 적이 1명이라 영향 미미.)

    Args:
        target:   Planet — 시뮬 대상
        eta:      int    — sim horizon (turns). target 의 함선 변화는 이 시점 기준.
        planets:  list[Planet] — 모든 행성 (좌표/공전 정보용)
        fleets:   list[Fleet]  — 모든 in-flight fleet
        av:       float — angular_velocity (obs). fleet 도착 예측에 필요.
        pos_cache: PositionCache — 있으면 fleet_dst_and_eta 캐싱 가속.

    Returns:
        (proj_owner, proj_ships) — eta 시점 owner (-1/0/1) 와 함선 수
    """
    sim_owner = target.owner
    sim_ships = float(target.ships)

    arrivals = []
    for f in fleets:
        dst_pid, f_eta = fleet_dst_and_eta(f, planets, av=av, pos_cache=pos_cache)
        if dst_pid == target.id and f_eta <= eta:
            arrivals.append((f_eta, f))
    arrivals.sort(key=lambda x: x[0])

    last_t = 0
    for arrive_t, f in arrivals:
        # A3: 중립(owner=-1) 구간엔 production 없음 (GAME_RULES).
        if sim_owner != -1:
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

    if sim_owner != -1:
        sim_ships += target.production * (eta - last_t)
    return sim_owner, sim_ships


SUPPORT_SOURCE_CAP_FRAC = 0.35   # support launch 가 source 에서 소진할 수 있는 최대 비율
SUPPORT_HOLD_PROJ_FRAC  = 0.20   # proj_owner==own 인 경우 보강량 = ceil(proj_ships * 이 비율)


def compute_support_required(src, dst, proj_owner, proj_ships, acting_player):
    """Support launch 의 최소 필요 병력 (mask + decoder 공통).

    게임 의미:
        proj_owner == acting_player → 보강 (이미 버티지만 적 incoming 있음)
            required = max(1, ceil(proj_ships * 0.20), ceil(dst.production))
        proj_owner != acting_player → rescue/recapture (도착 시 빼앗길 예정)
            required = max(1, proj_ships + 1)

    source cap:
        required > ceil(src.ships * 0.35) 이면 None 반환 — mask 차단 신호.
        decoder 가 같은 cap 으로 ships 를 clip 한다 (parity).

    None 반환 의미: 이 (src, dst) 는 support 로 의미 있는 launch 가 불가능
        (source 를 35% 이상 비워야 함 → 다른 곳 방어 무너짐).
    """
    src_ships = max(1, int(src.ships))
    if proj_owner == acting_player:
        required = max(
            1,
            int(math.ceil(int(proj_ships) * SUPPORT_HOLD_PROJ_FRAC)),
            int(math.ceil(dst.production)),
        )
    else:
        required = max(1, int(proj_ships) + 1)

    max_support = max(1, int(math.ceil(src_ships * SUPPORT_SOURCE_CAP_FRAC)))
    if required > max_support:
        return None
    return required


def resolve_ships_for_support(src, dst, angular_velocity, bin_value, src_ships,
                               required, pos_cache=None, amount_mode="multiplier"):
    """Support launch 의 ships_needed 결정 (mask 통과 후 decoder 호출).

    capture 와 다른 점:
      - required 는 compute_support_required 결과 (보강/rescue 의미).
      - source cap = ceil(src.ships * 0.35) — 방어가 source 비우는 행동 방지.
      - 고정점 반복 없음: required 가 mask 단계에서 결정됐고, ships 변동에
        따른 ETA 변화는 support 의미상 무시 가능 (점령이 아니라 보강).
    """
    src_ships = int(src_ships)
    angle, tx, ty, turns = aim(src, dst, angular_velocity, max(1, src_ships), pos_cache=pos_cache)
    if src_ships <= 0 or required <= 0:
        return 0, angle, tx, ty, turns, required

    if amount_mode == "multiplier":
        raw = required * bin_value
    else:  # "surplus"
        surplus = max(0, src_ships - required)
        raw     = required + bin_value * surplus
    ships = max(1, int(math.ceil(raw)))
    cap   = max(1, int(math.ceil(src_ships * SUPPORT_SOURCE_CAP_FRAC)))
    ships = min(src_ships, cap, ships)
    return ships, angle, tx, ty, turns, required


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
            proj_owner, proj_ships = project_target_at_eta(
                dst, eff_turns, planets, fleets,
                av=angular_velocity, pos_cache=pos_cache,
            )
            if proj_owner == src.owner:
                # 도착 시점에 이미 내 거 — 점령 의미 없음 (호출자가 mask off)
                return 0
            return max(1, int(proj_ships) + 1)
        # A4: static fallback — 중립(owner=-1) 은 production 없음 (GAME_RULES).
        if dst.owner == -1:
            return dst.ships + 1
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
