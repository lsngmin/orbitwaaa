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
                            max_turns=120):
    """launch 전 경로 유효성: 그 angle/ships로 날릴 때 첫 충돌을 예측.

    엔진과 동일 순서로 체크: out → sun → planet direct.
    각 flight turn t에서 planet 위치는 (t-1)번 회전한 상태로 본다
    (엔진은 fleet movement 시점에 아직 그 step의 planet rotation이 적용 안 됨).
    Sweep은 여기선 안 본다(첫 컷). Source 자기자신과의 충돌은 무시.

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
            px, py = predict_position(planet, av, t - 1)
            if point_to_segment_distance((px, py),
                                         (cur_x, cur_y), (new_x, new_y)) < planet.radius:
                return ("planet", planet.id)

        cur_x, cur_y = new_x, new_y

    return ("none", None)


def aim(src_planet, dst_planet, angular_velocity, num_ships):
    """
    dst_planet에 fleet이 도착할 시점의 위치를 예측해서
    (angle, arrival_x, arrival_y, turns) 반환.
    정적 행성은 현재 위치 그대로 조준.
    """
    if not is_orbiting(dst_planet):
        dx = dst_planet.x - src_planet.x
        dy = dst_planet.y - src_planet.y
        dist = math.hypot(dx, dy)
        turns = estimate_arrival_turn(dist, num_ships)
        return math.atan2(dy, dx), dst_planet.x, dst_planet.y, turns

    dist = math.hypot(dst_planet.x - src_planet.x, dst_planet.y - src_planet.y)
    turns = estimate_arrival_turn(dist, num_ships)

    for _ in range(10):
        tx, ty = predict_position(dst_planet, angular_velocity, turns)
        dist = math.hypot(tx - src_planet.x, ty - src_planet.y)
        new_turns = estimate_arrival_turn(dist, num_ships)
        if new_turns == turns:
            break
        turns = new_turns

    tx, ty = predict_position(dst_planet, angular_velocity, turns)
    return math.atan2(ty - src_planet.y, tx - src_planet.x), tx, ty, turns
