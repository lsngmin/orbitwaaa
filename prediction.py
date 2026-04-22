import math

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
    """ships 수에 따른 fleet 속도."""
    if num_ships <= 1:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(num_ships) / math.log(1000)) ** 1.5


def estimate_arrival_turn(distance, num_ships):
    """fleet이 도착하는 데 걸리는 턴 수."""
    speed = fleet_speed(num_ships)
    return math.ceil(distance / speed)


def crosses_sun(src_x, src_y, dst_x, dst_y, sun_radius=13.0):
    """fleet 직선 경로가 태양(중심 50,50)을 통과하는지 체크. 선분-원 교차 판정."""
    dx = dst_x - src_x
    dy = dst_y - src_y
    fx = src_x - CENTER_X
    fy = src_y - CENTER_Y

    a = dx * dx + dy * dy
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


def aim(src_planet, dst_planet, angular_velocity, num_ships):
    """
    dst_planet에 fleet이 도착할 시점의 위치를 예측해서 각도 반환.
    정적 행성은 현재 위치 그대로 조준.
    """
    if not is_orbiting(dst_planet):
        dx = dst_planet.x - src_planet.x
        dy = dst_planet.y - src_planet.y
        return math.atan2(dy, dx)

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
    return math.atan2(ty - src_planet.y, tx - src_planet.x)
