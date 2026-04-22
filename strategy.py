import math
from prediction import (
    aim, crosses_sun, estimate_arrival_turn,
    is_orbiting, predict_position
)

# 가중치 (나중에 자동 튜닝 대상)
W_PRODUCTION = 3.0   # 생산량 높을수록 좋음
W_SHIPS      = 0.3   # 방어력 높을수록 나쁨 (비용)
W_DISTANCE   = 0.05  # 멀수록 나쁨
W_ENEMY      = 2.0   # 적 행성 보너스

# 게임 페이즈
EARLY_END = 150
LATE_START = 350

_turn = 0


def get_state(obs):
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
    if isinstance(obs, dict):
        player     = obs.get("player", 0)
        raw_planets = obs.get("planets", [])
        raw_fleets  = obs.get("fleets", [])
        av          = obs.get("angular_velocity", 0)
        comet_ids   = set(obs.get("comet_planet_ids", []))
    else:
        player      = obs.player
        raw_planets = obs.planets
        raw_fleets  = obs.fleets
        av          = obs.angular_velocity
        comet_ids   = set(obs.comet_planet_ids or [])

    planets = [Planet(*p) for p in raw_planets]
    fleets  = [Fleet(*f) for f in raw_fleets]
    return player, planets, fleets, av, comet_ids


def fleet_will_hit(fleet, planet):
    """fleet 직선 경로가 planet에 도달하는지 체크."""
    dx = math.cos(fleet.angle)
    dy = math.sin(fleet.angle)
    fx = fleet.x - planet.x
    fy = fleet.y - planet.y
    t = -(fx * dx + fy * dy)
    if t < 0:
        return False
    cx = fleet.x + t * dx
    cy = fleet.y + t * dy
    return math.hypot(cx - planet.x, cy - planet.y) <= planet.radius * 1.5


def incoming_threat(planet, fleets, player):
    """내 행성으로 향하는 적 fleet ships 합산."""
    return sum(f.ships for f in fleets if f.owner != player and fleet_will_hit(f, planet))


def find_angle(src, dst, av, num_ships):
    """
    조준 각도 반환. 태양이 막으면 None.
    fleet은 직선만 이동 가능 — 우회 불가.
    """
    angle = aim(src, dst, av, num_ships)

    # aim()이 수렴한 목표 위치로 태양 충돌 체크
    tx = src.x + math.cos(angle) * math.hypot(dst.x - src.x, dst.y - src.y)
    ty = src.y + math.sin(angle) * math.hypot(dst.x - src.x, dst.y - src.y)
    if crosses_sun(src.x, src.y, tx, ty):
        return None

    return angle


def target_score(target, distance, is_comet, phase):
    """target 공격 가치 점수."""
    score = target.production * W_PRODUCTION
    score -= target.ships * W_SHIPS
    score -= distance * W_DISTANCE

    if target.owner >= 0:   # 적 행성이면 보너스
        score += W_ENEMY

    if is_comet:            # 코멧은 곧 사라질 수 있어서 페널티
        score -= 1.5

    if phase == 'late':     # 후반엔 공격 소극적
        score *= 0.7

    return score


def decide(obs):
    global _turn
    _turn += 1

    player, planets, fleets, av, comet_ids = get_state(obs)

    my_planets = [p for p in planets if p.owner == player]
    targets    = [p for p in planets if p.owner != player]

    if not targets:
        return []

    if _turn < EARLY_END:
        phase = 'early'
    elif _turn < LATE_START:
        phase = 'mid'
    else:
        phase = 'late'

    # 이미 내 fleet이 향하고 있는 target과 ships 합계
    in_transit = {}
    for f in fleets:
        if f.owner == player:
            for t in targets:
                if fleet_will_hit(f, t):
                    in_transit[t.id] = in_transit.get(t.id, 0) + f.ships

    moves = []
    assigned = {}  # target.id → 이번 턴 보내기로 한 ships 합계

    for mine in my_planets:
        threat = incoming_threat(mine, fleets, player)
        # 방어 후 사용 가능한 ships (최소 1은 남김)
        available = mine.ships - threat - 1
        if available <= 0:
            continue

        best_score  = float('-inf')
        best_target = None
        best_angle  = None
        best_needed = None

        for t in targets:
            ships_needed = t.ships + 1

            # 이미 fleet이 충분히 가고 있거나 이번 턴에 충분히 보냈으면 스킵
            if in_transit.get(t.id, 0) + assigned.get(t.id, 0) >= ships_needed:
                continue

            if available < ships_needed:
                continue

            angle = find_angle(mine, t, av, ships_needed)
            if angle is None:   # 태양이 막음
                continue

            dist  = math.hypot(mine.x - t.x, mine.y - t.y)
            score = target_score(t, dist, t.id in comet_ids, phase)

            if score > best_score:
                best_score  = score
                best_target = t
                best_angle  = angle
                best_needed = ships_needed

        if best_target is None:
            continue

        moves.append([mine.id, best_angle, best_needed])
        assigned[best_target.id] = assigned.get(best_target.id, 0) + best_needed

    return moves
