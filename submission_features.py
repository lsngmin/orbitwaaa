"""
Submission-only feature encoder for Orbit Wars.

제출 경로에서 학습용 env wrapper 의존성을 제거하기 위해
encode_planets / encode_fleets와 관련 상수만 분리한다.
"""

import os
import math
import numpy as np

MAX_PLANETS = 40
MAX_FLEETS = 100
HISTORY = 20
PLANET_DIM = 18   # +3: min_eta_norm, pred_x, pred_y / +2: sun_block, sun_dist_norm
FLEET_DIM = 7

ETA_NEAR = 5
ETA_MID = 15


def encode_planets(raw_planets, raw_fleets, player, comet_ids, angular_velocity=0.0):
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
    from prediction import is_orbiting, estimate_arrival_turn, predict_position, crosses_sun, sun_approach_distance

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if p.owner == player]

    enemy_near = {p.id: 0 for p in planets}
    enemy_mid = {p.id: 0 for p in planets}
    mine_near = {p.id: 0 for p in planets}
    mine_mid = {p.id: 0 for p in planets}

    for f in fleets:
        dx = math.cos(f.angle)
        dy = math.sin(f.angle)

        first_planet = None
        first_t = math.inf
        for p in planets:
            fx = f.x - p.x
            fy = f.y - p.y
            t = -(fx * dx + fy * dy)
            if t <= 0:
                continue
            cx = f.x + t * dx
            cy = f.y + t * dy
            if math.hypot(cx - p.x, cy - p.y) > p.radius * 1.5:
                continue
            if t < first_t:
                first_t = t
                first_planet = p

        if first_planet is None:
            continue

        eta = estimate_arrival_turn(first_t, f.ships)

        if f.owner == player:
            if eta <= ETA_NEAR:
                mine_near[first_planet.id] += f.ships
            elif eta <= ETA_MID:
                mine_mid[first_planet.id] += f.ships
        else:
            if eta <= ETA_NEAR:
                enemy_near[first_planet.id] += f.ships
            elif eta <= ETA_MID:
                enemy_mid[first_planet.id] += f.ships

    arr = np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)
    for i, p in enumerate(planets[:MAX_PLANETS]):
        owner_me = 1.0 if p.owner == player else 0.0
        owner_enemy = 1.0 if p.owner not in (-1, player) else 0.0
        owner_neutral = 1.0 if p.owner == -1 else 0.0

        # ETA feature: 내 행성 중 가장 가까운 곳에서 이 행성까지의 최소 ETA
        # 자기 자신은 스킵 (공격 대상이 아님 → 거리 0이면 feature 무의미)
        min_eta = 50.0
        if my_planets:
            for src in my_planets:
                if src.id == p.id:
                    continue
                dist = math.hypot(p.x - src.x, p.y - src.y)
                eta_to = estimate_arrival_turn(dist, 50)  # 50 ships 기준 근사
                if eta_to < min_eta:
                    min_eta = float(eta_to)
        min_eta_norm = min(min_eta / 50.0, 1.0)

        # 예상 도착 위치: min_eta 턴 후 이 행성의 위치
        pred_x, pred_y = predict_position(p, angular_velocity, int(min_eta))

        # 태양 위험도: 내 행성 → 이 행성 경로 기준 (자기 자신은 스킵)
        sun_block = 0.0
        sun_dist_min = 50.0
        if my_planets:
            for src in my_planets:
                if src.id == p.id:
                    continue
                sd = sun_approach_distance(src.x, src.y, pred_x, pred_y)
                if sd < sun_dist_min:
                    sun_dist_min = sd
                if crosses_sun(src.x, src.y, pred_x, pred_y):
                    sun_block = 1.0
        sun_dist_norm = min(sun_dist_min / 50.0, 1.0)

        arr[i] = [
            p.x / 100.0,
            p.y / 100.0,
            owner_me,
            owner_enemy,
            owner_neutral,
            min(p.ships / 1000.0, 1.0),
            p.production / 5.0,
            1.0 if is_orbiting(p) else 0.0,
            1.0 if p.id in comet_ids else 0.0,
            min(enemy_near[p.id] / 1000.0, 1.0),
            min(enemy_mid[p.id] / 1000.0, 1.0),
            min(mine_near[p.id] / 1000.0, 1.0),
            min(mine_mid[p.id] / 1000.0, 1.0),
            # ── ETA / 궤도 ──
            min_eta_norm,
            pred_x / 100.0,
            pred_y / 100.0,
            # ── 태양 위험도 ──
            sun_block,
            sun_dist_norm,
        ]
    return arr


def encode_fleets(raw_fleets, player):
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet

    fleets = [Fleet(*f) for f in raw_fleets]
    arr = np.zeros((MAX_FLEETS, FLEET_DIM), dtype=np.float32)
    for i, f in enumerate(fleets[:MAX_FLEETS]):
        arr[i] = [
            f.x / 100.0,
            f.y / 100.0,
            math.cos(f.angle),
            math.sin(f.angle),
            min(f.ships / 1000.0, 1.0),
            1.0 if f.owner == player else 0.0,
            1.0 if f.owner not in (-1, player) else 0.0,
        ]
    return arr
