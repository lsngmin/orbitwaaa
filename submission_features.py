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
PLANET_DIM      = 16
PLANET_FEAT_DIM = 15
FLEET_DIM      = 8
FLEET_FEAT_DIM = 7

ETA_NEAR = 5
ETA_MID = 15


def new_fleet_slot_state():
    """env_wrapper.new_fleet_slot_state 와 동일 — fleet_id → 고정 슬롯."""
    return {
        "fleet_to_slot": {},
        "free_slots": list(range(MAX_FLEETS)),
    }


def update_fleet_slots(fleets, slot_state):
    """env_wrapper.update_fleet_slots 미러. slot_state in-place 갱신."""
    fleet_to_slot = slot_state["fleet_to_slot"]
    free_slots    = slot_state["free_slots"]

    alive_ids = {f.id for f in fleets}
    dead_ids  = set(fleet_to_slot.keys()) - alive_ids
    for fid in dead_ids:
        slot = fleet_to_slot.pop(fid)
        free_slots.append(slot)
    free_slots.sort()

    fid_to_slot     = {}
    newly_allocated = set()
    for f in fleets:
        if f.id in fleet_to_slot:
            fid_to_slot[f.id] = fleet_to_slot[f.id]
            continue
        if not free_slots:
            continue
        slot = free_slots.pop(0)
        fleet_to_slot[f.id] = slot
        fid_to_slot[f.id]   = slot
        newly_allocated.add(slot)
    return fid_to_slot, newly_allocated


def clear_fleet_history_at_slots(fleet_history, slots):
    """env_wrapper.clear_fleet_history_at_slots 미러."""
    if not slots:
        return
    for arr in fleet_history:
        for slot in slots:
            arr[slot] = 0.0
            arr[slot, -1] = -2.0   # empty-slot sentinel (real fleet w/ src miss 는 -1)

def encode_planets(raw_planets, raw_fleets, player, comet_ids, comets=None,
                   angular_velocity=0.0):
    """env_wrapper.encode_planets 와 동일해야 함 (학습/제출 parity).

    velocity (idx 9,10) PR3:
      - comet → engine 의 paths 다음 step 변위 / MAX_SPEED
      - 일반 orbiting → ω 기반 원형 접선 속도 / vel_scale
      - 정적 → 0
    """
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
    from prediction import CENTER_X, CENTER_Y, MAX_SPEED, is_orbiting, estimate_arrival_turn

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    vel_scale = max(1.0, 50.0 * abs(float(angular_velocity)))

    comet_vel = {}
    if comets:
        for group in comets:
            pids   = group.get("planet_ids", []) if isinstance(group, dict) else getattr(group, "planet_ids", [])
            paths  = group.get("paths", [])      if isinstance(group, dict) else getattr(group, "paths", [])
            idx    = group.get("path_index", -1) if isinstance(group, dict) else getattr(group, "path_index", -1)
            if idx < 0:
                continue
            for j, pid in enumerate(pids):
                if j >= len(paths):
                    continue
                p_path = paths[j]
                if idx + 1 >= len(p_path):
                    continue
                cur = p_path[idx]
                nxt = p_path[idx + 1]
                comet_vel[pid] = (float(nxt[0] - cur[0]), float(nxt[1] - cur[1]))

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
        is_comet = (p.id in comet_ids)
        # comet 의 타원 궤도는 sun 거리가 변동 → is_orbiting 이 깜빡임.
        # encoder feature 차원에선 "ω 로 회전 중인 일반 행성" 만 1 로 둔다.
        # (env_wrapper 와 parity)
        orbiting = is_orbiting(p) and not is_comet
        owner_me = 1.0 if p.owner == player else 0.0
        owner_enemy = 1.0 if p.owner not in (-1, player) else 0.0
        owner_neutral = 1.0 if p.owner == -1 else 0.0
        if is_comet:
            cvx, cvy = comet_vel.get(p.id, (0.0, 0.0))
            vx_norm = float(np.clip(cvx / MAX_SPEED, -1.0, 1.0))
            vy_norm = float(np.clip(cvy / MAX_SPEED, -1.0, 1.0))
        elif orbiting:
            vx = -float(angular_velocity) * (p.y - CENTER_Y)
            vy =  float(angular_velocity) * (p.x - CENTER_X)
            vx_norm = float(np.clip(vx / vel_scale, -1.0, 1.0))
            vy_norm = float(np.clip(vy / vel_scale, -1.0, 1.0))
        else:
            vx_norm = 0.0
            vy_norm = 0.0

        arr[i] = [
            p.x / 100.0,
            p.y / 100.0,
            owner_me,
            owner_enemy,
            owner_neutral,
            min(p.ships / 1000.0, 1.0),
            p.production / 5.0,
            1.0 if orbiting else 0.0,
            1.0 if is_comet else 0.0,
            vx_norm,
            vy_norm,
            min(enemy_near[p.id] / 1000.0, 1.0),
            min(enemy_mid[p.id] / 1000.0, 1.0),
            min(mine_near[p.id] / 1000.0, 1.0),
            min(mine_mid[p.id] / 1000.0, 1.0),
            1.0,
        ]
    return arr


def encode_fleets(raw_fleets, raw_planets, player, fid_to_slot=None):
    """env_wrapper.encode_fleets 미러. fid_to_slot 으로 슬롯 안정화."""
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet
    from prediction import fleet_speed, MAX_SPEED

    fleets    = [Fleet(*f)  for f in raw_fleets]
    planets   = [Planet(*p) for p in raw_planets]
    id_to_idx = {p.id: idx for idx, p in enumerate(planets[:MAX_PLANETS])}

    arr = np.zeros((MAX_FLEETS, FLEET_DIM), dtype=np.float32)
    arr[:, -1] = -2.0   # empty-slot sentinel (real fleet 의 lookup miss 는 -1)
    for i, f in enumerate(fleets):
        if fid_to_slot is None:
            if i >= MAX_FLEETS:
                break
            slot = i
        else:
            slot = fid_to_slot.get(f.id)
            if slot is None or slot >= MAX_FLEETS:
                continue
        src_idx = id_to_idx.get(f.from_planet_id, -1)
        speed   = fleet_speed(f.ships)
        vx      = speed * math.cos(f.angle)
        vy      = speed * math.sin(f.angle)
        arr[slot] = [
            f.x / 100.0,
            f.y / 100.0,
            vx / MAX_SPEED,
            vy / MAX_SPEED,
            min(f.ships / 1000.0, 1.0),
            1.0 if f.owner == player else 0.0,
            1.0 if f.owner not in (-1, player) else 0.0,
            float(src_idx),
        ]
    return arr
