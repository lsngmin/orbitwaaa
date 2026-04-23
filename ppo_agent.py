"""
PPO inference agent for Kaggle submission.

torch 없는 환경에서도 동작 (numpy-only inference).
가중치 파일: weights.npz  (convert_weights.py로 생성)
"""

import os
import math
import numpy as np
from pathlib import Path
from collections import deque

from submission_features import (
    encode_planets, encode_fleets,
    MAX_PLANETS, MAX_FLEETS, PLANET_DIM, FLEET_DIM, HISTORY,
)
from prediction import aim, crosses_sun
from numpy_model import get_action, load_weights

# ── 가중치 로드 ───────────────────────────────────────────────────────────────

def _find_weights():
    base = Path(__file__).parent
    for p in [
        base / "weights.npz",
        Path("/kaggle_simulations/agent/weights.npz"),
    ]:
        if p.exists():
            return p
    raise FileNotFoundError(f"weights.npz not found near {base}")

_weights = load_weights(_find_weights())


# ── History buffer (플레이어별) ───────────────────────────────────────────────

_history: dict = {}

def _get_history(player: int):
    if player not in _history:
        _history[player] = (
            deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY),
            deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY),
        )
    return _history[player]


# ── Action decoder ────────────────────────────────────────────────────────────

def _decode(action_np, raw_planets, av, acting_player):
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

    planets = [Planet(*p) for p in raw_planets]
    moves   = []

    for i, p in enumerate(planets[:MAX_PLANETS]):
        if p.owner != acting_player:
            continue

        if action_np[i, 0] < 0.5:
            continue

        ships_ratio = float(action_np[i, 1])
        target_idx  = int(np.argmax(action_np[i, 2:2 + len(planets)]))
        if target_idx >= len(planets):
            continue

        target = planets[target_idx]
        if target.owner == acting_player:
            continue

        ships_needed = max(1, int(p.ships * ships_ratio))
        ships_needed = min(ships_needed, p.ships)
        if ships_needed <= 0:
            continue

        angle = aim(p, target, av, ships_needed)
        tx = p.x + math.cos(angle) * math.hypot(target.x - p.x, target.y - p.y)
        ty = p.y + math.sin(angle) * math.hypot(target.x - p.x, target.y - p.y)
        if crosses_sun(p.x, p.y, tx, ty):
            continue

        moves.append([p.id, angle, ships_needed])

    return moves


# ── Public agent entry point ──────────────────────────────────────────────────

def ppo_agent(obs):
    if isinstance(obs, dict):
        player      = obs.get("player", 0)
        raw_planets = obs.get("planets", [])
        raw_fleets  = obs.get("fleets", [])
        av          = obs.get("angular_velocity", 0)
        comet_ids   = set(obs.get("comet_planet_ids", []) or [])
    else:
        player      = obs.player
        raw_planets = list(obs.planets)
        raw_fleets  = list(obs.fleets)
        av          = obs.angular_velocity
        comet_ids   = set(obs.comet_planet_ids or [])

    history_p, history_f = _get_history(player)
    history_p.append(encode_planets(raw_planets, raw_fleets, player, comet_ids, av))
    history_f.append(encode_fleets(raw_fleets, player))

    p_hist = np.stack(list(history_p), axis=0)
    f_hist = np.stack(list(history_f), axis=0)
    flat   = np.concatenate([p_hist.flatten(), f_hist.flatten()]).astype(np.float32)

    action = get_action(flat[np.newaxis], _weights)   # (P, ACTION_DIM)
    return _decode(action, raw_planets, av, acting_player=player)
