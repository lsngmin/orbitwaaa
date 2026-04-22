"""
PPO inference agent for submission.

전역에서 모델을 한 번 로드하고, 매 turn history buffer를 유지하며
deterministic greedy action을 반환합니다.

제출 번들에 필요한 파일:
  ppo_agent.py, model.py, env_wrapper.py, prediction.py, config.yaml,
  mid_run/main_final.pt (또는 WEIGHTS_PATH 경로 수정)
"""

import os
import math
import torch
import numpy as np
from collections import deque

from model import OrbitWarsPolicy
from env_wrapper import (
    encode_planets, encode_fleets,
    MAX_PLANETS, MAX_FLEETS, PLANET_DIM, FLEET_DIM, HISTORY,
)
from prediction import aim, crosses_sun

# ── 가중치 경로 ───────────────────────────────────────────────────────────────

_base = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = (
    os.path.join(_base, "mid_run", "main_final.pt")
    if os.path.exists(os.path.join(_base, "mid_run", "main_final.pt"))
    else os.path.join(_base, "main_final.pt")
)

def _load_model():
    m = OrbitWarsPolicy()
    m.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    m.eval()
    # warmup: 첫 forward pass JIT 컴파일을 import 시점에 처리
    _dummy = torch.zeros(1, HISTORY * (MAX_PLANETS * PLANET_DIM + MAX_FLEETS * FLEET_DIM))
    with torch.no_grad():
        m.forward(_dummy)
    return m


_model = _load_model()


def _get_model():
    return _model


# ── History buffer (플레이어별) ───────────────────────────────────────────────

_history: dict = {}


def _get_history(player: int):
    if player not in _history:
        _history[player] = (
            deque(
                [np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY,
                maxlen=HISTORY,
            ),
            deque(
                [np.zeros((MAX_FLEETS, FLEET_DIM), dtype=np.float32)] * HISTORY,
                maxlen=HISTORY,
            ),
        )
    return _history[player]


# ── Deterministic action decode ───────────────────────────────────────────────

def _decode_greedy(action_logits, raw_planets, av, acting_player):
    """action_logits (MAX_PLANETS, ACTION_DIM) → moves list.

    launch  : sigmoid(logit) > 0.5  (no sampling)
    ships   : sigmoid(logit)        (mean of Normal)
    target  : argmax(logits)        (no sampling)
    """
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

    planets = [Planet(*p) for p in raw_planets]
    moves = []

    for i, p in enumerate(planets[:MAX_PLANETS]):
        if p.owner != acting_player:
            continue

        launch_prob = torch.sigmoid(action_logits[i, 0]).item()
        if launch_prob < 0.5:
            continue

        ships_ratio = float(torch.sigmoid(action_logits[i, 1]).clamp(0.0, 1.0).item())
        target_idx  = int(action_logits[i, 2:2 + len(planets)].argmax().item())

        if target_idx >= len(planets) or planets[target_idx].owner == acting_player:
            continue

        target       = planets[target_idx]
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
    """Kaggle submission agent. obs는 dict 또는 namedtuple 모두 지원."""
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
    history_p.append(encode_planets(raw_planets, raw_fleets, player, comet_ids))
    history_f.append(encode_fleets(raw_fleets, player))

    p_hist = np.stack(list(history_p), axis=0)
    f_hist = np.stack(list(history_f), axis=0)
    flat   = np.concatenate([p_hist.flatten(), f_hist.flatten()]).astype(np.float32)
    obs_t  = torch.from_numpy(flat).unsqueeze(0)

    model = _get_model()
    with torch.no_grad():
        action_logits, _ = model.forward(obs_t)

    return _decode_greedy(action_logits.squeeze(0), raw_planets, av, acting_player=player)
