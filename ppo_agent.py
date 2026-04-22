"""
PPO inference agent for submission.

전역에서 모델을 한 번 로드하고, 매 turn history buffer를 유지하며
학습된 정책 분포에서 샘플한 action을 그대로 실행합니다.

제출 번들에 필요한 파일:
  ppo_agent.py, submission_model.py, env_wrapper.py, prediction.py, config.yaml,
  mid_run/main_final.pt (또는 WEIGHTS_PATH 경로 수정)
"""

import os
import torch
import numpy as np
from collections import deque

from submission_model import OrbitWarsActor
from env_wrapper import (
    encode_planets, encode_fleets,
    MAX_PLANETS, MAX_FLEETS, PLANET_DIM, FLEET_DIM, HISTORY,
)
from prediction import aim

# ── 가중치 경로 ───────────────────────────────────────────────────────────────

_base = os.path.dirname(os.path.abspath(__file__))

def _find_weights():
    candidates = [
        os.path.join(_base, "checkpoints", "main_final.pt"),
        os.path.join(_base, "checkpoints", "main_latest.pt"),
        os.path.join(_base, "main_final.pt"),
        os.path.join(_base, "main_latest.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No weights found. Tried: {candidates}")

WEIGHTS_PATH = _find_weights()

def _load_model():
    m = OrbitWarsActor()
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    m.load_state_dict(state, strict=False)
    m.eval()
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


# ── Sampled action decode ─────────────────────────────────────────────────────

def _decode_sampled_action(action_np, raw_planets, av, acting_player):
    """샘플된 action_np (MAX_PLANETS, ACTION_DIM) → env moves list.

    모델 샘플은 그대로 사용하되, 액션 유효성만 보장.
    """
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
    from prediction import crosses_sun
    import math

    planets = [Planet(*p) for p in raw_planets]
    moves = []

    for i, p in enumerate(planets[:MAX_PLANETS]):
        # 내 행성에서만 발사 (유효성)
        if p.owner != acting_player:
            continue

        launch = action_np[i, 0]
        if launch < 0.5:
            continue

        ships_ratio = float(action_np[i, 1])
        target_idx  = int(np.argmax(action_np[i, 2:2 + len(planets)]))

        if target_idx >= len(planets):
            continue
        target = planets[target_idx]

        # 자기 행성 공격 금지 (유효성)
        if target.owner == acting_player:
            continue

        # ships 수 보장 (유효성)
        ships_needed = max(1, int(p.ships * ships_ratio))
        ships_needed = min(ships_needed, p.ships)
        if ships_needed <= 0:
            continue

        angle = aim(p, target, av, ships_needed)

        # 태양 충돌 금지 (유효성)
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
        action = model.get_action(obs_t)

    return _decode_sampled_action(action.squeeze(0).cpu().numpy(), raw_planets, av, acting_player=player)
