"""
Kaggle submission entry point for Orbit Wars PPO agent (pure RL).

rule-based 없음 — 모델 forward + 학습과 *동일한* mask/sample/decode 사용:
  encode → forward (torch) → analyze_action_space → masked sample
        → train.decode_action_to_moves
학습-제출 parity 를 위해 inference 도 train.py 의
  - analyze_action_space
  - decode_action_to_moves
를 그대로 재사용한다.

Kaggle 환경:
  - CPU-only, actTimeout=1s/turn, overageTime=60s (누적)
  - 첫 턴 overageTime 여유 있음 → `_load_model()` 은 lazy (첫 agent() 호출 시 로드)
  - stderr 에 turn wall time 찍어서 `kaggle competitions logs` 로 회수 가능
"""

import os
import sys
import time
from collections import deque

import numpy as np
import torch

# Kaggle sandbox 에서 로컬 import 안정화
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from submission_features import (
    encode_planets, encode_fleets,
    new_fleet_slot_state, update_fleet_slots, clear_fleet_history_at_slots,
    MAX_PLANETS, MAX_FLEETS, PLANET_DIM, FLEET_DIM, HISTORY,
)
from submission_actor import OrbitWarsActor, NUM_SHIPS_BINS, SHIPS_MULTIPLIER_BINS
from train import analyze_action_space, decode_action_to_moves


_DEVICE = torch.device("cpu")     # Kaggle = CPU-only
_MODEL  = None
_HIST: dict = {}                  # player_id -> (planet_hist_deque, fleet_hist_deque, slot_state)
_LAST_STEP: dict = {}             # player_id -> 마지막 본 obs.step (game boundary 감지)


# ── Weights loader ───────────────────────────────────────────────────────────

def _weights_path():
    """Kaggle / 로컬 모두 지원. 우선순위:
      1) 환경 변수 ORBIT_WEIGHTS  — 테스트 시 override 용
      2) <이 파일 폴더>/weights.pt
      3) /kaggle_simulations/agent/weights.pt
    """
    env_path = os.environ.get("ORBIT_WEIGHTS")
    if env_path and os.path.exists(env_path):
        return env_path
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "weights.pt"),
        "/kaggle_simulations/agent/weights.pt",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"weights.pt not found. tried: {candidates} (env ORBIT_WEIGHTS={env_path!r})"
    )


def _load_model():
    global _MODEL
    if _MODEL is None:
        t0 = time.time()
        model = OrbitWarsActor().to(_DEVICE)
        ckpt  = torch.load(_weights_path(), map_location=_DEVICE)
        # critic.* 키는 actor 모델에 없으므로 unexpected → strict=False 로 수용.
        result = model.load_state_dict(ckpt, strict=False)
        if result.missing_keys:
            print(
                f"[agent] WARN missing keys: {result.missing_keys[:5]}"
                f" (+{max(0, len(result.missing_keys) - 5)} more)",
                file=sys.stderr,
            )
        model.eval()
        _MODEL = model
        print(f"[agent] model loaded in {time.time() - t0:.3f}s", file=sys.stderr)
    return _MODEL


# ── History buffer (plater 별 분리) ──────────────────────────────────────────

def _fresh_history():
    f_hist = deque(
        [np.zeros((MAX_FLEETS, FLEET_DIM), dtype=np.float32)] * HISTORY,
        maxlen=HISTORY,
    )
    for arr in f_hist:
        arr[:, -1] = -1.0   # fleet sentinel
    return (
        deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY),
        f_hist,
        new_fleet_slot_state(),
    )


def _history(player, current_step):
    """같은 프로세스에서 p0/p1 둘 다 agent 로 불릴 수 있음 → player 별 분리.

    Game boundary 감지: 같은 player 의 obs.step 이 비단조(0 으로 떨어지거나 후퇴)면
    새 게임 시작으로 간주, 이전 게임의 history/slot_state 폐기.
    """
    last = _LAST_STEP.get(player)
    new_game = (last is None) or (current_step is None) or (current_step <= last)
    if new_game or player not in _HIST:
        _HIST[player] = _fresh_history()
    if current_step is not None:
        _LAST_STEP[player] = current_step
    return _HIST[player]


# ── Action sampling (학습 get_action_and_value 와 parity) ────────────────────

def _sample_action(action_logits, analysis):
    """학습 OrbitWarsPolicy.get_action_and_value 와 동일한 mask+sampling.

    action_logits: (1, MAX_PLANETS, ACTION_DIM) CPU tensor
    return: (MAX_PLANETS, ACTION_DIM) numpy
    """
    launch_logits = action_logits[0, :, 0].clone()
    ships_logits  = action_logits[0, :, 1:1 + NUM_SHIPS_BINS]
    target_logits = action_logits[0, :, 1 + NUM_SHIPS_BINS:].clone()

    launch_logits = launch_logits.masked_fill(~analysis.launch_mask, -1e9)
    target_logits = target_logits.masked_fill(~analysis.target_mask, -1e9)

    launch = torch.distributions.Bernoulli(logits=launch_logits).sample()
    ships  = torch.distributions.Categorical(logits=ships_logits).sample()
    target = torch.distributions.Categorical(logits=target_logits).sample()

    ships_onehot = torch.zeros_like(ships_logits)
    ships_onehot.scatter_(-1, ships.unsqueeze(-1), 1.0)
    target_onehot = torch.zeros_like(target_logits)
    target_onehot.scatter_(-1, target.unsqueeze(-1), 1.0)

    action = torch.cat([
        launch.unsqueeze(-1),
        ships_onehot,
        target_onehot,
    ], dim=-1)
    return action.numpy()


# ── Public agent ─────────────────────────────────────────────────────────────

def agent(obs):
    t0 = time.time()

    # obs 는 dict 또는 namedtuple-like 양쪽 지원
    if isinstance(obs, dict):
        player      = obs.get("player", 0)
        raw_planets = obs.get("planets", [])
        raw_fleets  = obs.get("fleets", [])
        av          = obs.get("angular_velocity", 0)
        comet_ids   = set(obs.get("comet_planet_ids", []) or [])
        comets      = obs.get("comets", []) or []
        cur_step    = obs.get("step")
    else:
        player      = obs.player
        raw_planets = list(obs.planets)
        raw_fleets  = list(obs.fleets)
        av          = obs.angular_velocity
        comet_ids   = set(obs.comet_planet_ids or [])
        comets      = getattr(obs, "comets", []) or []
        cur_step    = getattr(obs, "step", None)

    model = _load_model()

    # history 업데이트 (encode 는 학습용 env_wrapper 와 동일한 submission_features 사용)
    p_hist, f_hist, slot_state = _history(player, cur_step)
    p_hist.append(encode_planets(raw_planets, raw_fleets, player, comet_ids, comets, av))

    # slot-stable fleet encoding: 안정 매핑 + 새 슬롯 history 클리어
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet
    fleets_nt = [Fleet(*f) for f in raw_fleets]
    fid_to_slot, newly = update_fleet_slots(fleets_nt, slot_state)
    clear_fleet_history_at_slots(f_hist, newly)
    f_hist.append(encode_fleets(raw_fleets, raw_planets, player, fid_to_slot))

    # 학습 obs 레이아웃: 턴별 interleave (env_wrapper._build_tensor 와 동일).
    p_stack  = np.stack(list(p_hist), axis=0)            # (H, P, PD)
    f_stack  = np.stack(list(f_hist), axis=0)            # (H, F, FD)
    p_flat   = p_stack.reshape(HISTORY, MAX_PLANETS * PLANET_DIM)
    f_flat   = f_stack.reshape(HISTORY, MAX_FLEETS  * FLEET_DIM)
    per_turn = np.concatenate([p_flat, f_flat], axis=1)
    flat     = per_turn.reshape(-1).astype(np.float32)
    obs_t    = torch.from_numpy(flat).unsqueeze(0).to(_DEVICE)  # (1, ...)

    with torch.no_grad():
        action_logits = model(obs_t).cpu()               # (1, P, ACTION_DIM)

    analysis = analyze_action_space(raw_planets, av, acting_player=player)
    action_np = _sample_action(action_logits, analysis)
    moves = decode_action_to_moves(
        action_np, raw_planets, av, acting_player=player, analysis=analysis
    )

    dt = time.time() - t0
    # 첫 턴은 모델 로드 포함이라 큼. 이후 turn time 이 Kaggle overageTime 소진 페이스.
    print(
        f"[agent] p={player} t={dt:.3f}s planets={len(raw_planets)} "
        f"fleets={len(raw_fleets)} moves={len(moves)}",
        file=sys.stderr,
    )
    return moves
