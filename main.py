"""
Kaggle submission entry point for Orbit Wars PPO agent (pure RL).

rule-based 없음 — 모델 forward + 학습과 *동일* decode 사용:
  encode → forward (torch) → argmax → resolve_ships_for_capture → crosses_sun filter
학습-제출 parity 를 위해 decode 내부에서도 학습 코드(train.decode_action_to_moves)
와 같은 `resolve_ships_for_capture` / `crosses_sun` 를 호출.

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
    MAX_PLANETS, MAX_FLEETS, PLANET_DIM, FLEET_DIM, HISTORY,
)
from submission_actor import OrbitWarsActor, NUM_SHIPS_BINS, SHIPS_MULTIPLIER_BINS
from prediction import crosses_sun, resolve_ships_for_capture


_DEVICE = torch.device("cpu")     # Kaggle = CPU-only
_MODEL  = None
_HIST: dict = {}                  # player_id -> (planet_hist_deque, fleet_hist_deque)


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

def _history(player):
    """같은 프로세스에서 p0/p1 둘 다 agent 로 불릴 수 있음 → player 별로 deque 분리."""
    if player not in _HIST:
        _HIST[player] = (
            deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY),
            deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY),
        )
    return _HIST[player]


# ── Action decoder (학습 decode 와 parity) ────────────────────────────────────

def _decode(action_logits_np, raw_planets, av, player):
    """
    action_logits_np: (MAX_PLANETS, ACTION_DIM) numpy, already on CPU.
    학습 decode_action_to_moves 와 같은 필터 순서:
      1) own planet + ships>0 아니면 skip
      2) launch logit ≤ 0 → skip  (Bernoulli threshold 0.5 ≡ logit 0)
      3) target argmax — 자기/아군은 사전 mask (argmax 에서 -inf 처리)
      4) resolve_ships_for_capture → ships_needed
      5) ships_needed <= 0 → skip
      6) crosses_sun → skip
    argmax (deterministic) — sample 대비 안정적.
    """
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
    planets = [Planet(*p) for p in raw_planets]
    moves = []

    # target mask: 자기(i==j) + 아군 행성 → -inf, 나머지는 그대로.
    # crosses_sun 은 resolver 이후에 한 번 더 체크 (target 중심이 아닌 실제 aim 지점 기준).
    n_planets = len(planets)

    for i, p in enumerate(planets[:MAX_PLANETS]):
        if p.owner != player or p.ships <= 0:
            continue

        launch_logit = float(action_logits_np[i, 0])
        if launch_logit <= 0.0:
            continue   # Bernoulli threshold — 학습 시 sample 이지만 inference 는 mode

        # ships bin (K bins 중 argmax)
        ships_bin = int(np.argmax(action_logits_np[i, 1:1 + NUM_SHIPS_BINS]))
        multiplier = float(SHIPS_MULTIPLIER_BINS[ships_bin])

        # target argmax — 아군/자기 는 -inf 로 masking
        tgt_logits = action_logits_np[i, 1 + NUM_SHIPS_BINS:1 + NUM_SHIPS_BINS + n_planets].copy()
        for j, tp in enumerate(planets):
            if i == j or tp.owner == player:
                tgt_logits[j] = -1e9
        if not np.isfinite(tgt_logits).any() or tgt_logits.max() <= -1e8:
            continue   # 전부 masked (단독 생존 등)
        target_idx = int(np.argmax(tgt_logits))
        target = planets[target_idx]

        # 고정점 resolver — 학습과 동일 구현
        ships_needed, angle, tx, ty, _turns, _required, _conv = resolve_ships_for_capture(
            p, target, av, multiplier, p.ships,
        )
        if ships_needed <= 0:
            continue
        if crosses_sun(p.x, p.y, tx, ty):
            continue

        moves.append([p.id, angle, ships_needed])
    return moves


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
    else:
        player      = obs.player
        raw_planets = list(obs.planets)
        raw_fleets  = list(obs.fleets)
        av          = obs.angular_velocity
        comet_ids   = set(obs.comet_planet_ids or [])

    model = _load_model()

    # history 업데이트 (encode 는 학습용 env_wrapper 와 동일한 submission_features 사용)
    p_hist, f_hist = _history(player)
    p_hist.append(encode_planets(raw_planets, raw_fleets, player, comet_ids, av))
    f_hist.append(encode_fleets(raw_fleets, player))

    # 학습 obs 레이아웃: (H * (P*PD + F*FD),) flat
    p_stack = np.stack(list(p_hist), axis=0)              # (H, P, PD)
    f_stack = np.stack(list(f_hist), axis=0)              # (H, F, FD)
    flat    = np.concatenate([p_stack.reshape(-1), f_stack.reshape(-1)]).astype(np.float32)
    obs_t   = torch.from_numpy(flat).unsqueeze(0).to(_DEVICE)   # (1, ...)

    with torch.no_grad():
        action_logits = model(obs_t)                     # (1, P, ACTION_DIM)
    action_logits_np = action_logits[0].cpu().numpy()

    moves = _decode(action_logits_np, raw_planets, av, player)

    dt = time.time() - t0
    # 첫 턴은 모델 로드 포함이라 큼. 이후 turn time 이 Kaggle overageTime 소진 페이스.
    print(
        f"[agent] p={player} t={dt:.3f}s planets={len(raw_planets)} "
        f"fleets={len(raw_fleets)} moves={len(moves)}",
        file=sys.stderr,
    )
    return moves
