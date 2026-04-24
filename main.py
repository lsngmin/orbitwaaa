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


# ── Action masks (학습 analyze_action_space 와 parity) ───────────────────────

def _build_masks(planets, player):
    """학습 analyze_action_space 의 경량 버전.

    동일:
      - target_mask[i,j]: i 가 own + j 가 non-own + crosses_sun 아님
      - launch_mask[i]:   target_mask[i].any()
      - all-false row fallback: target_mask[i,i]=True (Categorical NaN 방지)

    생략: first_collision_on_path (aim 경로가 다른 행성을 먼저 맞을 가능성).
    → 실제 launch 시 resolve_ships_for_capture + crosses_sun 필터로 2차 검증.
      학습 mask 보다 살짝 permissive 하지만 decode 단계에서 걸러짐.
    """
    target_mask = torch.zeros(MAX_PLANETS, MAX_PLANETS, dtype=torch.bool)
    launch_mask = torch.zeros(MAX_PLANETS, dtype=torch.bool)

    for i, p in enumerate(planets[:MAX_PLANETS]):
        if p.owner != player or p.ships <= 0:
            continue
        for j, tp in enumerate(planets[:MAX_PLANETS]):
            if i == j or tp.owner == player:
                continue
            if crosses_sun(p.x, p.y, tp.x, tp.y):
                continue
            target_mask[i, j] = True
        if target_mask[i].any():
            launch_mask[i] = True

    # All-false row fallback — 학습 analyze_action_space 와 동일 처리
    for i in range(MAX_PLANETS):
        if not target_mask[i].any():
            target_mask[i, i] = True

    return launch_mask, target_mask


# ── Action decoder (학습 decode 와 parity: mask + sample) ────────────────────

def _decode(action_logits, raw_planets, av, player):
    """action_logits: (1, MAX_PLANETS, ACTION_DIM) torch on CPU.

    학습 rollout 과 동일 흐름:
      1) mask 계산 (launch/target)
      2) logit 에 mask 적용 (-1e9 masked_fill)
      3) Bernoulli(launch) / Categorical(ships_bin) / Categorical(target) 샘플
      4) launch=1 인 own planet 마다 resolve_ships_for_capture → crosses_sun 필터
    argmax 대신 샘플링 쓰는 이유: 초기 학습 logit 이 0 근처라 argmax 가
    false negative 대량 발생. 학습 rollout 이 샘플링이므로 inference 도 동일.
    """
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
    planets = [Planet(*p) for p in raw_planets]

    launch_mask, target_mask = _build_masks(planets, player)

    launch_logits    = action_logits[0, :, 0].clone()                      # (P,)
    ships_bin_logits = action_logits[0, :, 1:1 + NUM_SHIPS_BINS]           # (P, K)
    target_logits    = action_logits[0, :, 1 + NUM_SHIPS_BINS:].clone()    # (P, P)

    # Mask 적용 — 학습과 동일
    launch_logits = launch_logits.masked_fill(~launch_mask, -1e9)
    target_logits = target_logits.masked_fill(~target_mask, -1e9)

    # Sampling (no_grad 는 caller 에서)
    launch    = torch.distributions.Bernoulli(logits=launch_logits).sample()         # (P,)
    ships_bin = torch.distributions.Categorical(logits=ships_bin_logits).sample()    # (P,)
    target    = torch.distributions.Categorical(logits=target_logits).sample()       # (P,)

    launch_np    = launch.numpy()
    ships_bin_np = ships_bin.numpy()
    target_np    = target.numpy()

    moves = []
    for i, p in enumerate(planets[:MAX_PLANETS]):
        if p.owner != player or p.ships <= 0:
            continue
        if launch_np[i] < 0.5:
            continue

        target_idx = int(target_np[i])
        if target_idx >= len(planets) or target_idx == i:
            continue
        tgt_planet = planets[target_idx]
        if tgt_planet.owner == player:
            continue   # 자기편 샘플 fallback 방어

        multiplier = float(SHIPS_MULTIPLIER_BINS[int(ships_bin_np[i])])

        ships_needed, angle, tx, ty, _turns, _required, _conv = resolve_ships_for_capture(
            p, tgt_planet, av, multiplier, p.ships,
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
        action_logits = model(obs_t).cpu()               # (1, P, ACTION_DIM)

    moves = _decode(action_logits, raw_planets, av, player)

    dt = time.time() - t0
    # 첫 턴은 모델 로드 포함이라 큼. 이후 turn time 이 Kaggle overageTime 소진 페이스.
    print(
        f"[agent] p={player} t={dt:.3f}s planets={len(raw_planets)} "
        f"fleets={len(raw_fleets)} moves={len(moves)}",
        file=sys.stderr,
    )
    return moves
