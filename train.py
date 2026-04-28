"""
PPO + League Training for Orbit Wars.

League 구성:
  Main agent      — 지속 학습, league 전체 상대
  Main exploiter  — Main 약점 공략, 주기적 리셋
  League exploiter— 과거 모든 버전 상대

실행:
  python train.py                          # GPU 0, 단일 env
  python train.py --gpu 1 --n-envs 8      # GPU 1, 8개 병렬 env
  python train.py --run-dir checkpoints2  # 별도 체크포인트 디렉토리
"""

import os
import copy
import random
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import TrainingLogger, save_checkpoint, load_checkpoint
from utils.hit_tracker import HitRateTracker
import yaml
from collections import deque, Counter
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
import multiprocessing as mp

from model import OrbitWarsPolicy
from env_wrapper import (
    encode_planets, encode_fleets,
    MAX_PLANETS, MAX_FLEETS, PLANET_DIM, FLEET_DIM, HISTORY,
    SHIPS_SURPLUS_BINS, SHIPS_MULTIPLIERS, SHIPS_BINS, AMOUNT_MODE,
    NUM_SHIPS_BINS, NUM_ACTIONS,
)
from model import TARGET_PAIR_FEAT_DIM, AMOUNT_BIN_FEAT_DIM
from prediction import (
    aim, crosses_sun, first_collision_on_path, PositionCache,
    resolve_ships_for_capture, project_target_at_eta,
)
from mask import MaskContext, build_target_mask, build_action_mask

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_cfg_path) as f:
    CFG = yaml.safe_load(f)

T  = CFG["training"]
SP = CFG["selfplay"]

# opponent_mix 합 검증 — fallthrough 설계 버그(마지막 분기 unconditional) 방지.
# 합이 1 미만이면 남은 확률이 전부 exploiter로 쏠리고, 초과면 exploiter가 squeeze됨.
_mix_sum = sum(SP["opponent_mix"].values())
assert abs(_mix_sum - 1.0) < 1e-3, (
    f"config.yaml opponent_mix 합이 1이 아님: {SP['opponent_mix']} → sum={_mix_sum:.4f}"
)

# 기본값 — __main__ 에서 CLI 인자로 덮어씀
DEVICE   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "checkpoints"
N_ENVS   = 1


# ── League Pool ──────────────────────────────────────────────────────────────

class LeaguePool:
    """과거 체크포인트 관리."""

    def __init__(self, pool_size=SP["pool_size"]):
        self.pool_size = pool_size
        self.agents    = []   # [(path, win_rate, generation)]

    def add(self, model, generation, win_rate=0.0):
        """pool에 snapshot 편입. pool_size 초과 시 가장 오래된 항목 pop + 디스크 파일 삭제.

        A policy (매 gen 편입): win_rate 생략 가능 — eval gate로 쓰지 않음.
        """
        path = os.path.join(SAVE_DIR, f"gen_{generation:04d}.pt")
        torch.save(model.state_dict(), path)
        self.agents.append((path, win_rate, generation))
        while len(self.agents) > self.pool_size:
            old_path, _, _ = self.agents.pop(0)
            if os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except OSError:
                    pass
        print(f"League: gen {generation} 추가 (pool {len(self.agents)}/{self.pool_size})")

    def sample_opponent(self):
        if not self.agents:
            return None
        weights = [a[2] + 1 for a in self.agents]
        path, _, _ = random.choices(self.agents, weights=weights, k=1)[0]
        model = OrbitWarsPolicy().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        return model

    def __len__(self):
        return len(self.agents)


# ── Dense Reward ─────────────────────────────────────────────────────────────

def state_score(raw_obs, player):
    """행성 경제 상태를 단일 스칼라로 요약.

    timeout 승자 판정(total ships)과 더 가깝게 정렬하되,
    production / planet_count는 약한 shaping 힌트로만 유지한다.
    """
    if isinstance(raw_obs, dict):
        planets = raw_obs.get("planets", [])
        fleets  = raw_obs.get("fleets", [])
    else:
        planets = getattr(raw_obs, "planets", [])
        fleets  = getattr(raw_obs, "fleets", [])

    total_ships = total_prod = planet_count = 0.0
    for p in planets:
        owner = p[1] if isinstance(p, (list, tuple)) else p.owner
        ships = p[5] if isinstance(p, (list, tuple)) else p.ships
        prod  = p[6] if isinstance(p, (list, tuple)) else p.production
        if owner == player:
            total_ships  += ships
            total_prod   += prod
            planet_count += 1.0

    for f in fleets:
        owner = f[1] if isinstance(f, (list, tuple)) else f.owner
        ships = f[6] if isinstance(f, (list, tuple)) else f.ships
        if owner == player:
            total_ships += ships

    ship_w   = float(T.get("state_score_ship_weight", 0.01))
    prod_w   = float(T.get("state_score_prod_weight", 1.0))
    planet_w = float(T.get("state_score_planet_weight", 2.0))
    return total_ships * ship_w + total_prod * prod_w + planet_count * planet_w


def _snapshot_planet_owners(raw_obs):
    """env.state observation → {planet_id: (owner, prod)} 딕셔너리.

    env.step() 전에 호출해서 in-place mutation을 방지한다.
    """
    if isinstance(raw_obs, dict):
        planets = raw_obs.get("planets", [])
    else:
        planets = getattr(raw_obs, "planets", [])
    result = {}
    for p in planets:
        pid   = p[0] if isinstance(p, (list, tuple)) else p.id
        owner = p[1] if isinstance(p, (list, tuple)) else p.owner
        prod  = p[6] if isinstance(p, (list, tuple)) else p.production
        result[pid] = (owner, prod)
    return result


def _snapshot_obs_for_resolve(raw_obs):
    """env.step() 직전 obs의 planets/fleets/next_fleet_id를 깊은 복사해 고정.

    resolve_step이 prev_obs로 참조할 용도. env가 in-place mutate하므로 필수.
    """
    if isinstance(raw_obs, dict):
        planets = raw_obs.get("planets", [])
        fleets  = raw_obs.get("fleets", [])
        nfid    = raw_obs.get("next_fleet_id", 0)
    else:
        planets = getattr(raw_obs, "planets", [])
        fleets  = getattr(raw_obs, "fleets", [])
        nfid    = getattr(raw_obs, "next_fleet_id", 0)
    return {
        "planets": [tuple(p) for p in planets],
        "fleets":  [tuple(f) for f in fleets],
        "next_fleet_id": nfid,
    }


def neutral_capture_bonus(prev_map, curr_raw_obs, player, turn_norm=0.0):
    """중립 행성 점령 보너스: production에 비례한 즉각 보상.

    prev_map: _snapshot_planet_owners()로 미리 추출한 {pid: (owner, prod)}.
              env.step() 이후 참조 오염을 피하기 위해 raw_obs 직접 참조 대신 사용.

    계수는 config.yaml의 training.cap_bonus_gain / cap_bonus_loss로 관리.

    early_boost (multiplicative, neutral GAIN 한정):
      cap_bonus_early_multiplier (default 1.0 = off) ≥ 1.0 일 때만 의미 있음.
      early_boost = 1.0 + (multiplier - 1.0) × max(0, 1 - turn_norm)
      → turn_norm=0 (게임 초반) 일 때 multiplier 그대로, turn_norm=1 (말기) 일 때 1.0.
      base bonus 를 대체하지 않고 곱해서 amplify 만 함 (말기에도 base 는 보존).
      enemy capture / own loss 는 boost 안 적용 — 초반 중립 race 만 가속하려는 의도.
    """
    gain_coef = T.get("cap_bonus_gain", 0.05)
    loss_coef = T.get("cap_bonus_loss", 0.025)
    early_mult = float(T.get("cap_bonus_early_multiplier", 1.0))
    tn         = float(max(0.0, min(1.0, turn_norm)))
    early_boost = 1.0 + (early_mult - 1.0) * max(0.0, 1.0 - tn)

    if isinstance(curr_raw_obs, dict):
        curr_planets = curr_raw_obs.get("planets", [])
    else:
        curr_planets = getattr(curr_raw_obs, "planets", [])

    bonus = 0.0
    for p in curr_planets:
        pid   = p[0] if isinstance(p, (list, tuple)) else p.id
        owner = p[1] if isinstance(p, (list, tuple)) else p.owner
        prev_owner, prod = prev_map.get(pid, (-1, 0))
        if prev_owner == -1 and owner == player:        # 중립 → 내 것
            # neutral GAIN 만 early_boost 적용 (multiplicative — base 보존).
            bonus += prod * gain_coef * early_boost
        elif prev_owner == player and owner != player:  # 내 것 → 잃음
            # own loss 는 boost 없음 — phase 무관 일정 페널티.
            bonus -= prod * loss_coef
    return bonus


# ── Action masking ───────────────────────────────────────────────────────────

class ActionSpace:
    """analyze_action_space의 step-local 분석 결과.

    공유 가능한 자원:
      planets   — Planet 객체 리스트 (raw_planets 1회 파싱)
      pos_cache — PositionCache (mask 단계에서 (pid, turn) 미리 채워짐)

    Mask:
      action_mask — (P, NUM_ACTIONS) bool. skip(idx 0) 은 항상 허용,
                    bin(idx 1..K) 은 source viability 따라 마스킹.
      target_mask — (P, P) bool, ships_rep=src.ships 기준 viability
                    (decode는 actual ships로 별도 viability 재확인)
    """

    __slots__ = ("planets", "fleets", "pos_cache", "action_mask", "target_mask",
                 "av", "acting_player", "self_fallback_active",
                 "self_fallback_by_thresh", "active_by_thresh",
                 "mask_block_by_thresh",
                 "pair_features_target", "amount_features_full")

    def __init__(self, planets, fleets, pos_cache, action_mask, target_mask, av, acting_player,
                 self_fallback_active=0,
                 self_fallback_by_thresh=None, active_by_thresh=None,
                 mask_block_by_thresh=None,
                 pair_features_target=None, amount_features_full=None):
        self.planets       = planets
        self.fleets        = fleets
        self.pos_cache     = pos_cache
        self.action_mask   = action_mask
        self.target_mask   = target_mask
        self.av            = av
        self.acting_player = acting_player
        # self_fallback_active: 이 step 의 active source (acting_player 소유 + ships > 0)
        # 중 valid target 이 0 개라 self-fallback 이 적용된 src 수.
        # launchable=False → action_mask 로 skip 강제, 학습 contamination 없음.
        # 빈도 측정만 — 너무 크면 mask viability 보수화 검토 필요.
        self.self_fallback_active = self_fallback_active
        # 진단: src.ships threshold 별 self_fallback / active 분해. 작은 src 가
        # denominator 를 부풀리는 false alarm 인지 진짜 mask 위기인지 구분용.
        # dict {1: int, 5: int, 10: int, 20: int}.
        self.self_fallback_by_thresh = self_fallback_by_thresh or {}
        self.active_by_thresh        = active_by_thresh or {}
        # 진단: mask first-failure 분해 (per src.ships threshold).
        # {gate_name: {1: int, 5: int, 10: int, 20: int}, ...}
        # gate 우선순위: owner_rule → enemy_neutral_filter → sun_path → capacity_short
        # 각 (src, dst) pair 는 첫 실패 게이트에만 카운트됨.
        self.mask_block_by_thresh    = mask_block_by_thresh or {}
        # ── Phase A explicit cost features ───────────────────────────────────
        # pair_features_target: (P, P, TARGET_PAIR_FEAT_DIM=8) float32
        #   ch0~3: capacity-axis (req/src, req/dst, eta_norm, proj/src)
        #   ch4~5: production-axis (log1p(req/prod), log1p(req/(prod×eta)))
        #   ch6:   eta_advantage_norm (race signal)
        #   ch7:   turn_norm (phase)
        #   target_mask 가 False 인 (i,j) 도 텐서엔 들어가지만 logits 자체가
        #   -1e9 로 마스크되므로 무시됨. 학습-제출 parity 위해 항상 채움.
        # amount_features_full: (P, P, NUM_SHIPS_BINS, AMOUNT_BIN_FEAT_DIM=8) float32
        #   per-(src, dst, bin_k) candidate 지표. 모델은 chosen target 으로 gather.
        self.pair_features_target  = pair_features_target
        self.amount_features_full  = amount_features_full


def analyze_action_space(raw_planets, raw_fleets, av, acting_player, turn_norm=0.0):
    """공통 분석 1회 + masks 생성.

    재사용 의도:
      - planets/pos_cache → decode_action_to_moves(analysis=...)에 전달
      - action_mask/target_mask → 모델 forward에 전달

    viability는 ships_rep=src.ships(최대값, permissive)로 평가.
    decode는 actual ships_needed로 fcop를 다시 호출 (속도가 다르면 path도 다름).

    동적 mask (Guard A):
      - in-flight fleet 효과를 ETA 까지 시뮬해 도착 시점 owner 가 acting_player 면
        target 에서 제외 (이미 내 거 될 예정 → 추가 launch 무의미).
        repeat 발사 / 같은 target 중복 commit 을 mask 차원에서 차단.

    Mask first-failure tracking:
      - 게이트 우선순위: owner_rule (i==j or tgt.owner==acting_player)
        → enemy_neutral_filter (Guard A: proj_owner == acting_player)
        → sun_path (crosses_sun OR path collision)
        → capacity_short (src.ships < required)
      - 각 (src, dst) pair 가 첫 실패한 게이트에 1회만 카운트 (중복 X).
      - src.ships threshold 4종 (1/5/10/20) × gate 4종 = 16 카운터.
      - 진짜 mask 위기 (cap_ge20 ↑) vs 게임 룰 (own ↑) 분리 진단용.

    turn_norm:
      - 현재 step / episodeSteps. 0.0 = 초반, 1.0 = 종료.
      - pair_feats[i,j,7] / amount_feats[i,j,k,7] 채널로 전달 → bias_mlp 가
        phase-conditional weighting 자동 학습.
      - default 0.0 (제출 inference 에서 안 넘기면 초반 가정 — 안전한 fallback).
    """
    planets      = [Planet(*p) for p in raw_planets[:MAX_PLANETS]]
    fleets       = [Fleet(*f)  for f in raw_fleets]
    pos_cache    = PositionCache(planets, av)

    # Phase A: explicit cost-feature tensors. mask 가 False 인 (i,j) 위치는
    # 0 으로 둠 (logit -1e9 마스크가 우선이라 영향 없음).
    pair_feats   = torch.zeros(MAX_PLANETS, MAX_PLANETS, TARGET_PAIR_FEAT_DIM,
                                dtype=torch.float32)
    amount_feats = torch.zeros(MAX_PLANETS, MAX_PLANETS, NUM_SHIPS_BINS, AMOUNT_BIN_FEAT_DIM,
                                dtype=torch.float32)
    # 정규화 상수
    ETA_NORM       = 30.0      # 일반 eta 5~30 turn 분포
    RATIO_CLIP     = 5.0       # required/src 등의 발산 방지
    ENEMY_ETA_CAP  = 60.0      # enemy 가 도달 못 하는 dst (sun-path 등) 의 cap.
                                # eta_advantage = (enemy_eta - my_eta) / 30 의 분자에서
                                # +∞ 대신 60 turn 사용 → race 신호가 saturate 되도록.
    tn             = float(max(0.0, min(1.0, turn_norm)))   # phase channel value

    # ── Enemy-side ETA pre-compute (race signal 용) ─────────────────────────
    # 각 dst 에 대해 "현재 살아있는 enemy 어떤 source 든 가장 빨리 보낼 수 있는 ETA".
    # crosses_sun / path collision 이 막힌 경로는 무한대 → ENEMY_ETA_CAP 으로 cap.
    # eta_advantage = (enemy_min_eta - my_eta) / ETA_NORM ∈ [-1, 1] (clamp).
    #   양수 = "내가 적보다 빠름" → 일찍 잡으면 race 우위.
    #   음수 = "적이 더 빠름" → 보내봤자 경합 손실.
    # 비용: P_enemy × P_dst × aim() ≈ 추가 100~400 aim 호출 / step (cached). 무시 가능.
    dst_to_enemy_eta = [ENEMY_ETA_CAP] * len(planets)
    for j, tgt in enumerate(planets):
        if tgt.owner == acting_player:
            # 내 행성은 enemy race 의미 없음 (race signal off → 0 nominal advantage).
            # tgt 가 owner_rule 게이트로 어차피 mask off 라 의미 없지만 shape 유지.
            dst_to_enemy_eta[j] = ENEMY_ETA_CAP
            continue
        best = ENEMY_ETA_CAP
        for src in planets:
            if src.owner == acting_player or src.owner == -1 or src.ships <= 0:
                continue
            if src.id == tgt.id:
                continue
            if crosses_sun(src.x, src.y, tgt.x, tgt.y):
                continue
            ships_rep = int(src.ships)
            _, _, _, e_turns = aim(src, tgt, av, ships_rep, pos_cache=pos_cache)
            if e_turns is None or e_turns <= 0:
                continue
            if e_turns < best:
                best = float(e_turns)
        dst_to_enemy_eta[j] = min(best, ENEMY_ETA_CAP)

    # ── Build masks via mask/ module ────────────────────────────────────────
    # 게이트 우선순위: owner_rule → sun_path → enemy_neutral_filter → capacity_short
    # ctx.scratch 에 aim/proj/required 캐시 저장 → 아래 feature 루프가 재활용.
    ctx = MaskContext(
        planets=planets, fleets=fleets, av=av, acting_player=acting_player,
        pos_cache=pos_cache, num_planets=MAX_PLANETS, num_actions=NUM_ACTIONS,
    )
    target_result = build_target_mask(ctx)
    action_result = build_action_mask(ctx, target_result)

    # ── Mask first-failure decomposition counters ───────────────────────────
    # 4 게이트 × 4 src.ships threshold (1/5/10/20). 각 (src, dst) pair 는 첫 실패
    # 게이트 한 곳에만 카운트 (target_result.blocked_by 가 이미 first-fail idx 보유).
    THRESHOLDS = (1, 5, 10, 20)
    mask_block_by_thresh = {g: {t: 0 for t in THRESHOLDS} for g in target_result.gate_names}
    for i, src in enumerate(planets):
        if src.owner != acting_player or src.ships <= 0:
            continue
        sh = int(src.ships)
        for j in range(len(planets)):
            gate_idx = int(target_result.blocked_by[i, j].item())
            if gate_idx < 0:
                continue
            gate_name = target_result.gate_names[gate_idx]
            for t in THRESHOLDS:
                if sh >= t:
                    mask_block_by_thresh[gate_name][t] += 1

    # ── Pair / amount cost features ─────────────────────────────────────────
    # mask 통과한 (i,j) 만 채움. ctx.scratch 의 aim/proj/required 캐시 재활용.
    # 채널 의미는 pair_feats 0~7 / amount_feats 0~7 모두 동일 (capacity / production /
    # eta_adv / phase). 자세한 채널 정의는 ActionSpace 도크스트링 참조.
    for i in range(len(planets)):
        for j in range(len(planets)):
            if not target_result.mask[i, j]:
                continue
            src = planets[i]
            tgt = planets[j]
            _, _, _, turns          = ctx.scratch[("aim",  i, j)]
            _, proj_ships, eff_turns = ctx.scratch[("proj", i, j)]
            required                = ctx.scratch[("required", i, j)]
            src_ships_safe = max(int(src.ships), 1)
            tgt_ships_safe = max(int(tgt.ships), 1)
            src_prod_safe  = max(float(src.production), 1.0)
            eta_safe       = max(int(eff_turns), 1)
            req_over_src   = min(required / src_ships_safe, RATIO_CLIP)
            req_over_dst   = min(required / tgt_ships_safe, RATIO_CLIP)
            eta_norm       = min(eff_turns / ETA_NORM, 1.5)
            proj_over_src  = min(max(proj_ships, 0.0) / src_ships_safe, RATIO_CLIP)
            req_over_prod      = math.log1p(required / src_prod_safe)
            req_over_prod_eta  = math.log1p(required / (src_prod_safe * eta_safe))
            eta_adv_raw        = (dst_to_enemy_eta[j] - float(eff_turns)) / ETA_NORM
            eta_adv_norm       = max(-1.0, min(1.0, eta_adv_raw))
            pair_feats[i, j, 0] = req_over_src
            pair_feats[i, j, 1] = req_over_dst
            pair_feats[i, j, 2] = eta_norm
            pair_feats[i, j, 3] = proj_over_src
            pair_feats[i, j, 4] = req_over_prod
            pair_feats[i, j, 5] = req_over_prod_eta
            pair_feats[i, j, 6] = eta_adv_norm
            pair_feats[i, j, 7] = tn
            for k, bv in enumerate(SHIPS_BINS):
                if AMOUNT_MODE == "multiplier":
                    raw = required * bv
                    cand = max(1, math.ceil(raw))
                else:
                    surplus = max(src.ships - required, 0)
                    raw = required + bv * surplus
                    cand = max(1, int(round(raw)))
                cand = min(cand, src.ships)
                rem  = max(src.ships - cand, 0)
                cand_over_src = min(cand / src_ships_safe, 1.0)
                rem_over_src  = min(rem  / src_ships_safe, 1.0)
                cand_over_req = min(cand / max(required, 1), RATIO_CLIP)
                cand_over_prod      = math.log1p(cand / src_prod_safe)
                cand_over_prod_eta  = math.log1p(cand / (src_prod_safe * eta_safe))
                amount_feats[i, j, k, 0] = cand_over_src
                amount_feats[i, j, k, 1] = rem_over_src
                amount_feats[i, j, k, 2] = cand_over_req
                amount_feats[i, j, k, 3] = eta_norm
                amount_feats[i, j, k, 4] = cand_over_prod
                amount_feats[i, j, k, 5] = cand_over_prod_eta
                amount_feats[i, j, k, 6] = eta_adv_norm
                amount_feats[i, j, k, 7] = tn

    # ── self_fallback / active 진단 ──────────────────────────────────────────
    # 작은 src(1~2 ships)가 capacity-short 로 거의 항상 막혀 denominator 를 부풀리는지
    # (false alarm) vs 큰 src 도 진짜 막히는지 src.ships threshold 별로 구분.
    self_fallback_active    = 0
    self_fallback_by_thresh = {1: 0, 5: 0, 10: 0, 20: 0}
    active_by_thresh        = {1: 0, 5: 0, 10: 0, 20: 0}
    for i, src in enumerate(planets):
        if src.owner != acting_player or src.ships <= 0:
            continue
        has_target = bool(target_result.mask[i].any())
        if not has_target:
            # valid target 0 개 → self-fallback 대상. action_mask 가 launch 차단하므로
            # 학습 contamination 없음 — 빈도 계측용.
            self_fallback_active += 1
        sh = int(src.ships)
        for thresh in (1, 5, 10, 20):
            if sh >= thresh:
                active_by_thresh[thresh] += 1
                if not has_target:
                    self_fallback_by_thresh[thresh] += 1

    # All-false row fallback: Categorical NaN 방지용 self 허용
    # (launchable[i]=False → action_mask 가 skip 만 허용 → target lp 게이팅으로 학습 영향 없음).
    # raw mask 는 build_action_mask 가 이미 launchable 도출에 사용했으므로 여기서 clone 후 수정.
    target_mask = target_result.mask.clone()
    for i in range(MAX_PLANETS):
        if not target_mask[i].any():
            target_mask[i, i] = True

    return ActionSpace(planets, fleets, pos_cache, action_result.mask, target_mask, av, acting_player,
                       self_fallback_active=self_fallback_active,
                       self_fallback_by_thresh=self_fallback_by_thresh,
                       active_by_thresh=active_by_thresh,
                       mask_block_by_thresh=mask_block_by_thresh,
                       pair_features_target=pair_feats,
                       amount_features_full=amount_feats)


# ── Agent 행동 생성 ───────────────────────────────────────────────────────────

def decode_action_to_moves(action_np, raw_planets, av, acting_player,
                           return_counts=False, analysis=None, turn_norm=0.0):
    """Pure function: 샘플된 action_np → env moves 리스트. 모델 접근 없음.

    acting_player: 절대 owner ID (0 or 1). 행성 소유 판정에 사용.
    return_counts: True면 (moves, counts, launches) 튜플 반환 (hit rate tracking용).
        launches: moves와 1:1 대응하는 메타 dict 리스트 (source_id, target_id,
        ships, angle, start_x, start_y, +진단 전용 src_ships_at_launch / required /
        ships_bin). fleet_id 매핑/resolve_step 입력 + req_over_src/send_frac×bin 진단.
    analysis: ActionSpace. 있으면 planets/pos_cache 재사용 (mask 단계와 공유).
    turn_norm: ep_step / episode_steps_max ∈ [0, 1]. early-neutral 진단 (turn_norm <
        EARLY_PHASE_THRESHOLD = 0.25 이고 target.owner == -1) 에서 nearest_rank /
        eta / req_over_src / eta_advantage 분포를 별도 집계 — “초반에 가까운 저비용
        중립 잡는가” 검증용.
    """
    if analysis is not None:
        planets   = analysis.planets
        pos_cache = analysis.pos_cache
    else:
        planets   = [Planet(*p) for p in raw_planets]
        pos_cache = PositionCache(planets, av)
    moves   = []
    launches = []
    counts  = {"attempts": 0, "filtered_invalid_target": 0,
               "filtered_zero_ships": 0, "filtered_sun": 0,
               "filtered_path": 0, "launched": 0, "launched_high_prod": 0,
               # ── ships 분포 실측 (Categorical surplus-fraction head) ──────
               # chosen_surplus_frac_*: bin 선택값 (0~1). capacity-short 케이스에서는
               #   bin 이 무의미 (ships_needed=src.ships 강제) 라 분모/분자에서 제외 →
               #   bin_effective_count = launched - under_invested 로 정규화.
               # send_fraction_of_src_*: floor 포함 실제 자원 소진률 (ships_sent/src.ships).
               #   bin=0&여유 → required/src, bin=1 OR capacity-short → 1.0.
               "chosen_surplus_frac_sum": 0.0,
               "chosen_surplus_frac_sq_sum": 0.0,
               "bin_effective_count": 0,         # capacity-short 제외 launch 수 (chosen_surplus_frac 분모)
               "send_fraction_of_src_sum": 0.0,
               "send_fraction_of_src_sq_sum": 0.0,
               "ships_to_send_sum": 0,           # 실제 발사 ships 수 평균
               "required_ships_sum": 0.0,        # 필요 병력 추정치 평균
               "send_required_ratio_sum": 0.0,   # ships_to_send / required 평균
               "under_invested_count": 0,        # src.ships < required (capacity short) 횟수
               # ── over-send (다중 source 협조 실패): per-target Σships > required ───
               #   excess_sum    : Σ max(0, total_sent_to_target − required_at_target)
               #   target_count  : 이번 step 에서 over-send 발생한 distinct target 수
               "over_send_excess_sum": 0,
               "over_send_target_count": 0,
               # ── target-type 분리 (neutral=prod 무시 가능 / enemy=prod 회복) ──
               "ships_to_send_sum_neutral": 0,   "ships_to_send_sum_enemy": 0,
               "required_ships_sum_neutral": 0.0, "required_ships_sum_enemy": 0.0,
               "send_required_ratio_sum_neutral": 0.0, "send_required_ratio_sum_enemy": 0.0,
               "under_invested_count_neutral": 0, "under_invested_count_enemy": 0,
               # ── 1차 진단 metric (자원 보존 측정) ──────────────────────────
               "all_in_launches": 0,
               "remaining_ships_after_launch_sum": 0,
               # ── launch cost (continuous) ─────────────────────────────────
               # max(0, req/src - 0.5) 를 누적 → reward 에 곱해서 penalty.
               # 0~0.5 : cheap launch (penalty 0)
               # 0.5~1.0 : 50%~100% src use (penalty 단조↑)
               # 1.0+   : capacity short (penalty cap = req/src 그대로)
               # all_in_penalty (binary 80%) 와 별개 channel — cost feature
               # 와 reward 사이 continuous gradient 다리.
               "launch_cost_excess_sum": 0.0,
               # ── Phase C support (자기 행성 지원 launch) ───────────────────
               # decode 의 target.owner == acting_player 분기. 이전엔 invalid_target
               # 으로 폐기됐던 launch 가 fa784e9 fix 후 정상 처리. neutral/enemy
               # 통계와 분리해서 집계 — launch_to_cap_rate_enemy=1.43 같은 stat
               # pollution 방지.
               "support_launches": 0,
               "support_ships_to_send_sum": 0,
               # ── 초반 중립 확장 진단 (turn_norm < 0.25 AND target.owner == -1) ─
               # "가까운 저비용 중립을 빠르게 먹는가" 검증용. 평균 위주 stat 만
               # 으론 nearest_rank=1 (이상적) 인지 rank=8 (멀고 비싼) 인지 구분 X.
               # 모두 launched (실제 발사된) 기준 — under_invested / filtered 제외.
               "early_neutral_count": 0,                  # 분모
               "early_neutral_eta_sum": 0.0,              # 평균 eta_turns
               "early_neutral_req_over_src_sum": 0.0,     # 평균 cost
               "early_neutral_nearest_rank_sum": 0.0,     # 평균 rank (1=가장 가까운)
               "early_neutral_eta_advantage_sum": 0.0}    # 평균 race signal (>0=내가 빠름)
    # action 5-way 선택 히스토그램: idx 0 = skip, idx 1..K = bin (k-1).
    # ships_bin_hist_k 는 발사된 launch 들의 bin (0..K-1) 분포.
    # action_skip_count 는 skip 선택된 source step 수 (per-source 발사 보류 빈도).
    counts["action_skip_count"] = 0
    for k in range(NUM_SHIPS_BINS):
        counts[f"ships_bin_hist_{k}"] = 0
    target_prods = [t.production for t in planets if t.owner != acting_player]
    high_prod_threshold = np.quantile(target_prods, 0.75) if target_prods else None

    # per-target sends: over-send penalty 산출용 (다중 source 협조 측정)
    target_sends = {}        # target_id -> total ships sent this step
    target_required = {}     # target_id -> required (snapshot at first launch)

    for i, p in enumerate(planets[:MAX_PLANETS]):
        if p.owner != acting_player:
            continue
        # Action layout: [action_5way(NUM_ACTIONS), target_onehot(P)]
        #   action_idx 0 = skip (발사 안 함), 1..K = bin (k-1).
        action_idx = int(np.argmax(action_np[i, :NUM_ACTIONS]))
        target_idx = int(np.argmax(action_np[i, NUM_ACTIONS:
                                              NUM_ACTIONS + len(planets)]))

        if action_idx == 0:
            counts["action_skip_count"] += 1
            continue
        ships_bin = action_idx - 1
        # SHIPS_BINS = SHIPS_MULTIPLIERS (multiplier mode) or SHIPS_SURPLUS_BINS (legacy)
        bin_value = float(SHIPS_BINS[ships_bin])
        counts["attempts"] += 1

        # mask-decoder parity (Phase C 이후):
        #   기존 owner 체크 (target.owner==acting_player → invalid) 는 capture-only
        #   가정. Phase C 가 own-planet support 를 mask 에서 허용하면서 decoder 가
        #   support launch 를 전부 폐기 → mean_filtered_invalid_target 폭증 (0→0.87).
        #
        # 새 규칙:
        #   1) target_idx 범위 밖 → invalid (out-of-bounds)
        #   2) target_idx == i (self-self) → invalid (mask 의 self_fallback sentinel
        #      = "valid target 없음" 신호. 학습 에서 skip 으로 해석되어야 함)
        #   3) analysis 가 있고 target_mask[i, target_idx]==False → invalid (mask 와
        #      sample 의 동기 깨진 케이스. softmax masked_fill 으로 보통 안 일어나나
        #      defensive)
        #   own planet target 인데 mask 가 통과시킨 경우 = Phase C support → 정상 launch.
        #   resolve_ships_for_capture 의 _required_at 가 proj_owner==enemy 만 정상 required
        #   반환하므로 위협 없는 own 은 어차피 ships_needed=0 으로 자연 차단됨.
        if target_idx >= len(planets) or i == target_idx:
            counts["filtered_invalid_target"] += 1
            continue
        if analysis is not None and not bool(analysis.target_mask[i, target_idx]):
            counts["filtered_invalid_target"] += 1
            continue

        target = planets[target_idx]

        # 고정점 반복: surplus formula 로 ships_needed 결정.
        # 동적: fleets/planets 전달 시 in-flight 효과를 ETA forward sim 으로 반영.
        ships_needed, angle, tx, ty, turns, required, _ = resolve_ships_for_capture(
            p, target, av, bin_value, p.ships, pos_cache=pos_cache,
            fleets=(analysis.fleets if analysis is not None else None),
            planets=(planets if analysis is not None else None),
            amount_mode=AMOUNT_MODE,
        )
        # under_invested: src.ships < required → 1-A 가 ships_needed=0 으로 막은 케이스.
        # filtered_zero_ships 와 분리해서 *발사 시도* 시점에 집계 (학습 페널티/지표용).
        # required=0 (target 이 ETA 시점 self-owned) 은 점령 자체 의미 없음 → 제외.
        # Phase C: self-target (support) 는 capture 가 아니므로 neutral/enemy 분기에서 제외.
        if required > 0 and p.ships < required:
            counts["under_invested_count"] += 1
            if target.owner == -1:
                counts["under_invested_count_neutral"] += 1
            elif target.owner != acting_player:  # enemy
                counts["under_invested_count_enemy"] += 1
            # self-target under_invested 는 support_launches 카운트에서 함의됨
        if ships_needed <= 0:
            counts["filtered_zero_ships"] += 1
            continue

        if crosses_sun(p.x, p.y, tx, ty):
            counts["filtered_sun"] += 1
            continue

        max_turns = min((turns or 0) + 2, 120) if turns else 120
        cause, hit_pid = first_collision_on_path(
            p, angle, ships_needed, planets, av, max_turns=max_turns,
            pos_cache=pos_cache,
        )
        if cause != "planet" or hit_pid != target.id:
            counts["filtered_path"] += 1
            continue

        counts["launched"] += 1
        if high_prod_threshold is not None and target.production >= high_prod_threshold:
            counts["launched_high_prod"] += 1

        # ── ships 실측 (launched 기준 집계) ──────────────────────────────────
        # send_required_ratio = ships_needed / required  (실제 공급 비율).
        # under_invested 는 위에서 *발사 시도* 시점에 이미 집계됨 (1-A 가 ships_needed=0
        # 으로 차단했기 때문에 여기 도달하면 항상 src.ships >= required).
        srr = ships_needed / max(required, 1)
        send_frac_src = ships_needed / max(p.ships, 1)
        counts["chosen_surplus_frac_sum"]    += bin_value
        counts["chosen_surplus_frac_sq_sum"] += bin_value ** 2
        counts["bin_effective_count"]        += 1
        counts["send_fraction_of_src_sum"]    += send_frac_src
        counts["send_fraction_of_src_sq_sum"] += send_frac_src ** 2
        counts["ships_to_send_sum"]        += ships_needed
        counts["required_ships_sum"]       += required
        counts["send_required_ratio_sum"]  += srr
        counts[f"ships_bin_hist_{ships_bin}"] += 1
        # target-type 분리 (neutral / enemy / self-support).
        # self-support 는 launch_to_cap_rate / send_required_ratio 정의가 다름
        # (방어 회복 vs 점령) → 별도 카운트, capture 통계 (neutral/enemy) 와 분리.
        if target.owner == -1:
            counts["ships_to_send_sum_neutral"]       += ships_needed
            counts["required_ships_sum_neutral"]      += required
            counts["send_required_ratio_sum_neutral"] += srr
        elif target.owner != acting_player:  # enemy
            counts["ships_to_send_sum_enemy"]       += ships_needed
            counts["required_ships_sum_enemy"]      += required
            counts["send_required_ratio_sum_enemy"] += srr
        else:  # self / Phase C support
            counts["support_launches"]            += 1
            counts["support_ships_to_send_sum"]   += ships_needed
        # per-target 누적 (over-send 산출용 — 같은 target 두 번째 launch 부터는 required 고정)
        target_sends[target.id]   = target_sends.get(target.id, 0) + ships_needed
        target_required.setdefault(target.id, required)

        # ── 1차 진단: all-in / 잔여 ships ─────────────────────────────────
        # all-in: 한 번에 source의 80%+ 비우는 발사 (자원 무시 직접 지표).
        # remaining_ships: 발사 후 source 잔여 (= 방어 reserve / 다음 턴 여력).
        if p.ships > 0 and ships_needed >= HitRateTracker.ALL_IN_THRESHOLD * p.ships:
            counts["all_in_launches"] += 1
        counts["remaining_ships_after_launch_sum"] += max(p.ships - ships_needed, 0)
        # launch cost (continuous): req/src 가 0.5 초과 시 비례 penalty 누적.
        #   threshold 0.5 → 절반 이하 비용은 무료 (정상 launch 인센티브 보존).
        #   excess = max(0, required/src - 0.5)
        #   cheap launch  (req/src=0.4) → 0
        #   moderate      (req/src=0.7) → 0.2
        #   all-in        (req/src=1.0) → 0.5
        #   capacity short(req/src=1.5) → 1.0
        # 이 값을 reward 에서 -coef × Σ 로 곱함 → cost feature 와 reward 의
        # 연속 gradient 다리 (alpha 가 자라야 할 reward signal 생성).
        if p.ships > 0:
            req_over_src_lin = float(required) / float(p.ships)
            counts["launch_cost_excess_sum"] += max(0.0, req_over_src_lin - 0.5)

        # 초반 중립 확장 진단: turn_norm < 0.25 AND target.owner == -1.
        #   nearest_rank: src 에서 launchable 한 모든 중립을 ETA 오름차순 정렬,
        #     선택한 target 의 순위. 1=가장 가까운 중립, 8=8번째로 가까운 중립.
        #   eta_advantage: pair_features ch6, >0 = 적보다 빨리 도달.
        # analysis 가 None 이면 pair_features 접근 불가 → skip.
        EARLY_PHASE_THRESHOLD = 0.25
        if (turn_norm < EARLY_PHASE_THRESHOLD
                and target.owner == -1
                and analysis is not None):
            counts["early_neutral_count"] += 1
            counts["early_neutral_eta_sum"] += float(turns) if turns else 1.0
            counts["early_neutral_req_over_src_sum"] += float(required) / max(p.ships, 1)
            # nearest_rank 산출: src=i 에서 launchable 한 중립들을 eta_norm (ch2) 으로 정렬.
            pair_feats = analysis.pair_features_target  # (P, P, F_t)
            tmask_row = analysis.target_mask[i]          # (P,) bool
            candidates = []
            for j in range(len(planets)):
                if bool(tmask_row[j]) and planets[j].owner == -1:
                    eta_val = float(pair_feats[i, j, 2])  # ch2 = eta_norm
                    candidates.append((j, eta_val))
            candidates.sort(key=lambda x: x[1])
            rank = next((idx + 1 for idx, (j, _) in enumerate(candidates)
                         if j == target_idx), 0)
            if rank > 0:
                counts["early_neutral_nearest_rank_sum"] += rank
            # eta_advantage = pair_features ch6.
            counts["early_neutral_eta_advantage_sum"] += float(pair_feats[i, target_idx, 6])

        moves.append([p.id, angle, ships_needed])
        start_x = p.x + math.cos(angle) * (p.radius + 0.1)
        start_y = p.y + math.sin(angle) * (p.radius + 0.1)
        launches.append({
            "source_id": p.id,
            "target_id": target.id,
            "source_idx": i,                # planets[] index (eta_advantage lookup 용)
            "target_idx": target_idx,       # planets[] index
            "target_owner": target.owner,   # -1: neutral, 그 외: 적 (우리는 self 마스킹됨)
            "ships": ships_needed,
            "angle": angle,
            "start_x": start_x,
            "start_y": start_y,
            # 진단 전용 metadata (CSV logging 용, env 동작 무관):
            #   src_ships_at_launch — launch 시점 source 잔존 ships (req_over_src 분모)
            #   required             — launch 시점 required (req_over_src 분자)
            #   ships_bin            — 0..K-1 (send_frac × bin 매트릭스 분류)
            #   src_prod_at_launch   — production-axis 분모 (req/prod, req/(prod×eta))
            #   eta_turns            — required 산출에 쓰인 turns 와 동일 (정합성)
            "src_ships_at_launch": int(p.ships),
            "required": int(required),
            "ships_bin": int(ships_bin),
            "src_prod_at_launch": float(p.production),
            "eta_turns": int(turns) if turns else 1,
        })

    # over-send 정산: target 별 합산 ships 가 required 초과한 만큼 누적.
    # 다중 source 에서 같은 target 으로 보낸 경우 협조 실패 신호.
    for tid, sent_total in target_sends.items():
        req = target_required.get(tid, 0)
        excess = sent_total - req
        if excess > 0:
            counts["over_send_excess_sum"]   += excess
            counts["over_send_target_count"] += 1

    if return_counts:
        return moves, counts, launches
    return moves


def _opp_moves(opponent_model, obs_tensor, raw_planets, raw_fleets, av, device, turn_norm=0.0):
    """상대(player 1) 행동 생성 (PPO 저장 불필요 — 별도 샘플링 허용).

    turn_norm: 호출자 (rollout) 의 ep_step 기반 phase 신호. 동일 step 의 메인과 동기화.
    """
    if opponent_model is None:
        return []
    analysis = analyze_action_space(raw_planets, raw_fleets, av, acting_player=1,
                                    turn_norm=turn_norm)
    with torch.no_grad():
        action, *_ = opponent_model.get_action_and_value(
            obs_tensor.unsqueeze(0).to(device),
            action_mask=analysis.action_mask.unsqueeze(0).to(device),
            target_mask=analysis.target_mask.unsqueeze(0).to(device),
            target_pair_features=analysis.pair_features_target.unsqueeze(0).to(device),
            amount_features=analysis.amount_features_full.unsqueeze(0).to(device),
        )
    return decode_action_to_moves(
        action.squeeze(0).cpu().numpy(), raw_planets, av,
        acting_player=1, analysis=analysis, turn_norm=turn_norm,
    )


def get_obs_tensor(raw_obs, player, history_p, history_f):
    if isinstance(raw_obs, dict):
        raw_planets = raw_obs.get("planets", [])
        raw_fleets  = raw_obs.get("fleets", [])
        av          = raw_obs.get("angular_velocity", 0)
        comet_ids   = set(raw_obs.get("comet_planet_ids", []) or [])
        comets      = raw_obs.get("comets", []) or []
    else:
        raw_planets = getattr(raw_obs, "planets", [])
        raw_fleets  = getattr(raw_obs, "fleets", [])
        av          = getattr(raw_obs, "angular_velocity", 0)
        comet_ids   = set(getattr(raw_obs, "comet_planet_ids", []) or [])
        comets      = getattr(raw_obs, "comets", []) or []

    history_p.append(encode_planets(raw_planets, raw_fleets, player, comet_ids, comets, av))
    history_f.append(encode_fleets(raw_fleets, raw_planets, player))

    p_hist = np.stack(list(history_p), axis=0)
    f_hist = np.stack(list(history_f), axis=0)
    flat   = np.concatenate([p_hist.flatten(), f_hist.flatten()]).astype(np.float32)
    return torch.from_numpy(flat), raw_planets, raw_fleets, av


# ── 단일 env rollout (CPU/GPU 모두 지원) ─────────────────────────────────────

def _collect_single(main_model, opponent_model, n_steps, device):
    obs_list, act_list, rew_list, done_list, logp_list, val_list = [], [], [], [], [], []
    logp_heads_list = []
    action_mask_list, target_mask_list = [], []
    # Phase A: per-step cost feature buffers.
    #   pair_feat_list: (P, P, F_t) full — evaluate_actions 가 그대로 사용
    #   amount_feat_list: (P, K, F_a) — rollout 의 chosen target 으로 이미 gather 된 형태.
    #     evaluate_actions 시 _gather_amount_features 의 4D pass-through 분기 탐.
    pair_feat_list, amount_feat_list = [], []
    hit_tracker = HitRateTracker()
    sum_dense = sum_cap = sum_terminal = 0.0
    sum_all_in_penalty = 0.0   # Sprint 2: 발사 시 자원 보존 인센티브 (음수 누적)
    sum_over_send_penalty = 0.0   # 다중 source 협조 실패 페널티 (음수 누적)
    sum_under_invested_penalty = 0.0   # capacity-short launch 시도 페널티 (음수 누적)
    sum_launch_cost_penalty = 0.0   # Phase B: per-launch req/src cost penalty (continuous)
    # Step 4 계측: active source (acting_player 소유 + ships>0) 중 valid target=0 인 비율.
    # 학습 영향 없음 (skip 강제) 이지만 mask viability 가 너무 보수/공격적인지 진단.
    sum_self_fallback_active = 0
    sum_active_sources       = 0
    # 진단: src.ships threshold 별 분해 (작은 src 가 false alarm 만드는지 확인용)
    sum_self_fallback_by_thresh = {1: 0, 5: 0, 10: 0, 20: 0}
    sum_active_by_thresh        = {1: 0, 5: 0, 10: 0, 20: 0}
    # 진단: mask first-failure 분해 — 4 게이트 × 4 src.ships threshold (1/5/10/20).
    # gate: owner_rule | enemy_neutral_filter | sun_path | capacity_short
    # 분모는 sum_active_by_thresh (위와 동일) — 동일 step active source 수 기준 rate 산출.
    sum_mask_block_by_thresh = {
        g: {1: 0, 5: 0, 10: 0, 20: 0}
        for g in ("owner_rule", "enemy_neutral_filter", "sun_path", "capacity_short")
    }
    # 진단: per-launch eta_advantage_norm 분포 (race signal). 양수 = 적보다 빨리 도달.
    # 가설: 초반 launch 의 p50/p75 가 양수 → 정책이 race-favorable target 우선 선택.
    eta_adv_launched_list = []
    # 진단: per-launch req/src.ships 분포 (target 비용 측면). p50/p75/p90/p95 산출용.
    # source 깊은 깊이로 비싼 target 을 잡으면 reserve 가 줄어 다음 턴 방어 곤란 → all-in_rate 와 양의 상관.
    req_over_src_list = []
    # 진단: production-axis 분포. req_over_src 가 p90=1.0 으로 포화된 상태에서
    # "회복 가능한 베팅 vs 자살 베팅" 을 가르는 보조 신호.
    #   req_over_prod      = required / max(src.production, 1)         — 경제 비용 (몇 턴어치 생산)
    #   req_over_prod_eta  = required / max(src.production × eta, 1)   — 도착 전 회복 가능성
    # ETA 는 launch 시점 turns 와 동일 (analyze 의 eff_turns 와 의미 동일).
    req_over_prod_list      = []
    req_over_prod_eta_list  = []
    # 진단: send_frac × ships_bin 매트릭스. 각 bin 의 send_frac 분포 (multiplier 모드 동작 확인).
    # bin0 (1.00×) → send_frac ≈ required/src ; bin3 (1.20×) → send_frac ≈ ceil(req·1.20)/src
    send_frac_by_bin = {k: [] for k in range(NUM_SHIPS_BINS)}
    # win_rate 계측: episode 종료 시 main(player=0) reward로 win/draw/loss 집계.
    # draw는 0.5 가중치 (eval과 동일 규약).
    sum_wins = 0.0

    env = make("orbit_wars", debug=False)
    env.reset()
    hit_tracker.reset_episode(env.state[0].observation)
    # episode-level step counter (buffer 의 `step` 과 다름 — 에피소드 내 0..episodeSteps-1).
    # turn_norm = ep_step / episode_steps_max ∈ [0, 1] phase 신호로 모델/reward 에 전달.
    episode_steps_max = int(getattr(env.configuration, "episodeSteps", 500) or 500)
    ep_step = 0

    history_p     = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
    history_f     = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)
    history_p_opp = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
    history_f_opp = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)

    dense_coef = T["dense_reward_coef"]
    terminal_win_reward = float(T.get("terminal_win_reward", 1.0))
    # Sprint 2: 발사 시 source 80%+ 비우는 발사당 페널티 (음수). 0 이면 비활성.
    all_in_penalty_coef = float(T.get("all_in_penalty", 0.0))
    # 다중 source over-send 페널티 (per-excess-ship). 0 이면 비활성.
    over_send_penalty_coef = float(T.get("over_send_penalty", 0.0))
    # under-invested: src.ships < required (capacity short — 점령 수학적 불가).
    # 발사 *시도* 당 페널티. 1-A 가 decode 단에서 ships=0 으로 차단하지만
    # mask 가 못 잡는 edge (target_mask 이후 fixed-point oscillation 등) 를 reward 로 억제.
    under_invested_penalty_coef = float(T.get("under_invested_penalty", 0.0))
    # Phase B: launch cost penalty (continuous). 0 이면 비활성.
    # all_in_penalty (binary 80%) 와 별개 — req/src > 0.5 부터 단조 페널티.
    # cost feature 가 reward 와 묶여 target_bias_alpha 가 자랄 신호 생성.
    launch_cost_penalty_coef = float(T.get("launch_cost_penalty", 0.0))
    prev_score = (state_score(env.state[0].observation, player=0)
                - state_score(env.state[1].observation, player=1))

    # Episode-grained collection:
    #   - 바깥 while: target_steps 이상 채우고 현재 에피소드도 완주했으면 종료
    #   - 안쪽 while: 한 에피소드가 done=True 찍을 때까지 무조건 완주
    #   결과: buffer 마지막 step은 항상 done=True → last_value bootstrap 불필요.
    #   n_steps 파라미터는 "target"으로 해석 (실제 수집량은 ≥ n_steps).
    step = 0
    while True:
        ep_done = False
        while not ep_done:
            raw_obs_main = env.state[0].observation
            obs_t, raw_planets, raw_fleets, av = get_obs_tensor(raw_obs_main, 0, history_p, history_f)

            # turn_norm: 0.0 = 에피소드 시작, 1.0 = 종료. analyze 의 ch7 + bonus early_boost 에 쓰임.
            turn_norm = ep_step / max(episode_steps_max, 1)
            analysis = analyze_action_space(raw_planets, raw_fleets, av, acting_player=0,
                                            turn_norm=turn_norm)
            action_mask, target_mask = analysis.action_mask, analysis.target_mask
            pair_feats   = analysis.pair_features_target            # (P, P, F_t)
            amount_full  = analysis.amount_features_full            # (P, P, K, F_a)
            with torch.no_grad():
                action_t, log_prob, value, lp_heads = main_model.get_action_and_value(
                    obs_t.unsqueeze(0).to(device),
                    action_mask=action_mask.unsqueeze(0).to(device),
                    target_mask=target_mask.unsqueeze(0).to(device),
                    target_pair_features=pair_feats.unsqueeze(0).to(device),
                    amount_features=amount_full.unsqueeze(0).to(device),
                )
            action_np  = action_t.squeeze(0).cpu().numpy()
            moves_main, decode_counts, launches_main = decode_action_to_moves(
                action_np, raw_planets, av, acting_player=0, return_counts=True,
                analysis=analysis, turn_norm=turn_norm,
            )
            hit_tracker.record(decode_counts)

            raw_obs_opp = env.state[1].observation
            obs_opp, raw_planets_opp, raw_fleets_opp, av_opp = get_obs_tensor(raw_obs_opp, 1, history_p_opp, history_f_opp)
            moves_opp = _opp_moves(opponent_model, obs_opp, raw_planets_opp, raw_fleets_opp, av_opp, device,
                                   turn_norm=turn_norm)

            # env.step() 전에 snapshot — in-place mutation 방지 (P2 fix)
            prev_map      = _snapshot_planet_owners(env.state[0].observation)
            prev_obs_snap = _snapshot_obs_for_resolve(env.state[0].observation)
            hit_tracker.register_launches(launches_main, prev_obs_snap["next_fleet_id"])
            env.step([moves_main, moves_opp])
            ep_done = env.done

            curr_obs_main  = env.state[0].observation
            max_speed      = env.configuration.shipSpeed
            hit_tracker.resolve_step(prev_obs_snap, curr_obs_main, max_speed)
            curr_score     = (state_score(curr_obs_main, player=0)
                            - state_score(env.state[1].observation, player=1))
            dense_r        = dense_coef * (curr_score - prev_score)
            cap_bonus      = neutral_capture_bonus(prev_map, curr_obs_main, player=0,
                                                   turn_norm=turn_norm)
            # Sprint 2: 이번 step 의 all-in 발사 수에 비례한 페널티 (decode 시 이미 카운트됨).
            #   penalty = -coef × n_all_in   (coef=0 이면 비활성, Sprint 1 baseline 동일)
            all_in_penalty = -all_in_penalty_coef * decode_counts.get("all_in_launches", 0)
            # over-send: per-target Σships - required 의 양수 초과분에 비례한 페널티.
            #   다중 source 에서 같은 target 에 redundant 발사 시 함선 단위로 줄임.
            over_send_penalty = -over_send_penalty_coef * decode_counts.get("over_send_excess_sum", 0)
            # under-invested: src.ships < required 상태에서의 launch 시도 수에 비례한 페널티.
            under_invested_penalty = -under_invested_penalty_coef * decode_counts.get("under_invested_count", 0)
            # Phase B: launch cost (continuous). max(0, req/src - 0.5) 의 step 합계.
            #   target cost feature 와 reward 사이 직접 다리 — 비싼 target 고를수록 손해.
            launch_cost_penalty = -launch_cost_penalty_coef * decode_counts.get("launch_cost_excess_sum", 0.0)
            terminal_r     = 0.0
            reward         = (dense_r + cap_bonus + all_in_penalty + over_send_penalty
                              + under_invested_penalty + launch_cost_penalty)
            prev_score     = curr_score

            if ep_done:
                r          = env.state[0].reward
                terminal_r = terminal_win_reward if r == 1 else (
                    -terminal_win_reward if r == -1 else 0.0
                )
                reward    += terminal_r
                sum_wins  += 1.0 if r == 1 else (0.5 if r == 0 else 0.0)

            sum_dense    += dense_r
            sum_cap      += cap_bonus
            sum_all_in_penalty += all_in_penalty
            sum_over_send_penalty += over_send_penalty
            sum_under_invested_penalty += under_invested_penalty
            sum_launch_cost_penalty += launch_cost_penalty
            sum_terminal += terminal_r
            # Step 4 계측: 이 step 의 self_fallback 빈도. active_sources 는 분모 (acting
            # player 소유 + ships>0 인 src 수) — 매 step launchable[i] 결정 후 알 수 있음.
            sum_self_fallback_active += analysis.self_fallback_active
            sum_active_sources       += int(analysis.action_mask[:, 1:].any(dim=-1).sum().item()) \
                                        + analysis.self_fallback_active
            # 진단: threshold-bucketed self_fallback / active. analysis 가 dict 로 보관.
            for thresh in (1, 5, 10, 20):
                sum_self_fallback_by_thresh[thresh] += analysis.self_fallback_by_thresh.get(thresh, 0)
                sum_active_by_thresh[thresh]        += analysis.active_by_thresh.get(thresh, 0)
            # 진단: mask first-failure 분해 (4 게이트 × 4 threshold). 각 (src,dst) pair 가
            # 첫 실패 게이트에만 카운트되어 들어옴 (analyze_action_space 가 보장).
            mb_bt = analysis.mask_block_by_thresh
            for g in ("owner_rule", "enemy_neutral_filter", "sun_path", "capacity_short"):
                gate_dict = mb_bt.get(g, {})
                for thresh in (1, 5, 10, 20):
                    sum_mask_block_by_thresh[g][thresh] += gate_dict.get(thresh, 0)
            # 진단: per-launch req/src.ships 와 send_frac × bin. launches_main 에서 추출.
            # prod-axis 도 같은 launch metadata 에서 산출 (분모는 src.production / src.production×ETA).
            pair_feats_step = analysis.pair_features_target  # (P,P,F_t) — ch6 = eta_adv_norm
            for lc in launches_main:
                src_sh   = max(int(lc["src_ships_at_launch"]), 1)
                src_prod = max(float(lc["src_prod_at_launch"]), 1.0)
                eta_t    = max(int(lc["eta_turns"]), 1)
                req      = int(lc["required"])
                req_over_src_list.append(req / src_sh)
                req_over_prod_list.append(req / src_prod)
                req_over_prod_eta_list.append(req / (src_prod * eta_t))
                bk = int(lc["ships_bin"])
                if 0 <= bk < NUM_SHIPS_BINS:
                    send_frac_by_bin[bk].append(int(lc["ships"]) / src_sh)
                # eta_advantage_norm (race signal). pair_feats[i,j,6] 직접 참조 — analyze 와 정합.
                si = int(lc.get("source_idx", -1))
                ti = int(lc.get("target_idx", -1))
                if 0 <= si < MAX_PLANETS and 0 <= ti < MAX_PLANETS:
                    eta_adv_launched_list.append(float(pair_feats_step[si, ti, 6].item()))

            obs_list.append(obs_t)
            act_list.append(action_t.squeeze(0).cpu())
            rew_list.append(torch.tensor(reward, dtype=torch.float32))
            done_list.append(torch.tensor(float(ep_done), dtype=torch.float32))
            logp_list.append(log_prob.squeeze(0).cpu())
            logp_heads_list.append(lp_heads.squeeze(0).cpu())
            val_list.append(value.squeeze(0).cpu())
            action_mask_list.append(action_mask)
            target_mask_list.append(target_mask)
            # Phase A: cost features 저장.
            # pair_feats 는 full (P,P,F_t) 그대로 — evaluate_actions 가 사용.
            # amount_feats 는 rollout 에서 sample 된 chosen target 으로 미리 gather:
            #   (P, P, K, F_a) → (P, K, F_a). evaluate 시 _gather_amount_features 4D
            #   pass-through 으로 그대로 들어감.
            chosen_target = action_t[0, :, NUM_ACTIONS:].argmax(dim=-1).cpu()  # (P,) long
            gather_idx_5d = chosen_target.view(MAX_PLANETS, 1, 1, 1).expand(
                MAX_PLANETS, 1, NUM_SHIPS_BINS, AMOUNT_BIN_FEAT_DIM
            )
            amount_gathered = amount_full.gather(1, gather_idx_5d).squeeze(1)  # (P, K, F_a)
            pair_feat_list.append(pair_feats)
            amount_feat_list.append(amount_gathered)

            step += 1
            ep_step += 1

        # 에피소드 완주 후 target 도달 여부 확인
        if step >= n_steps:
            break

        # 다음 에피소드 준비 (env/history 리셋)
        env = make("orbit_wars", debug=False)
        env.reset()
        hit_tracker.reset_episode(env.state[0].observation)
        episode_steps_max = int(getattr(env.configuration, "episodeSteps", 500) or 500)
        ep_step       = 0
        prev_score    = (state_score(env.state[0].observation, player=0)
                       - state_score(env.state[1].observation, player=1))
        history_p     = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
        history_f     = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)
        history_p_opp = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
        history_f_opp = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)

    # Buffer 마지막 step은 항상 done=True (episode-grained collection).
    # → bootstrap 불필요, last_value = 0.0 고정.
    last_value = torch.tensor(0.0, dtype=torch.float32)

    rewards = torch.stack(rew_list)
    dones   = torch.stack(done_list)
    values  = torch.stack(val_list)
    # Full-episode collection 불변식: buffer 마지막 step은 항상 episode 끝.
    assert dones[-1].item() == 1.0, (
        "full-episode collection violated: buffer 마지막 step이 done=False. "
        "rollout loop 구조 확인 필요."
    )
    advantages, returns = compute_gae(rewards, dones, values, last_value=last_value)

    # Raw state 반환 (collect_rollout에서 worker 합산 후 한 번에 normalize).
    # worker별 episode 길이가 달라도 합산된 raw counters/step/episode로
    # 분모를 정확히 반영해 편향 없는 metric 산출.
    raw_stats = {
        "counters":     dict(hit_tracker.counters),
        "n_steps":      hit_tracker.n_steps,
        "episodes":     hit_tracker.episodes,
        "sum_dense":    sum_dense,
        "sum_cap":      sum_cap,
        "sum_all_in_penalty": sum_all_in_penalty,
        "sum_over_send_penalty": sum_over_send_penalty,
        "sum_under_invested_penalty": sum_under_invested_penalty,
        "sum_launch_cost_penalty": sum_launch_cost_penalty,
        "sum_terminal": sum_terminal,
        "sum_wins":     sum_wins,
        "sum_self_fallback_active": sum_self_fallback_active,
        "sum_active_sources":       sum_active_sources,
        # 진단: threshold-bucketed (1/5/10/20) self_fallback / active 합산.
        "sum_self_fallback_by_thresh": dict(sum_self_fallback_by_thresh),
        "sum_active_by_thresh":        dict(sum_active_by_thresh),
        # 진단: mask first-failure 분해 (4 게이트 × 4 threshold).
        "sum_mask_block_by_thresh": {
            g: dict(t) for g, t in sum_mask_block_by_thresh.items()
        },
        # 진단: per-launch raw values (worker 합산 후 분포 산출용).
        "req_over_src_list":      req_over_src_list,
        "req_over_prod_list":     req_over_prod_list,
        "req_over_prod_eta_list": req_over_prod_eta_list,
        "eta_adv_launched_list":  eta_adv_launched_list,
        "send_frac_by_bin":  {k: list(v) for k, v in send_frac_by_bin.items()},
    }
    return (
        torch.stack(obs_list),
        torch.stack(act_list),
        advantages,
        returns,
        torch.stack(logp_list),
        torch.stack(logp_heads_list),
        torch.stack(action_mask_list),
        torch.stack(target_mask_list),
        torch.stack(pair_feat_list),     # (T, P, P, F_t)
        torch.stack(amount_feat_list),   # (T, P, K, F_a) — chosen target 으로 gather 됨
        raw_stats,   # ← normalize는 collect_rollout에서 합산 후 수행
    )


# ── 병렬 rollout worker (별도 프로세스, CPU 전용) ───────────────────────────

def _init_worker():
    """worker 프로세스 초기화: CUDA 숨기기, 스레드 수 제한, 로그 억제."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # 1 thread/worker: n_envs × 1 = CPU core 수에 맞춤 (oversubscription 회피)
    torch.set_num_threads(1)
    import logging
    logging.getLogger("kaggle_environments").setLevel(logging.ERROR)


def _rollout_worker(args):
    """spawn된 자식 프로세스에서 실행. GPU 없이 CPU만 사용."""
    main_state, opp_state, n_steps = args
    cpu = torch.device("cpu")

    try:
        main_model = OrbitWarsPolicy().to(cpu)
        main_model.load_state_dict(main_state)
        main_model.eval()

        opponent_model = None
        if opp_state is not None:
            opponent_model = OrbitWarsPolicy().to(cpu)
            opponent_model.load_state_dict(opp_state)
            opponent_model.eval()

        return _collect_single(main_model, opponent_model, n_steps, cpu)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise


def _finalize_reward_stats(raw_list):
    """Worker별 raw_stats 리스트를 합산해서 normalized reward_stats dict 구성.

    단순 worker 평균은 episode 길이가 worker마다 다를 때 편향되므로
    raw counters/n_steps/episodes/sum_* 을 먼저 합산하고, 그 합산된 분모로
    한 번에 정규화한다 (step-weighted / episode-weighted / launched-weighted
    모두 HitRateTracker.summary_from_counters가 올바르게 처리).
    """
    from collections import defaultdict
    total_counters = defaultdict(float)
    total_n_steps = 0
    total_episodes = 0
    total_dense = total_cap = total_terminal = 0.0
    total_all_in_penalty = 0.0
    total_over_send_penalty = 0.0
    total_under_invested_penalty = 0.0
    total_launch_cost_penalty = 0.0
    total_wins = 0.0
    total_self_fallback_active = 0
    total_active_sources       = 0
    total_self_fallback_by_thresh = {1: 0, 5: 0, 10: 0, 20: 0}
    total_active_by_thresh        = {1: 0, 5: 0, 10: 0, 20: 0}
    total_mask_block_by_thresh    = {
        g: {1: 0, 5: 0, 10: 0, 20: 0}
        for g in ("owner_rule", "enemy_neutral_filter", "sun_path", "capacity_short")
    }
    total_req_over_src_list      = []
    total_req_over_prod_list     = []
    total_req_over_prod_eta_list = []
    total_eta_adv_launched_list  = []
    total_send_frac_by_bin       = {k: [] for k in range(NUM_SHIPS_BINS)}
    for r in raw_list:
        for k, v in r["counters"].items():
            total_counters[k] += v
        total_n_steps  += r["n_steps"]
        total_episodes += r["episodes"]
        total_dense    += r["sum_dense"]
        total_cap      += r["sum_cap"]
        total_all_in_penalty += r.get("sum_all_in_penalty", 0.0)
        total_over_send_penalty += r.get("sum_over_send_penalty", 0.0)
        total_under_invested_penalty += r.get("sum_under_invested_penalty", 0.0)
        total_launch_cost_penalty += r.get("sum_launch_cost_penalty", 0.0)
        total_terminal += r["sum_terminal"]
        total_wins     += r.get("sum_wins", 0.0)
        total_self_fallback_active += r.get("sum_self_fallback_active", 0)
        total_active_sources       += r.get("sum_active_sources", 0)
        # 진단: threshold-bucketed
        sf_bt = r.get("sum_self_fallback_by_thresh", {})
        ac_bt = r.get("sum_active_by_thresh",        {})
        for thresh in (1, 5, 10, 20):
            total_self_fallback_by_thresh[thresh] += sf_bt.get(thresh, 0)
            total_active_by_thresh[thresh]        += ac_bt.get(thresh, 0)
        # 진단: mask first-failure (gate × threshold) — worker 단순 합산.
        mb_bt = r.get("sum_mask_block_by_thresh", {})
        for g in ("owner_rule", "enemy_neutral_filter", "sun_path", "capacity_short"):
            gd = mb_bt.get(g, {})
            for thresh in (1, 5, 10, 20):
                total_mask_block_by_thresh[g][thresh] += gd.get(thresh, 0)
        # 진단: per-launch lists 합산
        total_req_over_src_list.extend(r.get("req_over_src_list", []))
        total_req_over_prod_list.extend(r.get("req_over_prod_list", []))
        total_req_over_prod_eta_list.extend(r.get("req_over_prod_eta_list", []))
        total_eta_adv_launched_list.extend(r.get("eta_adv_launched_list", []))
        sfb = r.get("send_frac_by_bin", {})
        for k in range(NUM_SHIPS_BINS):
            total_send_frac_by_bin[k].extend(sfb.get(k, []))

    stats = HitRateTracker.summary_from_counters(
        total_counters, total_n_steps, total_episodes,
    )
    steps_safe = max(total_n_steps, 1)
    stats["mean_dense"]    = total_dense    / steps_safe
    stats["mean_cap"]      = total_cap      / steps_safe
    stats["mean_terminal"] = total_terminal / steps_safe
    # Sprint 2: per-step all-in launch penalty (음수). all_in_penalty=0 이면 0.
    stats["mean_all_in_penalty"] = total_all_in_penalty / steps_safe
    # 다중 source over-send 페널티 (음수). over_send_penalty=0 이면 0.
    stats["mean_over_send_penalty"] = total_over_send_penalty / steps_safe
    # capacity-short launch 시도 페널티 (음수). under_invested_penalty=0 이면 0.
    stats["mean_under_invested_penalty"] = total_under_invested_penalty / steps_safe
    # Phase B: per-step launch cost penalty (음수). launch_cost_penalty=0 이면 0.
    stats["mean_launch_cost_penalty"] = total_launch_cost_penalty / steps_safe
    # Step 4 계측: active source (acting_player 소유 + ships>0) 중 valid target=0 인
    # source 의 비율. 학습 contamination 없음 (skip 강제) 이지만 mask viability 진단:
    #   - <1% : 거의 무시 가능
    #   - 1-5% : 노이즈 가능, 모니터링
    #   - >5% : mask 가 너무 보수적, ships_rep / Guard A,B 재검토
    stats["self_fallback_active_rate"] = (
        total_self_fallback_active / max(total_active_sources, 1)
    )
    # 진단 A: src.ships threshold 별 self_fallback_active_rate 분해.
    # 가설: 전체 86% 가 작은 src(1~2 ships) 의 false alarm 인지 vs 큰 src 도 진짜 막히는지 구분.
    # 만약 ge20 에서 5~10% 이하면 → 작은 src 영향 (false alarm). 반대로 ge20 이 여전히 30%+ →
    # mask 가 진짜 보수적 → ships_rep / Guard A,B 재검토.
    for thresh in (1, 5, 10, 20):
        stats[f"self_fallback_active_rate_ge{thresh}"] = (
            total_self_fallback_by_thresh[thresh] / max(total_active_by_thresh[thresh], 1)
        )
    # 진단 A': mask first-failure 분해 — gate × threshold 별 차단 rate.
    # 분모: 같은 threshold 의 active_by_thresh 합 (acting_player 소유 + ships≥thresh src 수).
    # 한 src 가 P 개 dst 와 비교되니 분모×P 가 (src,dst) pair 총수지만,
    # 분모를 active_src 로 두면 "src 당 평균 막힌 dst 수" 의미가 됨 — 모니터링 친숙.
    # 가설:
    #   - cap_short_ge20 ↑↑ : 진짜 mask 위기 (큰 src 가 비싼 target 에 막혀 idle).
    #   - own_ge20 ↑       : 게임 룰 (대부분의 행성이 acting_player 소유 — 후반).
    #   - enemy_neutral_filter ↑ : Guard A 가 in-flight 중복 방지로 자주 컷.
    #   - sun_path ↑       : 사선/태양 차폐 — geometric (mask issue 아님).
    for g in ("owner_rule", "enemy_neutral_filter", "sun_path", "capacity_short"):
        for thresh in (1, 5, 10, 20):
            stats[f"mask_block_{g}_ge{thresh}"] = (
                total_mask_block_by_thresh[g][thresh]
                / max(total_active_by_thresh[thresh], 1)
            )
    # 진단 B: per-launch req/src.ships 분포 (target 비용 측면).
    # 가설: send_required_ratio≈1.17 / send_fraction≈0.80 → req/src ≈ 0.68. 너무 비싼 target →
    # 발사 후 reserve 부족 → 다음 턴 방어 곤란. p90/p95 가 높으면 target 선택이 한계 비용에 몰림.
    if total_req_over_src_list:
        arr = np.asarray(total_req_over_src_list, dtype=np.float64)
        stats["req_over_src_launched_mean"] = float(arr.mean())
        for q, name in [(50, "p50"), (75, "p75"), (90, "p90"), (95, "p95")]:
            stats[f"req_over_src_launched_{name}"] = float(np.percentile(arr, q))
    else:
        stats["req_over_src_launched_mean"] = 0.0
        for name in ("p50", "p75", "p90", "p95"):
            stats[f"req_over_src_launched_{name}"] = 0.0
    # 진단 B': production-axis (req/prod, req/(prod×ETA)) 분포.
    # req_over_src 가 p90=1.0 으로 saturated 일 때, prod-axis 가 "회복 가능 vs 자살" 가른다.
    #   req_over_prod      높음 → 경제적으로 비싼 commit (생산량 많이 태움).
    #   req_over_prod_eta  높음 → 도착 전까지 회복 불가능 (장기 손실 확정).
    if total_req_over_prod_list:
        arr_p = np.asarray(total_req_over_prod_list, dtype=np.float64)
        stats["req_over_prod_launched_mean"] = float(arr_p.mean())
        for q, name in [(50, "p50"), (75, "p75"), (90, "p90"), (95, "p95")]:
            stats[f"req_over_prod_launched_{name}"] = float(np.percentile(arr_p, q))
    else:
        stats["req_over_prod_launched_mean"] = 0.0
        for name in ("p50", "p75", "p90", "p95"):
            stats[f"req_over_prod_launched_{name}"] = 0.0
    if total_req_over_prod_eta_list:
        arr_pe = np.asarray(total_req_over_prod_eta_list, dtype=np.float64)
        stats["req_over_prod_eta_launched_mean"] = float(arr_pe.mean())
        for q, name in [(50, "p50"), (75, "p75"), (90, "p90"), (95, "p95")]:
            stats[f"req_over_prod_eta_launched_{name}"] = float(np.percentile(arr_pe, q))
    else:
        stats["req_over_prod_eta_launched_mean"] = 0.0
        for name in ("p50", "p75", "p90", "p95"):
            stats[f"req_over_prod_eta_launched_{name}"] = 0.0
    # 진단 C: per-launch eta_advantage_norm 분포 (race signal).
    # 양수 = 적보다 빨리 도달 (early neutral race 우위), 음수 = 늦음 (보내봤자 경합 손실).
    # 가설: 정책이 race-favorable target 선호 시 launched p50/p75 → 양수 쪽으로 shift.
    if total_eta_adv_launched_list:
        arr_e = np.asarray(total_eta_adv_launched_list, dtype=np.float64)
        stats["eta_advantage_launched_mean"] = float(arr_e.mean())
        for q, name in [(50, "p50"), (75, "p75"), (90, "p90"), (95, "p95")]:
            stats[f"eta_advantage_launched_{name}"] = float(np.percentile(arr_e, q))
    else:
        stats["eta_advantage_launched_mean"] = 0.0
        for name in ("p50", "p75", "p90", "p95"):
            stats[f"eta_advantage_launched_{name}"] = 0.0
    # 진단 D: send_frac × ships_bin 분포 (multiplier 모드 동작 확인).
    # bin0 (1.00×) → send_frac ≈ required/src ; bin3 (1.20×) → send_frac ≈ ceil(req·1.20)/src.
    # bin0 p90 가 0.5+ 면 작은 src 가 just-capture 도 비싼 상태. bin3 p90 가 1.0 근접 →
    # over-send 가 over_send_penalty 로 충분히 억제됐는지 사후 확인.
    for k in range(NUM_SHIPS_BINS):
        bin_list = total_send_frac_by_bin[k]
        if bin_list:
            barr = np.asarray(bin_list, dtype=np.float64)
            stats[f"send_frac_bin{k}_mean"] = float(barr.mean())
            stats[f"send_frac_bin{k}_count"] = int(len(bin_list))
            # bin0/bin3 만 p90 추적 (양 극단). 중간은 mean 만.
            if k == 0 or k == NUM_SHIPS_BINS - 1:
                stats[f"send_frac_bin{k}_p90"] = float(np.percentile(barr, 90))
        else:
            stats[f"send_frac_bin{k}_mean"]  = 0.0
            stats[f"send_frac_bin{k}_count"] = 0
            if k == 0 or k == NUM_SHIPS_BINS - 1:
                stats[f"send_frac_bin{k}_p90"] = 0.0
    # rollout 내 main(player=0) 승률. on-policy sampling이라 eval보다 noisy지만,
    # match_type별 log로 분포별 성능을 바로 볼 수 있는 이점.
    stats["win_rate"]      = total_wins / max(total_episodes, 1)
    # Episode-grained collection에서 generation마다 실제 수집량이 달라지므로
    # target_steps 대비 초과량으로 "unusually 길었던 generation" 감지 가능.
    stats["actual_steps"]  = total_n_steps
    return stats


def collect_rollout(main_model, opponent_model, n_steps=512, n_envs=1, pool=None,
                    opp_states=None):
    """n_envs=1이면 단일 env, 그 이상이면 multiprocessing 병렬 rollout.

    opp_states: optional list[state_dict_cpu | None] of length n_envs. If provided,
                each worker receives its own opponent — enables per-env opponent
                mixing (self / league / exploiter all in one rollout batch).
                When None, opponent_model is shared across all workers (legacy /
                exploiter rollout / single opponent eval).

    _collect_single은 raw_stats(counters/n_steps/episodes/sum_*)를 반환.
    여기서 worker들을 합산한 뒤 step-weighted로 정확히 정규화해서
    최종 reward_stats를 만든다 (full-episode collection에서 worker별
    episode 길이가 달라도 편향 없음).
    """
    if n_envs <= 1 or pool is None:
        if opp_states is not None:
            opp_state = opp_states[0]
            opp = None
            if opp_state is not None:
                opp = OrbitWarsPolicy().to(DEVICE)
                opp.load_state_dict(opp_state)
                opp.eval()
        else:
            opp = opponent_model
        result = _collect_single(main_model, opp, n_steps, DEVICE)
        # tuple layout: 0=obs, 1=act, 2=adv, 3=ret, 4=logp, 5=logp_heads,
        #               6=action_mask, 7=target_mask, 8=pair_feats, 9=amount_feats, 10=raw_stats
        merged_stats = _finalize_reward_stats([result[10]])
        return (*result[:10], merged_stats)

    main_state = {k: v.cpu() for k, v in main_model.state_dict().items()}

    if opp_states is None:
        opp_state = ({k: v.cpu() for k, v in opponent_model.state_dict().items()}
                     if opponent_model is not None else None)
        opp_states = [opp_state] * n_envs
    elif len(opp_states) != n_envs:
        raise ValueError(
            f"opp_states length {len(opp_states)} != n_envs {n_envs}"
        )

    steps_per_env = max(1, n_steps // n_envs)
    worker_args   = [(main_state, opp_states[i], steps_per_env) for i in range(n_envs)]

    results = pool.map(_rollout_worker, worker_args)

    merged_stats = _finalize_reward_stats([r[10] for r in results])
    return (
        torch.cat([r[0] for r in results]),
        torch.cat([r[1] for r in results]),
        torch.cat([r[2] for r in results]),
        torch.cat([r[3] for r in results]),
        torch.cat([r[4] for r in results]),
        torch.cat([r[5] for r in results]),
        torch.cat([r[6] for r in results]),
        torch.cat([r[7] for r in results]),
        torch.cat([r[8] for r in results]),
        torch.cat([r[9] for r in results]),
        merged_stats,
    )


# ── Phase 기반 self-play 비율 ─────────────────────────────────────────────────

def _sample_opponent(main_model, exploiter, league):
    """opponent_mix 비율로 self/league/exploiter 중 하나 샘플.

    league.sample_opponent()이 None(비어 있음)이면 그 턴의 league 몫을 self로
    재할당 — 초반 pool 빌 때 분포가 main 자신과 완전히 동떨어지지 않도록.
    fallback은 "self"와 구분되는 별도 라벨("self_fallback")로 기록해
    "초반 self 비율이 높다"는 오해를 방지한다.
    반환: (opponent_model, match_type) — match_type ∈ {self, self_fallback, league, exploiter}.
    """
    mix = SP["opponent_mix"]
    r   = random.random()
    if r < mix["self"]:
        opp = copy.deepcopy(main_model); opp.eval()
        return opp, "self"
    if r < mix["self"] + mix["league"]:
        opp = league.sample_opponent()
        if opp is None:
            opp = copy.deepcopy(main_model); opp.eval()
            return opp, "self_fallback"
        return opp, "league"
    opp = copy.deepcopy(exploiter); opp.eval()
    return opp, "exploiter"


def _sample_opp_states_for_workers(main_state, exploiter_state, league, n_envs):
    """Per-env opponent sampling for parallel rollout workers.

    Returns list[(opp_state_dict_or_None, match_type)] of length n_envs.
    A single PPO update batch then mixes self / league / exploiter trajectories
    instead of being locked to one match_type per generation.

    main_state, exploiter_state: pre-extracted CPU state_dict snapshots — shared
    across "self" / "exploiter" envs to avoid n_envs deepcopies. League opponents
    are loaded fresh from disk (already independent snapshots).

    match_type labels match _sample_opponent: {self, self_fallback, league, exploiter}.
    """
    mix = SP["opponent_mix"]
    out = []
    for _ in range(n_envs):
        r = random.random()
        if r < mix["self"]:
            out.append((main_state, "self"))
            continue
        if r < mix["self"] + mix["league"]:
            opp = league.sample_opponent()
            if opp is None:
                out.append((main_state, "self_fallback"))
            else:
                opp_state = {k: v.detach().cpu() for k, v in opp.state_dict().items()}
                out.append((opp_state, "league"))
            continue
        out.append((exploiter_state, "exploiter"))
    return out


# ── PPO 업데이트 ──────────────────────────────────────────────────────────────

def compute_gae(rewards, dones, values, last_value=0.0, gamma=T["gamma"], lam=T["gae_lambda"]):
    """GAE(γ, λ) advantage + returns 계산.

    last_value: rollout 마지막 다음 상태의 critic value.
                non-terminal truncation 시 bootstrap에 사용.
    returns = advantages + values (critic target으로 사용)
    """
    T_len      = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae   = 0.0
    for t in reversed(range(T_len)):
        if t + 1 < T_len:
            next_val = values[t + 1].squeeze()
        else:
            next_val = last_value if isinstance(last_value, float) else last_value.squeeze()
        delta         = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t].squeeze()
        last_gae      = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values.squeeze(-1)
    return advantages, returns


def ppo_update(model, optimizer, obs, actions, old_log_probs, returns, advantages,
               old_logp_heads=None, action_masks=None, target_masks=None,
               pair_features=None, amount_features=None,
               clip_range=T["clip_range"], n_epochs=T["n_epochs"], minibatch_size=T["minibatch_size"],
               target_kl=T.get("target_kl")):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs           = obs.to(DEVICE)
    actions       = actions.to(DEVICE)
    old_log_probs = old_log_probs.to(DEVICE)
    advantages    = advantages.to(DEVICE)
    returns       = returns.to(DEVICE)
    if old_logp_heads is not None:
        old_logp_heads = old_logp_heads.to(DEVICE)
    if action_masks is not None:
        action_masks = action_masks.to(DEVICE)
    if target_masks is not None:
        target_masks = target_masks.to(DEVICE)
    if pair_features is not None:
        pair_features = pair_features.to(DEVICE)
    if amount_features is not None:
        amount_features = amount_features.to(DEVICE)

    N = len(obs)
    p_losses, v_losses, e_losses = [], [], []
    approx_kls, clip_fracs       = [], []
    ent_actions, ent_targets = [], []
    kl_a_hist, kl_t_hist = [], []
    cf_a_hist, cf_t_hist = [], []
    epochs_done = 0

    for epoch in range(n_epochs):
        early_stop = False
        idx = torch.randperm(N, device=DEVICE)
        for start in range(0, N, minibatch_size):
            mb = idx[start:start + minibatch_size]

            am_mb = action_masks[mb] if action_masks is not None else None
            tm_mb = target_masks[mb] if target_masks is not None else None
            pf_mb = pair_features[mb]   if pair_features   is not None else None
            af_mb = amount_features[mb] if amount_features is not None else None
            log_probs, entropy, values, ent_a, ent_t, lp_heads = model.evaluate_actions(
                obs[mb], actions[mb], action_mask=am_mb, target_mask=tm_mb,
                target_pair_features=pf_mb, amount_features=af_mb,
            )

            ratio        = (log_probs - old_log_probs[mb]).exp()
            adv_mb       = advantages[mb]
            surr1        = ratio * adv_mb
            surr2        = ratio.clamp(1 - clip_range, 1 + clip_range) * adv_mb
            policy_loss  = -torch.min(surr1, surr2).mean()
            value_loss   = nn.functional.mse_loss(values.squeeze(-1), returns[mb])
            entropy_loss = -entropy.mean()

            loss = policy_loss + T["vf_coef"] * value_loss + T["ent_coef"] * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), T["max_grad_norm"])
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - ratio.log()).mean().item()
                clip_frac = ((ratio - 1).abs() > clip_range).float().mean().item()

                if old_logp_heads is not None:
                    # per-head diagnostic: ratio/kl/cf를 head별로 분리 (loss에는 안 씀)
                    diff_h   = lp_heads - old_logp_heads[mb]           # (B, 2)
                    ratio_h  = diff_h.exp()
                    kl_h     = ((ratio_h - 1) - diff_h).mean(dim=0)    # (2,)
                    cf_h     = ((ratio_h - 1).abs() > clip_range).float().mean(dim=0)
                    kl_a_hist.append(kl_h[0].item())
                    kl_t_hist.append(kl_h[1].item())
                    cf_a_hist.append(cf_h[0].item())
                    cf_t_hist.append(cf_h[1].item())

            p_losses.append(policy_loss.item())
            v_losses.append(value_loss.item())
            e_losses.append(entropy_loss.item())
            approx_kls.append(approx_kl)
            clip_fracs.append(clip_frac)
            ent_actions.append(ent_a.mean().item())
            ent_targets.append(ent_t.mean().item())

            if target_kl is not None and approx_kl > target_kl:
                early_stop = True
                break

        epochs_done += 1
        if early_stop:
            break

    def _avg(lst):
        return (sum(lst) / len(lst)) if lst else 0.0

    head_metrics = {
        "kl_action": _avg(kl_a_hist), "kl_target": _avg(kl_t_hist),
        "cf_action": _avg(cf_a_hist), "cf_target": _avg(cf_t_hist),
    }

    return (
        sum(p_losses) / len(p_losses),
        sum(v_losses) / len(v_losses),
        sum(e_losses) / len(e_losses),
        sum(approx_kls) / len(approx_kls),
        sum(clip_fracs) / len(clip_fracs),
        epochs_done,
        sum(ent_actions) / len(ent_actions),
        sum(ent_targets) / len(ent_targets),
        head_metrics,
    )


# ── 평가 ──────────────────────────────────────────────────────────────────────

def evaluate(main_model, opponent_model, n_games=20):
    """Eval: win rate + win/loss split summary (under_invested, s/r per game).

    가설: under-invest가 패배와 상관 있다면 loss 게임의 under/sr이 win 게임보다 나쁨.
    - 전체 평균(rollout)은 노이즈가 많아서 인과 분석 어려움
    - 승패 split이 더 날카로운 신호 (같은 policy에서 win vs loss 차이)
    """
    wins = 0
    # 게임별 decode 카운트 누적 (평균을 내기 위해 합계 + launched 저장)
    # bucket: "win" / "loss" (draw는 버림)
    bucket_under_cnt  = {"win": 0, "loss": 0}
    bucket_launched   = {"win": 0, "loss": 0}
    bucket_sr_sum     = {"win": 0.0, "loss": 0.0}
    bucket_under_enm  = {"win": 0, "loss": 0}
    bucket_enemy_lch  = {"win": 0, "loss": 0}

    for _ in range(n_games):
        env = make("orbit_wars", debug=False)
        env.reset()
        history_p     = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
        history_f     = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)
        history_p_opp = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
        history_f_opp = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)

        # per-game 카운터 (승패 분류 후 bucket에 합산)
        game_launched = 0
        game_under    = 0
        game_sr_sum   = 0.0
        game_under_e  = 0
        game_enemy_l  = 0
        ep_step_e        = 0
        ep_steps_max_e   = int(getattr(env.configuration, "episodeSteps", 500) or 500)

        while not env.done:
            raw_main = env.state[0].observation
            obs_t, raw_p, raw_f, av = get_obs_tensor(raw_main, 0, history_p, history_f)
            turn_norm_e = ep_step_e / max(ep_steps_max_e, 1)
            analysis_e = analyze_action_space(raw_p, raw_f, av, acting_player=0,
                                              turn_norm=turn_norm_e)
            with torch.no_grad():
                action_t, _, _, _ = main_model.get_action_and_value(
                    obs_t.unsqueeze(0).to(DEVICE),
                    action_mask=analysis_e.action_mask.unsqueeze(0).to(DEVICE),
                    target_mask=analysis_e.target_mask.unsqueeze(0).to(DEVICE),
                    target_pair_features=analysis_e.pair_features_target.unsqueeze(0).to(DEVICE),
                    amount_features=analysis_e.amount_features_full.unsqueeze(0).to(DEVICE),
                )
            moves_main, counts, _launches = decode_action_to_moves(
                action_t.squeeze(0).cpu().numpy(), raw_p, av,
                acting_player=0, analysis=analysis_e, return_counts=True,
                turn_norm=turn_norm_e,
            )
            game_launched += counts.get("launched", 0)
            game_under    += counts.get("under_invested_count", 0)
            game_sr_sum   += counts.get("send_required_ratio_sum", 0.0)
            game_under_e  += counts.get("under_invested_count_enemy", 0)
            # enemy launch 수 = 전체 launched − neutral — 여기서는 직접 계산.
            # target_neutral/enemy counter가 decode counts에 없으므로 launches 순회는
            # 이미 소비됨. 대신 under_invested_count_enemy + (enemy에 대한 not-under)
            # 를 알 길이 없으므로, ships_to_send_sum_enemy>0 여부 등으로는 부정확.
            # 가장 단순: enemy launches = launched - neutral_launches. neutral 카운트 하자.
            # (counts dict에 직접 접근해서 enemy 분모 카운트)
            # → 실제로는 counts에 target_{neutral,enemy}가 없어서 별도 계산 필요.
            # 대안: launches 리스트에서 target_owner로 센다.
            for l in _launches:
                if l.get("target_owner", -1) != -1:
                    game_enemy_l += 1

            raw_opp = env.state[1].observation
            obs_o, raw_po, raw_fo, avo = get_obs_tensor(raw_opp, 1, history_p_opp, history_f_opp)
            moves_opp = _opp_moves(opponent_model, obs_o, raw_po, raw_fo, avo, DEVICE,
                                   turn_norm=turn_norm_e)

            env.step([moves_main, moves_opp])
            ep_step_e += 1

        r = env.state[0].reward
        if r == 1:
            wins += 1
            key = "win"
        elif r == 0:
            wins += 0.5
            key = None   # draw는 분석에서 제외 (noise만 추가)
        else:
            key = "loss"

        if key is not None:
            bucket_under_cnt[key] += game_under
            bucket_launched[key]  += game_launched
            bucket_sr_sum[key]    += game_sr_sum
            bucket_under_enm[key] += game_under_e
            bucket_enemy_lch[key] += game_enemy_l

    def _rate(num, den):
        return num / den if den > 0 else 0.0

    split = {
        "eval_under_win":        _rate(bucket_under_cnt["win"],  bucket_launched["win"]),
        "eval_under_loss":       _rate(bucket_under_cnt["loss"], bucket_launched["loss"]),
        "eval_sr_win":           _rate(bucket_sr_sum["win"],     bucket_launched["win"]),
        "eval_sr_loss":          _rate(bucket_sr_sum["loss"],    bucket_launched["loss"]),
        "eval_under_enemy_win":  _rate(bucket_under_enm["win"],  bucket_enemy_lch["win"]),
        "eval_under_enemy_loss": _rate(bucket_under_enm["loss"], bucket_enemy_lch["loss"]),
    }
    return wins / n_games, split


def _run_eval_game(p0_model, p1_model, tracker):
    """한 게임 실행 — p0 시점 action을 tracker(HitRateTracker, player_id=0)에 누적.

    rollout-style stats(under/sr/cap/by_tgt/combo …)를 모으기 위해 rollout과 동일한
    pattern: decode.record → register_launches → env.step → resolve_step.
    동시에 win/loss split용 per-game counter도 함께 반환.

    evaluate_pair에서 같은 seed를 두 번 쓰면서 양쪽 모델을 p0 자리에 번갈아 세움.

    return: (p0_reward ∈ {-1,0,1}, per_game_counters)
    """
    env = make("orbit_wars", debug=False)
    env.reset()
    tracker.reset_episode(env.state[0].observation)

    history_p     = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
    history_f     = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)
    history_p_opp = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
    history_f_opp = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)

    game_launched = 0
    game_under    = 0
    game_sr_sum   = 0.0
    game_under_e  = 0
    game_enemy_l  = 0
    ep_step_v      = 0
    ep_steps_max_v = int(getattr(env.configuration, "episodeSteps", 500) or 500)

    while not env.done:
        raw_obs_p0 = env.state[0].observation
        obs_t, raw_p, raw_f, av = get_obs_tensor(raw_obs_p0, 0, history_p, history_f)
        turn_norm_v = ep_step_v / max(ep_steps_max_v, 1)
        analysis = analyze_action_space(raw_p, raw_f, av, acting_player=0,
                                        turn_norm=turn_norm_v)
        with torch.no_grad():
            action_t, _, _, _ = p0_model.get_action_and_value(
                obs_t.unsqueeze(0).to(DEVICE),
                action_mask=analysis.action_mask.unsqueeze(0).to(DEVICE),
                target_mask=analysis.target_mask.unsqueeze(0).to(DEVICE),
                target_pair_features=analysis.pair_features_target.unsqueeze(0).to(DEVICE),
                amount_features=analysis.amount_features_full.unsqueeze(0).to(DEVICE),
            )
        moves_p0, counts, launches = decode_action_to_moves(
            action_t.squeeze(0).cpu().numpy(), raw_p, av,
            acting_player=0, analysis=analysis, return_counts=True,
            turn_norm=turn_norm_v,
        )
        tracker.record(counts)

        game_launched += counts.get("launched", 0)
        game_under    += counts.get("under_invested_count", 0)
        game_sr_sum   += counts.get("send_required_ratio_sum", 0.0)
        game_under_e  += counts.get("under_invested_count_enemy", 0)
        for l in launches:
            if l.get("target_owner", -1) != -1:
                game_enemy_l += 1

        raw_obs_p1 = env.state[1].observation
        obs_o, raw_po, raw_fo, avo = get_obs_tensor(raw_obs_p1, 1, history_p_opp, history_f_opp)
        moves_p1 = _opp_moves(p1_model, obs_o, raw_po, raw_fo, avo, DEVICE,
                              turn_norm=turn_norm_v)

        prev_obs_snap = _snapshot_obs_for_resolve(raw_obs_p0)
        tracker.register_launches(launches, prev_obs_snap["next_fleet_id"])
        env.step([moves_p0, moves_p1])
        ep_step_v += 1

        curr_obs  = env.state[0].observation
        max_speed = env.configuration.shipSpeed
        tracker.resolve_step(prev_obs_snap, curr_obs, max_speed)

    return env.state[0].reward, {
        "launched":       game_launched,
        "under":          game_under,
        "sr_sum":         game_sr_sum,
        "under_enemy":    game_under_e,
        "enemy_launched": game_enemy_l,
    }


def evaluate_pair(model_a, model_b, n_pairs=10):
    """Paired seat-swap eval: 같은 seed로 2게임씩 돌려 map-variance 제거.

    각 pair:
      Game 1: A=p0, B=p1 → A의 rollout-style stats 기록
      Game 2: B=p0, A=p1 (같은 seed로 같은 맵 재생성) → B의 stats 기록
    즉 A와 B 모두 "p0 자리에서 같은 맵"을 상대. win_rate/stats 비교가 깔끔.

    orbit_wars env는 global `random` 모듈로 맵 생성하므로, env.reset() 직전
    random.seed(...)를 고정하면 동일 맵이 나온다. random 상태는 호출 후 풀어줌
    (외부 self-play loop의 샘플링에 영향 없게).

    return dict:
      a_win_rate, b_win_rate  — 각 모델의 seat-평균 승률 (p0+p1 양쪽 자리 결과 반영).
                                모델 i의 WR = (i가 p0일 때 win) + (i가 p1일 때, 즉 상대가 p0
                                이고 졌을 때)를 game 수로 나눈 값. 두 값은 draw 제외 시
                                합이 1.0이 되도록 정의 (win_rate 컬럼 semantic 통일).
      a_p0_win_rate, b_p0_win_rate — 각 모델의 p0 시점 승률 (first-player 편향 디버그용).
      a_stats, b_stats        — HitRateTracker.summary() (rollout schema, p0 시점 수집).
      a_split, b_split        — win/loss split (eval_under_* / eval_sr_*, p0 시점 수집).
    """
    tracker_a = HitRateTracker(player_id=0)
    tracker_b = HitRateTracker(player_id=0)
    # seat-평균 승률용(전 게임 합산)과 p0-only 승률(debug)을 분리해서 추적.
    a_wins_total = b_wins_total = 0.0   # 양쪽 seat 전체 wins (2*n_pairs 게임 기준)
    a_p0_wins    = b_p0_wins    = 0.0   # 각자 p0일 때 wins (n_pairs 게임 기준)

    def _new_bucket():
        return {
            "under_cnt":       {"win": 0, "loss": 0},
            "launched":        {"win": 0, "loss": 0},
            "sr_sum":          {"win": 0.0, "loss": 0.0},
            "under_enemy":     {"win": 0, "loss": 0},
            "enemy_launched": {"win": 0, "loss": 0},
        }

    def _bucket_add(bucket, reward, pg):
        if reward == 1:    key = "win"
        elif reward == -1: key = "loss"
        else:              return    # draw 제외 (noise)
        bucket["under_cnt"][key]      += pg["under"]
        bucket["launched"][key]       += pg["launched"]
        bucket["sr_sum"][key]         += pg["sr_sum"]
        bucket["under_enemy"][key]    += pg["under_enemy"]
        bucket["enemy_launched"][key] += pg["enemy_launched"]

    def _to_wins(reward):
        if reward == 1:  return 1.0
        if reward == 0:  return 0.5
        return 0.0

    def _split_from(b):
        def _rate(n, d): return n / d if d > 0 else 0.0
        return {
            "eval_under_win":        _rate(b["under_cnt"]["win"],      b["launched"]["win"]),
            "eval_under_loss":       _rate(b["under_cnt"]["loss"],     b["launched"]["loss"]),
            "eval_sr_win":           _rate(b["sr_sum"]["win"],         b["launched"]["win"]),
            "eval_sr_loss":          _rate(b["sr_sum"]["loss"],        b["launched"]["loss"]),
            "eval_under_enemy_win":  _rate(b["under_enemy"]["win"],    b["enemy_launched"]["win"]),
            "eval_under_enemy_loss": _rate(b["under_enemy"]["loss"],   b["enemy_launched"]["loss"]),
        }

    a_bucket = _new_bucket()
    b_bucket = _new_bucket()
    rng_state = random.getstate()
    try:
        for _ in range(n_pairs):
            seed = random.randint(0, 2**31 - 1)

            # Game 1: A=p0, B=p1. r_a는 A(p0)의 reward; B는 p1이므로 reward=-r_a.
            random.seed(seed)
            r_a, pg_a = _run_eval_game(model_a, model_b, tracker_a)
            a_p0_wins    += _to_wins(r_a)
            a_wins_total += _to_wins(r_a)     # A as p0
            b_wins_total += _to_wins(-r_a)    # B as p1 (zero-sum 역수)
            _bucket_add(a_bucket, r_a, pg_a)

            # Game 2: B=p0, A=p1. 같은 seed → 같은 맵. r_b는 B(p0)의 reward.
            random.seed(seed)
            r_b, pg_b = _run_eval_game(model_b, model_a, tracker_b)
            b_p0_wins    += _to_wins(r_b)
            b_wins_total += _to_wins(r_b)     # B as p0
            a_wins_total += _to_wins(-r_b)    # A as p1
            _bucket_add(b_bucket, r_b, pg_b)
    finally:
        random.setstate(rng_state)

    total_games = 2 * max(n_pairs, 1)   # 모델당 p0 + p1 각 n_pairs 게임
    return {
        "a_win_rate":    a_wins_total / total_games,    # seat-평균 (2*n_pairs 게임 기준)
        "b_win_rate":    b_wins_total / total_games,    # a_win_rate + b_win_rate ≈ 1.0
        "a_p0_win_rate": a_p0_wins / max(n_pairs, 1),   # p0-only (first-player 편향 확인용)
        "b_p0_win_rate": b_p0_wins / max(n_pairs, 1),
        "a_stats":       tracker_a.summary(),
        "b_stats":       tracker_b.summary(),
        "a_split":       _split_from(a_bucket),
        "b_split":       _split_from(b_bucket),
    }


# ── Main Training Loop ────────────────────────────────────────────────────────

def train(n_envs=1, total_timesteps=None, eval_interval=None, n_games=None, rollout_steps=None):
    global DEVICE, SAVE_DIR

    total_timesteps = total_timesteps or T["total_timesteps"]
    eval_interval   = eval_interval   or SP["eval_interval"]
    n_games         = n_games         or 20
    rollout_steps   = rollout_steps   or 512

    print(f"Device: {DEVICE} | run_dir: {SAVE_DIR} | n_envs: {n_envs} | "
          f"total_timesteps: {total_timesteps} | eval_interval: {eval_interval} | "
          f"n_games: {n_games} | rollout_steps: {rollout_steps}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    logger = TrainingLogger(log_dir=os.path.join(SAVE_DIR, "..", "logs"))

    main_model = OrbitWarsPolicy().to(DEVICE)
    optimizer  = optim.Adam(main_model.parameters(), lr=T["learning_rate"])

    exploiter     = OrbitWarsPolicy().to(DEVICE)
    exploiter_opt = optim.Adam(exploiter.parameters(), lr=T["learning_rate"])

    league = LeaguePool()

    generation      = 0
    total_steps     = 0
    exploiter_reset = 0

    ckpt_path = os.path.join(SAVE_DIR, "resume.pt")
    # PARTIAL_TRANSFER=1: 모델 구조 변경 (e.g. Step 3 → Step 4 target head 교체) 시
    # 한 번만 strict=False 로 로드. 매칭 안 되는 key (target_q/k 등) 는 새로 init.
    strict_load = os.environ.get("PARTIAL_TRANSFER", "0") != "1"
    # RESET_ALPHA=1: target_bias_alpha 만 fresh init (0.1) 으로 강제 시동.
    # 5gen 진단 결과 기존 ckpt 의 target_bias_alpha=-0.014 가 0.1 init 을 덮어쓰면
    # 효과 없음 → 한 번만 reset 후 학습. amount_bin_alpha 는 이미 일하는 중이라 보존.
    # 한 번 사용 후 끄는 것 권장 (다음 재개 시 ckpt 의 새 학습값 살리려면).
    reset_keys = None
    if os.environ.get("RESET_ALPHA", "0") == "1":
        reset_keys = ["target_bias_alpha"]
    result = load_checkpoint(ckpt_path, main_model, optimizer, DEVICE,
                             strict=strict_load, reset_keys=reset_keys)
    if result:
        generation, total_steps, league.agents = result
    else:
        league.add(main_model, generation=0, win_rate=0.0)

    # 병렬 pool 생성 (n_envs > 1일 때만)
    pool = None
    if n_envs > 1:
        ctx  = mp.get_context("spawn")
        pool = ctx.Pool(n_envs, initializer=_init_worker)
        print(f"병렬 rollout pool: {n_envs}개 worker")

    try:
        while total_steps < total_timesteps:
            generation += 1
            gen_t0 = time.time()

            # Per-env opponent mixing: each rollout worker gets its own opponent
            # so a single PPO batch sees self / league / exploiter trajectories
            # together (avoids one-update-per-opponent-type overfitting).
            if n_envs > 1 and pool is not None:
                main_state_cpu = {k: v.detach().cpu() for k, v in main_model.state_dict().items()}
                exp_state_cpu  = {k: v.detach().cpu() for k, v in exploiter.state_dict().items()}
                opp_pairs   = _sample_opp_states_for_workers(main_state_cpu, exp_state_cpu, league, n_envs)
                opp_states  = [s for s, _ in opp_pairs]
                mt_counts   = Counter(t for _, t in opp_pairs)
                _abbr = {"self": "s", "self_fallback": "sf", "league": "l", "exploiter": "e"}
                if len(mt_counts) == 1:
                    match_type = next(iter(mt_counts))
                else:
                    match_type = "/".join(
                        f"{_abbr.get(k, k)}{v}" for k, v in sorted(mt_counts.items())
                    )
                (obs, actions, advantages, returns, log_probs, logp_heads, amasks, tmasks,
                 pair_feats, amount_feats, rew_stats) = collect_rollout(
                    main_model, opponent_model=None,
                    n_steps=rollout_steps, n_envs=n_envs, pool=pool,
                    opp_states=opp_states,
                )
            else:
                opponent, match_type = _sample_opponent(main_model, exploiter, league)
                (obs, actions, advantages, returns, log_probs, logp_heads, amasks, tmasks,
                 pair_feats, amount_feats, rew_stats) = collect_rollout(
                    main_model, opponent, n_steps=rollout_steps, n_envs=n_envs, pool=pool
                )
            p_loss, v_loss, e_loss, approx_kl, clip_frac, epochs_done, ent_a, ent_t, head_metrics = ppo_update(
                main_model, optimizer, obs, actions, log_probs, returns, advantages,
                old_logp_heads=logp_heads, action_masks=amasks, target_masks=tmasks,
                pair_features=pair_feats, amount_features=amount_feats,
            )
            total_steps += len(obs)

            exp_opp = copy.deepcopy(main_model)
            exp_opp.eval()
            (obs_e, act_e, adv_e, ret_e, logp_e, logp_heads_e, amasks_e, tmasks_e,
             pair_feats_e, amount_feats_e, _) = collect_rollout(
                exploiter, exp_opp, n_steps=max(1, rollout_steps // 2), n_envs=max(1, n_envs // 2), pool=pool
            )
            ppo_update(exploiter, exploiter_opt, obs_e, act_e, logp_e, ret_e, adv_e,
                       old_logp_heads=logp_heads_e, action_masks=amasks_e, target_masks=tmasks_e,
                       pair_features=pair_feats_e, amount_features=amount_feats_e)

            # 진단 C: Phase A additive bias channel 의 학습 강도 (scalar α).
            # 초기값 0 (residual=off) → 학습 진행에 따라 |α| 증가 = bias 가 정책에 영향 시작.
            # 너무 크면 (e.g. >2) bias 가 main logit 을 압도 → 단순 cost greedy. 작으면 (≈0)
            # bias 채널이 무시됨 → MLP feature 만으로 충분 / 또는 학습 자체가 안 됨.
            target_bias_alpha = float(main_model.target_bias_alpha.detach().cpu().item())
            amount_bin_alpha  = float(main_model.amount_bin_alpha.detach().cpu().item())

            logger.log(
                generation=generation, total_steps=total_steps, match_type=match_type,
                policy_loss=p_loss, value_loss=v_loss, entropy_loss=e_loss,
                approx_kl=approx_kl, clip_frac=clip_frac, epochs_done=epochs_done,
                ent_action=ent_a, ent_target=ent_t,
                league_size=len(league),
                # rollout 내 승률 (on-policy, noisy지만 match_type별로 분포별 성능 확인 가능).
                win_rate=rew_stats.get("win_rate", 0.0),
                mean_dense_rew=rew_stats["mean_dense"],
                mean_cap_bonus=rew_stats["mean_cap"],
                mean_terminal_rew=rew_stats["mean_terminal"],
                # Sprint 2: 발사 시 자원 보존 페널티 (음수). all_in_penalty=0 이면 0.
                mean_all_in_penalty=rew_stats.get("mean_all_in_penalty", 0.0),
                # Phase B: per-launch cost penalty (continuous, req/src>0.5 부터). 0 이면 비활성.
                mean_launch_cost_penalty=rew_stats.get("mean_launch_cost_penalty", 0.0),
                mean_attempts=rew_stats["mean_attempts"],
                mean_launched=rew_stats["mean_launched"],
                launch_rate=rew_stats["launch_rate"],
                # Step 4 측정: active source(자기 행성+ships>0) 중 valid target 없어
                # skip 으로 빠진 비율. 높으면 mask 가 너무 조이거나 dst 제약(파라볼라/부족)
                # 으로 행동공간이 막힌 상태. target_q/k pair-aware 도입 효과 확인용.
                self_fallback_active_rate=rew_stats.get("self_fallback_active_rate", 0.0),
                # 진단 A: src.ships threshold 별 self_fallback rate (작은 src false alarm 식별).
                self_fallback_active_rate_ge1=rew_stats.get("self_fallback_active_rate_ge1", 0.0),
                self_fallback_active_rate_ge5=rew_stats.get("self_fallback_active_rate_ge5", 0.0),
                self_fallback_active_rate_ge10=rew_stats.get("self_fallback_active_rate_ge10", 0.0),
                self_fallback_active_rate_ge20=rew_stats.get("self_fallback_active_rate_ge20", 0.0),
                # 진단 A': mask first-failure 분해 (4 게이트 × 4 src.ships threshold = 16 cols).
                # finalize 가 stats 에 채워준 키를 logger 로 forward — 안 빼먹어야 CSV 채워짐.
                **{f"mask_block_{g}_ge{t}": rew_stats.get(f"mask_block_{g}_ge{t}", 0.0)
                   for g in ("owner_rule", "enemy_neutral_filter", "sun_path", "capacity_short")
                   for t in (1, 5, 10, 20)},
                # 진단 B: per-launch req/src.ships 분포 (target 비용 측면).
                req_over_src_launched_mean=rew_stats.get("req_over_src_launched_mean", 0.0),
                req_over_src_launched_p50=rew_stats.get("req_over_src_launched_p50", 0.0),
                req_over_src_launched_p75=rew_stats.get("req_over_src_launched_p75", 0.0),
                req_over_src_launched_p90=rew_stats.get("req_over_src_launched_p90", 0.0),
                req_over_src_launched_p95=rew_stats.get("req_over_src_launched_p95", 0.0),
                # 진단 B': production-axis (req/prod, req/(prod×ETA)).
                # req_over_src 가 p90=1.0 saturated 일 때 "회복 가능 vs 자살" 가르는 신호.
                req_over_prod_launched_mean=rew_stats.get("req_over_prod_launched_mean", 0.0),
                req_over_prod_launched_p50=rew_stats.get("req_over_prod_launched_p50", 0.0),
                req_over_prod_launched_p75=rew_stats.get("req_over_prod_launched_p75", 0.0),
                req_over_prod_launched_p90=rew_stats.get("req_over_prod_launched_p90", 0.0),
                req_over_prod_launched_p95=rew_stats.get("req_over_prod_launched_p95", 0.0),
                req_over_prod_eta_launched_mean=rew_stats.get("req_over_prod_eta_launched_mean", 0.0),
                req_over_prod_eta_launched_p50=rew_stats.get("req_over_prod_eta_launched_p50", 0.0),
                req_over_prod_eta_launched_p75=rew_stats.get("req_over_prod_eta_launched_p75", 0.0),
                req_over_prod_eta_launched_p90=rew_stats.get("req_over_prod_eta_launched_p90", 0.0),
                req_over_prod_eta_launched_p95=rew_stats.get("req_over_prod_eta_launched_p95", 0.0),
                # 진단 C: per-launch eta_advantage_norm 분포 (race signal).
                # 양수 = 적보다 빨리 도달 (race 우위), 음수 = 늦음 (보내봤자 경합 손실).
                eta_advantage_launched_mean=rew_stats.get("eta_advantage_launched_mean", 0.0),
                eta_advantage_launched_p50=rew_stats.get("eta_advantage_launched_p50", 0.0),
                eta_advantage_launched_p75=rew_stats.get("eta_advantage_launched_p75", 0.0),
                eta_advantage_launched_p90=rew_stats.get("eta_advantage_launched_p90", 0.0),
                eta_advantage_launched_p95=rew_stats.get("eta_advantage_launched_p95", 0.0),
                # 진단 D: Phase A additive bias channel scalar α (model parameter, learned).
                target_bias_alpha=target_bias_alpha,
                amount_bin_alpha=amount_bin_alpha,
                # 진단 D: send_frac × ships_bin 매트릭스 (bin 별 mean + 양 극단 p90).
                **{f"send_frac_bin{k}_mean": rew_stats.get(f"send_frac_bin{k}_mean", 0.0)
                   for k in range(NUM_SHIPS_BINS)},
                # bin0 (just-capture) / bin{K-1} (top multiplier) p90 만 추적 — 양 극단 over/under 지표.
                # NUM_SHIPS_BINS==1 edge: 동일 bin → set 으로 중복 제거.
                **{f"send_frac_bin{k}_p90": rew_stats.get(f"send_frac_bin{k}_p90", 0.0)
                   for k in {0, NUM_SHIPS_BINS - 1}},
                mean_filtered_invalid_target=rew_stats["mean_filtered_invalid_target"],
                mean_filtered_zero_ships=rew_stats["mean_filtered_zero_ships"],
                mean_filtered_sun=rew_stats["mean_filtered_sun"],
                mean_filtered_path=rew_stats.get("mean_filtered_path", 0.0),
                mean_out=rew_stats.get("mean_out", 0.0),
                mean_sun_crash=rew_stats.get("mean_sun_crash", 0.0),
                mean_target_hit_exclusive=rew_stats.get("mean_target_hit_exclusive", 0.0),
                mean_target_hit_ambiguous=rew_stats.get("mean_target_hit_ambiguous", 0.0),
                mean_hit_other_exclusive=rew_stats.get("mean_hit_other_exclusive", 0.0),
                mean_hit_other_ambiguous=rew_stats.get("mean_hit_other_ambiguous", 0.0),
                mean_captured_exclusive=rew_stats.get("mean_captured_exclusive", 0.0),
                mean_captured_ambiguous=rew_stats.get("mean_captured_ambiguous", 0.0),
                mean_unknown_removal=rew_stats.get("mean_unknown_removal", 0.0),
                mean_launched_high_prod=rew_stats.get("mean_launched_high_prod", 0.0),
                mean_captured_neutral=rew_stats.get("mean_captured_neutral", 0.0),
                mean_captured_enemy=rew_stats.get("mean_captured_enemy", 0.0),
                mean_early_home_expand=rew_stats.get("mean_early_home_expand", 0.0),
                noop_rate=rew_stats.get("noop_rate", 0.0),
                high_prod_target_rate=rew_stats.get("high_prod_target_rate", 0.0),
                neutral_capture_rate=rew_stats.get("neutral_capture_rate", 0.0),
                enemy_capture_rate=rew_stats.get("enemy_capture_rate", 0.0),
                early_home_expand_per_episode=rew_stats.get("early_home_expand_per_episode", 0.0),
                mean_target_neutral=rew_stats.get("mean_target_neutral", 0.0),
                mean_target_enemy=rew_stats.get("mean_target_enemy", 0.0),
                mean_early_neutral_attempts=rew_stats.get("mean_early_neutral_attempts", 0.0),
                mean_early_enemy_attempts=rew_stats.get("mean_early_enemy_attempts", 0.0),
                mean_early_neutral_captured=rew_stats.get("mean_early_neutral_captured", 0.0),
                target_neutral_rate=rew_stats.get("target_neutral_rate", 0.0),
                target_enemy_rate=rew_stats.get("target_enemy_rate", 0.0),
                early_neutral_attempts_per_episode=rew_stats.get("early_neutral_attempts_per_episode", 0.0),
                early_enemy_attempts_per_episode=rew_stats.get("early_enemy_attempts_per_episode", 0.0),
                early_neutral_captured_per_episode=rew_stats.get("early_neutral_captured_per_episode", 0.0),
                mean_early_launch_neutral_captured=rew_stats.get("mean_early_launch_neutral_captured", 0.0),
                early_launch_neutral_captured_per_episode=rew_stats.get("early_launch_neutral_captured_per_episode", 0.0),
                early_neutral_launch_to_cap_rate=rew_stats.get("early_neutral_launch_to_cap_rate", 0.0),
                # 초반 중립 확장 품질 (Phase E 진단)
                early_neutral_eta_mean=rew_stats.get("early_neutral_eta_mean", 0.0),
                early_neutral_req_over_src_mean=rew_stats.get("early_neutral_req_over_src_mean", 0.0),
                early_neutral_nearest_rank_mean=rew_stats.get("early_neutral_nearest_rank_mean", 0.0),
                early_neutral_eta_advantage_mean=rew_stats.get("early_neutral_eta_advantage_mean", 0.0),
                # Phase C support stats
                support_launches_per_step=rew_stats.get("support_launches_per_step", 0.0),
                support_ships_to_send_mean=rew_stats.get("support_ships_to_send_mean", 0.0),
                chosen_surplus_frac_mean=rew_stats.get("chosen_surplus_frac_mean", 0.0),
                chosen_surplus_frac_std=rew_stats.get("chosen_surplus_frac_std", 0.0),
                send_fraction_of_src_mean=rew_stats.get("send_fraction_of_src_mean", 0.0),
                send_fraction_of_src_std=rew_stats.get("send_fraction_of_src_std", 0.0),
                ships_to_send_mean=rew_stats.get("ships_to_send_mean", 0.0),
                required_ships_mean=rew_stats.get("required_ships_mean", 0.0),
                send_required_ratio_mean=rew_stats.get("send_required_ratio_mean", 0.0),
                under_invested_rate=rew_stats.get("under_invested_rate", 0.0),
                over_send_excess_per_launch=rew_stats.get("over_send_excess_per_launch", 0.0),
                over_send_target_rate=rew_stats.get("over_send_target_rate", 0.0),
                mean_over_send_penalty=rew_stats.get("mean_over_send_penalty", 0.0),
                # target-type 분리: neutral(prod 없음) vs enemy(prod 회복) — waste 상관관계 분석용
                send_required_ratio_mean_neutral=rew_stats.get("send_required_ratio_mean_neutral", 0.0),
                send_required_ratio_mean_enemy=rew_stats.get("send_required_ratio_mean_enemy", 0.0),
                under_invested_rate_neutral=rew_stats.get("under_invested_rate_neutral", 0.0),
                under_invested_rate_enemy=rew_stats.get("under_invested_rate_enemy", 0.0),
                ships_to_send_mean_neutral=rew_stats.get("ships_to_send_mean_neutral", 0.0),
                ships_to_send_mean_enemy=rew_stats.get("ships_to_send_mean_enemy", 0.0),
                required_ships_mean_neutral=rew_stats.get("required_ships_mean_neutral", 0.0),
                required_ships_mean_enemy=rew_stats.get("required_ships_mean_enemy", 0.0),
                # 연계 공격: 단발 실패 vs 계획된 연속 압박 구분
                repeat_target_rate=rew_stats.get("repeat_target_rate", 0.0),
                launch_to_cap_rate_neutral=rew_stats.get("launch_to_cap_rate_neutral", 0.0),
                launch_to_cap_rate_enemy=rew_stats.get("launch_to_cap_rate_enemy", 0.0),
                # 1차 진단 metric 묶음 (단발 점령 + 유지 + 자원 보존)
                # 이 6개가 main rollout row(self/league/exploiter)에 찍혀야
                # Sprint 1 baseline → Sprint 2 비교가 가능. 빠지면 row 빈칸 → 측정 의미 상실.
                single_shot_capture_rate=rew_stats.get("single_shot_capture_rate", 0.0),
                capture_hold_k_rate=rew_stats.get("capture_hold_k_rate", 0.0),
                post_capture_reloss_rate_k=rew_stats.get("post_capture_reloss_rate_k", 0.0),
                all_in_launch_rate=rew_stats.get("all_in_launch_rate", 0.0),
                remaining_ships_after_launch_mean=rew_stats.get("remaining_ships_after_launch_mean", 0.0),
                distinct_targets_per_turn=rew_stats.get("distinct_targets_per_turn", 0.0),
                **{f"ships_bin_rate_{k}": rew_stats.get(f"ships_bin_rate_{k}", 0.0)
                   for k in range(NUM_SHIPS_BINS)},
                **head_metrics,
            )

            # crash 복구용: 매 gen마다 latest weights overwrite (용량 증가 없음).
            # eval_interval 주기를 기다리지 않고 최근 모델 항상 유지.
            torch.save(main_model.state_dict(), os.path.join(SAVE_DIR, "main_latest.pt"))

            # eval: 모니터링 전용. league.add 전에 수행 — 방금 저장한 자기 자신을
            # 샘플링해서 self-vs-self 50% 고정 승률을 찍는 걸 방지. 여기 pool은
            # 아직 이전 gen들만 포함하므로 "main vs 과거 pool" 의미가 깔끔함.
            if generation % eval_interval == 0:
                opp_eval = league.sample_opponent() or exploiter
                eval_t0  = time.time()
                win_rate, eval_split = evaluate(main_model, opp_eval, n_games=n_games)
                eval_wall_s = time.time() - eval_t0

                gen_wall_s = time.time() - gen_t0
                logger.log(
                    generation=generation, total_steps=total_steps, match_type="eval",
                    policy_loss=p_loss, value_loss=v_loss, entropy_loss=e_loss,
                    win_rate=win_rate, league_size=len(league),
                    eval_wall_s=eval_wall_s,
                    gen_wall_s=gen_wall_s,
                    **eval_split,
                )

                # exploiter_eval: main vs exploiter 전용 모니터링.
                #   exploiter_eval_main: main의 seat-평균 승률 + main 시점(p0) stats
                #   exploiter_eval_opp : exploiter의 seat-평균 승률 + exploiter 시점(p0) stats
                # win_rate 컬럼 semantic 통일 — 각 row의 win_rate는 "그 row 주체의
                # 양쪽 seat 평균 승률 vs 상대". 두 값 합 ≈ 1.0 (draw 제외).
                # stats/split는 여전히 "그 주체가 p0일 때" 수집 — 한쪽 seat stats지만
                # 같은 맵이라 두 row 비교는 공정.
                # evaluate_pair로 같은 seed에서 seat-swap paired 실행 → map-variance 제거.
                # rollout-style stats(HitRateTracker) 포함 → training row(self/league/exploiter)와
                # 같은 schema로 under/sr/cap/by_tgt/combo 직접 비교 가능.
                # exploiter_reset 직전에 수행 — fresh exploiter 찍는 것 방지.
                # n_pairs 절반: eval 비용 제한 (노이즈는 이동평균으로 읽을 것).
                ee_pairs = max(2, n_games // 2)
                ee_t0 = time.time()
                pair = evaluate_pair(main_model, exploiter, n_pairs=ee_pairs)
                ee_wall_s = time.time() - ee_t0
                # eval_wall_s: pair 전체(2×n_pairs 게임) 총 시간을 _main row에만 1회 기록.
                # _opp row는 비워둠 — 같은 값을 두 row에 찍으면 SUM/AVG 집계가 2배로 왜곡됨.
                logger.log(
                    generation=generation, total_steps=total_steps,
                    match_type="exploiter_eval_main",
                    win_rate=pair["a_win_rate"], league_size=len(league),
                    eval_wall_s=ee_wall_s,
                    **pair["a_stats"], **pair["a_split"],
                )
                logger.log(
                    generation=generation, total_steps=total_steps,
                    match_type="exploiter_eval_opp",
                    win_rate=pair["b_win_rate"], league_size=len(league),
                    **pair["b_stats"], **pair["b_split"],
                )

                exploiter_reset += 1
                if exploiter_reset % 5 == 0:
                    exploiter     = OrbitWarsPolicy().to(DEVICE)
                    exploiter_opt = optim.Adam(exploiter.parameters(), lr=T["learning_rate"])
                    print("  Exploiter 리셋")

            # A policy: 매 gen league 편입 (eval 이후). pool_size로 자동 rotation.
            league.add(main_model, generation, win_rate=0.0)

            # resume 상태 갱신: pool membership이 매 gen 바뀌므로 crash 시 정합성 유지.
            save_checkpoint(ckpt_path, main_model, optimizer, generation, total_steps, league.agents)

    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    print("학습 완료")
    torch.save(main_model.state_dict(), os.path.join(SAVE_DIR, "main_final.pt"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",              type=int,   default=0,             help="GPU index")
    parser.add_argument("--run-dir",          type=str,   default="checkpoints", help="checkpoint 저장 디렉토리")
    parser.add_argument("--n-envs",           type=int,   default=None,          help="병렬 env 수 (기본: config)")
    parser.add_argument("--total-timesteps",  type=int,   default=None,          help="학습 총 스텝 (기본: config)")
    parser.add_argument("--eval-interval",    type=int,   default=None,          help="평가 주기 세대 수 (기본: config)")
    parser.add_argument("--n-games",          type=int,   default=None,          help="평가 게임 수 (기본: 20)")
    parser.add_argument("--rollout-steps",    type=int,   default=None,          help="rollout 스텝 수 (기본: config)")
    parser.add_argument("--smoke",            action="store_true",               help="smoke test 세팅 (3000 steps, eval 2, 4 games, n_envs 4, rollout 512)")
    cli = parser.parse_args()

    DEVICE   = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    SAVE_DIR = cli.run_dir

    # smoke 우선 적용 — CLI 명시값(=non-None) 은 보존, 미지정만 작은 값으로 채움.
    # 이후 config-default fallback 은 None 만 채우므로 smoke 값 덮어쓰지 않음.
    if cli.smoke:
        cli.total_timesteps = cli.total_timesteps or 3000
        cli.eval_interval   = cli.eval_interval   or 2
        cli.n_games         = cli.n_games         or 4
        cli.n_envs          = cli.n_envs          or 4
        cli.rollout_steps   = cli.rollout_steps   or 512

    # config 기본값 적용 (CLI/smoke 미지정 시)
    cli.n_envs        = cli.n_envs        if cli.n_envs        is not None else T.get("n_envs", 1)
    cli.rollout_steps = cli.rollout_steps if cli.rollout_steps is not None else T.get("rollout_steps", 512)

    train(
        n_envs=cli.n_envs,
        total_timesteps=cli.total_timesteps,
        eval_interval=cli.eval_interval,
        n_games=cli.n_games,
        rollout_steps=cli.rollout_steps,
    )
