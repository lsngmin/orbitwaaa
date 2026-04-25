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
from collections import deque
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
import multiprocessing as mp

from model import OrbitWarsPolicy
from env_wrapper import (
    encode_planets, encode_fleets,
    MAX_PLANETS, MAX_FLEETS, PLANET_DIM, FLEET_DIM, HISTORY,
    SHIPS_MULTIPLIER_BINS, NUM_SHIPS_BINS,
)
from prediction import aim, crosses_sun, first_collision_on_path, PositionCache, resolve_ships_for_capture

with open("config.yaml") as f:
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


def neutral_capture_bonus(prev_map, curr_raw_obs, player):
    """중립 행성 점령 보너스: production에 비례한 즉각 보상.

    prev_map: _snapshot_planet_owners()로 미리 추출한 {pid: (owner, prod)}.
              env.step() 이후 참조 오염을 피하기 위해 raw_obs 직접 참조 대신 사용.

    계수는 config.yaml의 training.cap_bonus_gain / cap_bonus_loss로 관리.
    """
    gain_coef = T.get("cap_bonus_gain", 0.05)
    loss_coef = T.get("cap_bonus_loss", 0.025)

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
            bonus += prod * gain_coef
        elif prev_owner == player and owner != player:  # 내 것 → 잃음
            bonus -= prod * loss_coef
    return bonus


# ── Action masking ───────────────────────────────────────────────────────────

class ActionSpace:
    """analyze_action_space의 step-local 분석 결과.

    공유 가능한 자원:
      planets   — Planet 객체 리스트 (raw_planets 1회 파싱)
      pos_cache — PositionCache (mask 단계에서 (pid, turn) 미리 채워짐)

    Mask:
      launch_mask, target_mask — ships_rep=src.ships 기준 viability
                                  (decode는 actual ships로 별도 viability 재확인)
    """

    __slots__ = ("planets", "pos_cache", "launch_mask", "target_mask",
                 "av", "acting_player")

    def __init__(self, planets, pos_cache, launch_mask, target_mask, av, acting_player):
        self.planets       = planets
        self.pos_cache     = pos_cache
        self.launch_mask   = launch_mask
        self.target_mask   = target_mask
        self.av            = av
        self.acting_player = acting_player


def analyze_action_space(raw_planets, av, acting_player):
    """공통 분석 1회 + masks 생성.

    재사용 의도:
      - planets/pos_cache → decode_action_to_moves(analysis=...)에 전달
      - launch_mask/target_mask → 모델 forward에 전달

    viability는 ships_rep=src.ships(최대값, permissive)로 평가.
    decode는 actual ships_needed로 fcop를 다시 호출 (속도가 다르면 path도 다름).
    """
    planets      = [Planet(*p) for p in raw_planets[:MAX_PLANETS]]
    pos_cache    = PositionCache(planets, av)
    launch_mask  = torch.zeros(MAX_PLANETS, dtype=torch.bool)
    target_mask  = torch.zeros(MAX_PLANETS, MAX_PLANETS, dtype=torch.bool)

    for i, src in enumerate(planets):
        if src.owner != acting_player or src.ships <= 0:
            continue
        for j, tgt in enumerate(planets):
            if i == j or tgt.owner == acting_player:
                continue
            if crosses_sun(src.x, src.y, tgt.x, tgt.y):
                continue
            ships_rep = int(src.ships)
            angle, tx, ty, turns = aim(src, tgt, av, ships_rep, pos_cache=pos_cache)
            max_turns = min((turns or 0) + 2, 120) if turns else 120
            cause, hit_pid = first_collision_on_path(
                src, angle, ships_rep, planets, av, max_turns=max_turns,
                pos_cache=pos_cache,
            )
            if cause == "planet" and hit_pid == tgt.id:
                target_mask[i, j] = True

    for i, src in enumerate(planets):
        if src.owner != acting_player or src.ships <= 0:
            continue
        if target_mask[i].any():
            launch_mask[i] = True

    # All-false row fallback: Categorical NaN 방지용 self 허용
    # (launch_mask[i]=False이므로 launch=0으로 게이팅되어 학습 영향 없음)
    for i in range(MAX_PLANETS):
        if not target_mask[i].any():
            target_mask[i, i] = True

    return ActionSpace(planets, pos_cache, launch_mask, target_mask, av, acting_player)


def build_action_masks(raw_planets, av, acting_player):
    """Backward-compat 래퍼: analyze_action_space의 (launch_mask, target_mask)만 노출.

    신규 코드는 analyze_action_space를 직접 사용하고 ActionSpace를
    decode_action_to_moves에 그대로 넘겨 planets/pos_cache 재사용 권장.
    """
    a = analyze_action_space(raw_planets, av, acting_player)
    return a.launch_mask, a.target_mask


# ── Agent 행동 생성 ───────────────────────────────────────────────────────────

def decode_action_to_moves(action_np, raw_planets, av, acting_player,
                           return_counts=False, analysis=None):
    """Pure function: 샘플된 action_np → env moves 리스트. 모델 접근 없음.

    acting_player: 절대 owner ID (0 or 1). 행성 소유 판정에 사용.
    return_counts: True면 (moves, counts, launches) 튜플 반환 (hit rate tracking용).
        launches: moves와 1:1 대응하는 메타 dict 리스트 (source_id, target_id,
        ships, angle, start_x, start_y). fleet_id 매핑/resolve_step 입력으로 사용.
    analysis: ActionSpace. 있으면 planets/pos_cache 재사용 (mask 단계와 공유).
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
               # ── ships 분포 실측 (commit 2: Categorical multiplier head) ──
               "chosen_multiplier_sum": 0.0,     # 선택된 배수 평균 (1.10~2.00)
               "chosen_multiplier_sq_sum": 0.0,  # std 계산용
               "ships_to_send_sum": 0,           # 실제 발사 ships 수 평균
               "required_ships_sum": 0.0,        # 필요 병력 추정치 평균
               "send_required_ratio_sum": 0.0,   # ships_to_send / required 평균
               "under_invested_count": 0,        # ships_needed < int(required × multiplier) 횟수 (src.ships clip으로 nominal margin 미달)
               # ── target-type 분리 (neutral=prod 무시 가능 / enemy=prod 회복) ──
               # 도메인 차이: 중립은 prod 없음 → under-invest해도 단발 손실만, 반면 적은
               # prod로 재생산 → 같은 ratio라도 적 대상 under-invest가 장기적으로 더 큰 waste.
               # 승패 상관관계 분석: under_invested_rate_enemy가 지표로서 더 날카로움.
               "ships_to_send_sum_neutral": 0,   "ships_to_send_sum_enemy": 0,
               "required_ships_sum_neutral": 0.0, "required_ships_sum_enemy": 0.0,
               "send_required_ratio_sum_neutral": 0.0, "send_required_ratio_sum_enemy": 0.0,
               "under_invested_count_neutral": 0, "under_invested_count_enemy": 0,
               # ── 1차 진단 metric (자원 보존 측정) ──────────────────────────
               # all_in_launches: ships_needed >= 0.8 * src.ships (source를 거의 비움)
               # remaining_ships_after_launch_sum: 발사 후 source에 남은 ships 합
               #   둘 다 launched 분모로 나눠 rate/mean 산출.
               "all_in_launches": 0,
               "remaining_ships_after_launch_sum": 0}
    # ships_bin 선택 히스토그램 (K bins): counts["ships_bin_hist_k"] = count
    for k in range(NUM_SHIPS_BINS):
        counts[f"ships_bin_hist_{k}"] = 0
    target_prods = [t.production for t in planets if t.owner != acting_player]
    high_prod_threshold = np.quantile(target_prods, 0.75) if target_prods else None

    for i, p in enumerate(planets[:MAX_PLANETS]):
        if p.owner != acting_player:
            continue
        launch     = action_np[i, 0]
        # Action layout: [launch(1), ships_bin_onehot(K), target_onehot(P)]
        ships_bin  = int(np.argmax(action_np[i, 1:1 + NUM_SHIPS_BINS]))
        target_idx = int(np.argmax(action_np[i, 1 + NUM_SHIPS_BINS:
                                              1 + NUM_SHIPS_BINS + len(planets)]))
        multiplier = float(SHIPS_MULTIPLIER_BINS[ships_bin])

        if launch < 0.5:
            continue
        counts["attempts"] += 1

        if target_idx >= len(planets) or planets[target_idx].owner == acting_player:
            counts["filtered_invalid_target"] += 1
            continue

        target = planets[target_idx]

        # 고정점 반복으로 (ships_needed, required) 동시 해결.
        # 과거 1-pass 근사(p.ships로 turns 추정)는 느린 함대의 추가 production을
        # 과소평가해 bin=1.10x가 상습 under-invested였음 → commit 3에서 수정.
        ships_needed, angle, tx, ty, turns, required, _ = resolve_ships_for_capture(
            p, target, av, multiplier, p.ships, pos_cache=pos_cache,
        )
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
        # send_required_ratio = ships_needed(clip 후) / required  (실제 공급 비율)
        # under_invested     = ships_needed < int(required × multiplier)
        #                      즉 src.ships clip으로 nominal multiplier margin을 못 채운 경우.
        #                      commit 3 resolver가 margin을 보장하므로 이 분기는
        #                      정확히 "src.ships clip" 시점과 일치 (bin 선택이 과도히 ambitious).
        srr = ships_needed / max(required, 1)
        nominal_need = int(required * multiplier)
        under_invested = ships_needed < nominal_need
        counts["chosen_multiplier_sum"]    += multiplier
        counts["chosen_multiplier_sq_sum"] += multiplier ** 2
        counts["ships_to_send_sum"]        += ships_needed
        counts["required_ships_sum"]       += required
        counts["send_required_ratio_sum"]  += srr
        counts[f"ships_bin_hist_{ships_bin}"] += 1
        if under_invested:
            counts["under_invested_count"] += 1
        # target-type 분리 (neutral vs enemy)
        suffix = "neutral" if target.owner == -1 else "enemy"
        counts[f"ships_to_send_sum_{suffix}"]       += ships_needed
        counts[f"required_ships_sum_{suffix}"]      += required
        counts[f"send_required_ratio_sum_{suffix}"] += srr
        if under_invested:
            counts[f"under_invested_count_{suffix}"] += 1

        # ── 1차 진단: all-in / 잔여 ships ─────────────────────────────────
        # all-in: 한 번에 source의 80%+ 비우는 발사 (자원 무시 직접 지표).
        # remaining_ships: 발사 후 source 잔여 (= 방어 reserve / 다음 턴 여력).
        if p.ships > 0 and ships_needed >= HitRateTracker.ALL_IN_THRESHOLD * p.ships:
            counts["all_in_launches"] += 1
        counts["remaining_ships_after_launch_sum"] += max(p.ships - ships_needed, 0)

        moves.append([p.id, angle, ships_needed])
        start_x = p.x + math.cos(angle) * (p.radius + 0.1)
        start_y = p.y + math.sin(angle) * (p.radius + 0.1)
        launches.append({
            "source_id": p.id,
            "target_id": target.id,
            "target_owner": target.owner,   # -1: neutral, 그 외: 적 (우리는 self 마스킹됨)
            "ships": ships_needed,
            "angle": angle,
            "start_x": start_x,
            "start_y": start_y,
        })

    if return_counts:
        return moves, counts, launches
    return moves


def _opp_moves(opponent_model, obs_tensor, raw_planets, av, device):
    """상대(player 1) 행동 생성 (PPO 저장 불필요 — 별도 샘플링 허용)."""
    if opponent_model is None:
        return []
    analysis = analyze_action_space(raw_planets, av, acting_player=1)
    with torch.no_grad():
        action, *_ = opponent_model.get_action_and_value(
            obs_tensor.unsqueeze(0).to(device),
            launch_mask=analysis.launch_mask.unsqueeze(0).to(device),
            target_mask=analysis.target_mask.unsqueeze(0).to(device),
        )
    return decode_action_to_moves(
        action.squeeze(0).cpu().numpy(), raw_planets, av,
        acting_player=1, analysis=analysis,
    )


def get_obs_tensor(raw_obs, player, history_p, history_f):
    if isinstance(raw_obs, dict):
        raw_planets = raw_obs.get("planets", [])
        raw_fleets  = raw_obs.get("fleets", [])
        av          = raw_obs.get("angular_velocity", 0)
        comet_ids   = set(raw_obs.get("comet_planet_ids", []) or [])
    else:
        raw_planets = getattr(raw_obs, "planets", [])
        raw_fleets  = getattr(raw_obs, "fleets", [])
        av          = getattr(raw_obs, "angular_velocity", 0)
        comet_ids   = set(getattr(raw_obs, "comet_planet_ids", []) or [])

    history_p.append(encode_planets(raw_planets, raw_fleets, player, comet_ids, av))
    history_f.append(encode_fleets(raw_fleets, player))

    p_hist = np.stack(list(history_p), axis=0)
    f_hist = np.stack(list(history_f), axis=0)
    flat   = np.concatenate([p_hist.flatten(), f_hist.flatten()]).astype(np.float32)
    return torch.from_numpy(flat), raw_planets, av


# ── 단일 env rollout (CPU/GPU 모두 지원) ─────────────────────────────────────

def _collect_single(main_model, opponent_model, n_steps, device):
    obs_list, act_list, rew_list, done_list, logp_list, val_list = [], [], [], [], [], []
    logp_heads_list = []
    launch_mask_list, target_mask_list = [], []
    hit_tracker = HitRateTracker()
    sum_dense = sum_cap = sum_terminal = 0.0
    sum_all_in_penalty = 0.0   # Sprint 2: 발사 시 자원 보존 인센티브 (음수 누적)
    # win_rate 계측: episode 종료 시 main(player=0) reward로 win/draw/loss 집계.
    # draw는 0.5 가중치 (eval과 동일 규약).
    sum_wins = 0.0

    env = make("orbit_wars", debug=False)
    env.reset()
    hit_tracker.reset_episode(env.state[0].observation)

    history_p     = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
    history_f     = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)
    history_p_opp = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
    history_f_opp = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)

    dense_coef = T["dense_reward_coef"]
    terminal_win_reward = float(T.get("terminal_win_reward", 1.0))
    # Sprint 2: 발사 시 source 80%+ 비우는 발사당 페널티 (음수). 0 이면 비활성.
    all_in_penalty_coef = float(T.get("all_in_penalty", 0.0))
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
            obs_t, raw_planets, av = get_obs_tensor(raw_obs_main, 0, history_p, history_f)

            analysis = analyze_action_space(raw_planets, av, acting_player=0)
            launch_mask, target_mask = analysis.launch_mask, analysis.target_mask
            with torch.no_grad():
                action_t, log_prob, value, lp_heads = main_model.get_action_and_value(
                    obs_t.unsqueeze(0).to(device),
                    launch_mask=launch_mask.unsqueeze(0).to(device),
                    target_mask=target_mask.unsqueeze(0).to(device),
                )
            action_np  = action_t.squeeze(0).cpu().numpy()
            moves_main, decode_counts, launches_main = decode_action_to_moves(
                action_np, raw_planets, av, acting_player=0, return_counts=True,
                analysis=analysis,
            )
            hit_tracker.record(decode_counts)

            raw_obs_opp = env.state[1].observation
            obs_opp, raw_planets_opp, av_opp = get_obs_tensor(raw_obs_opp, 1, history_p_opp, history_f_opp)
            moves_opp = _opp_moves(opponent_model, obs_opp, raw_planets_opp, av_opp, device)

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
            cap_bonus      = neutral_capture_bonus(prev_map, curr_obs_main, player=0)
            # Sprint 2: 이번 step 의 all-in 발사 수에 비례한 페널티 (decode 시 이미 카운트됨).
            #   penalty = -coef × n_all_in   (coef=0 이면 비활성, Sprint 1 baseline 동일)
            all_in_penalty = -all_in_penalty_coef * decode_counts.get("all_in_launches", 0)
            terminal_r     = 0.0
            reward         = dense_r + cap_bonus + all_in_penalty
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
            sum_terminal += terminal_r

            obs_list.append(obs_t)
            act_list.append(action_t.squeeze(0).cpu())
            rew_list.append(torch.tensor(reward, dtype=torch.float32))
            done_list.append(torch.tensor(float(ep_done), dtype=torch.float32))
            logp_list.append(log_prob.squeeze(0).cpu())
            logp_heads_list.append(lp_heads.squeeze(0).cpu())
            val_list.append(value.squeeze(0).cpu())
            launch_mask_list.append(launch_mask)
            target_mask_list.append(target_mask)

            step += 1

        # 에피소드 완주 후 target 도달 여부 확인
        if step >= n_steps:
            break

        # 다음 에피소드 준비 (env/history 리셋)
        env = make("orbit_wars", debug=False)
        env.reset()
        hit_tracker.reset_episode(env.state[0].observation)
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
        "sum_terminal": sum_terminal,
        "sum_wins":     sum_wins,
    }
    return (
        torch.stack(obs_list),
        torch.stack(act_list),
        advantages,
        returns,
        torch.stack(logp_list),
        torch.stack(logp_heads_list),
        torch.stack(launch_mask_list),
        torch.stack(target_mask_list),
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
    total_wins = 0.0
    for r in raw_list:
        for k, v in r["counters"].items():
            total_counters[k] += v
        total_n_steps  += r["n_steps"]
        total_episodes += r["episodes"]
        total_dense    += r["sum_dense"]
        total_cap      += r["sum_cap"]
        total_all_in_penalty += r.get("sum_all_in_penalty", 0.0)
        total_terminal += r["sum_terminal"]
        total_wins     += r.get("sum_wins", 0.0)

    stats = HitRateTracker.summary_from_counters(
        total_counters, total_n_steps, total_episodes,
    )
    steps_safe = max(total_n_steps, 1)
    stats["mean_dense"]    = total_dense    / steps_safe
    stats["mean_cap"]      = total_cap      / steps_safe
    stats["mean_terminal"] = total_terminal / steps_safe
    # Sprint 2: per-step all-in launch penalty (음수). all_in_penalty=0 이면 0.
    stats["mean_all_in_penalty"] = total_all_in_penalty / steps_safe
    # rollout 내 main(player=0) 승률. on-policy sampling이라 eval보다 noisy지만,
    # match_type별 log로 분포별 성능을 바로 볼 수 있는 이점.
    stats["win_rate"]      = total_wins / max(total_episodes, 1)
    # Episode-grained collection에서 generation마다 실제 수집량이 달라지므로
    # target_steps 대비 초과량으로 "unusually 길었던 generation" 감지 가능.
    stats["actual_steps"]  = total_n_steps
    return stats


def collect_rollout(main_model, opponent_model, n_steps=512, n_envs=1, pool=None):
    """n_envs=1이면 단일 env, 그 이상이면 multiprocessing 병렬 rollout.

    _collect_single은 raw_stats(counters/n_steps/episodes/sum_*)를 반환.
    여기서 worker들을 합산한 뒤 step-weighted로 정확히 정규화해서
    최종 reward_stats를 만든다 (full-episode collection에서 worker별
    episode 길이가 달라도 편향 없음).
    """
    if n_envs <= 1 or pool is None:
        result = _collect_single(main_model, opponent_model, n_steps, DEVICE)
        merged_stats = _finalize_reward_stats([result[8]])
        return (*result[:8], merged_stats)

    main_state = {k: v.cpu() for k, v in main_model.state_dict().items()}
    opp_state  = ({k: v.cpu() for k, v in opponent_model.state_dict().items()}
                  if opponent_model is not None else None)

    steps_per_env = max(1, n_steps // n_envs)
    worker_args   = [(main_state, opp_state, steps_per_env)] * n_envs

    results = pool.map(_rollout_worker, worker_args)

    merged_stats = _finalize_reward_stats([r[8] for r in results])
    return (
        torch.cat([r[0] for r in results]),
        torch.cat([r[1] for r in results]),
        torch.cat([r[2] for r in results]),
        torch.cat([r[3] for r in results]),
        torch.cat([r[4] for r in results]),
        torch.cat([r[5] for r in results]),
        torch.cat([r[6] for r in results]),
        torch.cat([r[7] for r in results]),
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
               old_logp_heads=None, launch_masks=None, target_masks=None,
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
    if launch_masks is not None:
        launch_masks = launch_masks.to(DEVICE)
    if target_masks is not None:
        target_masks = target_masks.to(DEVICE)

    N = len(obs)
    p_losses, v_losses, e_losses = [], [], []
    approx_kls, clip_fracs       = [], []
    ent_launches, ent_ships_l, ent_targets = [], [], []
    kl_l_hist, kl_s_hist, kl_t_hist = [], [], []
    cf_l_hist, cf_s_hist, cf_t_hist = [], [], []
    epochs_done = 0

    for epoch in range(n_epochs):
        early_stop = False
        idx = torch.randperm(N, device=DEVICE)
        for start in range(0, N, minibatch_size):
            mb = idx[start:start + minibatch_size]

            lm_mb = launch_masks[mb] if launch_masks is not None else None
            tm_mb = target_masks[mb] if target_masks is not None else None
            log_probs, entropy, values, ent_l, ent_s, ent_t, lp_heads = model.evaluate_actions(
                obs[mb], actions[mb], launch_mask=lm_mb, target_mask=tm_mb,
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
                    diff_h   = lp_heads - old_logp_heads[mb]           # (B, 3)
                    ratio_h  = diff_h.exp()
                    kl_h     = ((ratio_h - 1) - diff_h).mean(dim=0)    # (3,)
                    cf_h     = ((ratio_h - 1).abs() > clip_range).float().mean(dim=0)
                    kl_l_hist.append(kl_h[0].item())
                    kl_s_hist.append(kl_h[1].item())
                    kl_t_hist.append(kl_h[2].item())
                    cf_l_hist.append(cf_h[0].item())
                    cf_s_hist.append(cf_h[1].item())
                    cf_t_hist.append(cf_h[2].item())

            p_losses.append(policy_loss.item())
            v_losses.append(value_loss.item())
            e_losses.append(entropy_loss.item())
            approx_kls.append(approx_kl)
            clip_fracs.append(clip_frac)
            ent_launches.append(ent_l.mean().item())
            ent_ships_l.append(ent_s.mean().item())
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
        "kl_launch": _avg(kl_l_hist), "kl_ships": _avg(kl_s_hist), "kl_target": _avg(kl_t_hist),
        "cf_launch": _avg(cf_l_hist), "cf_ships": _avg(cf_s_hist), "cf_target": _avg(cf_t_hist),
    }

    return (
        sum(p_losses) / len(p_losses),
        sum(v_losses) / len(v_losses),
        sum(e_losses) / len(e_losses),
        sum(approx_kls) / len(approx_kls),
        sum(clip_fracs) / len(clip_fracs),
        epochs_done,
        sum(ent_launches) / len(ent_launches),
        sum(ent_ships_l)  / len(ent_ships_l),
        sum(ent_targets)  / len(ent_targets),
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

        while not env.done:
            raw_main = env.state[0].observation
            obs_t, raw_p, av = get_obs_tensor(raw_main, 0, history_p, history_f)
            analysis_e = analyze_action_space(raw_p, av, acting_player=0)
            with torch.no_grad():
                action_t, _, _, _ = main_model.get_action_and_value(
                    obs_t.unsqueeze(0).to(DEVICE),
                    launch_mask=analysis_e.launch_mask.unsqueeze(0).to(DEVICE),
                    target_mask=analysis_e.target_mask.unsqueeze(0).to(DEVICE),
                )
            moves_main, counts, _launches = decode_action_to_moves(
                action_t.squeeze(0).cpu().numpy(), raw_p, av,
                acting_player=0, analysis=analysis_e, return_counts=True,
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
            obs_o, raw_po, avo = get_obs_tensor(raw_opp, 1, history_p_opp, history_f_opp)
            moves_opp = _opp_moves(opponent_model, obs_o, raw_po, avo, DEVICE)

            env.step([moves_main, moves_opp])

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

    while not env.done:
        raw_obs_p0 = env.state[0].observation
        obs_t, raw_p, av = get_obs_tensor(raw_obs_p0, 0, history_p, history_f)
        analysis = analyze_action_space(raw_p, av, acting_player=0)
        with torch.no_grad():
            action_t, _, _, _ = p0_model.get_action_and_value(
                obs_t.unsqueeze(0).to(DEVICE),
                launch_mask=analysis.launch_mask.unsqueeze(0).to(DEVICE),
                target_mask=analysis.target_mask.unsqueeze(0).to(DEVICE),
            )
        moves_p0, counts, launches = decode_action_to_moves(
            action_t.squeeze(0).cpu().numpy(), raw_p, av,
            acting_player=0, analysis=analysis, return_counts=True,
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
        obs_o, raw_po, avo = get_obs_tensor(raw_obs_p1, 1, history_p_opp, history_f_opp)
        moves_p1 = _opp_moves(p1_model, obs_o, raw_po, avo, DEVICE)

        prev_obs_snap = _snapshot_obs_for_resolve(raw_obs_p0)
        tracker.register_launches(launches, prev_obs_snap["next_fleet_id"])
        env.step([moves_p0, moves_p1])

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
    result = load_checkpoint(ckpt_path, main_model, optimizer, DEVICE)
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

            opponent, match_type = _sample_opponent(main_model, exploiter, league)

            obs, actions, advantages, returns, log_probs, logp_heads, lmasks, tmasks, rew_stats = collect_rollout(
                main_model, opponent, n_steps=rollout_steps, n_envs=n_envs, pool=pool
            )
            p_loss, v_loss, e_loss, approx_kl, clip_frac, epochs_done, ent_l, ent_s, ent_t, head_metrics = ppo_update(
                main_model, optimizer, obs, actions, log_probs, returns, advantages,
                old_logp_heads=logp_heads, launch_masks=lmasks, target_masks=tmasks,
            )
            total_steps += len(obs)

            exp_opp = copy.deepcopy(main_model)
            exp_opp.eval()
            obs_e, act_e, adv_e, ret_e, logp_e, logp_heads_e, lmasks_e, tmasks_e, _ = collect_rollout(
                exploiter, exp_opp, n_steps=max(1, rollout_steps // 2), n_envs=max(1, n_envs // 2), pool=pool
            )
            ppo_update(exploiter, exploiter_opt, obs_e, act_e, logp_e, ret_e, adv_e,
                       old_logp_heads=logp_heads_e, launch_masks=lmasks_e, target_masks=tmasks_e)

            logger.log(
                generation=generation, total_steps=total_steps, match_type=match_type,
                policy_loss=p_loss, value_loss=v_loss, entropy_loss=e_loss,
                approx_kl=approx_kl, clip_frac=clip_frac, epochs_done=epochs_done,
                ent_launch=ent_l, ent_ships=ent_s, ent_target=ent_t,
                league_size=len(league),
                # rollout 내 승률 (on-policy, noisy지만 match_type별로 분포별 성능 확인 가능).
                win_rate=rew_stats.get("win_rate", 0.0),
                mean_dense_rew=rew_stats["mean_dense"],
                mean_cap_bonus=rew_stats["mean_cap"],
                mean_terminal_rew=rew_stats["mean_terminal"],
                # Sprint 2: 발사 시 자원 보존 페널티 (음수). all_in_penalty=0 이면 0.
                mean_all_in_penalty=rew_stats.get("mean_all_in_penalty", 0.0),
                mean_attempts=rew_stats["mean_attempts"],
                mean_launched=rew_stats["mean_launched"],
                launch_rate=rew_stats["launch_rate"],
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
                chosen_multiplier_mean=rew_stats.get("chosen_multiplier_mean", 0.0),
                chosen_multiplier_std=rew_stats.get("chosen_multiplier_std", 0.0),
                ships_to_send_mean=rew_stats.get("ships_to_send_mean", 0.0),
                required_ships_mean=rew_stats.get("required_ships_mean", 0.0),
                send_required_ratio_mean=rew_stats.get("send_required_ratio_mean", 0.0),
                under_invested_rate=rew_stats.get("under_invested_rate", 0.0),
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
    parser.add_argument("--smoke",            action="store_true",               help="smoke test 세팅 (10000 steps, eval 5, 4 games)")
    cli = parser.parse_args()

    DEVICE   = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    SAVE_DIR = cli.run_dir

    # config 기본값 적용 (CLI 미지정 시)
    cli.n_envs        = cli.n_envs        if cli.n_envs        is not None else T.get("n_envs", 1)
    cli.rollout_steps = cli.rollout_steps if cli.rollout_steps is not None else T.get("rollout_steps", 512)

    if cli.smoke:
        cli.total_timesteps = cli.total_timesteps or 10000
        cli.eval_interval   = cli.eval_interval   or 5
        cli.n_games         = cli.n_games         or 4

    train(
        n_envs=cli.n_envs,
        total_timesteps=cli.total_timesteps,
        eval_interval=cli.eval_interval,
        n_games=cli.n_games,
        rollout_steps=cli.rollout_steps,
    )
