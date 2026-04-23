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
from env_wrapper import encode_planets, encode_fleets, MAX_PLANETS, MAX_FLEETS, PLANET_DIM, FLEET_DIM, HISTORY
from prediction import aim, crosses_sun, first_collision_on_path

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

T  = CFG["training"]
SP = CFG["selfplay"]

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

    def add(self, model, generation, win_rate):
        path = os.path.join(SAVE_DIR, f"gen_{generation:04d}.pt")
        torch.save(model.state_dict(), path)
        self.agents.append((path, win_rate, generation))
        if len(self.agents) > self.pool_size:
            self.agents.pop(0)
        print(f"League: gen {generation} 추가 (승률 {win_rate:.2%})")

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

    (planet_ships + fleet_ships) * 0.01 + production * 1.0 + planet_count * 2.0
    production / planet_count 가중치 상향: 초반 영토 확장의 학습 신호 강화.
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

    return total_ships * 0.01 + total_prod * 1.0 + planet_count * 2.0


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


# ── Agent 행동 생성 ───────────────────────────────────────────────────────────

def decode_action_to_moves(action_np, raw_planets, av, acting_player, return_counts=False):
    """Pure function: 샘플된 action_np → env moves 리스트. 모델 접근 없음.

    acting_player: 절대 owner ID (0 or 1). 행성 소유 판정에 사용.
    return_counts: True면 (moves, counts, launches) 튜플 반환 (hit rate tracking용).
        launches: moves와 1:1 대응하는 메타 dict 리스트 (source_id, target_id,
        ships, angle, start_x, start_y). fleet_id 매핑/resolve_step 입력으로 사용.
    """
    planets = [Planet(*p) for p in raw_planets]
    moves   = []
    launches = []
    counts  = {"attempts": 0, "filtered_invalid_target": 0,
               "filtered_zero_ships": 0, "filtered_sun": 0,
               "filtered_path": 0, "launched": 0}

    for i, p in enumerate(planets[:MAX_PLANETS]):
        if p.owner != acting_player:
            continue
        launch      = action_np[i, 0]
        ships_ratio = float(np.clip(action_np[i, 1], 0.0, 1.0))
        target_idx  = int(np.argmax(action_np[i, 2:2 + len(planets)]))

        if launch < 0.5:
            continue
        counts["attempts"] += 1

        if target_idx >= len(planets) or planets[target_idx].owner == acting_player:
            counts["filtered_invalid_target"] += 1
            continue

        target       = planets[target_idx]
        ships_needed = max(1, int(p.ships * ships_ratio))
        ships_needed = min(ships_needed, p.ships)
        if ships_needed <= 0:
            counts["filtered_zero_ships"] += 1
            continue

        angle, tx, ty, turns = aim(p, target, av, ships_needed)
        if crosses_sun(p.x, p.y, tx, ty):
            counts["filtered_sun"] += 1
            continue

        max_turns = min((turns or 0) + 2, 120) if turns else 120
        cause, hit_pid = first_collision_on_path(
            p, angle, ships_needed, planets, av, max_turns=max_turns,
        )
        if cause != "planet" or hit_pid != target.id:
            counts["filtered_path"] += 1
            continue

        counts["launched"] += 1
        moves.append([p.id, angle, ships_needed])
        start_x = p.x + math.cos(angle) * (p.radius + 0.1)
        start_y = p.y + math.sin(angle) * (p.radius + 0.1)
        launches.append({
            "source_id": p.id,
            "target_id": target.id,
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
    with torch.no_grad():
        action, *_ = opponent_model.get_action_and_value(obs_tensor.unsqueeze(0).to(device))
    return decode_action_to_moves(action.squeeze(0).cpu().numpy(), raw_planets, av, acting_player=1)


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
    hit_tracker = HitRateTracker()
    sum_dense = sum_cap = sum_terminal = 0.0

    env = make("orbit_wars", debug=False)
    env.reset()

    history_p     = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
    history_f     = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)
    history_p_opp = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
    history_f_opp = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)

    dense_coef = T["dense_reward_coef"]
    prev_score = (state_score(env.state[0].observation, player=0)
                - state_score(env.state[1].observation, player=1))

    step = 0
    while step < n_steps:
        raw_obs_main = env.state[0].observation
        obs_t, raw_planets, av = get_obs_tensor(raw_obs_main, 0, history_p, history_f)

        with torch.no_grad():
            action_t, log_prob, value, lp_heads = main_model.get_action_and_value(obs_t.unsqueeze(0).to(device))
        action_np  = action_t.squeeze(0).cpu().numpy()
        moves_main, decode_counts, launches_main = decode_action_to_moves(
            action_np, raw_planets, av, acting_player=0, return_counts=True,
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
        done = env.done

        curr_obs_main  = env.state[0].observation
        max_speed      = env.configuration.shipSpeed
        hit_tracker.resolve_step(prev_obs_snap, curr_obs_main, max_speed)
        curr_score     = (state_score(curr_obs_main, player=0)
                        - state_score(env.state[1].observation, player=1))
        dense_r        = dense_coef * (curr_score - prev_score)
        cap_bonus      = neutral_capture_bonus(prev_map, curr_obs_main, player=0)
        terminal_r     = 0.0
        reward         = dense_r + cap_bonus
        prev_score     = curr_score

        if done:
            r          = env.state[0].reward
            terminal_r = 1.0 if r == 1 else (-1.0 if r == -1 else 0.0)
            reward    += terminal_r

        sum_dense    += dense_r
        sum_cap      += cap_bonus
        sum_terminal += terminal_r

        obs_list.append(obs_t)
        act_list.append(action_t.squeeze(0).cpu())
        rew_list.append(torch.tensor(reward, dtype=torch.float32))
        done_list.append(torch.tensor(float(done), dtype=torch.float32))
        logp_list.append(log_prob.squeeze(0).cpu())
        logp_heads_list.append(lp_heads.squeeze(0).cpu())
        val_list.append(value.squeeze(0).cpu())

        step += 1
        if done:
            env = make("orbit_wars", debug=False)
            env.reset()
            hit_tracker.pending.clear()  # fleet id는 env 단위로 리셋됨
            prev_score    = (state_score(env.state[0].observation, player=0)
                           - state_score(env.state[1].observation, player=1))
            history_p     = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
            history_f     = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)
            history_p_opp = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
            history_f_opp = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)

    # rollout 마지막 다음 상태의 critic value (non-terminal bootstrap용)
    last_raw = env.state[0].observation
    last_obs_t, _, _ = get_obs_tensor(last_raw, 0, history_p, history_f)
    with torch.no_grad():
        _, _, last_value, _ = main_model.get_action_and_value(last_obs_t.unsqueeze(0).to(device))
    last_value = last_value.squeeze().cpu()

    rewards = torch.stack(rew_list)
    dones   = torch.stack(done_list)
    values  = torch.stack(val_list)
    advantages, returns = compute_gae(rewards, dones, values, last_value=last_value)

    reward_stats = {
        "mean_dense":    sum_dense    / max(n_steps, 1),
        "mean_cap":      sum_cap      / max(n_steps, 1),
        "mean_terminal": sum_terminal / max(n_steps, 1),
        **hit_tracker.summary(),
    }
    return (
        torch.stack(obs_list),
        torch.stack(act_list),
        advantages,
        returns,
        torch.stack(logp_list),
        torch.stack(logp_heads_list),
        reward_stats,
    )


# ── 병렬 rollout worker (별도 프로세스, CPU 전용) ───────────────────────────

def _init_worker():
    """worker 프로세스 초기화: CUDA 숨기기, 스레드 수 제한, 로그 억제."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
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


def collect_rollout(main_model, opponent_model, n_steps=512, n_envs=1, pool=None):
    """n_envs=1이면 단일 env, 그 이상이면 multiprocessing 병렬 rollout."""
    if n_envs <= 1 or pool is None:
        return _collect_single(main_model, opponent_model, n_steps, DEVICE)

    main_state = {k: v.cpu() for k, v in main_model.state_dict().items()}
    opp_state  = ({k: v.cpu() for k, v in opponent_model.state_dict().items()}
                  if opponent_model is not None else None)

    steps_per_env = max(1, n_steps // n_envs)
    worker_args   = [(main_state, opp_state, steps_per_env)] * n_envs

    results = pool.map(_rollout_worker, worker_args)

    # reward_stats: worker별 평균을 다시 평균
    keys = results[0][6].keys()
    merged_stats = {
        k: sum(r[6][k] for r in results) / len(results) for k in keys
    }
    return (
        torch.cat([r[0] for r in results]),
        torch.cat([r[1] for r in results]),
        torch.cat([r[2] for r in results]),
        torch.cat([r[3] for r in results]),
        torch.cat([r[4] for r in results]),
        torch.cat([r[5] for r in results]),
        merged_stats,
    )


# ── Phase 기반 self-play 비율 ─────────────────────────────────────────────────

def _self_play_prob(total_steps: int, league_size: int) -> float:
    """total_steps + league_size 기준으로 self-play 확률 반환.

    league_size < phase_min_league : early 강제 (pool 다양성 부족)
    total_steps < phase_early_steps: early  — self 0.8 / league 0.2
    total_steps < phase_mid_steps  : mid    — self 0.6 / league 0.4
    그 이후                         : late   — self 0.4 / league 0.6
    """
    if league_size == 0:
        return 1.0
    if league_size < SP["phase_min_league"] or total_steps < SP["phase_early_steps"]:
        return 0.8
    if total_steps < SP["phase_mid_steps"]:
        return 0.6
    return 0.4


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
               old_logp_heads=None,
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

            log_probs, entropy, values, ent_l, ent_s, ent_t, lp_heads = model.evaluate_actions(obs[mb], actions[mb])

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
    wins = 0
    for _ in range(n_games):
        env = make("orbit_wars", debug=False)
        env.reset()
        history_p     = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
        history_f     = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)
        history_p_opp = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
        history_f_opp = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)

        while not env.done:
            raw_main = env.state[0].observation
            obs_t, raw_p, av = get_obs_tensor(raw_main, 0, history_p, history_f)
            with torch.no_grad():
                action_t, _, _, _ = main_model.get_action_and_value(obs_t.unsqueeze(0).to(DEVICE))
            moves_main = decode_action_to_moves(action_t.squeeze(0).cpu().numpy(), raw_p, av, acting_player=0)

            raw_opp = env.state[1].observation
            obs_o, raw_po, avo = get_obs_tensor(raw_opp, 1, history_p_opp, history_f_opp)
            moves_opp = _opp_moves(opponent_model, obs_o, raw_po, avo, DEVICE)

            env.step([moves_main, moves_opp])

        r = env.state[0].reward
        if r == 1:
            wins += 1
        elif r == 0:
            wins += 0.5
    return wins / n_games


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

            self_prob  = _self_play_prob(total_steps, len(league))
            if random.random() < self_prob:
                opponent   = copy.deepcopy(main_model)
                opponent.eval()
                match_type = "self"
            else:
                opponent   = league.sample_opponent()
                match_type = "league"

            obs, actions, advantages, returns, log_probs, logp_heads, rew_stats = collect_rollout(
                main_model, opponent, n_steps=rollout_steps, n_envs=n_envs, pool=pool
            )
            p_loss, v_loss, e_loss, approx_kl, clip_frac, epochs_done, ent_l, ent_s, ent_t, head_metrics = ppo_update(
                main_model, optimizer, obs, actions, log_probs, returns, advantages,
                old_logp_heads=logp_heads,
            )
            total_steps += len(obs)

            exp_opp = copy.deepcopy(main_model)
            exp_opp.eval()
            obs_e, act_e, adv_e, ret_e, logp_e, logp_heads_e, _ = collect_rollout(
                exploiter, exp_opp, n_steps=max(1, rollout_steps // 2), n_envs=max(1, n_envs // 2), pool=pool
            )
            ppo_update(exploiter, exploiter_opt, obs_e, act_e, logp_e, ret_e, adv_e,
                       old_logp_heads=logp_heads_e)

            logger.log(
                generation=generation, total_steps=total_steps, match_type=match_type,
                policy_loss=p_loss, value_loss=v_loss, entropy_loss=e_loss,
                approx_kl=approx_kl, clip_frac=clip_frac, epochs_done=epochs_done,
                ent_launch=ent_l, ent_ships=ent_s, ent_target=ent_t,
                league_size=len(league),
                mean_dense_rew=rew_stats["mean_dense"],
                mean_cap_bonus=rew_stats["mean_cap"],
                mean_terminal_rew=rew_stats["mean_terminal"],
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
                **head_metrics,
            )

            if generation % eval_interval == 0:
                opp_eval = league.sample_opponent() or exploiter
                win_rate = evaluate(main_model, opp_eval, n_games=n_games)

                logger.log(
                    generation=generation, total_steps=total_steps, match_type="eval",
                    policy_loss=p_loss, value_loss=v_loss, entropy_loss=e_loss,
                    win_rate=win_rate, league_size=len(league),
                )

                if win_rate >= SP["win_threshold"]:
                    league.add(main_model, generation, win_rate)

                exploiter_reset += 1
                if exploiter_reset % 5 == 0:
                    exploiter     = OrbitWarsPolicy().to(DEVICE)
                    exploiter_opt = optim.Adam(exploiter.parameters(), lr=T["learning_rate"])
                    print("  Exploiter 리셋")

                save_checkpoint(ckpt_path, main_model, optimizer, generation, total_steps, league.agents)
                torch.save(main_model.state_dict(), os.path.join(SAVE_DIR, "main_latest.pt"))

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
    parser.add_argument("--n-envs",           type=int,   default=1,             help="병렬 env 수 (1=단일)")
    parser.add_argument("--total-timesteps",  type=int,   default=None,          help="학습 총 스텝 (기본: config)")
    parser.add_argument("--eval-interval",    type=int,   default=None,          help="평가 주기 세대 수 (기본: config)")
    parser.add_argument("--n-games",          type=int,   default=None,          help="평가 게임 수 (기본: 20)")
    parser.add_argument("--rollout-steps",    type=int,   default=None,          help="rollout 스텝 수 (기본: 512)")
    parser.add_argument("--smoke",            action="store_true",               help="smoke test 세팅 (10000 steps, eval 5, 4 games)")
    cli = parser.parse_args()

    DEVICE   = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    SAVE_DIR = cli.run_dir

    if cli.smoke:
        cli.total_timesteps = cli.total_timesteps or 10000
        cli.eval_interval   = cli.eval_interval   or 5
        cli.n_games         = cli.n_games         or 4
        cli.rollout_steps   = cli.rollout_steps   or 128

    train(
        n_envs=cli.n_envs,
        total_timesteps=cli.total_timesteps,
        eval_interval=cli.eval_interval,
        n_games=cli.n_games,
        rollout_steps=cli.rollout_steps,
    )
