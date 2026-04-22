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
import yaml
from collections import deque
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
import multiprocessing as mp

from model import OrbitWarsPolicy
from env_wrapper import encode_planets, encode_fleets, MAX_PLANETS, MAX_FLEETS, PLANET_DIM, FLEET_DIM, HISTORY
from prediction import aim, crosses_sun

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


# ── Agent 행동 생성 ───────────────────────────────────────────────────────────

def model_to_moves(model, obs_tensor, raw_planets, av, device=None):
    if device is None:
        device = DEVICE
    planets = [Planet(*p) for p in raw_planets]

    with torch.no_grad():
        action, _, _ = model.get_action_and_value(obs_tensor.unsqueeze(0).to(device))
    action = action.squeeze(0).cpu().numpy()

    moves = []
    for i, p in enumerate(planets[:MAX_PLANETS]):
        if p.owner != 0:
            continue
        launch      = action[i, 0]
        ships_ratio = (action[i, 1] + 1.0) / 2.0
        target_idx  = int(np.argmax(action[i, 2:2 + len(planets)]))

        if launch < 0.5:
            continue
        if target_idx >= len(planets) or planets[target_idx].owner == 0:
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

    history_p.append(encode_planets(raw_planets, raw_fleets, player, comet_ids))
    history_f.append(encode_fleets(raw_fleets, player))

    p_hist = np.stack(list(history_p), axis=0)
    f_hist = np.stack(list(history_f), axis=0)
    flat   = np.concatenate([p_hist.flatten(), f_hist.flatten()]).astype(np.float32)
    return torch.from_numpy(flat), raw_planets, av


# ── 단일 env rollout (CPU/GPU 모두 지원) ─────────────────────────────────────

def _collect_single(main_model, opponent_model, n_steps, device):
    obs_list, act_list, rew_list, done_list, logp_list, val_list = [], [], [], [], [], []

    env = make("orbit_wars", debug=False)
    env.reset()

    history_p     = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
    history_f     = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)
    history_p_opp = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
    history_f_opp = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)

    step = 0
    while step < n_steps:
        raw_obs_main = env.state[0].observation
        obs_t, raw_planets, av = get_obs_tensor(raw_obs_main, 0, history_p, history_f)

        with torch.no_grad():
            action_t, log_prob, value = main_model.get_action_and_value(obs_t.unsqueeze(0).to(device))
        moves_main = model_to_moves(main_model, obs_t, raw_planets, av, device)

        raw_obs_opp = env.state[1].observation
        obs_opp, raw_planets_opp, av_opp = get_obs_tensor(raw_obs_opp, 1, history_p_opp, history_f_opp)
        moves_opp = model_to_moves(opponent_model, obs_opp, raw_planets_opp, av_opp, device) if opponent_model else []

        env.step([moves_main, moves_opp])
        done = env.done

        reward = 0.0
        if done:
            r      = env.state[0].reward
            reward = 1.0 if r == 1 else (-1.0 if r == -1 else 0.0)

        obs_list.append(obs_t)
        act_list.append(action_t.squeeze(0).cpu())
        rew_list.append(torch.tensor(reward, dtype=torch.float32))
        done_list.append(torch.tensor(float(done), dtype=torch.float32))
        logp_list.append(log_prob.squeeze(0).cpu())
        val_list.append(value.squeeze(0).cpu())

        step += 1
        if done:
            env = make("orbit_wars", debug=False)
            env.reset()
            history_p     = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
            history_f     = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)
            history_p_opp = deque([np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY, maxlen=HISTORY)
            history_f_opp = deque([np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32)] * HISTORY, maxlen=HISTORY)

    return (
        torch.stack(obs_list),
        torch.stack(act_list),
        torch.stack(rew_list),
        torch.stack(done_list),
        torch.stack(logp_list),
        torch.stack(val_list),
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

    return (
        torch.cat([r[0] for r in results]),
        torch.cat([r[1] for r in results]),
        torch.cat([r[2] for r in results]),
        torch.cat([r[3] for r in results]),
        torch.cat([r[4] for r in results]),
        torch.cat([r[5] for r in results]),
    )


# ── PPO 업데이트 ──────────────────────────────────────────────────────────────

def compute_returns(rewards, dones, values, gamma=T["gamma"]):
    returns = torch.zeros_like(rewards)
    R = 0.0
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R * (1.0 - dones[t])
        returns[t] = R
    return returns


def ppo_update(model, optimizer, obs, actions, old_log_probs, returns, clip_range=T["clip_range"]):
    advantages = returns - returns.mean()
    advantages = advantages / (advantages.std() + 1e-8)

    obs           = obs.to(DEVICE)
    actions       = actions.to(DEVICE)
    old_log_probs = old_log_probs.to(DEVICE)
    advantages    = advantages.to(DEVICE)
    returns       = returns.to(DEVICE)

    log_probs, entropy, values = model.evaluate_actions(obs, actions)

    ratio       = (log_probs - old_log_probs).exp()
    surr1       = ratio * advantages
    surr2       = ratio.clamp(1 - clip_range, 1 + clip_range) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss  = nn.functional.mse_loss(values.squeeze(-1), returns)
    entropy_loss= -entropy.mean()

    loss = policy_loss + T["vf_coef"] * value_loss + T["ent_coef"] * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), T["max_grad_norm"])
    optimizer.step()

    return policy_loss.item(), value_loss.item(), entropy_loss.item()


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
            moves_main = model_to_moves(main_model, obs_t, raw_p, av)

            raw_opp = env.state[1].observation
            obs_o, raw_po, avo = get_obs_tensor(raw_opp, 1, history_p_opp, history_f_opp)
            moves_opp = model_to_moves(opponent_model, obs_o, raw_po, avo) if opponent_model else []

            env.step([moves_main, moves_opp])

        r = env.state[0].reward
        if r == 1:
            wins += 1
        elif r == 0:
            wins += 0.5
    return wins / n_games


# ── Main Training Loop ────────────────────────────────────────────────────────

def train(n_envs=1):
    global DEVICE, SAVE_DIR

    print(f"Device: {DEVICE} | run_dir: {SAVE_DIR} | n_envs: {n_envs}")
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
        while total_steps < T["total_timesteps"]:
            generation += 1

            if random.random() < 0.5 or len(league) == 0:
                opponent   = copy.deepcopy(main_model)
                opponent.eval()
                match_type = "self"
            else:
                opponent   = league.sample_opponent()
                match_type = "league"

            obs, actions, rewards, dones, log_probs, values = collect_rollout(
                main_model, opponent, n_steps=512, n_envs=n_envs, pool=pool
            )
            returns = compute_returns(rewards, dones, values)
            p_loss, v_loss, e_loss = ppo_update(main_model, optimizer, obs, actions, log_probs, returns)
            total_steps += len(obs)

            exp_opp = copy.deepcopy(main_model)
            exp_opp.eval()
            obs_e, act_e, rew_e, done_e, logp_e, val_e = collect_rollout(
                exploiter, exp_opp, n_steps=256, n_envs=max(1, n_envs // 2), pool=pool
            )
            ret_e = compute_returns(rew_e, done_e, val_e)
            ppo_update(exploiter, exploiter_opt, obs_e, act_e, logp_e, ret_e)

            logger.log(
                generation=generation, total_steps=total_steps, match_type=match_type,
                policy_loss=p_loss, value_loss=v_loss, entropy_loss=e_loss,
                league_size=len(league),
            )

            if generation % SP["eval_interval"] == 0:
                opp_eval = league.sample_opponent() or exploiter
                win_rate = evaluate(main_model, opp_eval, n_games=20)

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
    parser.add_argument("--gpu",     type=int, default=0,             help="GPU index")
    parser.add_argument("--run-dir", type=str, default="checkpoints", help="checkpoint 저장 디렉토리")
    parser.add_argument("--n-envs",  type=int, default=1,             help="병렬 env 수 (1=단일)")
    cli = parser.parse_args()

    DEVICE   = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    SAVE_DIR = cli.run_dir

    train(n_envs=cli.n_envs)
