"""bots/greedy_eval.py — Greedy 가중치 튜닝 인프라 (C7/C8/C9).

C7 evaluate(weights, seeds, opponents) → stats
C8 random_search(n_trials)             → CSV 로깅, 정렬된 결과 반환
C9 refine(top_results, more_seeds)     → 분산 줄여서 재측정

설계:
  - opponents = ['random', 'baseline'] (baseline = default WEIGHTS greedy)
    self-play 제거 — 정보 약함.
  - sides = (0, 1) — Q1/Q4 시작 위치 편향 제거. n_games 2배 비용 감수.
  - objective = win_rate; timeout_rate > 0.05 이면 -1 (hard reject).
  - noop_rate metric only — penalty 없음.
  - CSV trial 마다 즉시 append + flush — crash 에 강함, resume 지원.
  - sampler: 1차 search 는 log-uniform [0.32, 3.16] (= 10**uniform(-0.5, 0.5)).
    bias 만 uniform [-1, 1].
  - C9 seeds 는 superset (0~15 → 0~63) — 기존 결과 포함하면서 분산 축소.
"""
from __future__ import annotations

import csv
import os
import random
import time
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np
from kaggle_environments import make

from bots.greedy_expand_support import (
    WEIGHTS as DEFAULT_WEIGHTS,
    GreedyExpandSupportBot,
)


TIMEOUT_FRAC_HARD_REJECT = 0.05


# ── opponent factory ────────────────────────────────────────────────────

def make_opponent(spec):
    """spec → kaggle env 가 받을 agent (str 'random' / callable)."""
    if spec == "random":
        return "random"
    if spec == "baseline":
        return GreedyExpandSupportBot().act
    if isinstance(spec, dict):
        return GreedyExpandSupportBot(weights=spec).act
    raise ValueError(f"unknown opponent spec: {spec!r}")


def opponent_label(spec) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return "custom"
    return "custom"


# ── timed agent wrapper ─────────────────────────────────────────────────

def make_timed_agent(bot: GreedyExpandSupportBot):
    """bot.act 를 closure 로 감싸 호출 시간 누적. agent.step_times 에 초.

    kaggle env 가 클래스 인스턴스 __call__ 을 인식 못 함 → 일반 함수로 노출.
    """
    times: List[float] = []
    def agent(obs):
        t0 = time.time()
        moves = bot.act(obs)
        times.append(time.time() - t0)
        return moves
    agent.step_times = times
    return agent


# ── C7 evaluate ─────────────────────────────────────────────────────────

def evaluate(
    weights: dict,
    seeds: List[int],
    opponents: List = ("random", "baseline"),
    sides: Tuple[int, ...] = (0, 1),
) -> dict:
    """paired (seed, opponent, side) 전체 돌려 stats 집계.

    n_games = len(seeds) × len(opponents) × len(sides).
    같은 seeds 를 다른 weights 비교에도 재사용 → paired comparison.
    """
    per_opp = defaultdict(lambda: {"wins": 0, "n": 0})
    total_wins = total_games = 0
    total_noop = total_my_steps = total_moves = 0
    timeout_games = 0
    all_step_times: List[float] = []

    for seed in seeds:
        for opp_spec in opponents:
            opp_label = opponent_label(opp_spec)
            for my_side in sides:
                bot_inst = GreedyExpandSupportBot(weights=weights)
                timed = make_timed_agent(bot_inst)
                opp = make_opponent(opp_spec)

                # orbit_wars 는 configuration seed 안 받음 → 전역 seed.
                # 같은 (seed, side) 라도 opponent 가 'random' 이면 그쪽 randomness
                # 도 같이 결정 — paired comparison 성립.
                random.seed(int(seed))
                np.random.seed(int(seed))
                env = make("orbit_wars", debug=False)
                if my_side == 0:
                    env.run([timed, opp])
                else:
                    env.run([opp, timed])

                my_state  = env.state[my_side]
                opp_state = env.state[1 - my_side]
                if my_state.get("status") == "TIMEOUT":
                    timeout_games += 1
                my_r  = my_state.get("reward")  or 0
                opp_r = opp_state.get("reward") or 0
                won = my_r > opp_r
                if won:
                    total_wins += 1
                    per_opp[opp_label]["wins"] += 1
                per_opp[opp_label]["n"] += 1
                total_games += 1

                for s in env.steps:
                    action = s[my_side].get("action")
                    if not action:
                        total_noop += 1
                    else:
                        total_moves += len(action)
                    total_my_steps += 1
                all_step_times.extend(timed.step_times)

    n = max(1, total_games)
    win_rate     = total_wins / n
    timeout_rate = timeout_games / n
    noop_rate    = total_noop / max(1, total_my_steps)

    objective = -1.0 if timeout_rate > TIMEOUT_FRAC_HARD_REJECT else win_rate

    times_ms = sorted(t * 1000 for t in all_step_times)
    if times_ms:
        avg_ms = sum(times_ms) / len(times_ms)
        p95_ms = times_ms[int(0.95 * (len(times_ms) - 1))]
        max_ms = times_ms[-1]
    else:
        avg_ms = p95_ms = max_ms = 0.0

    return {
        "n_games":        total_games,
        "wins":           total_wins,
        "losses":         total_games - total_wins,
        "win_rate":       win_rate,
        "objective":      objective,
        "noop_rate":      noop_rate,
        "timeout_rate":   timeout_rate,
        "moves_per_turn": total_moves / max(1, total_my_steps),
        "avg_step_ms":    avg_ms,
        "p95_step_ms":    p95_ms,
        "max_step_ms":    max_ms,
        "per_opponent": {k: {"win_rate": v["wins"] / max(1, v["n"]), "n": v["n"]}
                          for k, v in per_opp.items()},
    }


# ── sampler ─────────────────────────────────────────────────────────────

def default_sampler(rng: random.Random | None = None) -> dict:
    """log-uniform [0.32, 3.16] for weights, uniform [-1, 1] for biases."""
    rng = rng or random
    lo, hi = -0.5, 0.5
    return {
        "common": {k: 10 ** rng.uniform(lo, hi) for k in DEFAULT_WEIGHTS["common"]},
        "neutral": {"bias": rng.uniform(-1, 1),
                    "nearest": 10 ** rng.uniform(lo, hi)},
        "enemy":   {"bias": rng.uniform(-1, 1),
                    "weakness": 10 ** rng.uniform(lo, hi)},
        "support": {"bias": rng.uniform(-1, 1),
                    "threat": 10 ** rng.uniform(lo, hi),
                    "prod_save": 10 ** rng.uniform(lo, hi)},
    }


def flatten_weights(w: dict) -> Dict[str, float]:
    out = {}
    for kind, sub in w.items():
        for name, val in sub.items():
            out[f"w_{kind}_{name}"] = val
    return out


def WEIGHT_FIELDS() -> List[str]:
    return sorted(flatten_weights(DEFAULT_WEIGHTS).keys())


# ── C8 random_search ────────────────────────────────────────────────────

def random_search(
    n_trials: int = 200,
    eval_seeds: List[int] | None = None,
    eval_opponents: List = ("random", "baseline"),
    output_csv: str = "agent_logs/greedy_search.csv",
    sampler: Callable = default_sampler,
    rng_seed: int = 0,
    verbose: bool = True,
) -> List[Tuple[dict, dict]]:
    """log-uniform random search. CSV resume 지원.

    output_csv 이미 있으면 그 행 수만큼 skip — 도중 중단해도 이어 돌릴 수 있음.
    """
    if eval_seeds is None:
        eval_seeds = list(range(16))
    eval_opponents = list(eval_opponents)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    weight_keys = WEIGHT_FIELDS()
    stat_keys = ["objective", "win_rate", "timeout_rate", "noop_rate",
                 "moves_per_turn", "avg_step_ms", "p95_step_ms", "max_step_ms",
                 "wins", "losses", "n_games"]
    per_opp_keys = [f"vs_{opponent_label(o)}_wr" for o in eval_opponents]
    fieldnames = ["trial_id"] + weight_keys + stat_keys + per_opp_keys

    # Resume: 기존 행 수 = 다음 trial id
    start_trial = 0
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        with open(output_csv) as f:
            start_trial = max(0, sum(1 for _ in f) - 1)

    rng = random.Random(rng_seed + start_trial)
    write_header = (not os.path.exists(output_csv)
                     or os.path.getsize(output_csv) == 0)

    results: List[Tuple[dict, dict]] = []
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush()

        for trial in range(start_trial, n_trials):
            w = sampler(rng)
            stats = evaluate(w, eval_seeds, eval_opponents)

            row = {"trial_id": trial}
            row.update(flatten_weights(w))
            row.update({k: stats.get(k, 0) for k in stat_keys})
            for op in eval_opponents:
                lbl = opponent_label(op)
                row[f"vs_{lbl}_wr"] = stats["per_opponent"].get(lbl, {}).get("win_rate", 0)
            writer.writerow(row)
            f.flush()

            results.append((w, stats))
            if verbose:
                print(f"trial {trial:3d}: obj={stats['objective']:+.3f} "
                      f"wr={stats['win_rate']:.2f} t/o={stats['timeout_rate']:.2f} "
                      f"max_ms={stats['max_step_ms']:.0f}")

    return sorted(results, key=lambda x: x[1]["objective"], reverse=True)


# ── C9 refine ───────────────────────────────────────────────────────────

def refine(
    top_results: List[Tuple[dict, dict]],
    more_seeds: List[int] | None = None,
    more_opponents: List = ("random", "baseline"),
    output_csv: str = "agent_logs/greedy_refine.csv",
    verbose: bool = True,
) -> List[Tuple[dict, dict]]:
    """top_results 의 weights 들을 더 많은 seed 로 재평가. 분산 축소.

    more_seeds: random_search 의 superset 권장 (0~15 → 0~63).
    """
    if more_seeds is None:
        more_seeds = list(range(64))
    more_opponents = list(more_opponents)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    weight_keys = WEIGHT_FIELDS()
    stat_keys = ["objective", "win_rate", "timeout_rate", "noop_rate",
                 "moves_per_turn", "avg_step_ms", "p95_step_ms", "max_step_ms",
                 "wins", "losses", "n_games"]
    per_opp_keys = [f"vs_{opponent_label(o)}_wr" for o in more_opponents]
    fieldnames = ["rank"] + weight_keys + stat_keys + per_opp_keys

    refined: List[Tuple[dict, dict]] = []
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, (w, _) in enumerate(top_results):
            stats = evaluate(w, more_seeds, more_opponents)
            row = {"rank": rank}
            row.update(flatten_weights(w))
            row.update({k: stats.get(k, 0) for k in stat_keys})
            for op in more_opponents:
                lbl = opponent_label(op)
                row[f"vs_{lbl}_wr"] = stats["per_opponent"].get(lbl, {}).get("win_rate", 0)
            writer.writerow(row)
            f.flush()
            refined.append((w, stats))
            if verbose:
                print(f"refine rank {rank:2d}: obj={stats['objective']:+.3f} "
                      f"wr={stats['win_rate']:.2f}")

    return sorted(refined, key=lambda x: x[1]["objective"], reverse=True)


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["evaluate", "search", "refine"])
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seeds",  type=int, default=16)
    ap.add_argument("--csv",    type=str, default=None)
    args = ap.parse_args()

    if args.cmd == "evaluate":
        stats = evaluate(DEFAULT_WEIGHTS, list(range(args.seeds)))
        print(stats)
    elif args.cmd == "search":
        out = args.csv or "agent_logs/greedy_search.csv"
        results = random_search(n_trials=args.trials,
                                  eval_seeds=list(range(args.seeds)),
                                  output_csv=out)
        print(f"top 5:")
        for w, s in results[:5]:
            print(f"  obj={s['objective']:+.3f} wr={s['win_rate']:.2f}")
    elif args.cmd == "refine":
        # 사전 search 결과 CSV 가 있어야 함 — top 10 읽어서 refine
        in_csv = args.csv or "agent_logs/greedy_search.csv"
        if not os.path.exists(in_csv):
            raise SystemExit(f"missing search CSV: {in_csv}")
        # 간단히: top 10 by objective 컬럼
        rows = []
        with open(in_csv) as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        rows.sort(key=lambda r: float(r["objective"]), reverse=True)
        top10 = []
        for r in rows[:10]:
            w = {"common": {}, "neutral": {}, "enemy": {}, "support": {}}
            for k, v in r.items():
                if k.startswith("w_"):
                    _, kind, name = k.split("_", 2)
                    w[kind][name] = float(v)
            top10.append((w, {"objective": float(r["objective"])}))
        refine(top10)
