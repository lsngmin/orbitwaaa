"""
Episode-grained collection에서 병렬 worker의 reward_stats 합산 정확성.

이전 버그: `collect_rollout`이 worker별 summary를 단순 평균(/ len(results))으로
합치면, 짧은 episode worker와 긴 episode worker가 같은 가중치를 받아
mean_dense, noop_rate, hit rate 등이 편향됨.

해결: `_collect_single`이 raw counters/n_steps/episodes/sum_* 을 반환하고
`collect_rollout._finalize_reward_stats`가 합산한 뒤 한 번에 normalize.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from collections import defaultdict

import pytest

from train import _finalize_reward_stats
from utils.hit_tracker import HitRateTracker


def _make_raw(n_steps, episodes, dense, cap, terminal, **counters):
    return {
        "counters":     defaultdict(float, counters),
        "n_steps":      n_steps,
        "episodes":     episodes,
        "sum_dense":    dense,
        "sum_cap":      cap,
        "sum_terminal": terminal,
    }


# ── 1. step-weighted: mean_dense ─────────────────────────────────────────────

def test_mean_dense_step_weighted_not_worker_averaged():
    """Worker A: 200 step, dense_sum=2.0 (평균 0.01)
    Worker B: 800 step, dense_sum=16.0 (평균 0.02)

    올바른 step-weighted 평균: (2 + 16) / (200 + 800) = 18/1000 = 0.018
    (잘못된) worker 평균: (0.01 + 0.02) / 2 = 0.015  ← 편향
    """
    raw_a = _make_raw(n_steps=200, episodes=1, dense=2.0,  cap=0.0, terminal=0.0)
    raw_b = _make_raw(n_steps=800, episodes=2, dense=16.0, cap=0.0, terminal=0.0)

    stats = _finalize_reward_stats([raw_a, raw_b])

    assert abs(stats["mean_dense"] - 0.018) < 1e-9, (
        f"mean_dense={stats['mean_dense']} — step-weighted 기대값 0.018"
    )
    # 편향된 worker 평균 0.015와 명확히 다름을 확인
    assert abs(stats["mean_dense"] - 0.015) > 1e-4
    # actual_steps: generation 실제 수집량 (target 대비 초과 감지용)
    assert stats["actual_steps"] == 1000


# ── 2. per-episode metric: episode-weighted ──────────────────────────────────

def test_early_home_expand_per_episode_episode_weighted():
    """per_episode metric은 episode 수로 normalize 되어야 함.

    Worker A: 1 episode, early_home_expand=3  →  3/1 = 3.0
    Worker B: 4 episodes, early_home_expand=8  →  8/4 = 2.0

    합산: early_home_expand=11, episodes=5 → 11/5 = 2.2 (step 수와 무관)
    """
    raw_a = _make_raw(n_steps=100, episodes=1, dense=0.0, cap=0.0, terminal=0.0,
                      early_home_expand=3)
    raw_b = _make_raw(n_steps=900, episodes=4, dense=0.0, cap=0.0, terminal=0.0,
                      early_home_expand=8)

    stats = _finalize_reward_stats([raw_a, raw_b])
    assert abs(stats["early_home_expand_per_episode"] - 2.2) < 1e-9


# ── 3. launched-weighted: neutral_capture_rate ───────────────────────────────

def test_neutral_capture_rate_launched_weighted():
    """launched 분모로 정규화되는 rate는 launched 합으로 올바르게 계산.

    Worker A: launched=10, captured_neutral=3  →  0.3
    Worker B: launched=90, captured_neutral=18 →  0.2

    합산: launched=100, captured=21 → 0.21
    (worker 평균 (0.3 + 0.2)/2 = 0.25 는 편향)
    """
    raw_a = _make_raw(n_steps=200, episodes=1, dense=0.0, cap=0.0, terminal=0.0,
                      launched=10, captured_neutral=3, attempts=20)
    raw_b = _make_raw(n_steps=800, episodes=3, dense=0.0, cap=0.0, terminal=0.0,
                      launched=90, captured_neutral=18, attempts=180)

    stats = _finalize_reward_stats([raw_a, raw_b])
    assert abs(stats["neutral_capture_rate"] - 0.21) < 1e-9
    assert abs(stats["launch_rate"] - 0.5) < 1e-9  # 100/200 attempts


# ── 4. single worker passthrough ─────────────────────────────────────────────

def test_single_worker_passthrough():
    """단일 worker인 경우: summary_from_counters와 결과가 동일해야 함."""
    raw = _make_raw(n_steps=500, episodes=2, dense=5.0, cap=1.0, terminal=-5.0,
                    launched=20, attempts=30, captured_neutral=5)

    stats = _finalize_reward_stats([raw])
    direct = HitRateTracker.summary_from_counters(
        raw["counters"], raw["n_steps"], raw["episodes"],
    )
    direct["mean_dense"]    = 5.0 / 500
    direct["mean_cap"]      = 1.0 / 500
    direct["mean_terminal"] = -5.0 / 500

    for k in direct:
        assert abs(stats[k] - direct[k]) < 1e-9, f"key={k}: {stats[k]} != {direct[k]}"


# ── 5. pooled variance for chosen_multiplier ─────────────────────────────────

def test_chosen_multiplier_pooled_variance():
    """E[X²] - E[X]² 공식은 counter 합산 후에도 유효.

    Worker A: launched=2, mult 값 [1.1, 1.3] → sum=2.4, sq_sum=2.9
    Worker B: launched=2, mult 값 [1.6, 2.0] → sum=3.6, sq_sum=6.56
    합산: launched=4, sum=6.0, sq_sum=9.46
      mean = 6.0/4 = 1.5
      var  = 9.46/4 - 1.5² = 2.365 - 2.25 = 0.115
      std  = sqrt(0.115) ≈ 0.339
    """
    import math
    raw_a = _make_raw(n_steps=100, episodes=1, dense=0.0, cap=0.0, terminal=0.0,
                      launched=2, chosen_multiplier_sum=2.4,
                      chosen_multiplier_sq_sum=1.1**2 + 1.3**2)
    raw_b = _make_raw(n_steps=100, episodes=1, dense=0.0, cap=0.0, terminal=0.0,
                      launched=2, chosen_multiplier_sum=3.6,
                      chosen_multiplier_sq_sum=1.6**2 + 2.0**2)

    stats = _finalize_reward_stats([raw_a, raw_b])
    expected_mean = 1.5
    expected_var  = 9.46 / 4 - 1.5**2
    expected_std  = math.sqrt(expected_var)

    assert abs(stats["chosen_multiplier_mean"] - expected_mean) < 1e-9
    assert abs(stats["chosen_multiplier_std"]  - expected_std)  < 1e-6


# ── 6. 회귀 방지: worker-평균이 편향된다는 사실 문서화 ────────────────────────

def test_regression_worker_average_is_biased():
    """이전 구현 (단순 worker 평균) 이 실제로 편향 결과를 낸다는 증거.

    짧은 worker (1 step)와 긴 worker (999 step)에서 mean_dense를 구하면
    worker 평균은 두 worker를 50:50으로 섞지만, 실제 분포는 1:999.
    """
    # Worker A: 1 step, dense_sum=5.0 → mean=5.0
    raw_a = _make_raw(n_steps=1,   episodes=1, dense=5.0, cap=0.0, terminal=0.0)
    # Worker B: 999 steps, dense_sum=0.0 → mean=0.0
    raw_b = _make_raw(n_steps=999, episodes=1, dense=0.0, cap=0.0, terminal=0.0)

    stats = _finalize_reward_stats([raw_a, raw_b])
    # 올바른 step-weighted: 5/(1+999) = 0.005
    assert abs(stats["mean_dense"] - 0.005) < 1e-9
    # worker 평균 (편향): (5.0 + 0.0)/2 = 2.5 — 500× 차이
    biased = (5.0 + 0.0) / 2
    assert abs(stats["mean_dense"] - biased) > 2.0, \
        "step-weighted 결과가 worker 평균과 명확히 달라야 함"
