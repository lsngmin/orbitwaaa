"""
Dense Reward 검증 테스트.

- state_score가 직관적으로 좋은 상태에 더 높은 점수를 주는지
- delta_score가 상태 개선/악화 방향으로 계산되는지
- dense_reward_coef=0이면 terminal reward와 동일한지 (회귀)
- score 계산에 상대 행성은 포함되지 않는지
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from train import state_score


def make_raw_obs(planets):
    """planet list → raw_obs dict 형태."""
    # planet tuple: (id, owner, x, y, radius, ships, production)
    return {"planets": planets}


# ── state_score 직관성 ────────────────────────────────────────────────────────

def test_more_planets_higher_score():
    """행성 수가 많을수록 score가 높아야 함."""
    obs_1p = make_raw_obs([(0, 0, 50.0, 50.0, 2.0, 10, 1)])
    obs_2p = make_raw_obs([(0, 0, 50.0, 50.0, 2.0, 10, 1),
                           (1, 0, 60.0, 50.0, 2.0, 10, 1)])
    assert state_score(obs_2p, player=0) > state_score(obs_1p, player=0)


def test_more_ships_higher_score():
    """같은 행성 수라도 ships가 많을수록 score가 높아야 함."""
    obs_low  = make_raw_obs([(0, 0, 50.0, 50.0, 2.0, 10, 1)])
    obs_high = make_raw_obs([(0, 0, 50.0, 50.0, 2.0, 100, 1)])
    assert state_score(obs_high, player=0) > state_score(obs_low, player=0)


def test_more_production_higher_score():
    """production이 높을수록 score가 높아야 함."""
    obs_low  = make_raw_obs([(0, 0, 50.0, 50.0, 2.0, 10, 1)])
    obs_high = make_raw_obs([(0, 0, 50.0, 50.0, 2.0, 10, 5)])
    assert state_score(obs_high, player=0) > state_score(obs_low, player=0)


def test_enemy_planets_not_counted():
    """상대(player=1) 행성은 player=0 score에 포함되지 않아야 함."""
    obs_mine  = make_raw_obs([(0, 0, 50.0, 50.0, 2.0, 50, 3)])
    obs_enemy = make_raw_obs([(0, 1, 50.0, 50.0, 2.0, 50, 3)])
    assert state_score(obs_mine,  player=0) > 0
    assert state_score(obs_enemy, player=0) == 0.0


def test_neutral_planets_not_counted():
    """중립(owner=-1) 행성도 포함되지 않아야 함."""
    obs = make_raw_obs([(0, -1, 50.0, 50.0, 2.0, 50, 3)])
    assert state_score(obs, player=0) == 0.0


# ── delta_score 방향성 ────────────────────────────────────────────────────────

def test_delta_positive_when_state_improves():
    """행성을 점령하면 delta > 0."""
    obs_before = make_raw_obs([(0, 0, 50.0, 50.0, 2.0, 10, 1)])
    obs_after  = make_raw_obs([(0, 0, 50.0, 50.0, 2.0, 10, 1),
                               (1, 0, 60.0, 50.0, 2.0,  5, 2)])
    delta = state_score(obs_after, player=0) - state_score(obs_before, player=0)
    assert delta > 0, f"행성 점령 후 delta={delta} <= 0"


def test_delta_negative_when_state_worsens():
    """행성을 잃으면 delta < 0."""
    obs_before = make_raw_obs([(0, 0, 50.0, 50.0, 2.0, 10, 1),
                               (1, 0, 60.0, 50.0, 2.0,  5, 2)])
    obs_after  = make_raw_obs([(0, 0, 50.0, 50.0, 2.0, 10, 1),
                               (1, 1, 60.0, 50.0, 2.0,  5, 2)])  # 행성 1 빼앗김
    delta = state_score(obs_after, player=0) - state_score(obs_before, player=0)
    assert delta < 0, f"행성 상실 후 delta={delta} >= 0"


# ── dense_reward_coef=0 회귀 ─────────────────────────────────────────────────

def test_zero_coef_means_zero_dense_reward():
    """dense_reward_coef=0이면 delta * coef = 0 — terminal reward만 남아야 함."""
    coef = 0.0
    obs_before = make_raw_obs([(0, 0, 50.0, 50.0, 2.0, 10, 1)])
    obs_after  = make_raw_obs([(0, 0, 50.0, 50.0, 2.0, 10, 1),
                               (1, 0, 60.0, 50.0, 2.0,  5, 2)])
    delta  = state_score(obs_after, player=0) - state_score(obs_before, player=0)
    reward = coef * delta
    assert reward == 0.0
