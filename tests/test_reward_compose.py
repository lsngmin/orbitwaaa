"""Reward 패키지 (mask-style ctx-based) 검증.

검증 항목:
- RewardBreakdown.total = 모든 component 필드 합 (parity)
- 각 component (ctx) -> float 의 단위 동작
- compose_rewards(ctx) 가 RewardBreakdown 의 모든 필드 채우는지
- neutral_capture_bonus 의 분리 전 train.py 동작 1:1 보존
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from reward import (
    RewardContext,
    RewardBreakdown,
    COMPONENTS,
    COMPONENT_NAMES,
    compose_rewards,
    dense_reward,
    neutral_capture_bonus,
    all_in_penalty,
    over_send_penalty,
    under_invested_penalty,
    launch_cost_penalty,
    terminal_reward,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _obs(planets):
    """planet tuple: (id, owner, x, y, radius, ships, production)."""
    return {"planets": planets}


# ── RewardBreakdown.total parity ──────────────────────────────────────────────

def test_breakdown_total_is_sum_of_all_fields():
    b = RewardBreakdown(
        dense=0.123, cap_bonus=0.456,
        all_in_penalty=-0.05, over_send_penalty=-0.02,
        under_invested_penalty=-0.01, launch_cost_penalty=-0.07,
        terminal=1.0,
    )
    expected = 0.123 + 0.456 - 0.05 - 0.02 - 0.01 - 0.07 + 1.0
    assert b.total == pytest.approx(expected, abs=1e-12)


def test_breakdown_default_total_zero():
    """모든 필드 기본값=0 → total=0."""
    assert RewardBreakdown().total == 0.0


# ── compose_rewards 통합 ──────────────────────────────────────────────────────

def test_compose_fills_all_breakdown_fields():
    """compose_rewards 결과의 각 필드가 component 직접 호출과 일치."""
    ctx = RewardContext(
        prev_score=0.0, curr_score=1.0,
        prev_map={0: (-1, 5)},
        curr_obs=_obs([(0, 0, 0, 0, 0, 10, 5)]),  # neutral → me
        decode_counts={
            "all_in_launches": 2,
            "over_send_excess_sum": 3,
            "under_invested_count": 1,
            "launch_cost_excess_sum": 0.5,
        },
        turn_norm=0.0, ep_done=True, raw_terminal=1, player=0,
        dense_coef=0.5,
        all_in_penalty_coef=0.1,
        over_send_penalty_coef=0.2,
        under_invested_penalty_coef=0.3,
        launch_cost_penalty_coef=0.4,
        cap_gain_coef=0.05, cap_loss_coef=0.025,
        cap_early_multiplier=1.0,
        terminal_win_reward=2.0,
    )
    b = compose_rewards(ctx)
    assert b.dense                  == pytest.approx(dense_reward(ctx))
    assert b.cap_bonus              == pytest.approx(neutral_capture_bonus(ctx))
    assert b.all_in_penalty         == pytest.approx(all_in_penalty(ctx))
    assert b.over_send_penalty      == pytest.approx(over_send_penalty(ctx))
    assert b.under_invested_penalty == pytest.approx(under_invested_penalty(ctx))
    assert b.launch_cost_penalty    == pytest.approx(launch_cost_penalty(ctx))
    assert b.terminal               == pytest.approx(terminal_reward(ctx))


def test_compose_total_matches_manual_sum():
    """compose_rewards(ctx).total == COMPONENTS 직접 합."""
    ctx = RewardContext(
        prev_score=0.0, curr_score=2.0,
        prev_map={0: (-1, 4), 1: (0, 3)},
        curr_obs=_obs([
            (0, 0, 0, 0, 0, 10, 4),  # neutral → me (gain)
            (1, 1, 0, 0, 0, 10, 3),  # me → enemy (loss)
        ]),
        decode_counts={
            "all_in_launches": 1, "over_send_excess_sum": 2,
            "under_invested_count": 0, "launch_cost_excess_sum": 0.3,
        },
        turn_norm=0.5, ep_done=True, raw_terminal=-1, player=0,
        dense_coef=1.0,
        all_in_penalty_coef=0.05, over_send_penalty_coef=0.02,
        under_invested_penalty_coef=0.01, launch_cost_penalty_coef=0.04,
        cap_gain_coef=0.10, cap_loss_coef=0.05, cap_early_multiplier=2.0,
        terminal_win_reward=1.0,
    )
    b = compose_rewards(ctx)
    manual = sum(c(ctx) for c in COMPONENTS)
    assert b.total == pytest.approx(manual, abs=1e-12)


def test_components_tuple_matches_names():
    """COMPONENTS 와 COMPONENT_NAMES 가 동기화 되어 있음."""
    assert COMPONENT_NAMES == tuple(c.__name__ for c in COMPONENTS)
    assert len(COMPONENTS) == 7  # dense, cap, 4 penalty, terminal


# ── component 단위 동작 ───────────────────────────────────────────────────────

def test_dense_reward_delta():
    ctx = RewardContext(prev_score=1.0, curr_score=3.0, dense_coef=0.5)
    assert dense_reward(ctx) == pytest.approx(1.0)


def test_dense_reward_zero_when_no_change():
    ctx = RewardContext(prev_score=2.0, curr_score=2.0, dense_coef=10.0)
    assert dense_reward(ctx) == 0.0


@pytest.mark.parametrize("fn,key,val", [
    (all_in_penalty,         "all_in_launches",       3),
    (over_send_penalty,      "over_send_excess_sum",  4.5),
    (under_invested_penalty, "under_invested_count",  2),
    (launch_cost_penalty,    "launch_cost_excess_sum", 1.7),
])
def test_penalties_nonpositive_and_zero_when_off(fn, key, val):
    """coef=0 또는 count=0 이면 0, 정상 입력이면 음수."""
    ctx_zero_coef = RewardContext(decode_counts={key: val})  # 모든 coef=0 (default)
    assert fn(ctx_zero_coef) == 0.0

    ctx_zero_count = RewardContext(decode_counts={key: 0},
                                    all_in_penalty_coef=1.0,
                                    over_send_penalty_coef=1.0,
                                    under_invested_penalty_coef=1.0,
                                    launch_cost_penalty_coef=1.0)
    assert fn(ctx_zero_count) == 0.0


def test_terminal_reward_only_when_ep_done():
    """ep_done=False 면 raw_terminal 무관하게 0."""
    ctx = RewardContext(ep_done=False, raw_terminal=1, terminal_win_reward=5.0)
    assert terminal_reward(ctx) == 0.0


@pytest.mark.parametrize("raw,expected_mult", [(1, 1), (-1, -1), (0, 0)])
def test_terminal_reward_mapping(raw, expected_mult):
    """ep_done=True 일 때 raw_terminal 부호에 따라 ±W / 0."""
    W = 2.5
    ctx = RewardContext(ep_done=True, raw_terminal=raw, terminal_win_reward=W)
    assert terminal_reward(ctx) == pytest.approx(W * expected_mult)


# ── neutral_capture_bonus parity ──────────────────────────────────────────────
#
# 분리 전 train.py 시절 동작 (line 181-218):
#   gain_coef × prod × early_boost   (중립 → player)
#   loss_coef × prod                 (player → 비player)
#   early_boost = 1 + (mult - 1) × max(0, 1 - turn_norm)   (gain 한정)

def test_capture_bonus_neutral_to_player_gain():
    ctx = RewardContext(
        prev_map={0: (-1, 5)},
        curr_obs=_obs([(0, 0, 0, 0, 0, 10, 5)]),
        player=0, turn_norm=0.0,
        cap_gain_coef=0.05, cap_loss_coef=0.025, cap_early_multiplier=1.0,
    )
    assert neutral_capture_bonus(ctx) == pytest.approx(5 * 0.05)


def test_capture_bonus_player_to_other_loss():
    ctx = RewardContext(
        prev_map={0: (0, 7)},
        curr_obs=_obs([(0, 1, 0, 0, 0, 10, 7)]),
        player=0, turn_norm=0.0,
        cap_gain_coef=0.05, cap_loss_coef=0.025,
        cap_early_multiplier=1.5,  # loss 에는 안 쓰임
    )
    assert neutral_capture_bonus(ctx) == pytest.approx(-0.025 * 7)


def test_capture_bonus_early_boost_gain_only():
    """early_multiplier > 1, turn_norm=0 → gain 만 amplify, loss 는 base."""
    ctx = RewardContext(
        prev_map={0: (-1, 4), 1: (0, 4)},
        curr_obs=_obs([
            (0, 0, 0, 0, 0, 10, 4),  # gain (boosted)
            (1, 1, 0, 0, 0, 10, 4),  # loss (NOT boosted)
        ]),
        player=0, turn_norm=0.0,
        cap_gain_coef=0.10, cap_loss_coef=0.05, cap_early_multiplier=2.0,
    )
    # gain: 4 × 0.10 × 2.0 = 0.8 ; loss: 4 × 0.05 = 0.2
    assert neutral_capture_bonus(ctx) == pytest.approx(0.8 - 0.2)


def test_capture_bonus_late_game_boost_off():
    """turn_norm=1.0 → early_boost=1.0, multiplier 무관."""
    ctx = RewardContext(
        prev_map={0: (-1, 3)},
        curr_obs=_obs([(0, 0, 0, 0, 0, 10, 3)]),
        player=0, turn_norm=1.0,
        cap_gain_coef=0.10, cap_loss_coef=0.05, cap_early_multiplier=5.0,
    )
    assert neutral_capture_bonus(ctx) == pytest.approx(3 * 0.10)


def test_capture_bonus_turn_norm_clamped():
    """turn_norm 이 [0,1] 밖이면 clamp."""
    base = RewardContext(
        prev_map={0: (-1, 1)},
        curr_obs=_obs([(0, 0, 0, 0, 0, 1, 1)]),
        player=0,
        cap_gain_coef=1.0, cap_loss_coef=0.0, cap_early_multiplier=2.0,
    )
    a = neutral_capture_bonus(RewardContext(**{**base.__dict__, "turn_norm": -0.5}))
    b = neutral_capture_bonus(RewardContext(**{**base.__dict__, "turn_norm":  0.0}))
    assert a == pytest.approx(b)


def test_capture_bonus_no_change_zero():
    ctx = RewardContext(
        prev_map={0: (0, 5), 1: (1, 5), 2: (-1, 5)},
        curr_obs=_obs([
            (0, 0, 0, 0, 0, 10, 5),
            (1, 1, 0, 0, 0, 10, 5),
            (2, -1, 0, 0, 0, 0, 5),
        ]),
        player=0, turn_norm=0.0,
        cap_gain_coef=0.5, cap_loss_coef=0.5, cap_early_multiplier=2.0,
    )
    assert neutral_capture_bonus(ctx) == 0.0


def test_capture_bonus_empty_prev_map_zero_prod():
    """prev_map 에 없으면 default (-1, 0) — prod=0 이라 bonus 0.

    분리 전 train.py:211 동일 동작: `prev_map.get(pid, (-1, 0))` 으로 prod=0 fallback.
    이 default 는 reset 직후 등 snapshot 누락 케이스를 silent 하게 무시하기 위한 것.
    """
    ctx = RewardContext(
        prev_map={},
        curr_obs=_obs([(0, 0, 0, 0, 0, 10, 5)]),
        player=0, cap_gain_coef=0.05, cap_loss_coef=0.025,
    )
    assert neutral_capture_bonus(ctx) == 0.0
