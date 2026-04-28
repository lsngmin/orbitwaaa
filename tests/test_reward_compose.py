"""Reward 패키지 (mask-style ctx-based) 검증.

검증 항목:
- RewardBreakdown.total = 모든 component 필드 합 (parity)
- 각 component (ctx) -> float 의 단위 동작
- compose_rewards(ctx) 가 RewardBreakdown 의 모든 필드 채우는지
- neutral_capture_bonus + own_planet_loss_penalty parity (분리 전 단일 함수와 동일 합)
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
    own_planet_loss_penalty,
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
        dense=0.123,
        neutral_capture_bonus=0.456,
        own_planet_loss_penalty=-0.10,
        all_in_penalty=-0.05, over_send_penalty=-0.02,
        under_invested_penalty=-0.01, launch_cost_penalty=-0.07,
        terminal=1.0,
    )
    expected = 0.123 + 0.456 - 0.10 - 0.05 - 0.02 - 0.01 - 0.07 + 1.0
    assert b.total == pytest.approx(expected, abs=1e-12)


def test_breakdown_default_total_zero():
    """모든 필드 기본값=0 → total=0."""
    assert RewardBreakdown().total == 0.0


# ── compose_rewards 통합 ──────────────────────────────────────────────────────

def test_compose_fills_all_breakdown_fields():
    """compose_rewards 결과의 각 필드가 component 직접 호출과 일치."""
    ctx = RewardContext(
        prev_score=0.0, curr_score=1.0,
        prev_map={0: (-1, 5), 1: (0, 3)},
        curr_obs=_obs([
            (0, 0, 0, 0, 0, 10, 5),  # neutral → me (gain)
            (1, 1, 0, 0, 0, 10, 3),  # me → enemy (loss)
        ]),
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
    assert b.dense                   == pytest.approx(dense_reward(ctx))
    assert b.neutral_capture_bonus   == pytest.approx(neutral_capture_bonus(ctx))
    assert b.own_planet_loss_penalty == pytest.approx(own_planet_loss_penalty(ctx))
    assert b.all_in_penalty          == pytest.approx(all_in_penalty(ctx))
    assert b.over_send_penalty       == pytest.approx(over_send_penalty(ctx))
    assert b.under_invested_penalty  == pytest.approx(under_invested_penalty(ctx))
    assert b.launch_cost_penalty     == pytest.approx(launch_cost_penalty(ctx))
    assert b.terminal                == pytest.approx(terminal_reward(ctx))


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
    """COMPONENTS 와 COMPONENT_NAMES 가 동기화 되어 있고, 분리 후 8개."""
    assert COMPONENT_NAMES == tuple(c.__name__ for c in COMPONENTS)
    assert len(COMPONENTS) == 8  # dense / neutral_cap / own_loss / 4 penalty / terminal


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


# ── neutral_capture_bonus (gain only) ─────────────────────────────────────────

def test_neutral_capture_bonus_only_gains():
    """중립 → 내것 만 처리. 내것 → 잃음 케이스는 0 반환 (loss 함수 책임)."""
    ctx = RewardContext(
        prev_map={0: (-1, 5)},
        curr_obs=_obs([(0, 0, 0, 0, 0, 10, 5)]),
        player=0, turn_norm=0.0,
        cap_gain_coef=0.05, cap_loss_coef=0.025, cap_early_multiplier=1.0,
    )
    assert neutral_capture_bonus(ctx) == pytest.approx(5 * 0.05)


def test_neutral_capture_bonus_ignores_loss_transition():
    """내 것 → 적/중립 transition 은 neutral_capture_bonus 가 무시 (loss 함수 책임)."""
    ctx = RewardContext(
        prev_map={0: (0, 7)},
        curr_obs=_obs([(0, 1, 0, 0, 0, 10, 7)]),  # me → enemy
        player=0, turn_norm=0.0,
        cap_gain_coef=0.05, cap_loss_coef=0.025,
    )
    assert neutral_capture_bonus(ctx) == 0.0


def test_neutral_capture_bonus_early_boost():
    """early_multiplier > 1 + turn_norm=0 → gain 만 amplify."""
    ctx = RewardContext(
        prev_map={0: (-1, 4)},
        curr_obs=_obs([(0, 0, 0, 0, 0, 10, 4)]),
        player=0, turn_norm=0.0,
        cap_gain_coef=0.10, cap_loss_coef=0.05, cap_early_multiplier=2.0,
    )
    assert neutral_capture_bonus(ctx) == pytest.approx(4 * 0.10 * 2.0)


def test_neutral_capture_bonus_late_game_boost_off():
    """turn_norm=1.0 → early_boost=1.0, multiplier 무관."""
    ctx = RewardContext(
        prev_map={0: (-1, 3)},
        curr_obs=_obs([(0, 0, 0, 0, 0, 10, 3)]),
        player=0, turn_norm=1.0,
        cap_gain_coef=0.10, cap_loss_coef=0.05, cap_early_multiplier=5.0,
    )
    assert neutral_capture_bonus(ctx) == pytest.approx(3 * 0.10)


def test_neutral_capture_bonus_turn_norm_clamped():
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


# ── own_planet_loss_penalty (loss only) ───────────────────────────────────────

def test_own_planet_loss_penalty_only_losses():
    """내것 → 적/중립 만 처리. 중립 → 내것 케이스는 0 (gain 함수 책임)."""
    ctx = RewardContext(
        prev_map={0: (0, 7)},
        curr_obs=_obs([(0, 1, 0, 0, 0, 10, 7)]),  # me → enemy
        player=0, cap_gain_coef=0.05, cap_loss_coef=0.025,
    )
    assert own_planet_loss_penalty(ctx) == pytest.approx(-0.025 * 7)


def test_own_planet_loss_penalty_ignores_gain_transition():
    """중립 → 내것 transition 은 loss 함수가 무시."""
    ctx = RewardContext(
        prev_map={0: (-1, 5)},
        curr_obs=_obs([(0, 0, 0, 0, 0, 10, 5)]),  # neutral → me
        player=0, cap_gain_coef=0.05, cap_loss_coef=0.025,
    )
    assert own_planet_loss_penalty(ctx) == 0.0


def test_own_planet_loss_penalty_no_early_boost():
    """early_multiplier 가 커도 loss 에는 영향 없음 (의도적 비대칭)."""
    base_ctx = dict(
        prev_map={0: (0, 4)},
        curr_obs=_obs([(0, 1, 0, 0, 0, 10, 4)]),
        player=0, cap_gain_coef=0.10, cap_loss_coef=0.05,
    )
    a = own_planet_loss_penalty(RewardContext(**base_ctx, turn_norm=0.0, cap_early_multiplier=1.0))
    b = own_planet_loss_penalty(RewardContext(**base_ctx, turn_norm=0.0, cap_early_multiplier=5.0))
    assert a == b == pytest.approx(-4 * 0.05)


def test_own_planet_loss_penalty_to_neutral_also_counted():
    """내 것 → 중립 (-1) 도 loss (점령 후 ships=0 으로 수직 빼앗기지 않고 풀려도)."""
    ctx = RewardContext(
        prev_map={0: (0, 6)},
        curr_obs=_obs([(0, -1, 0, 0, 0, 0, 6)]),  # me → neutral
        player=0, cap_loss_coef=0.025,
    )
    assert own_planet_loss_penalty(ctx) == pytest.approx(-0.025 * 6)


# ── 분리 전 단일 함수와의 parity ──────────────────────────────────────────────

def test_split_components_sum_matches_legacy_combined():
    """gain + loss 분리 후 합이 분리 전 단일 함수와 동일.

    분리 전 train.py:181-218 동작:
        for p in planets:
            if neutral → me: bonus += prod * gain * boost
            elif me → not me: bonus -= prod * loss
        return bonus

    분리 후: neutral_capture_bonus(ctx) + own_planet_loss_penalty(ctx)
    """
    ctx = RewardContext(
        prev_map={0: (-1, 5), 1: (0, 3), 2: (0, 4)},
        curr_obs=_obs([
            (0, 0, 0, 0, 0, 10, 5),   # neutral → me (gain)
            (1, 1, 0, 0, 0, 10, 3),   # me → enemy (loss)
            (2, -1, 0, 0, 0, 0, 4),   # me → neutral (loss)
        ]),
        player=0, turn_norm=0.3,
        cap_gain_coef=0.05, cap_loss_coef=0.025, cap_early_multiplier=1.5,
    )
    # 분리 전 expected:
    #   gain : 5 × 0.05 × early_boost(turn_norm=0.3, mult=1.5)
    #          = 5 × 0.05 × (1 + 0.5 × 0.7) = 5 × 0.05 × 1.35
    #   loss : -3 × 0.025 + -4 × 0.025 = -0.175
    early_boost = 1.0 + (1.5 - 1.0) * (1.0 - 0.3)
    expected_legacy = 5 * 0.05 * early_boost - 3 * 0.025 - 4 * 0.025
    actual_split = neutral_capture_bonus(ctx) + own_planet_loss_penalty(ctx)
    assert actual_split == pytest.approx(expected_legacy, abs=1e-12)


def test_no_change_zero_in_both():
    """소유자 변동 없으면 둘 다 0."""
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
    assert own_planet_loss_penalty(ctx) == 0.0


def test_empty_prev_map_zero_in_both():
    """prev_map default (-1, 0) 라 prod=0 → 둘 다 0 (분리 전 동작 동일)."""
    ctx = RewardContext(
        prev_map={},
        curr_obs=_obs([(0, 0, 0, 0, 0, 10, 5)]),
        player=0, cap_gain_coef=0.05, cap_loss_coef=0.025,
    )
    assert neutral_capture_bonus(ctx) == 0.0
    assert own_planet_loss_penalty(ctx) == 0.0


# ── B 단계 — capture_events parity ────────────────────────────────────────────
#
# B 도입 후에도 어떤 component 도 ctx.capture_events 를 소비하지 않음.
# 따라서 capture_events 가 비어있든 채워져 있든 reward 값은 동일해야 함 (parity).

def test_capture_events_does_not_affect_existing_rewards():
    """ctx.capture_events 가 채워져도 기존 component 의 출력은 변화 없음."""
    from reward import LaunchMetadata, CaptureEvent

    base_kwargs = dict(
        prev_score=0.0, curr_score=1.0,
        prev_map={0: (-1, 5)},
        curr_obs=_obs([(0, 0, 0, 0, 0, 10, 5)]),
        decode_counts={"all_in_launches": 1, "over_send_excess_sum": 2,
                       "under_invested_count": 0, "launch_cost_excess_sum": 0.3},
        turn_norm=0.0, ep_done=False, raw_terminal=0, player=0,
        dense_coef=1.0,
        all_in_penalty_coef=0.05, over_send_penalty_coef=0.02,
        under_invested_penalty_coef=0.01, launch_cost_penalty_coef=0.04,
        cap_gain_coef=0.10, cap_loss_coef=0.05, cap_early_multiplier=1.0,
        terminal_win_reward=1.0,
    )
    ctx_empty = RewardContext(**base_kwargs)

    fake_meta = LaunchMetadata(
        turn=0, source_id=99, target_id=0, target_owner_at_launch=-1,
        target_kind="attack", ships_sent=10, eta_turns=3,
        req_over_src=0.5, req_over_prod=2.0,
        nearest_neutral_rank=1, target_prod=5.0,
    )
    fake_event = CaptureEvent(
        turn=1, planet_id=0, prev_owner=-1, new_owner=0,
        target_prod=5.0, linked_launches=[fake_meta],
    )
    ctx_with_events = RewardContext(**base_kwargs, capture_events=[fake_event])

    b_empty = compose_rewards(ctx_empty)
    b_full  = compose_rewards(ctx_with_events)
    assert b_empty.total == pytest.approx(b_full.total, abs=1e-12)
    # field-by-field
    for f in ("dense", "neutral_capture_bonus", "own_planet_loss_penalty",
              "all_in_penalty", "over_send_penalty", "under_invested_penalty",
              "launch_cost_penalty", "terminal"):
        assert getattr(b_empty, f) == pytest.approx(getattr(b_full, f), abs=1e-12), (
            f"field {f} 가 capture_events 에 영향받음 — B 단계 parity 위반"
        )


def test_capture_events_default_empty_list():
    """RewardContext 기본값에 capture_events=[] 있어야 (현재 호출부 영향 없도록)."""
    ctx = RewardContext()
    assert ctx.capture_events == []
