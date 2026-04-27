"""
over-send penalty: 다중 source 에서 같은 target 에 redundant 발사하는
협조 실패 시그널을 reward shaping 으로 줄임.

decode_action_to_moves 가 per-target Σships - required 의 양수 초과분을
over_send_excess_sum / over_send_target_count 으로 누적.
"""
import os
import sys
import math
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env_wrapper import MAX_PLANETS, NUM_SHIPS_BINS, NUM_ACTIONS
from train import decode_action_to_moves

ACTION_DIM = NUM_ACTIONS + MAX_PLANETS


def _set_action(action_np, src_idx, ships_bin, target_idx, launch=1.0):
    action_idx = (1 + ships_bin) if launch >= 0.5 else 0
    action_np[src_idx, action_idx] = 1.0
    action_np[src_idx, NUM_ACTIONS + target_idx] = 1.0


class FakePlanet:
    def __init__(self, id, x, y, owner, ships, production, radius=3.0, *args):
        self.id = id
        self.x = x
        self.y = y
        self.owner = owner
        self.ships = ships
        self.production = production
        self.radius = radius


def make_raw(id_, x, y, owner, ships=20, production=2, radius=3.0):
    return (id_, x, y, owner, ships, production, radius)


@patch("train.Planet", FakePlanet)
@patch("prediction.aim", return_value=(math.pi / 4, 80.0, 80.0, 10))
@patch("train.crosses_sun", return_value=False)
@patch("train.first_collision_on_path", return_value=("planet", 2))
def test_over_send_excess_two_sources_same_target(*_):
    """두 source 가 같은 target 에 발사 → required 한 번만, ships 합산.
    multiplier mode 에서 bin=3 (top multiplier ≈ 1.20×) 두 발 → sent ≈ 2·ceil(req·1.20)
    > required → over-send 발생.
    """
    action_np = np.zeros((MAX_PLANETS, ACTION_DIM), dtype=np.float32)
    # bin=3 → top multiplier (e.g. 1.20×). 둘 다 top multiplier → 합산 sent > required.
    _set_action(action_np, src_idx=0, ships_bin=3, target_idx=2)
    _set_action(action_np, src_idx=1, ships_bin=3, target_idx=2)

    raw_planets = [
        make_raw(0, 10.0, 10.0, owner=0, ships=100),
        make_raw(1, 20.0, 20.0, owner=0, ships=100),
        make_raw(2, 80.0, 80.0, owner=1, ships=5, production=2),  # required ≈ 26
    ]

    moves, counts, launches = decode_action_to_moves(
        action_np, raw_planets, av=0.0, acting_player=0, return_counts=True,
    )

    assert counts["launched"] == 2
    sent_total = sum(l["ships"] for l in launches)
    # required 는 첫 launch 시점 snapshot (모든 launch 동일 target → 같은 required).
    # multiplier 모드에서도 두 launch 합산 > required 이면 excess 발생.
    required = counts["required_ships_sum"] / counts["launched"]
    assert sent_total > required, (
        f"sent_total={sent_total} should exceed required≈{required} for over-send to trigger"
    )
    assert counts["over_send_excess_sum"] > 0
    assert counts["over_send_target_count"] == 1, (
        f"distinct over-send target 1개 기대, got {counts['over_send_target_count']}"
    )


@patch("train.Planet", FakePlanet)
@patch("prediction.aim", return_value=(math.pi / 4, 80.0, 80.0, 10))
@patch("train.crosses_sun", return_value=False)
@patch("train.first_collision_on_path", return_value=("planet", 2))
def test_over_send_zero_when_single_source_just_capture(*_):
    """단일 source bin=0 (just-capture) → over-send 0 (sent == required)."""
    action_np = np.zeros((MAX_PLANETS, ACTION_DIM), dtype=np.float32)
    _set_action(action_np, src_idx=0, ships_bin=0, target_idx=2)

    raw_planets = [
        make_raw(0, 10.0, 10.0, owner=0, ships=100),
        make_raw(1, 20.0, 20.0, owner=0, ships=100),
        make_raw(2, 80.0, 80.0, owner=1, ships=5, production=2),
    ]

    moves, counts, launches = decode_action_to_moves(
        action_np, raw_planets, av=0.0, acting_player=0, return_counts=True,
    )

    assert counts["launched"] == 1
    # bin=0 → ships == required → excess == 0 (per-target sent=required)
    assert counts["over_send_excess_sum"] == 0
    assert counts["over_send_target_count"] == 0


@patch("train.Planet", FakePlanet)
@patch("prediction.aim", return_value=(math.pi / 4, 80.0, 80.0, 10))
@patch("train.crosses_sun", return_value=False)
def test_over_send_distinct_targets_count_separately(mock_sun, mock_aim):
    """두 source 가 서로 다른 target 에 over-send → target_count=2."""

    def fake_path(p, angle, ships, planets, av, max_turns=120, pos_cache=None):
        # source 0 → target 2, source 1 → target 3
        if p.id == 0:
            return ("planet", 2)
        return ("planet", 3)

    action_np = np.zeros((MAX_PLANETS, ACTION_DIM), dtype=np.float32)
    _set_action(action_np, src_idx=0, ships_bin=3, target_idx=2)  # top multiplier
    _set_action(action_np, src_idx=1, ships_bin=3, target_idx=3)  # top multiplier

    raw_planets = [
        make_raw(0, 10.0, 10.0, owner=0, ships=100),
        make_raw(1, 20.0, 20.0, owner=0, ships=100),
        make_raw(2, 80.0, 80.0, owner=1, ships=5, production=2),
        make_raw(3, 70.0, 70.0, owner=1, ships=5, production=2),
    ]

    with patch("train.first_collision_on_path", side_effect=fake_path):
        moves, counts, launches = decode_action_to_moves(
            action_np, raw_planets, av=0.0, acting_player=0, return_counts=True,
        )

    assert counts["launched"] == 2
    # 각 target 1번씩 top multiplier → target 별 excess = ceil(req·1.20) - req > 0
    assert counts["over_send_target_count"] == 2
    assert counts["over_send_excess_sum"] > 0


@patch("train.Planet", FakePlanet)
@patch("prediction.aim", return_value=(math.pi / 4, 80.0, 80.0, 10))
@patch("train.crosses_sun", return_value=False)
@patch("train.first_collision_on_path", return_value=("planet", 2))
def test_over_send_excess_value_matches_sum_minus_required(*_):
    """excess_sum 은 정확히 (sum_sent - required) 이어야 함."""
    action_np = np.zeros((MAX_PLANETS, ACTION_DIM), dtype=np.float32)
    # 둘 다 top multiplier (multiplier 모드에서 over-send 보장)
    _set_action(action_np, src_idx=0, ships_bin=3, target_idx=2)
    _set_action(action_np, src_idx=1, ships_bin=3, target_idx=2)

    raw_planets = [
        make_raw(0, 10.0, 10.0, owner=0, ships=100),
        make_raw(1, 20.0, 20.0, owner=0, ships=100),
        make_raw(2, 80.0, 80.0, owner=1, ships=5, production=2),
    ]

    moves, counts, launches = decode_action_to_moves(
        action_np, raw_planets, av=0.0, acting_player=0, return_counts=True,
    )

    sent_total = sum(l["ships"] for l in launches)
    # required 값은 first launch 시점에 snapshot
    required = counts["required_ships_sum"] / counts["launched"]  # mean (둘 다 same)
    expected_excess = sent_total - int(required)
    assert counts["over_send_excess_sum"] == expected_excess, (
        f"excess_sum={counts['over_send_excess_sum']}, "
        f"expected={expected_excess} (sent_total={sent_total}, required={required})"
    )
