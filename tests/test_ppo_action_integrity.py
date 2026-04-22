"""
PPO action sampling integrity tests.

P0 bug: _collect_single이 get_action_and_value를 스텝당 두 번 호출해
        저장된 action_t/log_prob와 실제 실행된 행동이 불일치하던 버그 검증.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import numpy as np
import torch
import pytest
from unittest.mock import patch, MagicMock

from train import decode_action_to_moves
from env_wrapper import MAX_PLANETS


# ── helpers ──────────────────────────────────────────────────────────────────

class FakePlanet:
    def __init__(self, id, x, y, owner, ships, production, *args):
        self.id         = id
        self.x          = x
        self.y          = y
        self.owner      = owner
        self.ships      = ships
        self.production = production


def make_raw(id_, x, y, owner, ships=20, production=2):
    return (id_, x, y, owner, ships, production)


# ── Unit test 2: decode correctness ──────────────────────────────────────────

@patch("train.Planet", FakePlanet)
@patch("train.aim", return_value=math.pi / 4)
@patch("train.crosses_sun", return_value=False)
def test_decode_no_launch(mock_sun, mock_aim):
    """launch < 0.5 → 빈 moves."""
    action_np = np.zeros((MAX_PLANETS, MAX_PLANETS + 2), dtype=np.float32)
    raw_planets = [make_raw(0, 10.0, 10.0, 0)]
    moves = decode_action_to_moves(action_np, raw_planets, av=0.0, acting_player=0)
    assert moves == []
    mock_aim.assert_not_called()


@patch("train.Planet", FakePlanet)
@patch("train.aim", return_value=math.pi / 4)
@patch("train.crosses_sun", return_value=False)
def test_decode_launch_to_enemy(mock_sun, mock_aim):
    """launch >= 0.5, enemy target → move 1개 생성."""
    action_np = np.zeros((MAX_PLANETS, MAX_PLANETS + 2), dtype=np.float32)
    action_np[0, 0] = 1.0   # launch
    action_np[0, 1] = 0.0   # ships_ratio = 0.5
    action_np[0, 3] = 10.0  # argmax → target_idx=1

    raw_planets = [
        make_raw(0, 10.0, 10.0, owner=0, ships=20),
        make_raw(1, 90.0, 90.0, owner=1, ships=5),
    ]
    moves = decode_action_to_moves(action_np, raw_planets, av=0.0, acting_player=0)
    assert len(moves) == 1
    assert moves[0][0] == 0  # from_planet_id


@patch("train.Planet", FakePlanet)
@patch("train.aim", return_value=math.pi / 4)
@patch("train.crosses_sun", return_value=False)
def test_decode_own_planet_target_skipped(mock_sun, mock_aim):
    """target이 내 행성이면 move 생성 안 함."""
    action_np = np.zeros((MAX_PLANETS, MAX_PLANETS + 2), dtype=np.float32)
    action_np[0, 0] = 1.0
    action_np[0, 3] = 10.0  # target_idx=1 (내 행성)

    raw_planets = [
        make_raw(0, 10.0, 10.0, owner=0, ships=20),
        make_raw(1, 90.0, 90.0, owner=0, ships=5),  # 내 행성
    ]
    moves = decode_action_to_moves(action_np, raw_planets, av=0.0, acting_player=0)
    assert moves == []


@patch("train.Planet", FakePlanet)
@patch("train.aim", return_value=math.pi / 4)
@patch("train.crosses_sun", return_value=True)  # 태양 차단
def test_decode_sun_blocked(mock_sun, mock_aim):
    """crosses_sun=True → move 생성 안 함."""
    action_np = np.zeros((MAX_PLANETS, MAX_PLANETS + 2), dtype=np.float32)
    action_np[0, 0] = 1.0
    action_np[0, 3] = 10.0

    raw_planets = [
        make_raw(0, 10.0, 10.0, owner=0, ships=20),
        make_raw(1, 90.0, 90.0, owner=1, ships=5),
    ]
    moves = decode_action_to_moves(action_np, raw_planets, av=0.0, acting_player=0)
    assert moves == []


# ── Unit test 1: 스텝당 get_action_and_value 1회 호출 보장 ──────────────────

def test_single_sample_per_step():
    """
    _collect_single에서 main agent의 get_action_and_value가
    스텝당 정확히 1회만 호출되는지 확인.
    """
    from train import _collect_single
    from model import OrbitWarsPolicy

    device = torch.device("cpu")
    model  = OrbitWarsPolicy().to(device)
    model.eval()

    call_count = [0]
    original   = model.get_action_and_value

    def counting_fn(*args, **kwargs):
        call_count[0] += 1
        return original(*args, **kwargs)

    model.get_action_and_value = counting_fn

    N_STEPS = 3
    _collect_single(model, None, n_steps=N_STEPS, device=device)

    assert call_count[0] == N_STEPS, (
        f"get_action_and_value called {call_count[0]} times for {N_STEPS} steps "
        f"(expected {N_STEPS} — double sampling bug re-introduced?)"
    )


# ── Trace integrity: action_t가 moves의 유일한 원천인지 확인 ─────────────────

@patch("train.Planet", FakePlanet)
@patch("train.aim", return_value=0.5)
@patch("train.crosses_sun", return_value=False)
def test_decode_is_pure_function(mock_sun, mock_aim):
    """
    동일한 action_np 입력 → 동일한 moves 출력.
    decode_action_to_moves가 외부 상태에 의존하지 않음을 확인.
    """
    action_np = np.zeros((MAX_PLANETS, MAX_PLANETS + 2), dtype=np.float32)
    action_np[0, 0] = 1.0
    action_np[0, 1] = 0.5
    action_np[0, 3] = 5.0

    raw_planets = [
        make_raw(0, 20.0, 20.0, owner=0, ships=30),
        make_raw(1, 80.0, 80.0, owner=1, ships=5),
    ]

    moves1 = decode_action_to_moves(action_np, raw_planets, av=0.0, acting_player=0)
    moves2 = decode_action_to_moves(action_np, raw_planets, av=0.0, acting_player=0)

    assert moves1 == moves2
    assert len(moves1) == 1


def test_collect_single_stored_action_matches_decoded_moves():
    """
    _collect_single에서 저장된 action_t와 decode_action_to_moves에
    전달된 action이 같은 샘플임을 trace로 확인.

    방법: decode_action_to_moves를 래핑해서 입력 action_np를 기록하고,
          get_action_and_value 출력의 action_t와 비교.
    """
    from train import _collect_single
    import train
    from model import OrbitWarsPolicy

    device     = torch.device("cpu")
    model      = OrbitWarsPolicy().to(device)
    model.eval()

    sampled_actions  = []
    decoded_actions  = []
    original_sample  = model.get_action_and_value
    original_decode  = train.decode_action_to_moves

    def tracing_sample(*args, **kwargs):
        action_t, lp, v = original_sample(*args, **kwargs)
        sampled_actions.append(action_t.squeeze(0).cpu().numpy().copy())
        return action_t, lp, v

    def tracing_decode(action_np, raw_planets, av, acting_player=0):
        decoded_actions.append(action_np.copy())
        return original_decode(action_np, raw_planets, av, acting_player=acting_player)

    model.get_action_and_value = tracing_sample

    with patch.object(train, "decode_action_to_moves", side_effect=tracing_decode):
        _collect_single(model, None, n_steps=3, device=device)

    assert len(sampled_actions) == len(decoded_actions) == 3

    for step, (sampled, decoded) in enumerate(zip(sampled_actions, decoded_actions)):
        np.testing.assert_array_equal(
            sampled, decoded,
            err_msg=f"Step {step}: stored action_t ≠ decoded action (double-sampling bug)"
        )
