import os
import sys
import math
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env_wrapper import MAX_PLANETS
from train import decode_action_to_moves
from utils.hit_tracker import HitRateTracker


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
@patch("train.aim", return_value=(math.pi / 4, 50.0, 50.0, 10))
def test_decode_action_counts_cover_filters_and_launch(mock_aim):
    action_np = np.zeros((MAX_PLANETS, MAX_PLANETS + 2), dtype=np.float32)

    # planet 0 -> invalid target (self-owned)
    action_np[0, 0] = 1.0
    action_np[0, 3] = 10.0  # target_idx=1

    # planet 2 -> zero ships after clipping/min
    action_np[2, 0] = 1.0
    action_np[2, 7] = 10.0  # target_idx=5

    # planet 3 -> sun filtered
    action_np[3, 0] = 1.0
    action_np[3, 7] = 10.0  # target_idx=5

    # planet 4 -> valid launch
    action_np[4, 0] = 1.0
    action_np[4, 7] = 10.0  # target_idx=5
    action_np[4, 1] = 0.5

    raw_planets = [
        make_raw(0, 10.0, 10.0, owner=0, ships=20),
        make_raw(1, 20.0, 20.0, owner=0, ships=10),   # self target
        make_raw(2, 30.0, 30.0, owner=0, ships=0),    # zero-ships source
        make_raw(3, 40.0, 40.0, owner=0, ships=20),
        make_raw(4, 50.0, 50.0, owner=0, ships=20),
        make_raw(5, 80.0, 80.0, owner=1, ships=5),    # enemy for valid launch
    ]

    def fake_crosses_sun(x1, y1, x2, y2):
        # Third launched source only is sun-blocked.
        return x1 == 40.0

    with patch("train.crosses_sun", side_effect=fake_crosses_sun), \
         patch("train.first_collision_on_path", return_value=("planet", 5)):
        moves, counts, launches = decode_action_to_moves(
            action_np, raw_planets, av=0.0, acting_player=0, return_counts=True
        )

    assert len(moves) == 1
    assert len(launches) == 1
    assert launches[0]["source_id"] == 4
    assert launches[0]["target_id"] == 5
    assert launches[0]["ships"] == 10
    assert math.isclose(launches[0]["angle"], math.pi / 4)
    expected_core = {
        "attempts": 4,
        "filtered_invalid_target": 1,
        "filtered_zero_ships": 1,
        "filtered_sun": 1,
        "filtered_path": 0,
        "launched": 1,
    }
    for k, v in expected_core.items():
        assert counts[k] == v, f"{k}: expected {v}, got {counts[k]}"
    # ships 실측 필드 존재 확인 (값 검증은 별도 테스트)
    for k in ("ships_ratio_sum", "ships_to_send_sum", "required_ships_sum",
              "send_required_ratio_sum", "under_invested_count"):
        assert k in counts, f"missing ships field: {k}"


def test_hit_rate_tracker_summary_returns_per_step_means():
    tracker = HitRateTracker()
    tracker.record({
        "attempts": 4,
        "filtered_invalid_target": 1,
        "filtered_zero_ships": 1,
        "filtered_sun": 1,
        "launched": 1,
    })
    tracker.record({
        "attempts": 2,
        "filtered_invalid_target": 0,
        "filtered_zero_ships": 0,
        "filtered_sun": 1,
        "launched": 1,
    })

    summary = tracker.summary()

    assert summary["mean_attempts"] == 3.0
    assert summary["mean_launched"] == 1.0
    assert summary["mean_filtered_invalid_target"] == 0.5
    assert summary["mean_filtered_zero_ships"] == 0.5
    assert summary["mean_filtered_sun"] == 1.0
    assert summary["launch_rate"] == 2 / 6


# ── V2: resolve_step ────────────────────────────────────────────────────────

def _make_launch(source_id, target_id, ships, angle, start_x, start_y):
    return {
        "source_id": source_id,
        "target_id": target_id,
        "ships": ships,
        "angle": angle,
        "start_x": start_x,
        "start_y": start_y,
    }


def _planet(pid, owner, x, y, radius=3.0, ships=10, production=1):
    return (pid, owner, x, y, radius, ships, production)


def _obs(planets, fleets, next_fleet_id=0):
    return {"planets": planets, "fleets": fleets, "next_fleet_id": next_fleet_id}


def test_register_launches_assigns_sequential_ids():
    tracker = HitRateTracker(player_id=0)
    launches = [
        _make_launch(0, 5, 10, 0.0, 10.0, 10.0),
        _make_launch(2, 7, 20, math.pi, 30.0, 30.0),
    ]
    tracker.register_launches(launches, next_fleet_id=42)
    assert set(tracker.pending.keys()) == {42, 43}
    assert tracker.pending[42]["target_id"] == 5
    assert tracker.pending[43]["target_id"] == 7


def test_resolve_out_of_bounds():
    tracker = HitRateTracker(player_id=0)
    # Fleet at (99, 50), angle 0, ships 10 → new_pos ≈ (100.96, 50) → out
    tracker.register_launches(
        [_make_launch(0, 1, 10, 0.0, 99.0, 50.0)],
        next_fleet_id=1,
    )
    prev_obs = _obs(planets=[_planet(0, 0, 99.0, 50.0)], fleets=[])
    curr_obs = _obs(planets=[_planet(0, 0, 99.0, 50.0)], fleets=[])
    tracker.resolve_step(prev_obs, curr_obs, max_speed=6)
    assert tracker.counters["out"] == 1
    assert 1 not in tracker.pending


def test_resolve_sun_crash():
    tracker = HitRateTracker(player_id=0)
    # Fleet near sun center, heading further in
    tracker.register_launches(
        [_make_launch(0, 1, 10, 0.0, 45.0, 50.0)],
        next_fleet_id=1,
    )
    prev_obs = _obs(planets=[_planet(0, 0, 45.0, 50.0)], fleets=[])
    curr_obs = _obs(planets=[_planet(0, 0, 45.0, 50.0)], fleets=[])
    tracker.resolve_step(prev_obs, curr_obs, max_speed=6)
    assert tracker.counters["sun_crash"] == 1


def test_resolve_target_hit_exclusive_and_captured():
    tracker = HitRateTracker(player_id=0)
    # 시작 위치는 엔진 규약대로 source planet radius+0.1 바깥이므로 source를 obs에 안 넣음
    # Fleet at (20, 50), angle 0, ships 10 → new_pos ≈ (21.96, 50). Target at (22, 50) radius 3.
    tracker.register_launches(
        [_make_launch(0, 5, 10, 0.0, 20.0, 50.0)],
        next_fleet_id=1,
    )
    prev_obs = _obs(
        planets=[
            _planet(5, 1, 22.0, 50.0, ships=1),  # enemy-owned, low ships
        ],
        fleets=[],
    )
    curr_obs = _obs(
        planets=[
            _planet(5, 0, 22.0, 50.0, ships=5),  # captured by us
        ],
        fleets=[],
    )
    tracker.resolve_step(prev_obs, curr_obs, max_speed=6)
    assert tracker.counters["target_hit_exclusive"] == 1
    assert tracker.counters["target_hit_ambiguous"] == 0
    assert tracker.counters["captured_exclusive"] == 1


def test_resolve_hit_other_when_wrong_planet():
    tracker = HitRateTracker(player_id=0)
    # Fleet intends target=9 but hits planet 5
    tracker.register_launches(
        [_make_launch(0, 9, 10, 0.0, 20.0, 50.0)],
        next_fleet_id=1,
    )
    prev_obs = _obs(
        planets=[
            _planet(5, 1, 22.0, 50.0),
            _planet(9, 1, 80.0, 50.0),
        ],
        fleets=[],
    )
    curr_obs = _obs(
        planets=[
            _planet(5, 1, 22.0, 50.0),
            _planet(9, 1, 80.0, 50.0),
        ],
        fleets=[],
    )
    tracker.resolve_step(prev_obs, curr_obs, max_speed=6)
    assert tracker.counters["hit_other_exclusive"] == 1
    assert tracker.counters["target_hit_exclusive"] == 0


def test_resolve_ambiguous_when_two_own_fleets_hit_same_planet():
    tracker = HitRateTracker(player_id=0)
    tracker.register_launches(
        [
            _make_launch(0, 5, 10, 0.0, 20.0, 50.0),
            _make_launch(1, 5, 10, 0.0, 20.0, 50.1),
        ],
        next_fleet_id=1,
    )
    prev_obs = _obs(
        planets=[_planet(5, 1, 22.0, 50.0, ships=1)],
        fleets=[],
    )
    curr_obs = _obs(
        planets=[_planet(5, 0, 22.0, 50.0, ships=19)],
        fleets=[],
    )
    tracker.resolve_step(prev_obs, curr_obs, max_speed=6)
    # 두 fleet 모두 planet 5 hit → ambiguous. target도 동일하므로 target_hit_ambiguous=2
    assert tracker.counters["target_hit_ambiguous"] == 2
    assert tracker.counters["target_hit_exclusive"] == 0
    assert tracker.counters["captured_ambiguous"] == 2


def test_resolve_ambiguous_when_enemy_fleet_hits_same_planet():
    tracker = HitRateTracker(player_id=0)
    tracker.register_launches(
        [_make_launch(0, 5, 10, 0.0, 20.0, 50.0)],
        next_fleet_id=1,
    )
    # 적 fleet (id=99, owner=1)가 prev_obs에서 같은 planet 5 쪽으로 날아가다 소멸
    enemy_fleet = (99, 1, 20.0, 50.0, 0.0, 0, 10)
    prev_obs = _obs(
        planets=[_planet(5, -1, 22.0, 50.0, ships=3)],
        fleets=[enemy_fleet],
    )
    curr_obs = _obs(
        planets=[_planet(5, 0, 22.0, 50.0, ships=7)],
        fleets=[],
    )
    tracker.resolve_step(prev_obs, curr_obs, max_speed=6)
    # 내 fleet는 target 맞음, 하지만 enemy도 같이 붙어서 ambiguous
    assert tracker.counters["target_hit_ambiguous"] == 1
    assert tracker.counters["target_hit_exclusive"] == 0


def test_target_owner_counters_split_neutral_and_enemy():
    """register_launches가 target_owner 기반으로 전체 launched 분포를 분류."""
    tracker = HitRateTracker(player_id=0)
    launches = [
        {**_make_launch(0, 5, 10, 0.0, 10.0, 10.0), "target_owner": -1},   # 중립
        {**_make_launch(1, 6, 10, 0.0, 20.0, 20.0), "target_owner": 1},    # 적
        {**_make_launch(2, 7, 10, 0.0, 30.0, 30.0), "target_owner": 1},    # 적
    ]
    tracker.register_launches(launches, next_fleet_id=1)
    assert tracker.counters["target_neutral"] == 1
    assert tracker.counters["target_enemy"] == 2
    # 초반 20턴이라 early_* attempts 도 집계되어야 함
    assert tracker.counters["early_neutral_attempts"] == 1
    assert tracker.counters["early_enemy_attempts"] == 2


def test_early_attempts_only_in_first_20_turns():
    """episode_turn >= 20이면 early_*_attempts 집계 안 함."""
    tracker = HitRateTracker(player_id=0)
    tracker.episode_turn = 20  # 초반 phase 벗어남
    launches = [
        {**_make_launch(0, 5, 10, 0.0, 10.0, 10.0), "target_owner": -1},
        {**_make_launch(1, 6, 10, 0.0, 20.0, 20.0), "target_owner": 1},
    ]
    tracker.register_launches(launches, next_fleet_id=1)
    # 전체 분포는 여전히 집계
    assert tracker.counters["target_neutral"] == 1
    assert tracker.counters["target_enemy"] == 1
    # 초반 attempts는 0
    assert tracker.counters["early_neutral_attempts"] == 0
    assert tracker.counters["early_enemy_attempts"] == 0


def test_early_neutral_captured_counts_within_first_20_turns():
    """초반 20턴 내 중립 점령은 거리 무관하게 early_neutral_captured 집계."""
    tracker = HitRateTracker(player_id=0)
    # home 설정은 멀리 (90, 90) — target 거리 멀게 해서 home20 영역 밖임을 보장
    tracker.reset_episode(_obs(
        planets=[_planet(99, 0, 90.0, 90.0)],   # my home
        fleets=[],
    ))
    # 원거리 중립 (20, 50) 공격
    tracker.register_launches(
        [{**_make_launch(0, 5, 10, 0.0, 20.0, 50.0), "target_owner": -1}],
        next_fleet_id=1,
    )
    prev_obs = _obs(planets=[
        _planet(99, 0, 90.0, 90.0),
        _planet(5, -1, 22.0, 50.0, ships=1),
    ])
    curr_obs = _obs(planets=[
        _planet(99, 0, 90.0, 90.0),
        _planet(5, 0, 22.0, 50.0, ships=9),   # 점령 성공
    ])
    tracker.resolve_step(prev_obs, curr_obs, max_speed=6)
    assert tracker.counters["captured_neutral"] == 1
    assert tracker.counters["early_neutral_captured"] == 1
    # home 영역 밖 → early_home_expand 는 올라가면 안 됨
    assert tracker.counters["early_home_expand"] == 0


def test_early_launch_neutral_captured_counts_when_launched_early_but_resolved_late():
    """
    핵심 진단 케이스: 초반 15턴에 발사한 fleet이 25턴에 도착해 중립 점령.
    - episode_turn=25 (resolve 시점) → 기존 early_neutral_captured 안 잡힘
    - launched_at_turn=15 < 20 → early_launch_neutral_captured 잡힘
    """
    tracker = HitRateTracker(player_id=0)
    tracker.reset_episode(_obs(
        planets=[_planet(99, 0, 90.0, 90.0)],  # my home
        fleets=[],
    ))
    tracker.episode_turn = 15  # 초반 구간에 발사
    tracker.register_launches(
        [{**_make_launch(0, 5, 10, 0.0, 20.0, 50.0), "target_owner": -1}],
        next_fleet_id=1,
    )
    # 시간 경과: resolve 시점은 25턴 (초반 구간 벗어남)
    tracker.episode_turn = 25
    prev_obs = _obs(planets=[
        _planet(99, 0, 90.0, 90.0),
        _planet(5, -1, 22.0, 50.0, ships=1),
    ])
    curr_obs = _obs(planets=[
        _planet(99, 0, 90.0, 90.0),
        _planet(5, 0, 22.0, 50.0, ships=9),
    ])
    tracker.resolve_step(prev_obs, curr_obs, max_speed=6)

    assert tracker.counters["captured_neutral"] == 1
    # resolve 시점(25) >= 20 이라 기존 지표는 0
    assert tracker.counters["early_neutral_captured"] == 0
    # launched_at_turn(15) < 20 이라 새 지표는 1
    assert tracker.counters["early_launch_neutral_captured"] == 1


def test_early_launch_neutral_captured_zero_when_launched_late():
    """대조군: 발사 자체가 22턴(초반 벗어남)에 일어난 경우 → 새 지표도 0."""
    tracker = HitRateTracker(player_id=0)
    tracker.reset_episode(_obs(
        planets=[_planet(99, 0, 90.0, 90.0)],
        fleets=[],
    ))
    tracker.episode_turn = 22
    tracker.register_launches(
        [{**_make_launch(0, 5, 10, 0.0, 20.0, 50.0), "target_owner": -1}],
        next_fleet_id=1,
    )
    prev_obs = _obs(planets=[
        _planet(99, 0, 90.0, 90.0),
        _planet(5, -1, 22.0, 50.0, ships=1),
    ])
    curr_obs = _obs(planets=[
        _planet(99, 0, 90.0, 90.0),
        _planet(5, 0, 22.0, 50.0, ships=9),
    ])
    tracker.resolve_step(prev_obs, curr_obs, max_speed=6)

    assert tracker.counters["captured_neutral"] == 1
    assert tracker.counters["early_launch_neutral_captured"] == 0


def test_ships_distribution_metrics_computed_from_launched_aggregates():
    """
    decode counts에 누적한 ships_ratio_sum/ships_to_send_sum/required_ships_sum/
    send_required_ratio_sum를 summary()가 launched로 나눠 평균/표준편차로 환산.
    """
    tracker = HitRateTracker()
    # 2번의 launch 시뮬: ratio 0.2 & 0.4, ships 10 & 20, required 50 & 40
    tracker.record({
        "attempts": 2,
        "launched": 2,
        "ships_ratio_sum": 0.2 + 0.4,         # mean 0.3
        "ships_ratio_sq_sum": 0.04 + 0.16,    # E[X²] 0.10 → std = sqrt(0.10 - 0.09) = 0.1
        "ships_to_send_sum": 10 + 20,         # mean 15
        "required_ships_sum": 50 + 40,        # mean 45
        "send_required_ratio_sum": 0.2 + 0.5, # mean 0.35
        "under_invested_count": 2,            # 둘 다 srr<1 → rate 100%
    })
    s = tracker.summary()
    assert abs(s["ships_ratio_mean"] - 0.3) < 1e-6
    assert abs(s["ships_ratio_std"] - 0.1) < 1e-6
    assert abs(s["ships_to_send_mean"] - 15.0) < 1e-6
    assert abs(s["required_ships_mean"] - 45.0) < 1e-6
    assert abs(s["send_required_ratio_mean"] - 0.35) < 1e-6
    assert abs(s["under_invested_rate"] - 1.0) < 1e-6


def test_ships_distribution_metrics_zero_when_no_launches():
    """launched=0이면 0으로 나누지 않고 모든 ships 지표를 0.0으로."""
    tracker = HitRateTracker()
    tracker.record({"attempts": 1, "launched": 0})
    s = tracker.summary()
    for k in ("ships_ratio_mean", "ships_ratio_std", "ships_to_send_mean",
              "required_ships_mean", "send_required_ratio_mean", "under_invested_rate"):
        assert s[k] == 0.0, f"{k} should be 0 when launched=0, got {s[k]}"


def test_resolve_keeps_alive_fleet_in_pending():
    tracker = HitRateTracker(player_id=0)
    tracker.register_launches(
        [_make_launch(0, 5, 10, 0.0, 10.0, 10.0)],
        next_fleet_id=1,
    )
    prev_obs = _obs(
        planets=[_planet(5, 1, 90.0, 90.0)],
        fleets=[],
    )
    # fleet 여전히 비행 중
    curr_obs = _obs(
        planets=[_planet(5, 1, 90.0, 90.0)],
        fleets=[(1, 0, 11.96, 10.0, 0.0, 0, 10)],
    )
    tracker.resolve_step(prev_obs, curr_obs, max_speed=6)
    assert 1 in tracker.pending
    assert math.isclose(tracker.pending[1]["last_x"], 11.96)
    # 아무 카운터도 증가하지 않아야 함
    for k in ["out", "sun_crash", "target_hit_exclusive", "hit_other_exclusive"]:
        assert tracker.counters[k] == 0
