import os
import sys
import math
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env_wrapper import MAX_PLANETS, NUM_SHIPS_BINS, NUM_ACTIONS
from train import decode_action_to_moves
from utils.hit_tracker import HitRateTracker

# Action layout: [action_5way(NUM_ACTIONS), target_onehot(P)]
#   action_5way: idx 0 = skip, idx 1..K = bin (k-1)
ACTION_DIM = NUM_ACTIONS + MAX_PLANETS


def _set_action(action_np, src_idx, ships_bin, target_idx, launch=1.0):
    """테스트 헬퍼: 5-way action layout 으로 설정.
    launch >= 0.5 → action_idx = 1 + ships_bin (발사). 아니면 0 (skip).
    """
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
@patch("train.aim", return_value=(math.pi / 4, 50.0, 50.0, 10))
def test_decode_action_counts_cover_filters_and_launch(mock_aim):
    """surplus-fraction head.
    ships_bin=0 (bin_value=0.0 → just-capture floor) 기준 시나리오:
      - target(enemy) ships=5, prod=2, turns=10 → required = 5 + 2×10 + 1 = 26
      - src.ships=20 < required=26 (capacity short) → ships_needed = src.ships = 20
    """
    action_np = np.zeros((MAX_PLANETS, ACTION_DIM), dtype=np.float32)

    # planet 0 -> invalid target (self-owned)
    _set_action(action_np, src_idx=0, ships_bin=0, target_idx=1)

    # planet 2 -> zero ships after clipping (src.ships=0)
    _set_action(action_np, src_idx=2, ships_bin=0, target_idx=5)

    # planet 3 -> sun filtered
    _set_action(action_np, src_idx=3, ships_bin=0, target_idx=5)

    # planet 4 -> valid launch (ships_bin=0 → bin_value=0.0, just-capture floor)
    _set_action(action_np, src_idx=4, ships_bin=0, target_idx=5)

    raw_planets = [
        make_raw(0, 10.0, 10.0, owner=0, ships=20),
        make_raw(1, 20.0, 20.0, owner=0, ships=10),   # self target
        make_raw(2, 30.0, 30.0, owner=0, ships=0),    # zero-ships source
        make_raw(3, 40.0, 40.0, owner=0, ships=20),
        make_raw(4, 50.0, 50.0, owner=0, ships=20),
        make_raw(5, 80.0, 80.0, owner=1, ships=5, production=2),
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
    # required = 5 + 2×10 + 1 = 26, src.ships=20 < required → capacity short → 20
    assert launches[0]["ships"] == 20
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
    for k in ("chosen_surplus_frac_sum", "chosen_surplus_frac_sq_sum",
              "bin_effective_count",
              "send_fraction_of_src_sum", "send_fraction_of_src_sq_sum",
              "ships_to_send_sum", "required_ships_sum",
              "send_required_ratio_sum", "under_invested_count",
              "over_send_excess_sum", "over_send_target_count"):
        assert k in counts, f"missing ships field: {k}"
    # ships_bin 히스토그램 필드 존재 확인
    for k in range(NUM_SHIPS_BINS):
        assert f"ships_bin_hist_{k}" in counts, f"missing bin hist field: ships_bin_hist_{k}"
    # bin 0 선택 1회, 다른 bin 0회
    assert counts["ships_bin_hist_0"] == 1
    for k in range(1, NUM_SHIPS_BINS):
        assert counts[f"ships_bin_hist_{k}"] == 0


@patch("train.Planet", FakePlanet)
@patch("prediction.aim", return_value=(math.pi / 4, 50.0, 50.0, 10))
@patch("train.crosses_sun", return_value=False)
@patch("train.first_collision_on_path", return_value=("planet", 1))
def test_under_invested_counts_capacity_short(mock_path, mock_sun, mock_aim):
    """
    surplus-fraction head 의 under_invested_count 시맨틱:
      src.ships < required (capacity short — bin 무관 점령 불가).

    회귀 방지: src 가 required 보다 적게 가지고 있으면 어떤 bin 으로도
    floor(required) 를 채울 수 없음. 이 케이스만 under_invested 로 집계.
      - src.ships=20, target.ships=5, prod=2, turns=10 → required=26
      - bin=2 (0.66) 무관: src 가 부족하므로 ships_needed = src.ships = 20
      - under_invested_count=1 (capacity short)

    prediction.aim을 patch해야 resolve_ships_for_capture 내부 aim 호출까지 덮음.
    """
    action_np = np.zeros((MAX_PLANETS, ACTION_DIM), dtype=np.float32)
    _set_action(action_np, src_idx=0, ships_bin=2, target_idx=1)

    raw_planets = [
        make_raw(0, 10.0, 10.0, owner=0, ships=20),
        make_raw(1, 80.0, 80.0, owner=1, ships=5, production=2),
    ]

    moves, counts, launches = decode_action_to_moves(
        action_np, raw_planets, av=0.0, acting_player=0, return_counts=True,
    )

    assert counts["launched"] == 1
    assert launches[0]["ships"] == 20          # capacity short → src.ships
    # required=26, send=20 → src.ships < required → under-invested
    assert counts["under_invested_count"] == 1, (
        "capacity-short under_invested 누락: "
        f"src.ships=20 < required=26 인데 카운트 안 됨"
    )


@patch("train.Planet", FakePlanet)
@patch("prediction.aim", return_value=(math.pi / 4, 50.0, 50.0, 10))
@patch("train.crosses_sun", return_value=False)
@patch("train.first_collision_on_path", return_value=("planet", 1))
def test_under_invested_zero_when_capacity_sufficient(mock_path, mock_sun, mock_aim):
    """
    대조군: src.ships >= required → under_invested_count=0.
      - src.ships=100, required=26, bin=0 (just-capture floor)
      - ships_needed = required = 26 (surplus=74 × 0.0 = 0)
      - 100 >= 26 → under=0
    """
    action_np = np.zeros((MAX_PLANETS, ACTION_DIM), dtype=np.float32)
    _set_action(action_np, src_idx=0, ships_bin=0, target_idx=1)  # bin=0.0

    raw_planets = [
        make_raw(0, 10.0, 10.0, owner=0, ships=100),
        make_raw(1, 80.0, 80.0, owner=1, ships=5, production=2),
    ]
    moves, counts, launches = decode_action_to_moves(
        action_np, raw_planets, av=0.0, acting_player=0, return_counts=True,
    )

    assert counts["launched"] == 1
    assert launches[0]["ships"] == 26           # bin=0 → just-capture (= required)
    assert counts["under_invested_count"] == 0  # capacity 충족


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
    surplus-fraction head: decode counts에 누적한 chosen_surplus_frac_sum/
    ships_to_send_sum/required_ships_sum/send_required_ratio_sum/ships_bin_hist_*를
    summary()가 launched로 나눠 평균/표준편차/히스토그램으로 환산.

    chosen_surplus_frac_*는 bin_effective_count(non-capacity-short)로 정규화,
    send_fraction_of_src_*는 launched로 정규화.
    """
    tracker = HitRateTracker()
    # 2번의 launch 시뮬: bin_value 0.0 & 0.66, ships 26 & 40, required 26 & 25
    # 둘 다 capacity 충족 → bin_effective_count=2
    # send_frac: 26/30=0.8667, 40/80=0.5 → mean=0.6833
    sf_a, sf_b = 26 / 30, 40 / 80
    record = {
        "attempts": 2,
        "launched": 2,
        "chosen_surplus_frac_sum": 0.0 + 0.66,        # mean 0.33
        "chosen_surplus_frac_sq_sum": 0.0 + 0.4356,   # E[X²]=0.2178, var=0.2178-0.1089=0.1089 → std=0.33
        "bin_effective_count": 2,                     # 둘 다 non-capacity-short
        "send_fraction_of_src_sum": sf_a + sf_b,
        "send_fraction_of_src_sq_sum": sf_a ** 2 + sf_b ** 2,
        "ships_to_send_sum": 26 + 40,                 # mean 33
        "required_ships_sum": 26 + 25,                # mean 25.5
        "send_required_ratio_sum": 1.0 + 1.60,        # mean 1.30
        "under_invested_count": 0,
        # bin 0 (0.0) 1회, bin 2 (0.66) 1회
        "ships_bin_hist_0": 1,
        "ships_bin_hist_2": 1,
    }
    tracker.record(record)
    s = tracker.summary()
    assert abs(s["chosen_surplus_frac_mean"] - 0.33) < 1e-6
    assert abs(s["chosen_surplus_frac_std"] - 0.33) < 1e-6
    expected_sf_mean = (sf_a + sf_b) / 2
    expected_sf_var  = (sf_a ** 2 + sf_b ** 2) / 2 - expected_sf_mean ** 2
    expected_sf_std  = math.sqrt(max(expected_sf_var, 0.0))
    assert abs(s["send_fraction_of_src_mean"] - expected_sf_mean) < 1e-6
    assert abs(s["send_fraction_of_src_std"] - expected_sf_std) < 1e-6
    assert abs(s["ships_to_send_mean"] - 33.0) < 1e-6
    assert abs(s["required_ships_mean"] - 25.5) < 1e-6
    assert abs(s["send_required_ratio_mean"] - 1.30) < 1e-6
    assert abs(s["under_invested_rate"] - 0.0) < 1e-6
    # bin 히스토그램 비율 (launched=2 기준)
    assert abs(s["ships_bin_rate_0"] - 0.5) < 1e-6
    assert abs(s["ships_bin_rate_2"] - 0.5) < 1e-6
    # 선택 안 된 bin들은 0
    assert s["ships_bin_rate_1"] == 0.0
    assert s["ships_bin_rate_3"] == 0.0


def test_ships_distribution_metrics_zero_when_no_launches():
    """launched=0이면 0으로 나누지 않고 모든 ships 지표를 0.0으로."""
    tracker = HitRateTracker()
    tracker.record({"attempts": 1, "launched": 0})
    s = tracker.summary()
    for k in ("chosen_surplus_frac_mean", "chosen_surplus_frac_std",
              "send_fraction_of_src_mean", "send_fraction_of_src_std",
              "ships_to_send_mean",
              "required_ships_mean", "send_required_ratio_mean", "under_invested_rate",
              "over_send_excess_per_launch"):
        assert s[k] == 0.0, f"{k} should be 0 when launched=0, got {s[k]}"
    for k in range(NUM_SHIPS_BINS):
        assert s[f"ships_bin_rate_{k}"] == 0.0, f"ships_bin_rate_{k} should be 0"


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
