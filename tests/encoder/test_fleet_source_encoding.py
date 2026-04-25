"""encode_fleets 의 source idx 매핑 회귀 (가벼운 인코딩 검증).

핵심 계약:
  - fleet.from_planet_id → planets[:MAX_PLANETS] 내 idx 로 정확히 매핑
  - 미존재 planet id → -1 sentinel
  - 빈 fleet 슬롯 → -1 sentinel
  - 마지막 dim 위치에 idx 저장
  - planets 가 비-순차 id 순서로 와도 매핑 정확
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math
import numpy as np
import pytest
from env_wrapper import encode_fleets, FLEET_DIM, FLEET_FEAT_DIM, MAX_FLEETS


# Planet(id, owner, x, y, radius, ships, production)
# Fleet(id, owner, x, y, angle, from_planet_id, ships)


def _planet(id_, owner=0, x=50.0, y=50.0):
    return (id_, owner, x, y, 5.0, 50, 2)


def _fleet(id_, owner, from_pid, ships=10, x=10.0, y=10.0, angle=0.0):
    return (id_, owner, x, y, angle, from_pid, ships)


def test_idx_at_last_dim_position():
    """마지막 dim (= FLEET_DIM - 1) 자리에 idx 저장."""
    raw_planets = [_planet(7)]
    raw_fleets  = [_fleet(0, owner=0, from_pid=7)]
    arr = encode_fleets(raw_fleets, raw_planets, player=0)

    assert arr.shape == (MAX_FLEETS, FLEET_DIM)
    # 마지막 dim 만 idx, 앞 FLEET_FEAT_DIM 은 numeric features
    assert arr[0, FLEET_DIM - 1] == 0.0   # planet 7 → planets[0] → idx 0


def test_id_to_idx_mapping_preserves_position():
    """planets 의 list 순서 == idx (id 자체가 아님)."""
    # 일부러 비-순차 id 로 줌
    raw_planets = [_planet(13), _planet(7), _planet(99)]
    raw_fleets = [
        _fleet(0, owner=0, from_pid=13),    # → idx 0
        _fleet(1, owner=0, from_pid=7),     # → idx 1
        _fleet(2, owner=1, from_pid=99),    # → idx 2
    ]
    arr = encode_fleets(raw_fleets, raw_planets, player=0)
    assert arr[0, -1] == 0.0
    assert arr[1, -1] == 1.0
    assert arr[2, -1] == 2.0


def test_unknown_id_maps_to_sentinel():
    """planets 에 없는 from_planet_id → -1."""
    raw_planets = [_planet(0), _planet(1)]
    raw_fleets  = [_fleet(0, owner=0, from_pid=999)]   # 없는 id
    arr = encode_fleets(raw_fleets, raw_planets, player=0)
    assert arr[0, -1] == -1.0


def test_empty_slots_have_sentinel():
    """빈 fleet 슬롯 (fleet 리스트가 짧음) → 마지막 dim = -2.0 (empty-slot sentinel)."""
    raw_planets = [_planet(0)]
    raw_fleets  = []   # 전부 빈 슬롯
    arr = encode_fleets(raw_fleets, raw_planets, player=0)
    assert arr.shape == (MAX_FLEETS, FLEET_DIM)
    # 모든 슬롯의 마지막 dim 은 -2 (empty-slot sentinel)
    assert np.all(arr[:, -1] == -2.0)
    # 앞 numeric feature 들은 0
    assert np.all(arr[:, :FLEET_FEAT_DIM] == 0.0)


def test_lookup_miss_real_fleet_keeps_minus_one():
    """real fleet 의 source lookup 실패 → -1 (NOT -2). padding mask 에 안 잡혀야 함.

    회귀: src_idx=-1 sentinel 이 빈 슬롯(-2)과 분리되어 attention 에 살아남는지.
    """
    raw_planets = [_planet(0), _planet(1)]
    raw_fleets  = [_fleet(0, owner=0, from_pid=999)]   # 없는 src id
    arr = encode_fleets(raw_fleets, raw_planets, player=0)
    # slot 0: real fleet, src lookup miss → -1
    assert arr[0, -1] == -1.0, "real fleet w/ src miss 는 -1 sentinel 유지해야 함"
    # 다른 슬롯들: 빈 슬롯 → -2
    assert np.all(arr[1:, -1] == -2.0), "빈 슬롯은 -2 sentinel"
    # real fleet 의 다른 feature 는 valid (위치/ships 등)
    assert arr[0, 4] > 0.0 or arr[0, 5] == 1.0, "real fleet 의 numeric feature 는 살아있어야 함"


def test_fleet_features_layout():
    """idx 0~6 layout 회귀 (좌표/속도벡터/ships/owner).

    PR2 변경: idx 2,3 = (cos, sin) → (vx/MAX_SPEED, vy/MAX_SPEED)
    where vx = fleet_speed(ships) × cos(angle).
    """
    from prediction import fleet_speed, MAX_SPEED

    raw_planets = [_planet(0)]
    raw_fleets  = [_fleet(0, owner=0, from_pid=0, ships=300, x=20.0, y=80.0, angle=0.0)]
    arr = encode_fleets(raw_fleets, raw_planets, player=0)

    speed_300 = fleet_speed(300)
    expected_vx = speed_300 * math.cos(0.0) / MAX_SPEED
    expected_vy = speed_300 * math.sin(0.0) / MAX_SPEED

    assert arr[0, 0] == pytest.approx(0.20)                  # x/100
    assert arr[0, 1] == pytest.approx(0.80)                  # y/100
    assert arr[0, 2] == pytest.approx(expected_vx, abs=1e-5) # vx/MAX_SPEED
    assert arr[0, 3] == pytest.approx(expected_vy, abs=1e-5) # vy/MAX_SPEED
    assert arr[0, 4] == pytest.approx(0.30)                  # 300/1000
    assert arr[0, 5] == 1.0                                  # owner == player
    assert arr[0, 6] == 0.0                                  # not enemy


def test_velocity_direction_recovered_via_atan2():
    """vx, vy 로부터 atan2 로 angle 복원 가능 (ships > 0 일 때)."""
    raw_planets = [_planet(0)]
    angles = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4,
              math.pi - 1e-3, -math.pi / 2, -math.pi / 4]
    raw_fleets = [
        _fleet(i, owner=0, from_pid=0, ships=200, angle=a)
        for i, a in enumerate(angles)
    ]
    arr = encode_fleets(raw_fleets, raw_planets, player=0)
    for i, a in enumerate(angles):
        vx = arr[i, 2]
        vy = arr[i, 3]
        recovered = math.atan2(vy, vx)
        # angle wrap (±π) 허용
        diff = (recovered - a + math.pi) % (2 * math.pi) - math.pi
        assert abs(diff) < 1e-4, f"angle={a}, recovered={recovered}"


def test_velocity_magnitude_scales_with_ships():
    """ships 가 많을수록 |v| 가 크다 (단조 증가)."""
    from prediction import fleet_speed, MAX_SPEED

    raw_planets = [_planet(0)]
    ship_counts = [10, 100, 500, 2000]   # 2000 은 cap (= MAX_SPEED)
    raw_fleets = [
        _fleet(i, owner=0, from_pid=0, ships=n, angle=0.0)
        for i, n in enumerate(ship_counts)
    ]
    arr = encode_fleets(raw_fleets, raw_planets, player=0)

    mags = [math.hypot(arr[i, 2], arr[i, 3]) for i in range(len(ship_counts))]

    # 단조 증가 (혹은 cap 에서 동률)
    for a, b in zip(mags, mags[1:]):
        assert a <= b + 1e-6, f"non-monotonic: {mags}"

    # cap (ships=2000) 에서 |v|/MAX_SPEED ≈ 1
    assert mags[-1] == pytest.approx(1.0, abs=1e-5)
    # ships=10 에서 fleet_speed(10)/MAX_SPEED 와 일치
    assert mags[0] == pytest.approx(fleet_speed(10) / MAX_SPEED, abs=1e-5)


def test_low_ships_yields_minimum_velocity():
    """ships ≤ 1 → fleet_speed fallback 1.0 (log(0)/log(1) 0-div 가드).
    |v|/MAX_SPEED = 1/6 (cos/sin scale 만 적용). encoder 가 안 터지는지 회귀.

    이름이 ‘zero_velocity’ 였던 이전 버전은 본문과 의미가 어긋났음 (실제로는
    nonzero minimum 단언) → 이름 정정.
    """
    from prediction import fleet_speed, MAX_SPEED

    raw_planets = [_planet(0)]
    # ships=1: fleet_speed(1) = 1.0 (가드)
    raw_fleets_1 = [_fleet(0, owner=0, from_pid=0, ships=1, angle=0.0)]
    arr_1 = encode_fleets(raw_fleets_1, raw_planets, player=0)
    assert arr_1[0, 2] == pytest.approx(fleet_speed(1) / MAX_SPEED, abs=1e-5)
    assert arr_1[0, 3] == pytest.approx(0.0, abs=1e-6)
    # 추가: ships=0 도 같은 fallback (가드 분기 robust 회귀)
    raw_fleets_0 = [_fleet(0, owner=0, from_pid=0, ships=0, angle=0.0)]
    arr_0 = encode_fleets(raw_fleets_0, raw_planets, player=0)
    assert arr_0[0, 2] == pytest.approx(1.0 / MAX_SPEED, abs=1e-5)
    assert arr_0[0, 3] == pytest.approx(0.0, abs=1e-6)


def test_planets_overflow_beyond_max_maps_to_sentinel():
    """planets 가 MAX_PLANETS 초과 → 잘린 후의 idx 만 valid, 그 이후 from_pid 는 -1.

    encode_planets/fleets 둘 다 [:MAX_PLANETS] slice 를 적용하므로
    잘린 행성을 source 로 갖는 fleet 은 lookup 불가 → -1.
    """
    from env_wrapper import MAX_PLANETS
    raw_planets = [_planet(i) for i in range(MAX_PLANETS + 5)]
    # 잘릴 행성 (idx ≥ MAX_PLANETS) 을 source 로
    raw_fleets = [_fleet(0, owner=0, from_pid=MAX_PLANETS + 2)]
    arr = encode_fleets(raw_fleets, raw_planets, player=0)
    assert arr[0, -1] == -1.0
