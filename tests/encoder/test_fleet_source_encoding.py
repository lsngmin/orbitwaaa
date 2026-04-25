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
    """빈 fleet 슬롯 (fleet 리스트가 짧음) → 마지막 dim = -1.0."""
    raw_planets = [_planet(0)]
    raw_fleets  = []   # 전부 빈 슬롯
    arr = encode_fleets(raw_fleets, raw_planets, player=0)
    assert arr.shape == (MAX_FLEETS, FLEET_DIM)
    # 모든 슬롯의 마지막 dim 은 -1
    assert np.all(arr[:, -1] == -1.0)
    # 앞 numeric feature 들은 0
    assert np.all(arr[:, :FLEET_FEAT_DIM] == 0.0)


def test_first_seven_features_unchanged_from_baseline():
    """0~6 dim 의 의미는 그대로 (좌표/각도/ships/owner)."""
    raw_planets = [_planet(0)]
    raw_fleets  = [_fleet(0, owner=0, from_pid=0, ships=300, x=20.0, y=80.0, angle=0.0)]
    arr = encode_fleets(raw_fleets, raw_planets, player=0)

    assert arr[0, 0] == pytest.approx(0.20)            # x/100
    assert arr[0, 1] == pytest.approx(0.80)            # y/100
    assert arr[0, 2] == pytest.approx(math.cos(0.0))   # cos(angle)
    assert arr[0, 3] == pytest.approx(math.sin(0.0))   # sin(angle)
    assert arr[0, 4] == pytest.approx(0.30)            # 300/1000
    assert arr[0, 5] == 1.0                            # owner == player
    assert arr[0, 6] == 0.0                            # not enemy


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
