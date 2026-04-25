"""
Source-side defensive feature 회귀 테스트 (F21/F22).

PLANET_DIM[21..22]:
  F21 = source_enemy_pressure_norm     — 적 행성(eta≤30) ships 합 / 1000
  F22 = source_nearest_enemy_eta_norm  — 가장 가까운 적 행성의 eta / 50

own-only 원칙:
  - p.owner != player → F21=0.0, F22=1.0 (해당없음 = 안전 reading)
  - p.owner == player → 적 행성만 대상 (중립/자기 제외)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from env_wrapper import encode_planets, PLANET_DIM
from prediction import estimate_arrival_turn

PLAYER = 0

# Planet tuple = (id, owner, x, y, radius, ships, production)
def planet(id_, owner, x=0.0, y=0.0, ships=50, prod=2, radius=5.0):
    return (id_, owner, x, y, radius, ships, prod)


# ── feature index ──────────────────────────────────────────────────────────
IDX_PRESSURE      = 21
IDX_NEAREST_ETA   = 22


# ── PLANET_DIM 검증 ────────────────────────────────────────────────────────

def test_planet_dim_is_23():
    """PLANET_DIM 23 (기존 21 + F21/F22 추가)."""
    assert PLANET_DIM == 23


# ── 비-아군 행성: 기본값 (해당없음) ────────────────────────────────────────

def test_enemy_planet_pressure_zero_eta_one():
    """적 행성에는 source 관점 적용 안 됨 → F21=0, F22=1.0."""
    src   = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=100, prod=2)
    enemy = planet(1, owner=1,      x=20.0, y=0.0, ships=50,  prod=2)
    arr = encode_planets([src, enemy], [], player=PLAYER, comet_ids=set())

    assert arr[1, IDX_PRESSURE]    == 0.0
    assert arr[1, IDX_NEAREST_ETA] == 1.0


def test_neutral_planet_pressure_zero_eta_one():
    """중립 행성에는 source 관점 적용 안 됨 → F21=0, F22=1.0."""
    src     = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=100, prod=2)
    neutral = planet(1, owner=-1,     x=20.0, y=0.0, ships=10,  prod=1)
    arr = encode_planets([src, neutral], [], player=PLAYER, comet_ids=set())

    # 적 행성이 없으니 src 자체도 0 (test_own_planet_no_enemies와 일관)
    assert arr[0, IDX_PRESSURE]    == 0.0
    assert arr[0, IDX_NEAREST_ETA] == 1.0
    # neutral은 own이 아니므로 default
    assert arr[1, IDX_PRESSURE]    == 0.0
    assert arr[1, IDX_NEAREST_ETA] == 1.0


# ── 아군 행성: 적 행성 인근 ────────────────────────────────────────────────

def test_own_planet_pressure_sums_nearby_enemy_ships():
    """own 행성의 F21 = 인근(eta≤30) 적 행성 ships 합 / 1000."""
    src   = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=100, prod=2)
    e1    = planet(1, owner=1,      x=10.0, y=0.0, ships=200, prod=2)
    e2    = planet(2, owner=1,      x=15.0, y=0.0, ships=300, prod=3)
    arr = encode_planets([src, e1, e2], [], player=PLAYER, comet_ids=set())

    # 두 적 모두 거리 짧음 → eta≤30, pressure_sum = 200 + 300 = 500
    assert abs(arr[0, IDX_PRESSURE] - 0.5) < 1e-6


def test_own_planet_nearest_enemy_eta():
    """own 행성의 F22 = 가장 가까운 적의 eta / 50."""
    src = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=100, prod=2)
    e1  = planet(1, owner=1,      x=10.0, y=0.0, ships=50,  prod=2)
    e2  = planet(2, owner=1,      x=30.0, y=0.0, ships=50,  prod=2)
    arr = encode_planets([src, e1, e2], [], player=PLAYER, comet_ids=set())

    expected_nearest_eta = estimate_arrival_turn(10.0, 50)
    assert abs(arr[0, IDX_NEAREST_ETA] - expected_nearest_eta / 50.0) < 1e-6


def test_own_planet_pressure_excludes_neutrals_and_own():
    """F21 합산은 적 행성만 — 중립/다른 아군 제외."""
    src   = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=100, prod=2)
    me2   = planet(1, owner=PLAYER, x=10.0, y=0.0, ships=999, prod=3)
    neu   = planet(2, owner=-1,     x=12.0, y=0.0, ships=999, prod=1)
    enemy = planet(3, owner=1,      x=15.0, y=0.0, ships=200, prod=2)
    arr = encode_planets([src, me2, neu, enemy], [], player=PLAYER, comet_ids=set())

    # 적 1개 (200 ships) → pressure = 200/1000 = 0.2
    assert abs(arr[0, IDX_PRESSURE] - 0.2) < 1e-6


def test_own_planet_pressure_excludes_far_enemy():
    """eta > 30 적 행성은 pressure 합산에서 제외 (단 nearest_eta는 무관)."""
    src = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=100, prod=2)
    # 거리 200 → fleet_speed(50) 약 4.6 → eta 약 44 > 30
    far = planet(1, owner=1,      x=200.0, y=0.0, ships=500, prod=3)
    arr = encode_planets([src, far], [], player=PLAYER, comet_ids=set())

    # eta가 윈도우 밖이면 pressure 합 0
    assert arr[0, IDX_PRESSURE] == 0.0
    # nearest_eta는 그래도 채워짐 (eta/50 capped at 1.0)
    assert arr[0, IDX_NEAREST_ETA] == 1.0


def test_own_planet_no_enemies_pressure_zero_eta_one():
    """적 행성이 없으면 own 행성도 F21=0, F22=1.0."""
    src = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=100, prod=2)
    me2 = planet(1, owner=PLAYER, x=10.0, y=0.0, ships=50,  prod=2)
    neu = planet(2, owner=-1,     x=20.0, y=0.0, ships=10,  prod=1)
    arr = encode_planets([src, me2, neu], [], player=PLAYER, comet_ids=set())

    for idx in (0, 1):
        assert arr[idx, IDX_PRESSURE]    == 0.0
        assert arr[idx, IDX_NEAREST_ETA] == 1.0


# ── 정규화 범위 ────────────────────────────────────────────────────────────

def test_pressure_clipped_at_one_when_overpressure():
    """적 ships ≫ 1000 → pressure_norm clip 1.0."""
    src   = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=100, prod=2)
    enemy = planet(1, owner=1,      x=10.0, y=0.0, ships=5000, prod=2)
    arr = encode_planets([src, enemy], [], player=PLAYER, comet_ids=set())

    assert arr[0, IDX_PRESSURE] == 1.0


def test_all_features_within_normalized_range():
    """모든 F21/F22 값이 [0, 1] 범위 안."""
    src   = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=500, prod=3)
    e1    = planet(1, owner=1,      x=10.0, y=0.0, ships=200, prod=2)
    e2    = planet(2, owner=1,      x=30.0, y=0.0, ships=400, prod=4)
    n1    = planet(3, owner=-1,     x=15.0, y=0.0, ships=10,  prod=1)
    arr = encode_planets([src, e1, e2, n1], [], player=PLAYER, comet_ids=set())

    for i in range(4):
        for idx in (IDX_PRESSURE, IDX_NEAREST_ETA):
            v = arr[i, idx]
            assert 0.0 <= v <= 1.0, f"planet {i} feature {idx} out of range: {v}"


# ── submission_features parity ────────────────────────────────────────────

def test_submission_features_parity():
    """submission_features.encode_planets와 학습용 encode_planets 결과 일치."""
    from submission_features import encode_planets as sub_encode

    src   = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=100, prod=2)
    e1    = planet(1, owner=1,      x=10.0, y=0.0, ships=200, prod=2)
    e2    = planet(2, owner=1,      x=15.0, y=0.0, ships=300, prod=3)

    train_arr = encode_planets([src, e1, e2], [], player=PLAYER, comet_ids=set())
    sub_arr   = sub_encode([src, e1, e2], [], player=PLAYER, comet_ids=set())

    # 두 인코더 21개 row × 23 col 모두 일치해야 함 (제출 parity)
    import numpy as np
    assert np.allclose(train_arr, sub_arr), "encode_planets parity 깨짐"
