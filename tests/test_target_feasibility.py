"""
Target feasibility 번들 (F19/F20/F21) 회귀 테스트.

PLANET_DIM[18..20]:
  F19 = required_norm        — (target.ships + production × min_eta + 1) / 1000
  F20 = best_src_ships_norm  — argmin-eta 기준 아군의 ships / 1000
  F21 = feasibility_norm     — (best_src.ships / required) / 4, clip 1.0

same-source 원칙: min_eta를 달성한 best_src 하나를 재사용해 F19/F20/F21 동일 기준.
자기 행성: F19/F21 = 0 (공격 의미 없음), F20만 "가장 가까운 다른 아군 ships".
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np

from env_wrapper import encode_planets, PLANET_DIM
from prediction import estimate_arrival_turn

PLAYER = 0

# Planet tuple = (id, owner, x, y, radius, ships, production)
def planet(id_, owner, x=0.0, y=0.0, ships=50, prod=2, radius=5.0):
    return (id_, owner, x, y, radius, ships, prod)


# ── feature index (21-dim 중 마지막 3개) ───────────────────────────────────
IDX_REQUIRED       = 18
IDX_BEST_SRC_SHIPS = 19
IDX_FEASIBILITY    = 20


# ── 자기 행성: F19/F21 = 0, F20만 유효 ──────────────────────────────────────

def test_own_planet_required_and_feasibility_are_zero():
    """자기 행성은 required_norm=0, feasibility_norm=0 (공격 개념 없음)."""
    src   = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=200, prod=3)
    other = planet(1, owner=PLAYER, x=10.0, y=0.0, ships=50,  prod=2)
    arr = encode_planets([src, other], [], player=PLAYER, comet_ids=set())

    assert arr[0, IDX_REQUIRED]    == 0.0
    assert arr[0, IDX_FEASIBILITY] == 0.0
    assert arr[1, IDX_REQUIRED]    == 0.0
    assert arr[1, IDX_FEASIBILITY] == 0.0


def test_own_planet_best_src_is_closest_other_own():
    """자기 행성의 F20 = 가장 가까운 다른 아군의 ships / 1000 (증원 원천 정보)."""
    src   = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=200, prod=3)
    other = planet(1, owner=PLAYER, x=10.0, y=0.0, ships=50,  prod=2)
    arr = encode_planets([src, other], [], player=PLAYER, comet_ids=set())

    # src(idx 0)의 가장 가까운 다른 아군 = other (50 ships) → 0.05
    assert abs(arr[0, IDX_BEST_SRC_SHIPS] - 0.05) < 1e-6
    # other(idx 1)의 가장 가까운 다른 아군 = src (200 ships) → 0.2
    assert abs(arr[1, IDX_BEST_SRC_SHIPS] - 0.2) < 1e-6


def test_only_one_own_planet_then_its_features_all_zero():
    """내 행성이 유일하면 그 행성의 F19/F20/F21 모두 0 (best_src=None)."""
    src   = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=200, prod=3)
    enemy = planet(1, owner=1,      x=50.0, y=0.0, ships=5,   prod=2)
    arr = encode_planets([src, enemy], [], player=PLAYER, comet_ids=set())

    # src(유일 아군): best_src=None → 모든 F19/F20/F21 = 0
    assert arr[0, IDX_REQUIRED]       == 0.0
    assert arr[0, IDX_BEST_SRC_SHIPS] == 0.0
    assert arr[0, IDX_FEASIBILITY]    == 0.0


# ── 적 행성: 공식대로 계산 ────────────────────────────────────────────────

def test_enemy_required_follows_formula():
    """required_norm = (ships + production × min_eta + 1) / 1000."""
    src   = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=100, prod=0)
    enemy = planet(1, owner=1,      x=50.0, y=0.0, ships=5,   prod=2)
    arr = encode_planets([src, enemy], [], player=PLAYER, comet_ids=set())

    # best_src = src (유일), dist=50, eta = estimate_arrival_turn(50, 50)
    eta = estimate_arrival_turn(50.0, 50)
    expected_req = (5 + 2 * eta + 1) / 1000.0
    assert abs(arr[1, IDX_REQUIRED] - expected_req) < 1e-5


def test_enemy_best_src_is_argmin_eta_not_argmin_distance():
    """
    best_src는 eta 기준 argmin. 거리 ≒ eta 근사라 실질적으로 일치하지만
    이 테스트는 거리 기준 argmin과 같은 src가 나오는 기본 케이스 검증.
    """
    # 본진: 거리 50, 탄약 10 / 서브: 거리 30, 탄약 500
    home = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=10,  prod=0)
    sub  = planet(2, owner=PLAYER, x=80.0, y=0.0, ships=500, prod=0)
    enemy = planet(1, owner=1,     x=50.0, y=0.0, ships=5,   prod=2)
    arr = encode_planets([home, enemy, sub], [], player=PLAYER, comet_ids=set())

    # enemy(idx 1)의 best_src: dist(home→enemy)=50, dist(sub→enemy)=30
    # 작은 dist → 작은 eta → best_src = sub (500 ships) → 0.5
    assert abs(arr[1, IDX_BEST_SRC_SHIPS] - 0.5) < 1e-6


def test_same_source_bundle_internally_consistent():
    """F19/F20/F21이 모두 동일 best_src 기준으로 일관 유도됨."""
    src   = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=100, prod=0)
    enemy = planet(1, owner=1,      x=50.0, y=0.0, ships=5,   prod=2)
    arr = encode_planets([src, enemy], [], player=PLAYER, comet_ids=set())

    eta      = estimate_arrival_turn(50.0, 50)
    required = 5 + 2 * eta + 1

    expected_req_norm      = required / 1000.0
    expected_best_src_norm = 100 / 1000.0                         # src.ships=100
    expected_feas_norm     = min((100 / required) / 4.0, 1.0)

    assert abs(arr[1, IDX_REQUIRED]       - expected_req_norm)      < 1e-5
    assert abs(arr[1, IDX_BEST_SRC_SHIPS] - expected_best_src_norm) < 1e-6
    assert abs(arr[1, IDX_FEASIBILITY]    - expected_feas_norm)     < 1e-5


# ── 중립 행성 ──────────────────────────────────────────────────────────────

def test_neutral_planet_treated_as_attackable():
    """중립(owner=-1)은 공격 대상 → 적과 동일 공식 적용."""
    src     = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=100, prod=0)
    neutral = planet(1, owner=-1,     x=50.0, y=0.0, ships=5,   prod=2)
    arr = encode_planets([src, neutral], [], player=PLAYER, comet_ids=set())

    assert arr[1, IDX_REQUIRED]       > 0.0
    assert arr[1, IDX_BEST_SRC_SHIPS] > 0.0
    assert arr[1, IDX_FEASIBILITY]    > 0.0


# ── feasibility clip ──────────────────────────────────────────────────────

def test_feasibility_clipped_at_1_when_overpowered():
    """best_src ≫ required → feasibility > 4 → /4 정규화 후 clip 1.0."""
    src   = planet(0, owner=PLAYER, x=0.0, y=0.0, ships=2000, prod=0)
    enemy = planet(1, owner=1,      x=5.0, y=0.0, ships=1,    prod=0)
    arr = encode_planets([src, enemy], [], player=PLAYER, comet_ids=set())

    # required ≈ 1 + 0*eta + 1 = 2, feasibility = 2000/2 = 1000 → clip 1.0
    assert arr[1, IDX_FEASIBILITY] == 1.0


def test_feasibility_near_zero_when_overwhelmed():
    """best_src ≪ required → feasibility 매우 작음."""
    src   = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=5,   prod=0)
    enemy = planet(1, owner=1,      x=50.0, y=0.0, ships=500, prod=3)
    arr = encode_planets([src, enemy], [], player=PLAYER, comet_ids=set())

    # required ≈ 500 + 3*eta + 1 ≈ 651, feasibility ≈ 5/651/4 ≈ 0.002
    assert 0.0 < arr[1, IDX_FEASIBILITY] < 0.01


# ── no own planets 엣지 ───────────────────────────────────────────────────

def test_no_own_planets_all_features_zero():
    """내 행성이 하나도 없으면 모든 행성의 F19/F20/F21 = 0."""
    enemy   = planet(0, owner=1,  x=10.0, y=0.0, ships=50, prod=2)
    neutral = planet(1, owner=-1, x=20.0, y=0.0, ships=5,  prod=1)
    arr = encode_planets([enemy, neutral], [], player=PLAYER, comet_ids=set())

    for i in range(2):
        assert arr[i, IDX_REQUIRED]       == 0.0
        assert arr[i, IDX_BEST_SRC_SHIPS] == 0.0
        assert arr[i, IDX_FEASIBILITY]    == 0.0


# ── 정규화 범위 ────────────────────────────────────────────────────────────

def test_all_features_within_normalized_range():
    """모든 F19/F20/F21 값이 [0, 1] 범위 안."""
    src   = planet(0, owner=PLAYER, x=0.0,  y=0.0, ships=500, prod=3)
    e1    = planet(1, owner=1,      x=30.0, y=0.0, ships=50,  prod=2)
    e2    = planet(2, owner=1,      x=60.0, y=0.0, ships=200, prod=4)
    n1    = planet(3, owner=-1,     x=15.0, y=0.0, ships=10,  prod=1)
    arr = encode_planets([src, e1, e2, n1], [], player=PLAYER, comet_ids=set())

    for i in range(4):
        for idx in (IDX_REQUIRED, IDX_BEST_SRC_SHIPS, IDX_FEASIBILITY):
            v = arr[i, idx]
            assert 0.0 <= v <= 1.0, f"planet {i} feature {idx} out of range: {v}"
