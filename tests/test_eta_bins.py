"""
P2-3 검증 테스트: ETA bin 기반 incoming 집계.

- near(1~5턴) / mid(6~15턴) 구간 분류 정확성
- 구간 초과 fleet는 집계되지 않음
- enemy/mine 각각 올바른 bin에 분류
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import numpy as np
import pytest
from env_wrapper import encode_planets, PLANET_DIM, ETA_NEAR, ETA_MID

# Planet(id, owner, x, y, radius, ships, production)
# Fleet(id, owner, x, y, angle, from_planet_id, ships)

PLAYER = 0

def make_planet(id_, owner=0, x=50.0, y=50.0, radius=5.0, ships=50, prod=2):
    return (id_, owner, x, y, radius, ships, prod)


def make_fleet_heading_to(pid, fleet_id, owner, planet_x, planet_y, distance, ships=1):
    """planet 바로 아래에서 위를 향하는 fleet — 직선 경로가 행성을 통과."""
    angle = math.pi / 2  # 위쪽(+y) 방향
    fx = planet_x
    fy = planet_y - distance
    return (fleet_id, owner, fx, fy, angle, pid, ships)


def get_incoming(raw_planets, raw_fleets):
    """encode_planets 결과에서 incoming bin 4개 추출 (idx 9~12)."""
    arr = encode_planets(raw_planets, raw_fleets, player=PLAYER, comet_ids=set())
    # [enemy_near, enemy_mid, mine_near, mine_mid] — 정규화 전 원본 아님 (×1000)
    return arr[0, 9], arr[0, 10], arr[0, 11], arr[0, 12]


# ── near bin (ETA ≤ 5) ────────────────────────────────────────────────────────

def test_enemy_near_bin():
    """ETA ≤ 5인 적 fleet → enemy_near에 집계."""
    planet = make_planet(0, owner=PLAYER)
    fleet  = make_fleet_heading_to(0, 1, owner=1, planet_x=50, planet_y=50, distance=4, ships=1)
    # ships=1, speed=1.0, ETA=ceil(4/1)=4 → near
    en, em, mn, mm = get_incoming([planet], [fleet])
    assert en > 0,  "enemy_near가 0 — near bin 미집계"
    assert em == 0, "enemy_mid가 0이어야 함"
    assert mn == 0
    assert mm == 0


def test_mine_near_bin():
    """ETA ≤ 5인 아군 fleet → mine_near에 집계."""
    planet = make_planet(0, owner=PLAYER)
    fleet  = make_fleet_heading_to(0, 1, owner=PLAYER, planet_x=50, planet_y=50, distance=3, ships=1)
    en, em, mn, mm = get_incoming([planet], [fleet])
    assert mn > 0,  "mine_near가 0 — near bin 미집계"
    assert mm == 0
    assert en == 0
    assert em == 0


# ── mid bin (6 ≤ ETA ≤ 15) ───────────────────────────────────────────────────

def test_enemy_mid_bin():
    """6 ≤ ETA ≤ 15인 적 fleet → enemy_mid에 집계."""
    planet = make_planet(0, owner=PLAYER)
    fleet  = make_fleet_heading_to(0, 1, owner=1, planet_x=50, planet_y=50, distance=10, ships=1)
    # ETA=10 → mid
    en, em, mn, mm = get_incoming([planet], [fleet])
    assert em > 0,  "enemy_mid가 0 — mid bin 미집계"
    assert en == 0, "enemy_near가 0이어야 함"


def test_mine_mid_bin():
    """6 ≤ ETA ≤ 15인 아군 fleet → mine_mid에 집계."""
    planet = make_planet(0, owner=PLAYER)
    fleet  = make_fleet_heading_to(0, 1, owner=PLAYER, planet_x=50, planet_y=50, distance=8, ships=1)
    en, em, mn, mm = get_incoming([planet], [fleet])
    assert mm > 0,  "mine_mid가 0 — mid bin 미집계"
    assert mn == 0


# ── far fleet (ETA > 15) — 어느 bin에도 집계 안 됨 ──────────────────────────

def test_far_fleet_not_counted():
    """ETA > 15인 fleet는 어느 bin에도 집계되지 않아야 함."""
    planet = make_planet(0, owner=PLAYER)
    fleet  = make_fleet_heading_to(0, 1, owner=1, planet_x=50, planet_y=50, distance=20, ships=1)
    # ETA=20 > ETA_MID=15
    en, em, mn, mm = get_incoming([planet], [fleet])
    assert en == 0 and em == 0 and mn == 0 and mm == 0


# ── PLANET_DIM 상수 확인 ──────────────────────────────────────────────────────

def test_planet_dim_is_13():
    """PLANET_DIM이 13인지 확인 (ETA bin 추가 반영)."""
    assert PLANET_DIM == 13


def test_encode_planets_output_shape():
    """encode_planets 출력이 (MAX_PLANETS, 13) shape인지 확인."""
    from env_wrapper import MAX_PLANETS
    planet = make_planet(0, owner=PLAYER)
    arr = encode_planets([planet], [], player=PLAYER, comet_ids=set())
    assert arr.shape == (MAX_PLANETS, 13)
