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

def test_planet_dim_is_21():
    """PLANET_DIM이 21인지 확인 (ETA + 궤도 예측 + 태양 위험도 + target feasibility)."""
    assert PLANET_DIM == 21


def test_encode_planets_output_shape():
    """encode_planets 출력이 (MAX_PLANETS, 21) shape인지 확인."""
    from env_wrapper import MAX_PLANETS
    planet = make_planet(0, owner=PLAYER)
    arr = encode_planets([planet], [], player=PLAYER, comet_ids=set())
    assert arr.shape == (MAX_PLANETS, 21)


# ── P1 회귀: fleet는 레이 상 첫 번째 행성에만 집계 ───────────────────────────

def test_fleet_hits_only_nearest_planet():
    """fleet 직선 경로 상에 두 행성이 있을 때 가까운 쪽에만 집계."""
    # fleet at (50, 30), angle=π/2(위), ships=1
    # planet A at (50, 35) — distance t=5, near bin
    # planet B at (50, 45) — distance t=15, mid bin
    # 게임 규칙상 A에서 소멸 → B에는 집계 안 됨
    planet_a = make_planet(0, owner=1, x=50.0, y=35.0, radius=3.0)
    planet_b = make_planet(1, owner=1, x=50.0, y=45.0, radius=3.0)
    fleet    = (10, 1, 50.0, 30.0, math.pi / 2, 99, 1)  # fleet heading up

    arr = encode_planets([planet_a, planet_b], [fleet], player=PLAYER, comet_ids=set())
    # planet_a(idx 0): enemy_near or enemy_mid 중 하나 > 0
    a_enemy = arr[0, 9] + arr[0, 10]
    # planet_b(idx 1): 아무 bin도 0이어야 함
    b_enemy = arr[1, 9] + arr[1, 10]

    assert a_enemy > 0, "가까운 planet_a에 집계되지 않음"
    assert b_enemy == 0, "fleet 소멸 이후 planet_b에 중복 집계됨 (P1 버그 재발)"


def test_eta_uses_ray_distance_not_center_distance():
    """ETA가 center-to-center 거리가 아닌 레이 진행 거리(t)로 계산되는지 확인.

    fleet가 행성 옆을 비스듬히 지나가는 경우,
    center-to-center 거리는 크지만 t(레이 투영)는 작을 수 있음.
    """
    # planet at (50, 50), radius=5
    # fleet at (47, 46), angle=π/2 (위) — 행성 좌측에서 위로 진행
    # t = -(fx*dx + fy*dy) = -((47-50)*0 + (46-50)*1) = 4  → near bin
    # center-to-center = hypot(3, 4) = 5 → 경계값이라 bin이 달라질 수 있음
    planet = make_planet(0, owner=1, x=50.0, y=50.0, radius=5.0)
    fleet  = (10, 1, 47.0, 46.0, math.pi / 2, 99, 1)

    arr = encode_planets([planet], [fleet], player=PLAYER, comet_ids=set())
    enemy_near = arr[0, 9]
    enemy_mid  = arr[0, 10]

    # t=4 → ETA=4 → near bin이어야 함
    assert enemy_near > 0, f"t=4이므로 near bin에 있어야 함 (near={enemy_near}, mid={enemy_mid})"
    assert enemy_mid  == 0
