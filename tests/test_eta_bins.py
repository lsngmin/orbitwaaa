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
    """encode_planets 결과에서 incoming bin 4개 추출 (idx 11~14)."""
    arr = encode_planets(raw_planets, raw_fleets, player=PLAYER, comet_ids=set())
    # [enemy_near, enemy_mid, mine_near, mine_mid] — 정규화 전 원본 아님 (×1000)
    return arr[0, 11], arr[0, 12], arr[0, 13], arr[0, 14]


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

def test_planet_dim_is_16():
    """PLANET_DIM=16 (15 features + 1 is_valid sentinel)."""
    assert PLANET_DIM == 16


def test_encode_planets_output_shape():
    """encode_planets 출력이 (MAX_PLANETS, PLANET_DIM) shape."""
    from env_wrapper import MAX_PLANETS
    planet = make_planet(0, owner=PLAYER)
    arr = encode_planets([planet], [], player=PLAYER, comet_ids=set())
    assert arr.shape == (MAX_PLANETS, PLANET_DIM)


def test_is_valid_sentinel():
    """채워진 행 마지막 dim = 1.0, 빈 슬롯 = 0.0."""
    from env_wrapper import MAX_PLANETS
    planet = make_planet(0, owner=PLAYER)
    arr = encode_planets([planet], [], player=PLAYER, comet_ids=set())
    assert arr[0, -1] == 1.0           # filled
    assert arr[1:, -1].sum() == 0.0    # rest empty


def test_orbit_velocity_features_for_orbiting_planet():
    """공전 행성은 현재 속도 벡터(vx, vy)를 feature로 가진다."""
    planet = make_planet(0, owner=PLAYER, x=70.0, y=50.0, radius=4.0)
    av = 0.10
    arr = encode_planets([planet], [], player=PLAYER, comet_ids=set(), angular_velocity=av)
    vx = arr[0, 9]
    vy = arr[0, 10]

    assert vx == pytest.approx(0.0, abs=1e-6)
    assert vy == pytest.approx(0.4, abs=1e-6)


def test_orbit_velocity_features_zero_for_static_planet():
    """정적 행성은 vx/vy가 0이어야 한다."""
    planet = make_planet(0, owner=PLAYER, x=95.0, y=50.0, radius=5.0)
    arr = encode_planets([planet], [], player=PLAYER, comet_ids=set(), angular_velocity=0.10)
    assert arr[0, 9] == pytest.approx(0.0)
    assert arr[0, 10] == pytest.approx(0.0)


def test_is_orbiting_flag_off_for_comet_regardless_of_distance():
    """comet 은 sun 거리가 perihelion/aphelion 사이에서 변동 → is_orbiting (idx 7)
    이 깜빡이지 않도록 comet 이면 강제 0.

    회귀: idx 7 의 의미를 "ω 로 회전 중인 일반 행성" 으로 고정.
    is_comet (idx 8) 이 comet 정체성을 안정적으로 알린다.
    """
    # 같은 comet 이 두 위치 (sun 가까이 / 멀리) 에 있을 때 둘 다 idx 7 = 0 이어야 함.
    near_sun = make_planet(7, owner=-1, x=55.0, y=50.0, radius=3.0)   # is_orbiting==True 인 거리
    far_sun  = make_planet(8, owner=-1, x=95.0, y=50.0, radius=3.0)   # is_orbiting==False 인 거리

    arr = encode_planets(
        [near_sun, far_sun], [], player=PLAYER,
        comet_ids={7, 8}, angular_velocity=0.0,
    )
    # idx 7 = is_orbiting, idx 8 = is_comet
    assert arr[0, 7] == 0.0, "comet (sun 가까이) 는 is_orbiting=0 이어야 함"
    assert arr[1, 7] == 0.0, "comet (sun 멀리) 는 is_orbiting=0 이어야 함"
    assert arr[0, 8] == 1.0
    assert arr[1, 8] == 1.0


def test_is_orbiting_flag_on_for_non_comet_orbiting_planet():
    """일반 (non-comet) 회전 행성은 idx 7 = 1 (sanity)."""
    p = make_planet(0, owner=PLAYER, x=70.0, y=50.0, radius=4.0)
    arr = encode_planets([p], [], player=PLAYER, comet_ids=set(), angular_velocity=0.05)
    assert arr[0, 7] == 1.0
    assert arr[0, 8] == 0.0


def test_comet_velocity_zero_when_no_path_data():
    """comet 인데 `comets` 인자가 비어있으면 (= path 미상) velocity 0 fallback.

    PR3 이전과 동일한 회귀: 학습 obs 가 comets 를 채우지 못하는 corner case
    (혹은 외부 호출자가 미제공) 에서도 안전하게 0 으로 떨어지는지.
    """
    comet = make_planet(7, owner=-1, x=40.0, y=50.0, radius=1.0)
    arr = encode_planets([comet], [], player=PLAYER, comet_ids={7}, angular_velocity=0.10)
    assert arr[0, 9] == pytest.approx(0.0)
    assert arr[0, 10] == pytest.approx(0.0)


def test_comet_velocity_from_paths():
    """comet velocity = `paths[i][idx+1] - paths[i][idx]`, MAX_SPEED 정규화."""
    from prediction import MAX_SPEED

    comet = make_planet(7, owner=-1, x=30.0, y=50.0, radius=1.0)
    # 한 group, 한 planet_id, idx=2 → cur=(30,50), next=(33,52) → vx=3, vy=2
    comets = [{
        "planet_ids": [7],
        "paths": [[
            [10.0, 50.0],
            [20.0, 50.0],
            [30.0, 50.0],   # idx=2 (현재)
            [33.0, 52.0],   # idx=3 (다음)
            [36.0, 56.0],
        ]],
        "path_index": 2,
    }]
    arr = encode_planets([comet], [], player=PLAYER, comet_ids={7},
                         comets=comets, angular_velocity=0.0)
    assert arr[0, 9]  == pytest.approx(3.0 / MAX_SPEED, abs=1e-5)
    assert arr[0, 10] == pytest.approx(2.0 / MAX_SPEED, abs=1e-5)


def test_comet_velocity_zero_at_last_path_index():
    """path_index 가 마지막 → 다음 턴 expire → velocity 0 (boundary)."""
    comet = make_planet(7, owner=-1, x=36.0, y=56.0, radius=1.0)
    comets = [{
        "planet_ids": [7],
        "paths": [[
            [10.0, 50.0],
            [20.0, 50.0],
            [36.0, 56.0],   # idx=2, 마지막 (len=3)
        ]],
        "path_index": 2,
    }]
    arr = encode_planets([comet], [], player=PLAYER, comet_ids={7},
                         comets=comets, angular_velocity=0.0)
    assert arr[0, 9]  == pytest.approx(0.0)
    assert arr[0, 10] == pytest.approx(0.0)


def test_comet_velocity_zero_at_negative_path_index():
    """`path_index < 0` → encoder 의 `if idx < 0: continue` 가드 → velocity 0.

    spawn 직후 미초기화나 미정의 corner case (예: 외부 호출자가 -1 default 를
    그대로 넘김) 에서 lookup 이 안 터지고 안전하게 0 으로 fallback 하는지 회귀.
    `_at_last_path_index` 와 짝을 이루는 boundary 회귀.
    """
    comet = make_planet(7, owner=-1, x=30.0, y=50.0, radius=1.0)
    comets = [{
        "planet_ids": [7],
        "paths": [[
            [30.0, 50.0],
            [33.0, 52.0],
            [36.0, 56.0],
        ]],
        "path_index": -1,   # ← 가드 대상
    }]
    arr = encode_planets([comet], [], player=PLAYER, comet_ids={7},
                         comets=comets, angular_velocity=0.0)
    assert arr[0, 9]  == pytest.approx(0.0)
    assert arr[0, 10] == pytest.approx(0.0)


def test_comet_velocity_independent_of_other_planets_orbit_velocity():
    """non-comet orbiting 행성은 comets 인자 영향 없음 — 기존 ω 기반 식 그대로."""
    from prediction import MAX_SPEED

    orbiter = make_planet(0, owner=PLAYER, x=70.0, y=50.0, radius=4.0)
    comet   = make_planet(7, owner=-1,   x=30.0, y=50.0, radius=1.0)
    comets  = [{
        "planet_ids": [7],
        "paths": [[[30.0, 50.0], [33.0, 52.0]]],
        "path_index": 0,
    }]
    av = 0.10
    arr = encode_planets([orbiter, comet], [], player=PLAYER, comet_ids={7},
                         comets=comets, angular_velocity=av)
    # orbiter (idx 0): PR1 회귀 — vx=0, vy=0.4
    assert arr[0, 9]  == pytest.approx(0.0, abs=1e-6)
    assert arr[0, 10] == pytest.approx(0.4, abs=1e-6)
    # comet (idx 1): PR3 — paths-based
    assert arr[1, 9]  == pytest.approx(3.0 / MAX_SPEED, abs=1e-5)
    assert arr[1, 10] == pytest.approx(2.0 / MAX_SPEED, abs=1e-5)


def test_comet_velocity_uses_correct_path_per_planet_id():
    """group["planet_ids"][j] ↔ group["paths"][j] 매핑 정확."""
    from prediction import MAX_SPEED

    # 두 comet, 같은 group: pid=7 → paths[0], pid=8 → paths[1]
    comet_a = make_planet(7, owner=-1, x=30.0, y=50.0, radius=1.0)
    comet_b = make_planet(8, owner=-1, x=70.0, y=50.0, radius=1.0)
    comets = [{
        "planet_ids": [7, 8],
        "paths": [
            [[30.0, 50.0], [34.0, 50.0]],   # pid=7, vx=4
            [[70.0, 50.0], [70.0, 53.0]],   # pid=8, vy=3
        ],
        "path_index": 0,
    }]
    arr = encode_planets([comet_a, comet_b], [], player=PLAYER,
                         comet_ids={7, 8}, comets=comets, angular_velocity=0.0)
    assert arr[0, 9]  == pytest.approx(4.0 / MAX_SPEED, abs=1e-5)
    assert arr[0, 10] == pytest.approx(0.0, abs=1e-6)
    assert arr[1, 9]  == pytest.approx(0.0, abs=1e-6)
    assert arr[1, 10] == pytest.approx(3.0 / MAX_SPEED, abs=1e-5)


def test_comet_velocity_clipped_when_exceeds_max_speed():
    """비정상적으로 큰 path 변위는 MAX_SPEED 로 clip."""
    from prediction import MAX_SPEED

    comet = make_planet(7, owner=-1, x=30.0, y=50.0, radius=1.0)
    # cometSpeed=4 정상보다 큰 변위 (방어적 clip 검증)
    comets = [{
        "planet_ids": [7],
        "paths": [[[30.0, 50.0], [30.0 + MAX_SPEED * 3, 50.0]]],
        "path_index": 0,
    }]
    arr = encode_planets([comet], [], player=PLAYER, comet_ids={7},
                         comets=comets, angular_velocity=0.0)
    assert arr[0, 9]  == pytest.approx(1.0)   # clip
    assert arr[0, 10] == pytest.approx(0.0)


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
    a_enemy = arr[0, 11] + arr[0, 12]
    # planet_b(idx 1): 아무 bin도 0이어야 함
    b_enemy = arr[1, 11] + arr[1, 12]

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
    enemy_near = arr[0, 11]
    enemy_mid  = arr[0, 12]

    # t=4 → ETA=4 → near bin이어야 함
    assert enemy_near > 0, f"t=4이므로 near bin에 있어야 함 (near={enemy_near}, mid={enemy_mid})"
    assert enemy_mid  == 0
