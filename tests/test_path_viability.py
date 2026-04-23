"""first_collision_on_path() 단위 테스트.

path viability는 launch 전에 (angle, ships)로 실제 첫 충돌을 예측한다.
엔진 순서 재현: out → sun → planet direct.
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from prediction import first_collision_on_path


def P(id, x, y, radius=3.0, owner=-1, ships=0, production=0):
    return Planet(id=id, owner=owner, x=x, y=y, radius=radius,
                  ships=ships, production=production)


def test_hits_target_directly():
    # 태양(50,50)에서 벗어난 y=10 라인 — sun 방해 없이 직진
    src = P(0, 20.0, 10.0)
    tgt = P(5, 80.0, 10.0)
    cause, pid = first_collision_on_path(
        src, angle=0.0, num_ships=10, planets=[src, tgt], av=0.0,
    )
    assert (cause, pid) == ("planet", 5)


def test_out_of_bounds_when_aimed_off_board():
    # 서쪽 가장자리에서 더 서쪽으로 발사
    src = P(0, 5.0, 50.0)
    cause, pid = first_collision_on_path(
        src, angle=math.pi, num_ships=10, planets=[src], av=0.0,
    )
    assert cause == "out"
    assert pid is None


def test_sun_crash_when_aimed_through_center():
    # (30, 50)에서 동쪽으로 → (50, 50) 태양 관통
    src = P(0, 30.0, 50.0)
    cause, pid = first_collision_on_path(
        src, angle=0.0, num_ships=10, planets=[src], av=0.0,
    )
    assert cause == "sun"
    assert pid is None


def test_hits_blocking_planet_before_target():
    # src (10, 10), blocker (30, 10) at radius 3, target (80, 10)
    src = P(0, 10.0, 10.0)
    blocker = P(1, 30.0, 10.0, radius=3.0)
    tgt = P(5, 80.0, 10.0)
    cause, pid = first_collision_on_path(
        src, angle=0.0, num_ships=10, planets=[src, blocker, tgt], av=0.0,
    )
    # blocker에 먼저 박혀야 함 → hit_other 사례
    assert (cause, pid) == ("planet", 1)


def test_none_when_path_empty_and_within_bounds():
    # 작은 반경, 아무 것도 없는 방향, 짧은 max_turns
    src = P(0, 50.0, 5.0)
    # 동쪽으로 발사, sun 위쪽/아래쪽 피하도록 y=5 → sun center (50,50)과 거리 45 > SUN_RADIUS
    # 1 턴만 시뮬 (max_turns=1). out/planet/sun 모두 아니어야 함.
    cause, pid = first_collision_on_path(
        src, angle=0.0, num_ships=10, planets=[src], av=0.0, max_turns=1,
    )
    assert cause == "none"
    assert pid is None


def test_source_planet_ignored():
    # 자기 자신(source)을 충돌 체크에서 무시하는지 확인. y=10라인 (sun 안전)
    src = P(0, 20.0, 10.0, radius=3.0)
    tgt = P(5, 80.0, 10.0)
    cause, pid = first_collision_on_path(
        src, angle=0.0, num_ships=10, planets=[src, tgt], av=0.0,
    )
    assert pid != src.id


def test_orbit_correction_affects_hit_decision():
    # 공전 중인 target: av=0이면 hit, av가 크면 회전해서 miss.
    # y=10 라인으로 sun 피하고, target은 ROTATION_RADIUS_LIMIT=50 안쪽으로 공전 궤도에 둠.
    src = P(0, 10.0, 20.0, radius=2.0)
    # target orbital radius: sqrt((30-50)^2 + (20-50)^2) = sqrt(400+900) ≈ 36 < 50 → 공전
    tgt = P(5, 30.0, 20.0, radius=2.0)
    # av=0: 정적이면 target 맞음
    cause0, pid0 = first_collision_on_path(
        src, angle=0.0, num_ships=10, planets=[src, tgt], av=0.0,
    )
    assert (cause0, pid0) == ("planet", 5)
    # av 크게: target이 회전해서 경로 벗어남
    cause1, pid1 = first_collision_on_path(
        src, angle=0.0, num_ships=10, planets=[src, tgt], av=1.0,
    )
    assert pid1 != 5
