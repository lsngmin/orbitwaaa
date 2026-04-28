"""Helper 동적화 회귀: project_target_at_eta + Guard A.

설계 의도:
  - project_target_at_eta: target 행성을 eta 시점까지 forward simulate.
    in-flight fleet 도착, ownership swap, production 누적 모두 반영.
  - Guard A: ETA 시점에 target 이 acting_player 소유로 예측되면 mask off
    (이미 내 거 될 거라 추가 launch 무의미 — repeat 행동 직접 차단).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
from prediction import project_target_at_eta, fleet_dst_and_eta


# Planet(id, owner, x, y, radius, ships, production)
# Fleet(id, owner, x, y, angle, from_planet_id, ships)


def _planet(id_, owner=-1, x=50.0, y=50.0, ships=10, production=2):
    return Planet(id_, owner, x, y, 5.0, ships, production)


def _fleet(id_, owner, x, y, angle, ships=20, from_pid=0):
    return Fleet(id_, owner, x, y, angle, from_pid, ships)


# ── project_target_at_eta ─────────────────────────────────────────────────


def test_no_fleets_only_production():
    """fleet 없으면 production 만 누적."""
    target = _planet(0, owner=1, x=50.0, y=50.0, ships=10, production=2)
    planets = [target]
    proj_owner, proj_ships = project_target_at_eta(target, eta=5, planets=planets, fleets=[])
    assert proj_owner == 1
    assert proj_ships == 10 + 2 * 5


def test_self_fleet_reinforces_owner():
    """같은 owner fleet 도착 → 함선 +."""
    # y=20 — sun(50,50) 안 가로지르는 위치 (행성이 sun 위에 있을 수 없으므로).
    target = _planet(0, owner=1, x=50.0, y=20.0, ships=10, production=0)
    src    = _planet(99, owner=1, x=10.0, y=20.0, ships=0)
    # fleet at (15, 20) angle=0 (east) → planet 0 방향
    fleet  = _fleet(0, owner=1, x=15.0, y=20.0, angle=0.0, ships=20, from_pid=99)
    proj_owner, proj_ships = project_target_at_eta(
        target, eta=30, planets=[target, src], fleets=[fleet],
    )
    assert proj_owner == 1
    # production=0 이므로 정확히 ships + fleet.ships
    assert proj_ships >= 30   # 10 + 20 (도착 후 production 가능)


def test_enemy_fleet_captures_when_overwhelming():
    """적 fleet 의 ships > defender → 점령 swap."""
    target = _planet(0, owner=1, x=50.0, y=20.0, ships=5, production=0)
    src    = _planet(99, owner=0, x=10.0, y=20.0, ships=0)
    fleet  = _fleet(0, owner=0, x=15.0, y=20.0, angle=0.0, ships=20, from_pid=99)
    proj_owner, proj_ships = project_target_at_eta(
        target, eta=30, planets=[target, src], fleets=[fleet],
    )
    assert proj_owner == 0   # 점령됨
    assert proj_ships >= 15  # 20 - 5 = 15 (도착 후 production 가능)


def test_enemy_fleet_attack_repelled_when_undermanned():
    """적 fleet 의 ships ≤ defender → 점령 실패, owner 유지."""
    target = _planet(0, owner=1, x=50.0, y=20.0, ships=50, production=0)
    src    = _planet(99, owner=0, x=10.0, y=20.0, ships=0)
    fleet  = _fleet(0, owner=0, x=15.0, y=20.0, angle=0.0, ships=20, from_pid=99)
    proj_owner, proj_ships = project_target_at_eta(
        target, eta=30, planets=[target, src], fleets=[fleet],
    )
    assert proj_owner == 1
    assert proj_ships == 50 - 20


def test_eta_zero_returns_current_state():
    """eta=0 → fleet 무시 (도착 시간 0 미만), 현재 상태 그대로."""
    target = _planet(0, owner=1, ships=42, production=3)
    proj_owner, proj_ships = project_target_at_eta(target, eta=0, planets=[target], fleets=[])
    assert proj_owner == 1
    assert proj_ships == 42


def test_fleet_after_eta_ignored():
    """fleet ETA > sim eta → 시뮬에 포함 안 됨."""
    target = _planet(0, owner=1, x=50.0, y=50.0, ships=10, production=0)
    src    = _planet(99, owner=0, x=10.0, y=50.0, ships=0)
    # 매우 멀리 떨어진 곳에서 발사 → 도착 오래 걸림
    fleet  = _fleet(0, owner=0, x=99.0, y=99.0, angle=math.pi, ships=99, from_pid=99)
    # eta=2 면 그 사이에 fleet 도착 못 함
    proj_owner, proj_ships = project_target_at_eta(
        target, eta=2, planets=[target, src], fleets=[fleet],
    )
    assert proj_owner == 1   # 점령 안 됨
    assert proj_ships == 10  # production=0, fleet 무시


def test_fleet_dst_and_eta_basic():
    """ray-cast 가 첫 충돌 행성 + 도착 turn 반환."""
    p = _planet(7, x=80.0, y=20.0)
    f = _fleet(0, owner=0, x=15.0, y=20.0, angle=0.0, ships=10)
    dst_pid, eta = fleet_dst_and_eta(f, [p])
    assert dst_pid == 7
    assert eta >= 1   # 도착 시간 양수


def test_fleet_dst_and_eta_no_hit():
    """충돌 없으면 (-1, inf)."""
    p = _planet(0, x=10.0, y=10.0)
    # 멀어지는 방향
    f = _fleet(0, owner=0, x=50.0, y=90.0, angle=0.0, ships=10)
    dst_pid, eta = fleet_dst_and_eta(f, [p])
    assert dst_pid == -1
    assert eta == math.inf


# ── Guard A: analyze_action_space 동작 회귀 ──────────────────────────────


def test_guard_a_masks_target_when_already_will_be_owned():
    """이미 내 fleet 이 도착해서 점령 예정 → target_mask off.

    핵심 조건: prior fleet 이 src launch 보다 *먼저* 도착해야 Guard A 가 발동.
      - prior fleet 을 target 가까이 배치 (도착 빠름)
      - src 는 target 에서 멀고 함선 적게 (도착 느림)
    좌표는 sun(중심 50,50, 반경 10.5) 안 가로지르는 위치로 잡음.
    """
    from train import analyze_action_space

    raw_planets = [
        (0, 0, 20.0, 20.0, 5.0, 10, 2),    # 내 행성 (적은 ships → 느린 launch)
        (1, 1, 80.0, 20.0, 5.0, 5, 0),     # 적 행성, 적은 ships
    ]
    # prior fleet — target 코앞 (70, 20), 큰 ships → 빠른 도착, 점령 가능
    raw_fleets = [
        (0, 0, 70.0, 20.0, 0.0, 0, 50),
    ]
    analysis = analyze_action_space(raw_planets, raw_fleets, av=0.0, acting_player=0)
    assert analysis.target_mask[0, 1] == False, \
        "prior fleet 이 src launch 보다 빨리 도착해 점령 예정 → mask off (Guard A)"


def test_guard_a_keeps_mask_when_target_still_enemy_at_eta():
    """내 fleet 부족 → target 여전히 적 → mask on 유지."""
    from train import analyze_action_space

    raw_planets = [
        (0, 0, 20.0, 20.0, 5.0, 100, 2),
        (1, 1, 80.0, 20.0, 5.0, 50, 0),    # 함선 50 (튼튼)
    ]
    # 내 fleet 부족 (ships=10 < defender 50)
    raw_fleets = [
        (0, 0, 30.0, 20.0, 0.0, 0, 10),
    ]
    analysis = analyze_action_space(raw_planets, raw_fleets, av=0.0, acting_player=0)
    assert analysis.target_mask[0, 1] == True, \
        "내 fleet 만으로 점령 불가 → mask 유지 (추가 launch 필요)"


def test_no_fleets_baseline_mask_unchanged():
    """fleet 없으면 mask 동작은 기존과 동일 (geometric only)."""
    from train import analyze_action_space

    raw_planets = [
        (0, 0, 20.0, 20.0, 5.0, 100, 2),
        (1, 1, 80.0, 20.0, 5.0, 5, 0),
    ]
    analysis = analyze_action_space(raw_planets, raw_fleets=[], av=0.0, acting_player=0)
    assert analysis.target_mask[0, 1] == True


# ── 동적 required_ships ─────────────────────────────────────────────────


def test_resolve_ships_dynamic_uses_inflight_fleets():
    """fleets 전달 시 in-flight 효과로 required 가 정적값과 달라짐.

    조건: src launch 가 prior fleet 보다 늦게 도착해야 inbound 효과 반영.
    """
    from prediction import resolve_ships_for_capture

    # src 멀리 + 적은 ships → 느린 launch
    src = _planet(0, owner=0, x=20.0, y=20.0, ships=10, production=0)
    dst = _planet(1, owner=1, x=80.0, y=20.0, ships=50, production=0)

    # prior fleet — target 코앞, 도착 매우 빠름
    inflight = _fleet(0, owner=0, x=70.0, y=20.0, angle=0.0, ships=20, from_pid=0)

    # 정적: required ≈ 50 + 1 = 51
    _, _, _, _, _, req_static, _ = resolve_ships_for_capture(
        src, dst, angular_velocity=0.0, bin_value=1.0, src_ships=src.ships,
    )

    # 동적: 내 inbound 20 → defender 도착 시점 30 → required ≈ 31
    _, _, _, _, _, req_dynamic, _ = resolve_ships_for_capture(
        src, dst, angular_velocity=0.0, bin_value=1.0, src_ships=src.ships,
        fleets=[inflight], planets=[src, dst],
    )

    assert req_dynamic < req_static, \
        f"동적 required ({req_dynamic}) 는 inbound 효과로 정적 ({req_static}) 보다 작아야 함"


def test_resolve_ships_dynamic_zero_when_already_owned():
    """ETA 시점에 이미 내 거 → required = 0."""
    from prediction import resolve_ships_for_capture

    # src 느림 + prior fleet 빠름
    src = _planet(0, owner=0, x=20.0, y=20.0, ships=10, production=0)
    dst = _planet(1, owner=1, x=80.0, y=20.0, ships=5, production=0)

    inflight = _fleet(0, owner=0, x=70.0, y=20.0, angle=0.0, ships=50, from_pid=0)

    _, _, _, _, _, req, _ = resolve_ships_for_capture(
        src, dst, angular_velocity=0.0, bin_value=1.0, src_ships=src.ships,
        fleets=[inflight], planets=[src, dst],
    )
    # 도착 시 이미 내 거 → required 0
    assert req == 0, f"도착 시 자기 행성 → required=0 기대, 받음 {req}"
