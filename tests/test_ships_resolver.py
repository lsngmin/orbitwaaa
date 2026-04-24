"""
commit 3: resolve_ships_for_capture 고정점 반복 검증.

과거 1-pass 근사 버그:
  - aim(..., p.ships)로 turns를 먼저 추정 → required 계산 → multiplier 적용
  - 실제 보낸 ships_needed가 p.ships보다 훨씬 적으면 fleet_speed ↓ → turns ↑
    → 실제 required ↑ → 의도한 multiplier margin이 소실 (bin=1.10x는 상습 미달)

수정: (ships_needed, required)를 고정점 반복으로 동시 해결.
"""

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from prediction import (
    resolve_ships_for_capture,
    fleet_speed,
    estimate_arrival_turn,
    aim,
)


class StaticPlanet:
    """정적 행성 (is_orbiting=False 경로용)."""
    def __init__(self, id_, x, y, owner, ships, production, radius=3.0):
        self.id         = id_
        self.x          = x
        self.y          = y
        self.owner      = owner
        self.ships      = ships
        self.production = production
        self.radius     = radius


def _analytical_required(dst_ships, production, ships_sent, distance):
    """fleet_speed formula로 실제 도착 턴의 required를 계산."""
    turns = estimate_arrival_turn(distance, ships_sent)
    return dst_ships + production * turns + 1


# ── 수렴성 ───────────────────────────────────────────────────────────────────

def test_converges_within_max_iter():
    """일반 케이스: 3회 이내 수렴."""
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=100, production=0)
    dst = StaticPlanet(1, 50.0,  0.0, owner=1, ships=5,   production=2)

    ships, angle, tx, ty, turns, required, converged = resolve_ships_for_capture(
        src, dst, angular_velocity=0.0, multiplier=1.10, src_ships=src.ships,
    )
    assert converged, "3회 반복에 수렴 실패"
    assert ships > 0
    assert required > 0


def test_clip_to_src_ships_when_need_exceeds():
    """multiplier × required > src.ships인 경우 즉시 src.ships로 clip."""
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=10, production=0)
    # 멀고 큰 target → required 큼
    dst = StaticPlanet(1, 80.0,  0.0, owner=1, ships=50, production=3)

    ships, _, _, _, _, required, converged = resolve_ships_for_capture(
        src, dst, angular_velocity=0.0, multiplier=2.00, src_ships=src.ships,
    )
    assert converged
    assert ships == src.ships    # src.ships로 clip
    assert int(required * 2.00) >= src.ships  # 실제로 clip이 작동한 상황


# ── under-investment 해소 검증 (P1 버그의 핵심) ──────────────────────────────

def test_no_under_invest_bin_110():
    """
    bin=1.10x 시나리오: src.ships=100, 먼 중립 target.
    기존 1-pass 근사에서는 bin=1.10이 상습 미달이었음.

    고정점 해결 후:
      ships_needed × (actual) ≥ required × 1.10 (margin 손실 없음)
      OR ships_needed == src.ships (clip된 경우)
    """
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=100, production=0)
    dst = StaticPlanet(1, 50.0,  0.0, owner=1, ships=5,   production=2)
    multiplier = 1.10

    ships, _, _, _, turns, required, _ = resolve_ships_for_capture(
        src, dst, angular_velocity=0.0, multiplier=multiplier, src_ships=src.ships,
    )

    # 실제 도착 시점의 required (고정점 값) 는 내부 required와 일치해야 함
    actual_required = _analytical_required(
        dst.ships, dst.production, ships, distance=50.0,
    )
    assert actual_required == required, (
        f"고정점 불일치: reported required={required}, actual={actual_required}"
    )

    # 고정점 조건: ships == min(max(1, int(required × mult)), src.ships)
    expected_ships = min(max(1, int(required * multiplier)), src.ships)
    assert ships == expected_ships

    # under-investment 없음: actual required × multiplier ≤ ships (클립 없는 경우)
    if ships < src.ships:
        # 고정점이므로 ships >= int(actual_required × multiplier)
        assert ships >= int(actual_required * multiplier), (
            f"bin={multiplier} under-invested: sent {ships}, "
            f"need {int(actual_required * multiplier)} for {multiplier}x margin "
            f"(actual required={actual_required})"
        )


def test_margin_consistent_across_bins():
    """모든 multiplier bin에서 under-investment 없음을 확인.

    수렴 여부와 무관하게 핵심 불변식:
      ships_needed × actual_speed가 실제 도착 시 required × multiplier ≥ 달성.
    oscillation 시에는 best_ships fallback이 margin을 보장.
    """
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=200, production=0)
    dst = StaticPlanet(1, 60.0,  0.0, owner=1, ships=8,   production=2)

    for mult in [1.10, 1.30, 1.60, 2.00]:
        ships, _, _, _, _, required, _ = resolve_ships_for_capture(
            src, dst, angular_velocity=0.0, multiplier=mult, src_ships=src.ships,
        )
        # required는 최종 ships로 계산된 값이어야 함 (reporting 일관성)
        actual_req = _analytical_required(dst.ships, dst.production, ships, 60.0)
        assert actual_req == required, f"bin={mult}: required 불일치"

        if ships < src.ships:  # clip 안 된 경우만 margin 검증
            assert ships >= int(actual_req * mult), (
                f"bin={mult}: under-invested "
                f"(ships={ships}, need={int(actual_req * mult)}, "
                f"actual_req={actual_req})"
            )


# ── 회귀 방지: 과거 1-pass 근사의 과소평가 재발 시 실패 ──────────────────────

def test_fixed_point_better_than_single_pass():
    """
    동일 시나리오에서 1-pass 근사와 고정점 해결 비교.
    bin=1.10x, 먼 target에서 실제 required gap을 확인.
    """
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=100, production=0)
    dst = StaticPlanet(1, 50.0,  0.0, owner=1, ships=5,   production=2)
    multiplier = 1.10

    # 1-pass 근사 (legacy 버그 재현)
    _, _, _, turns_fast = aim(src, dst, 0.0, src.ships)
    req_legacy          = dst.ships + dst.production * (turns_fast or 1) + 1
    ships_legacy        = min(max(1, int(req_legacy * multiplier)), src.ships)
    # 실제 도착 시 required
    actual_req_legacy   = _analytical_required(
        dst.ships, dst.production, ships_legacy, 50.0,
    )

    # 고정점 해결
    ships_fixed, _, _, _, _, req_fixed, _ = resolve_ships_for_capture(
        src, dst, 0.0, multiplier, src.ships,
    )

    # 1-pass는 req_legacy를 과소평가 → actual > req_legacy (bug)
    # 고정점은 req_fixed == actual_required (self-consistent)
    assert req_fixed >= req_legacy, (
        "고정점 required는 1-pass required보다 크거나 같아야 함"
    )
    # 고정점 ships도 1-pass ships보다 크거나 같아야 (margin 복원)
    assert ships_fixed >= ships_legacy, (
        f"고정점 ships({ships_fixed}) < 1-pass ships({ships_legacy}): "
        "margin 복원 실패"
    )


# ── 엣지 케이스 ───────────────────────────────────────────────────────────────

def test_zero_src_ships_clips_to_one():
    """src_ships=0 → ships_needed=1 (lower bound 보장)."""
    # src_ships=0이어도 max(1, ...) 때문에 1로 떨어지고, min(1, 0)=0.
    # 호출자(decode)가 ships<=0일 때 filtered_zero_ships로 거르는 패턴.
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=0, production=0)
    dst = StaticPlanet(1, 50.0,  0.0, owner=1, ships=5, production=2)

    ships, _, _, _, _, _, converged = resolve_ships_for_capture(
        src, dst, 0.0, multiplier=1.10, src_ships=0,
    )
    assert ships == 0, "src_ships=0이면 clip 후 0"


def test_adjacent_planets_trivial_case():
    """
    아주 가까운 행성: turns=1 vs turns=2 경계에서 oscillation 가능.
    converged=False여도 best_ships fallback이 margin을 보장해야 함.
    """
    src = StaticPlanet(0,  0.0, 0.0, owner=0, ships=100, production=0)
    dst = StaticPlanet(1,  2.0, 0.0, owner=1, ships=5,   production=2)

    ships, _, _, _, turns, required, _ = resolve_ships_for_capture(
        src, dst, 0.0, multiplier=1.10, src_ships=src.ships,
    )
    assert turns >= 1
    # 실제 도착 시점 required 기준 margin 유지 (under-invest 없음)
    actual_req = _analytical_required(dst.ships, dst.production, ships, 2.0)
    assert ships >= int(actual_req * 1.10), (
        f"adjacent planets under-invested: ships={ships}, need={int(actual_req * 1.10)}"
    )
