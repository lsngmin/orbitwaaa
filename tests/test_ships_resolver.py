"""
resolve_ships_for_capture 검증 (surplus-fraction head).

공식:
  required     = ETA forward sim 기반 도착 시 필요 함선
  surplus      = max(0, src_ships - required)
  ships_needed = clip(required + bin_value × surplus, 1, src_ships)

bin_value=0 → just-capture (= required)
bin_value=1 → 올인 (= src_ships)
src_ships < required → capacity short → ships_needed = src_ships
"""

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from prediction import (
    resolve_ships_for_capture,
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


# ── bin_value=0: just-capture floor ──────────────────────────────────────────

def test_bin_zero_returns_required_when_capacity_sufficient():
    """bin_value=0 → ships_needed == required (surplus 0 사용)."""
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=100, production=0)
    dst = StaticPlanet(1, 50.0,  0.0, owner=1, ships=5,   production=2)

    ships, _, _, _, _, required, converged = resolve_ships_for_capture(
        src, dst, angular_velocity=0.0, bin_value=0.0, src_ships=src.ships,
    )
    assert converged
    # bin=0 은 정확히 required 만큼 발사 (margin 0)
    assert ships == required, f"bin=0: ships={ships}, required={required}"
    assert required > 0


def test_bin_zero_self_consistent_required():
    """bin=0 의 고정점: required 가 ships=required 로 계산된 값과 일치."""
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=100, production=0)
    dst = StaticPlanet(1, 50.0,  0.0, owner=1, ships=5,   production=2)

    ships, _, _, _, _, required, _ = resolve_ships_for_capture(
        src, dst, 0.0, bin_value=0.0, src_ships=src.ships,
    )
    actual_req = _analytical_required(dst.ships, dst.production, ships, 50.0)
    assert actual_req == required, (
        f"고정점 불일치: reported required={required}, actual={actual_req}"
    )


# ── bin_value=1: all-in ──────────────────────────────────────────────────────

def test_bin_one_returns_all_src_ships():
    """bin_value=1 → ships_needed == src_ships (전부 발사)."""
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=100, production=0)
    dst = StaticPlanet(1, 50.0,  0.0, owner=1, ships=5,   production=2)

    ships, _, _, _, _, _, _ = resolve_ships_for_capture(
        src, dst, 0.0, bin_value=1.0, src_ships=src.ships,
    )
    assert ships == src.ships


# ── 중간 bin: required ≤ ships ≤ src_ships ───────────────────────────────────

def test_intermediate_bin_in_range():
    """0 < bin < 1 → required ≤ ships ≤ src_ships, 단조 증가."""
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=200, production=0)
    dst = StaticPlanet(1, 60.0,  0.0, owner=1, ships=8,   production=2)

    prev_ships = None
    for bv in [0.0, 0.33, 0.66, 1.0]:
        ships, _, _, _, _, required, _ = resolve_ships_for_capture(
            src, dst, 0.0, bin_value=bv, src_ships=src.ships,
        )
        # 범위: [required, src_ships]
        assert required <= ships <= src.ships, (
            f"bin={bv}: ships={ships} not in [{required}, {src.ships}]"
        )
        # 단조 비감소 (bin↑ → ships↑)
        if prev_ships is not None:
            assert ships >= prev_ships, (
                f"bin={bv}: ships={ships} < prev={prev_ships} (단조성 위배)"
            )
        prev_ships = ships


def test_intermediate_bin_formula_approximately():
    """bin_value=0.5 → ships ≈ required + 0.5 × surplus.

    converged 케이스만 정확 검증. 고정점 오실레이션 시 best_ships fallback.
    """
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=100, production=0)
    dst = StaticPlanet(1, 50.0,  0.0, owner=1, ships=5,   production=2)

    ships, _, _, _, _, required, converged = resolve_ships_for_capture(
        src, dst, 0.0, bin_value=0.5, src_ships=src.ships,
    )
    if converged:
        surplus = src.ships - required
        expected = min(src.ships, max(1, int(round(required + 0.5 * surplus))))
        assert ships == expected, (
            f"bin=0.5: ships={ships}, expected={expected} "
            f"(required={required}, surplus={surplus})"
        )


# ── capacity-short: src_ships < required ─────────────────────────────────────

def test_capacity_short_returns_zero():
    """src_ships < required 면 bin 무관 ships_needed = 0 (1-A: dominated action 차단).

    이전 동작은 src.ships 전부 보내는 것이었으나, 점령 수학적 불가 = dominated action
    이므로 0 으로 차단하고 호출자가 under_invested 로 집계 후 launch 폐기.
    """
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=10, production=0)
    # 멀고 큰 target → required 큼
    dst = StaticPlanet(1, 80.0,  0.0, owner=1, ships=50, production=3)

    for bv in [0.0, 0.33, 0.66, 1.0]:
        ships, _, _, _, _, required, _ = resolve_ships_for_capture(
            src, dst, 0.0, bin_value=bv, src_ships=src.ships,
        )
        assert ships == 0, (
            f"bin={bv}: capacity-short 이면 ships=0 이어야 함 (실제={ships})"
        )
        assert src.ships < required, (
            f"테스트 전제 위반: src.ships={src.ships} < required={required} 이어야 함"
        )


# ── 엣지 케이스 ───────────────────────────────────────────────────────────────

def test_zero_src_ships_returns_zero():
    """src_ships=0 → ships_needed=0 (호출자 filtered_zero_ships로 거름)."""
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=0, production=0)
    dst = StaticPlanet(1, 50.0,  0.0, owner=1, ships=5, production=2)

    ships, _, _, _, _, _, converged = resolve_ships_for_capture(
        src, dst, 0.0, bin_value=0.5, src_ships=0,
    )
    assert ships == 0


def test_adjacent_planets_trivial_case():
    """아주 가까운 행성: turns=1 vs turns=2 경계에서도 surplus 공식 유지."""
    src = StaticPlanet(0,  0.0, 0.0, owner=0, ships=100, production=0)
    dst = StaticPlanet(1,  2.0, 0.0, owner=1, ships=5,   production=2)

    ships, _, _, _, turns, required, _ = resolve_ships_for_capture(
        src, dst, 0.0, bin_value=0.33, src_ships=src.ships,
    )
    assert turns >= 1
    assert required <= ships <= src.ships


# ── 회귀: bin_value 가 monotone 보장 ─────────────────────────────────────────

def test_monotone_bin_to_ships_across_targets():
    """다양한 target 에서 bin↑ → ships↑ (단조성)."""
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=200, production=0)
    targets = [
        StaticPlanet(1, 30.0, 0.0, owner=1, ships=3,  production=1),
        StaticPlanet(2, 60.0, 0.0, owner=1, ships=8,  production=2),
        StaticPlanet(3, 90.0, 0.0, owner=1, ships=15, production=3),
    ]
    for dst in targets:
        prev = None
        for bv in [0.0, 0.33, 0.66, 1.0]:
            ships, *_ = resolve_ships_for_capture(
                src, dst, 0.0, bin_value=bv, src_ships=src.ships,
            )
            if prev is not None:
                assert ships >= prev, (
                    f"target dist={dst.x}, bin={bv}: ships={ships} < prev={prev}"
                )
            prev = ships
