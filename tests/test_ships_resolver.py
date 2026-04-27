"""
resolve_ships_for_capture 검증.

지원 모드 (Phase A):
  amount_mode="multiplier" (default, production):
    ships_needed = clip(ceil(required × multiplier), 1, src_ships)
    bin_value = multiplier ∈ ships_multipliers (e.g. [1.00, 1.05, 1.12, 1.20])
  amount_mode="surplus" (legacy):
    surplus      = max(0, src_ships - required)
    ships_needed = clip(round(required + bin × surplus), 1, src_ships)
    bin_value = bin ∈ ships_surplus_bins (e.g. [0.0, 0.33, 0.66, 1.0])

src_ships < required → capacity short → ships_needed = 0 (dominated action 차단).
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


# ═══════════════════════════════════════════════════════════════════════════
# multiplier mode (production default)
# ═══════════════════════════════════════════════════════════════════════════

def test_multiplier_one_returns_required_when_capacity_sufficient():
    """multiplier=1.0 → ships == ceil(required × 1.0) == required."""
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=100, production=0)
    dst = StaticPlanet(1, 50.0,  0.0, owner=1, ships=5,   production=2)

    ships, _, _, _, _, required, converged = resolve_ships_for_capture(
        src, dst, angular_velocity=0.0, bin_value=1.0, src_ships=src.ships,
        amount_mode="multiplier",
    )
    assert converged
    assert ships == required, f"mult=1.0: ships={ships}, required={required}"
    assert required > 0


def test_multiplier_self_consistent_required():
    """multiplier=1.0 의 고정점: required 가 ships=required 로 계산된 값과 일치."""
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=100, production=0)
    dst = StaticPlanet(1, 50.0,  0.0, owner=1, ships=5,   production=2)

    ships, _, _, _, _, required, _ = resolve_ships_for_capture(
        src, dst, 0.0, bin_value=1.0, src_ships=src.ships, amount_mode="multiplier",
    )
    actual_req = _analytical_required(dst.ships, dst.production, ships, 50.0)
    assert actual_req == required, (
        f"고정점 불일치: reported required={required}, actual={actual_req}"
    )


def test_multiplier_top_bin_exceeds_required():
    """multiplier=1.20 → ships = ceil(required × 1.20) > required."""
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=100, production=0)
    dst = StaticPlanet(1, 50.0,  0.0, owner=1, ships=5,   production=2)

    ships, _, _, _, _, required, converged = resolve_ships_for_capture(
        src, dst, 0.0, bin_value=1.20, src_ships=src.ships, amount_mode="multiplier",
    )
    assert converged
    assert ships == math.ceil(required * 1.20), (
        f"mult=1.20: ships={ships}, expected ceil({required}*1.20)={math.ceil(required*1.20)}"
    )
    # 핵심: top multiplier 라도 src_ships 전부 보내지 않음 (multiplier mode 의 이점)
    assert ships < src.ships


def test_multiplier_intermediate_bins_in_range():
    """multiplier ∈ [1.00, 1.05, 1.12, 1.20] → required ≤ ships ≤ src.ships.

    엄격 단조성은 보장되지 않음: ships ↑ → fleet 속도 ↑ → turns ↓ → required ↓ →
    ceil(req·mult) 가 ±1 흔들릴 수 있음 (fixed-point 결합). 핵심 invariant 만 검증.
    """
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=200, production=0)
    dst = StaticPlanet(1, 60.0,  0.0, owner=1, ships=8,   production=2)

    bins = [1.00, 1.05, 1.12, 1.20]
    ships_per_bin = []
    for mult in bins:
        ships, _, _, _, _, required, _ = resolve_ships_for_capture(
            src, dst, 0.0, bin_value=mult, src_ships=src.ships, amount_mode="multiplier",
        )
        assert ships >= required, (
            f"mult={mult}: ships={ships} < required={required}"
        )
        assert ships <= src.ships, (
            f"mult={mult}: ships={ships} > src.ships={src.ships}"
        )
        ships_per_bin.append(ships)

    # 약한 단조: 끝점은 확실히 강하게 비감소 (ceil(r·1.20) ≥ ceil(r·1.00) = r).
    assert ships_per_bin[-1] >= ships_per_bin[0], (
        f"top mult ships={ships_per_bin[-1]} should be ≥ base ships={ships_per_bin[0]}"
    )


def test_multiplier_capacity_short_returns_zero():
    """src_ships < required 이면 mult 무관 ships=0 (dominated action 차단)."""
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=10, production=0)
    dst = StaticPlanet(1, 80.0,  0.0, owner=1, ships=50, production=3)

    for mult in [1.00, 1.05, 1.12, 1.20]:
        ships, _, _, _, _, required, _ = resolve_ships_for_capture(
            src, dst, 0.0, bin_value=mult, src_ships=src.ships, amount_mode="multiplier",
        )
        assert ships == 0, (
            f"mult={mult}: capacity-short 이면 ships=0 이어야 함 (실제={ships})"
        )
        assert src.ships < required, (
            f"테스트 전제 위반: src.ships={src.ships} < required={required} 이어야 함"
        )


def test_multiplier_zero_src_ships_returns_zero():
    """src_ships=0 → ships=0 (mode 무관)."""
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=0, production=0)
    dst = StaticPlanet(1, 50.0,  0.0, owner=1, ships=5, production=2)

    ships, _, _, _, _, _, _ = resolve_ships_for_capture(
        src, dst, 0.0, bin_value=1.05, src_ships=0, amount_mode="multiplier",
    )
    assert ships == 0


def test_multiplier_adjacent_planets_trivial_case():
    """가까운 행성: turns=1 ~ 2 경계에서도 multiplier 공식 유지."""
    src = StaticPlanet(0,  0.0, 0.0, owner=0, ships=100, production=0)
    dst = StaticPlanet(1,  2.0, 0.0, owner=1, ships=5,   production=2)

    ships, _, _, _, turns, required, _ = resolve_ships_for_capture(
        src, dst, 0.0, bin_value=1.05, src_ships=src.ships, amount_mode="multiplier",
    )
    assert turns >= 1
    assert required <= ships <= src.ships


# ═══════════════════════════════════════════════════════════════════════════
# surplus mode (legacy, 명시적 amount_mode="surplus" 필요)
# ═══════════════════════════════════════════════════════════════════════════

def test_surplus_bin_zero_returns_required_when_capacity_sufficient():
    """surplus bin=0 → ships == required (surplus 0 사용)."""
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=100, production=0)
    dst = StaticPlanet(1, 50.0,  0.0, owner=1, ships=5,   production=2)

    ships, _, _, _, _, required, converged = resolve_ships_for_capture(
        src, dst, angular_velocity=0.0, bin_value=0.0, src_ships=src.ships,
        amount_mode="surplus",
    )
    assert converged
    assert ships == required, f"bin=0: ships={ships}, required={required}"
    assert required > 0


def test_surplus_bin_one_returns_all_src_ships():
    """surplus bin=1 → ships == src_ships (legacy all-in)."""
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=100, production=0)
    dst = StaticPlanet(1, 50.0,  0.0, owner=1, ships=5,   production=2)

    ships, _, _, _, _, _, _ = resolve_ships_for_capture(
        src, dst, 0.0, bin_value=1.0, src_ships=src.ships, amount_mode="surplus",
    )
    assert ships == src.ships


def test_surplus_intermediate_bin_in_range():
    """surplus bins ∈ [0, 1] → required ≤ ships ≤ src_ships, 단조 증가."""
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=200, production=0)
    dst = StaticPlanet(1, 60.0,  0.0, owner=1, ships=8,   production=2)

    prev_ships = None
    for bv in [0.0, 0.33, 0.66, 1.0]:
        ships, _, _, _, _, required, _ = resolve_ships_for_capture(
            src, dst, 0.0, bin_value=bv, src_ships=src.ships, amount_mode="surplus",
        )
        assert required <= ships <= src.ships, (
            f"bin={bv}: ships={ships} not in [{required}, {src.ships}]"
        )
        if prev_ships is not None:
            assert ships >= prev_ships, (
                f"bin={bv}: ships={ships} < prev={prev_ships} (단조성 위배)"
            )
        prev_ships = ships


def test_surplus_intermediate_bin_formula_approximately():
    """surplus bin=0.5 → ships ≈ required + 0.5 × surplus."""
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=100, production=0)
    dst = StaticPlanet(1, 50.0,  0.0, owner=1, ships=5,   production=2)

    ships, _, _, _, _, required, converged = resolve_ships_for_capture(
        src, dst, 0.0, bin_value=0.5, src_ships=src.ships, amount_mode="surplus",
    )
    if converged:
        surplus = src.ships - required
        expected = min(src.ships, max(1, int(round(required + 0.5 * surplus))))
        assert ships == expected, (
            f"bin=0.5: ships={ships}, expected={expected} "
            f"(required={required}, surplus={surplus})"
        )


def test_surplus_capacity_short_returns_zero():
    """surplus src_ships < required 이면 bin 무관 ships=0."""
    src = StaticPlanet(0,  0.0,  0.0, owner=0, ships=10, production=0)
    dst = StaticPlanet(1, 80.0,  0.0, owner=1, ships=50, production=3)

    for bv in [0.0, 0.33, 0.66, 1.0]:
        ships, _, _, _, _, required, _ = resolve_ships_for_capture(
            src, dst, 0.0, bin_value=bv, src_ships=src.ships, amount_mode="surplus",
        )
        assert ships == 0, (
            f"bin={bv}: capacity-short 이면 ships=0 이어야 함 (실제={ships})"
        )
        assert src.ships < required


def test_surplus_adjacent_planets_trivial_case():
    """surplus 모드 가까운 행성 turns=1~2 경계."""
    src = StaticPlanet(0,  0.0, 0.0, owner=0, ships=100, production=0)
    dst = StaticPlanet(1,  2.0, 0.0, owner=1, ships=5,   production=2)

    ships, _, _, _, turns, required, _ = resolve_ships_for_capture(
        src, dst, 0.0, bin_value=0.33, src_ships=src.ships, amount_mode="surplus",
    )
    assert turns >= 1
    assert required <= ships <= src.ships


def test_surplus_monotone_bin_to_ships_across_targets():
    """surplus 다양한 target 에서 bin↑ → ships↑."""
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
                src, dst, 0.0, bin_value=bv, src_ships=src.ships, amount_mode="surplus",
            )
            if prev is not None:
                assert ships >= prev, (
                    f"target dist={dst.x}, bin={bv}: ships={ships} < prev={prev}"
                )
            prev = ships
