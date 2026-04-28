"""Target mask — (P, P) bool. attack/support 양 종류를 한 mask 에 담음.

게이트 순서 (GATES):
    target_owner_allowed
    → flight_path_clear
    → projected_arrival_state
    → attack_still_needed
    → capacity_sufficient

scratch 캐시:
    ("target_kind", src, dst) = "attack" | "support"   (target_owner_allowed 통과 시)
    ("aim", src, dst)         = (angle, tx, ty, turns) (flight_path_clear 통과 시)
    ("proj", src, dst)        = (proj_owner, proj_ships, eff_turns)
                                                       (projected_arrival_state 통과 시,
                                                        kind 무관하게 항상 채움)
    ("required", src, dst)    = int                    (capacity_sufficient 통과 시;
                                                        attack=capture_required,
                                                        support=support_required)

설계 원칙 (Phase C support — 정식 action mode):
    support 는 "내 행성 + 적 incoming" 인 dst 에만 허용된다 (target_owner_allowed).
    support_required 는 prediction.compute_support_required 가 정의:
        proj_owner == own  → max(1, ceil(proj_ships * 0.20), ceil(dst.production))
        proj_owner != own  → max(1, proj_ships + 1)  (rescue/recapture)
        모든 경우 source 35% cap 초과 시 None → mask 차단.
    decoder 의 resolve_ships_for_support 가 동일한 함수를 호출 → mask/decoder parity.

원본 로직: train.py:378-417 (rename 전 owner_rule/sun_path/enemy_neutral_filter/
capacity_short).
"""
from __future__ import annotations
import torch

from prediction import (
    aim, crosses_sun, first_collision_on_path, project_target_at_eta,
    fleet_dst_and_eta, compute_support_required,
)
from . import MaskContext, MaskResult


def _has_enemy_incoming(ctx: MaskContext, dst: int) -> bool:
    """dst 행성으로 향하는 적 fleet 가 하나라도 있으면 True.

    fleet_dst_and_eta(f, planets) 는 ray-cast 로 fleet 의 첫 충돌 행성 id 를
    반환. cache 안 함 — 자기 행성은 P 의 작은 부분 (보통 5~10 개) 이라
    per-mask-build 에 비용 부담 없음.
    """
    dst_id = ctx.planets[dst].id
    for f in ctx.fleets:
        if f.owner == ctx.acting_player:
            continue
        target_id, _ = fleet_dst_and_eta(f, ctx.planets)
        if target_id == dst_id:
            return True
    return False


def target_owner_allowed(ctx: MaskContext, src: int, dst: int) -> bool:
    """target 종류 결정 gate.

    게임 의미:
        src 행성에서 dst 행성으로 보낼 수 있는 목적지 종류인가?

    통과:
        enemy/neutral 행성 → attack target.
        내 행성 + 적 incoming 있음 → support target.

    차단:
        padding index (src/dst >= P_actual)
        자기 자신 (src == dst)
        src 가 내 행성이 아님
        src.ships <= 0
        안전한 내 행성 (적 incoming 없음 — 지원 불필요)

    scratch:
        ("target_kind", src, dst) = "attack" | "support"
    """
    P_actual = len(ctx.planets)
    if src >= P_actual or dst >= P_actual or src == dst:
        return False
    src_p = ctx.planets[src]
    if src_p.owner != ctx.acting_player or src_p.ships <= 0:
        return False
    dst_p = ctx.planets[dst]
    if dst_p.owner == ctx.acting_player:
        if not _has_enemy_incoming(ctx, dst):
            return False
        ctx.scratch[("target_kind", src, dst)] = "support"
        return True
    ctx.scratch[("target_kind", src, dst)] = "attack"
    return True


def flight_path_clear(ctx: MaskContext, src: int, dst: int) -> bool:
    """공통 gate (attack/support).

    게임 의미:
        src 에서 dst 까지 함대가 실제로 도달 가능한가?

    통과:
        태양에 막히지 않고,
        first collision 이 의도한 dst 행성임.

    차단:
        태양을 지나거나,
        다른 행성에 먼저 충돌하거나,
        경로 계산상 도달 불가.

    scratch:
        ("aim", src, dst) = (angle, tx, ty, turns)
    """
    src_p = ctx.planets[src]
    tgt_p = ctx.planets[dst]
    if crosses_sun(src_p.x, src_p.y, tgt_p.x, tgt_p.y):
        return False
    ships_rep = int(src_p.ships)
    angle, tx, ty, turns = aim(src_p, tgt_p, ctx.av, ships_rep, pos_cache=ctx.pos_cache)
    max_turns = min((turns or 0) + 2, 120) if turns else 120
    cause, hit_pid = first_collision_on_path(
        src_p, angle, ships_rep, ctx.planets, ctx.av,
        max_turns=max_turns, pos_cache=ctx.pos_cache,
    )
    if cause != "planet" or hit_pid != tgt_p.id:
        return False
    ctx.scratch[("aim", src, dst)] = (angle, tx, ty, turns)
    return True


def projected_arrival_state(ctx: MaskContext, src: int, dst: int) -> bool:
    """공통 gate — projection 계산 책임만, 차단하지 않음.

    게임 의미:
        내 함대가 dst 에 도착할 ETA 시점에, 현재 예정된 함대들만 고려하면
        dst 의 owner/ships 가 어떻게 될 것인가?

    통과:
        항상 (kind 무관). proj 결과는 후행 게이트가 차단 결정에 사용.

    차단:
        없음.

    scratch:
        ("proj", src, dst) = (proj_owner, proj_ships, eff_turns)
    """
    _, _, _, turns = ctx.scratch[("aim", src, dst)]
    eff_turns = turns if turns else 1
    proj_owner, proj_ships = project_target_at_eta(
        ctx.planets[dst], eff_turns, ctx.planets, ctx.fleets,
    )
    ctx.scratch[("proj", src, dst)] = (proj_owner, proj_ships, eff_turns)
    return True


def attack_still_needed(ctx: MaskContext, src: int, dst: int) -> bool:
    """공격 target 전용 gate (support 는 통과).

    게임 의미:
        attack target 이라면, dst 에 추가 공격 함대를 보낼 필요가 있는가?

    통과:
        support target → 항상 통과 (이 gate 적용 대상 아님).
        attack target → 도착 시점 dst 가 아직 내 소유가 아닐 것으로 예측됨.

    차단:
        attack target 인데 도착 시점에 이미 내 소유가 될 예정.
        (이미 가는 내 함대가 점령할 예정이라 중복 공격.)
    """
    kind = ctx.scratch[("target_kind", src, dst)]
    if kind == "support":
        return True
    proj_owner, _, _ = ctx.scratch[("proj", src, dst)]
    if proj_owner == ctx.acting_player:
        return False
    return True


def capacity_sufficient(ctx: MaskContext, src: int, dst: int) -> bool:
    """공격/지원 병력 충분성 gate (kind 분기).

    게임 의미:
        src 가 이 target kind 에 맞는 최소 병력을 보낼 수 있는가?

    attack:
        capture_required = max(1, proj_ships + 1)  (도착 시점 점령 비용)
        src.ships >= capture_required 여야 통과.

    support:
        support_required = compute_support_required(...)
            proj_owner==own  → max(1, ceil(proj_ships*0.20), ceil(dst.production))
            proj_owner!=own  → max(1, proj_ships + 1)
        source 35% cap 초과 시 차단 (None 반환).
        통과 시 src.ships >= support_required (cap 안에서 자동 보장).

    차단:
        attack: src.ships < capture_required.
        support: compute_support_required 가 None (cap 초과) 또는 src 부족.

    scratch:
        ("required", src, dst) = int
        attack 와 support 의 required 는 서로 다른 의미. 읽는 쪽은
        ("target_kind", src, dst) 도 함께 확인해야 한다.
    """
    kind = ctx.scratch[("target_kind", src, dst)]
    proj_owner, proj_ships, _ = ctx.scratch[("proj", src, dst)]
    src_p = ctx.planets[src]
    src_ships = int(src_p.ships)
    dst_p = ctx.planets[dst]

    if kind == "attack":
        required = max(1, int(proj_ships) + 1)
    else:  # "support"
        required = compute_support_required(
            src_p, dst_p, proj_owner, proj_ships, ctx.acting_player,
        )
        if required is None:
            return False

    if src_ships < required:
        return False
    ctx.scratch[("required", src, dst)] = required
    return True


GATES = (
    target_owner_allowed,
    flight_path_clear,
    projected_arrival_state,
    attack_still_needed,
    capacity_sufficient,
)
GATE_NAMES = tuple(g.__name__ for g in GATES)


def build_target_mask(ctx: MaskContext) -> MaskResult:
    P = ctx.num_planets
    mask = torch.zeros(P, P, dtype=torch.bool)
    blocked_by = torch.full((P, P), -1, dtype=torch.long)
    for i in range(P):
        for j in range(P):
            for k, gate in enumerate(GATES):
                if not gate(ctx, i, j):
                    blocked_by[i, j] = k
                    break
            else:
                mask[i, j] = True
    return MaskResult(mask=mask, blocked_by=blocked_by, gate_names=GATE_NAMES)
