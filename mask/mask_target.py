"""Target mask — (P, P) bool. 4 게이트 순차 적용.

게이트 우선순위 (현 train.py 와 동일):
    owner_rule → sun_path → enemy_neutral_filter → capacity_short

원본 로직: train.py:378-417.

ctx.scratch 캐시 키 (다운스트림 feature 계산이 재활용):
    ("aim", src, dst)      = (angle, tx, ty, turns)              # sun_path 통과 시
    ("proj", src, dst)     = (proj_owner, proj_ships, eff_turns) # enemy_neutral_filter 통과 시
    ("required", src, dst) = int                                  # capacity_short 통과 시
"""
from __future__ import annotations
import torch

from prediction import aim, crosses_sun, first_collision_on_path, project_target_at_eta
from . import MaskContext, MaskResult


def owner_rule(ctx: MaskContext, src: int, dst: int) -> bool:
    """train.py:378-386 — padded idx / self-target / src 미소유·ships=0 / dst 내것 차단."""
    P_actual = len(ctx.planets)
    if src >= P_actual or dst >= P_actual or src == dst:
        return False
    src_p = ctx.planets[src]
    if src_p.owner != ctx.acting_player or src_p.ships <= 0:
        return False
    if ctx.planets[dst].owner == ctx.acting_player:
        return False
    return True


def sun_path(ctx: MaskContext, src: int, dst: int) -> bool:
    """train.py:387-401 — 직접 태양 차폐 + first_collision 이 의도한 dst 가 아닌 경우."""
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


def enemy_neutral_filter(ctx: MaskContext, src: int, dst: int) -> bool:
    """train.py:402-410 — Guard A. 도착 시점 owner == acting_player 면 차단."""
    angle, tx, ty, turns = ctx.scratch[("aim", src, dst)]
    eff_turns = turns if turns else 1
    proj_owner, proj_ships = project_target_at_eta(
        ctx.planets[dst], eff_turns, ctx.planets, ctx.fleets,
    )
    if proj_owner == ctx.acting_player:
        return False
    ctx.scratch[("proj", src, dst)] = (proj_owner, proj_ships, eff_turns)
    return True


def capacity_short(ctx: MaskContext, src: int, dst: int) -> bool:
    """train.py:411-416 — src.ships < required(=proj_ships+1) 차단."""
    _, proj_ships, _ = ctx.scratch[("proj", src, dst)]
    required = max(1, int(proj_ships) + 1)
    if ctx.planets[src].ships < required:
        return False
    ctx.scratch[("required", src, dst)] = required
    return True


GATES = (owner_rule, sun_path, enemy_neutral_filter, capacity_short)
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
