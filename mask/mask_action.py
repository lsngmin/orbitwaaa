"""Action mask — (P, A) bool. mask_target 와 동일한 게이트 리스트 패턴.

게이트 시그니처: def gate(ctx, src, action) -> bool  (True = 통과)
target_result 는 ctx.scratch["real_target_mask"] (decoder-valid target mask)
로 게이트에 전달. sample fallback/dummy target 은 real_target_mask 에
넣지 않는다 — 그래야 launchable 이 "decoder 가 받아들일 valid target 이
하나라도 있는가" 를 정확히 판단한다.

원본: train.py:486-489 + model.py:475-476.
"""
from __future__ import annotations
import torch

from . import MaskContext, MaskResult


def launchable(ctx: MaskContext, src: int, action: int) -> bool:
    """skip(action=0) 은 항상 허용. launch bin 은 src 가 valid target 보유 시만."""
    if action == 0:
        return True
    return bool(ctx.scratch["real_target_mask"][src].any())


GATES = (launchable,)
GATE_NAMES = tuple(g.__name__ for g in GATES)


def build_action_mask(ctx: MaskContext, target_result: MaskResult) -> MaskResult:
    ctx.scratch["real_target_mask"] = target_result.mask
    P, A = ctx.num_planets, ctx.num_actions
    mask = torch.zeros(P, A, dtype=torch.bool)
    blocked_by = torch.full((P, A), -1, dtype=torch.long)
    for i in range(P):
        for a in range(A):
            for k, gate in enumerate(GATES):
                if not gate(ctx, i, a):
                    blocked_by[i, a] = k
                    break
            else:
                mask[i, a] = True
    return MaskResult(mask=mask, blocked_by=blocked_by, gate_names=GATE_NAMES)
