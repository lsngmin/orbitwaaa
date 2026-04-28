"""Actor head 마스킹.

규약 (마스크 함수 인터페이스):
    def gate(ctx: MaskContext, src: int, dst: int) -> bool
        # True = 통과, False = 차단

build_target_mask / build_action_mask 가 게이트들을 순서대로 적용하고
첫 실패 게이트를 blocked_by 에 기록 (mask_block 진단 호환).
"""
from dataclasses import dataclass, field
from typing import Any
import torch


@dataclass
class MaskContext:
    planets: list
    fleets: list
    av: Any
    acting_player: int
    pos_cache: Any
    num_planets: int
    num_actions: int
    scratch: dict = field(default_factory=dict)  # 게이트 간 aim()/proj 캐시 공유


@dataclass
class MaskResult:
    mask: torch.Tensor          # bool, True = allowed
    blocked_by: torch.Tensor    # long, 첫 실패 게이트 idx (-1 = 통과)
    gate_names: tuple[str, ...]


from .mask_target import build_target_mask
from .mask_action import build_action_mask

__all__ = ["MaskContext", "MaskResult", "build_target_mask", "build_action_mask"]
