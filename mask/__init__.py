"""Actor head 마스킹 — 게이트 기반 (P, *) bool 마스크 빌드.

────────────────────────────────────────────────────────────────────────────
용어 규칙
    mask:
        최종 bool tensor (예: target_mask, action_mask).
        모델의 샘플링 후보를 직접 제한한다.
    gate:
        mask 를 만들기 위한 단일 책임 판정 함수.
        GATES 튜플에 들어간다. True = 통과, False = 차단.
    filter:
        신규 코드에서 사용하지 않는다. 의미가 불명확하므로 gate 로 통일.
        기존 filter 명칭은 CSV 호환이 필요한 경우에만 임시 유지.

파일 분할
    mask/mask_<dim>.py 한 파일 = 한 출력 차원.
        mask_target.py → (P, P)
        mask_action.py → (P, A)
    새 마스크 차원 추가 시 같은 패턴으로 파일 분리.

게이트 함수
    시그니처: def <gate_name>(ctx: MaskContext, src: int, dst_or_action: int) -> bool
    반환: True = 통과(allowed), False = 차단(blocked). 절대 뒤집지 말 것.
    이름:
        게임 의미가 드러나야 한다.
        통과 조건 중심 suffix 사용:
            *_allowed, *_valid, *_needed, *_sufficient, *_clear
        피할 suffix:
            *_filter, block_*, deny_*, not_*
        신규 게이트는 위 규칙 적용. 기존 게이트명은 CSV 호환상 유지 가능.
    원칙: 한 게이트는 한 차단 사유만 본다. 복합 조건은 별도 게이트로 분리.

게이트 docstring 규약
    함수 docstring 에 다음 항목을 반드시 적는다 (없으면 N/A 표시).
        게임 의미: src/dst 에서 어떤 게임 상황을 판정하는지 한 줄.
        통과:     True 가 되는 조건 (게임 용어로).
        차단:     False 가 되는 조건 + 그게 게임상 무엇을 의미하는지.
        scratch:  쓰는 키 / 읽는 키 (있으면 필수).
        주의:     attack/support 전용 등 적용 범위 (있으면 필수).
    "tensor 에서 무엇을 거른다" 가 아니라 "게임에서 무슨 상황을 막는다" 관점.

오케스트레이션
    모듈 최상단에 GATES = (gate1, gate2, …) 튜플로 우선순위 명시.
    GATE_NAMES = tuple(g.__name__ for g in GATES) 로 자동 도출.
    순서는 비용/의존성 기준.
        싸고 독립적인 게이트가 앞.
        비싼 계산/scratch 의존 게이트가 뒤.
    build_<dim>_mask(ctx, …) -> MaskResult 가 유일한 public entry.
        이중 루프 + 첫 실패 게이트 idx 를 blocked_by 에 기록.
        blocked_by 는 첫 실패 원인만 뜻한다.
        mask_block_* 통계는 게이트 순서에 의존한다.

새 게이트 추가
    기본은 GATES 끝에 append.
    단, 후행 게이트의 scratch 의존성이나 비용상 앞에 필요하면 중간 삽입 가능.
    이 경우 기존 mask_block_* 통계 의미가 바뀌므로 commit message 에 명시.
    기존 CSV 와 직접 비교하지 말 것.
    새 게이트 추가 시 logger.py 의 mask_block_<gate_name>_ge{1,5,10,20} 컬럼도 추가.

MaskContext.scratch 캐시 규약
    게이트 간 중간 계산 공유용 dict.
    키 컨벤션: ("<name>", src, dst) 튜플.
        ("target_kind", src, dst) = "attack" | "support"   ※ Phase C 로드맵 — 미구현
        ("aim", src, dst)         = (angle, tx, ty, turns)
        ("proj", src, dst)        = (proj_owner, proj_ships, eff_turns)
        ("required", src, dst)    = int
    ("required", src, dst) 는 target_kind 에 따라 capture_required 또는
        support_required 일 수 있다.
        그러므로 required 를 읽는 쪽은 target_kind 도 함께 확인해야 한다.
    선행 게이트가 통과한 (src, dst) 에서만 캐시를 채운다.
    후행 게이트는 캐시 존재를 가정한다.
    의존 관계는 모듈 docstring 에 명시한다.
    크로스 파일 공유는 스칼라 키 사용.
        "real_target_mask" = decoder-valid target mask.
    sample fallback/dummy target 은 real_target_mask 에 넣지 않는다.

수정 시 주의
    target_mask 가 True 로 연 target 은 decoder 가 반드시 valid move 로 처리해야 한다.
    decoder 가 invalid 로 버릴 target 은 target_mask 에서 열면 안 된다.
    action_mask 는 target owner/path/required 를 직접 판단하지 않는다.
    action_mask 는 real_target_mask[src].any() 만 보고 launch bin 을 연다.

원본 로직 위치는 각 mask_*.py 의 docstring 의 train.py:<lines> 참조 —
리팩터링 검증 시 그 라인과 동작 비교.

현재 게이트 (mask_target.py, GATES 순서)
    target_owner_allowed       ← 옛 owner_rule        (kind 결정: attack/support)
    flight_path_clear          ← 옛 sun_path          (경로)
    projected_arrival_state    (신규)                 (proj 계산만, 차단 안 함)
    attack_still_needed        ← 옛 enemy_neutral_filter
                                                       (attack 전용; support 는 통과)
    capacity_sufficient        ← 옛 capacity_short    (attack: capture_required;
                                                       support: support_required + 35% cap)

    `support_needed` 는 별도 게이트로 두지 않고 target_owner_allowed 가
    "내 행성 + 적 incoming" 조건으로 흡수 — 1-게이트-1-책임 원칙은 유지하되
    support 자격 판정은 kind 결정과 한 흐름이라 분리 이득 없음.

    support_required 는 prediction.compute_support_required 가 단일 정의:
        proj_owner == own  → max(1, ceil(proj_ships*0.20), ceil(dst.production))
        proj_owner != own  → max(1, proj_ships + 1)  (rescue/recapture)
        모든 경우 ceil(src.ships * 0.35) 초과 시 None → mask 차단.
    decoder 의 resolve_ships_for_support 가 같은 함수 호출 → mask/decoder parity.

현재 게이트 (mask_action.py)
    launchable             ← real_target_mask[src].any() 만 본다
                              (신규 네이밍은 launch_action_available — 미적용,
                               CSV 호환 위해 잠정 유지)

CSV 컬럼: rename 와 동시에 logger.py 동기화 완료.
    옛 mask_block_owner_rule_*           → mask_block_target_owner_allowed_*
    옛 mask_block_sun_path_*             → mask_block_flight_path_clear_*
    옛 mask_block_enemy_neutral_filter_* → mask_block_attack_still_needed_*
    옛 mask_block_capacity_short_*       → mask_block_capacity_sufficient_*
    rename 이전 CSV 와 직접 컬럼 비교 불가 — gen 단위 추세는 새 이름으로 시작.

향후 작업
    - support_required = 1 → enemy_pressure 기반 식으로 강화 (capacity_sufficient).
    - launchable → launch_action_available rename (action_mask CSV 컬럼은 현재
      logger.py 에 노출 안 됨 — rename 시 영향 적음).
────────────────────────────────────────────────────────────────────────────
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
