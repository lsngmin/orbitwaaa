"""Greedy expand-support 휴리스틱 봇.

전략 (rule-based, ML 없음):
  - 초반 (step <= EXPAND_PHASE_END): 중립 행성 우선 확장.
  - 중반: 확장 + 내 행성 방어 (적 incoming 있으면 support).
  - 후반 (step >= ATTACK_PHASE_START): 약한 적 행성도 공격 후보.
  - source 함선의 SOURCE_BUDGET_FRAC 이상 한 번에 안 비움.
  - 같은 target 에 over-send (required × DUPLICATE_BUFFER 초과) 차단.

진입점:
  - agent(obs): Kaggle / 학습 league opponent / BC trajectory 생성 동일 시그니처.
  - GreedyExpandSupportBot: 향후 phase tracking 등 stateful 확장용 클래스.

게임 물리는 prediction.py 의 helper 들 재사용 (aim / first_collision_on_path /
project_target_at_eta / resolve_ships_for_capture / resolve_ships_for_support /
compute_support_required). 직접 재구현 안 함.
"""
from __future__ import annotations


# ── 튜너블 상수 ─────────────────────────────────────────────────────────

SOURCE_BUDGET_FRAC = 0.65   # src 가 1 step 에 비울 최대 비율
EXPAND_PHASE_END   = 50     # step <= 이건: 중립만 후보
ATTACK_PHASE_START = 200    # step >= 이거부터: 약한 적도 후보
DUPLICATE_BUFFER   = 1.2    # 같은 target sum_planned >= required × 이값 → skip
ETA_PENALTY        = 0.05   # score 에서 eta 가중치


# ── 진입점 ───────────────────────────────────────────────────────────────

def agent(obs) -> list:
    """Kaggle 진입점. obs dict → [[from_pid, angle, ships], ...]."""
    return GreedyExpandSupportBot().act(obs)


class GreedyExpandSupportBot:
    """1-step greedy 봇. 현재는 stateless — 인스턴스 재사용해도 동일 동작."""

    def act(self, obs) -> list:
        """obs → moves. 다음 단계에서 본체 채움."""
        return []
