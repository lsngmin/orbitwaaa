"""Reward component 구현 (mask-style ctx-based 인터페이스).

규약:
    시그니처: def <name>(ctx: RewardContext) -> float
    반환:    step 단위 스칼라. penalty 는 음수 (또는 0), bonus 는 양수.
    원칙:    한 component 는 한 책임만. config / env / model 직접 참조 금지 — ctx 통해서만.

새 component 추가:
    1. 이 파일에 def <name>(ctx) -> float 작성
    2. COMPONENTS 튜플에 추가
    3. RewardBreakdown (reward/__init__.py) 에 동일 필드 추가
    4. compose_rewards 의 매핑에 한 줄 추가
    5. 필요 시 RewardContext 에 새 필드 추가 + train.py 에서 주입

참조 원본 라인 (분리 전 train.py)
    dense_reward            train.py:990 inline
    neutral_capture_bonus   train.py:181-218
    all_in_penalty          train.py:995 inline
    over_send_penalty       train.py:998 inline
    under_invested_penalty  train.py:1000 inline
    launch_cost_penalty     train.py:1003 inline
    terminal_reward         train.py:1009-1014 inline
"""
from __future__ import annotations

from reward import RewardContext, RewardBreakdown


# ── components ────────────────────────────────────────────────────────────────

def dense_reward(ctx: RewardContext) -> float:
    """Δstate_score 기반 dense reward. coef × (curr - prev)."""
    return ctx.dense_coef * (ctx.curr_score - ctx.prev_score)


def all_in_penalty(ctx: RewardContext) -> float:
    """source 80%+ 비우는 발사 수 × coef. 음수 반환."""
    return -ctx.all_in_penalty_coef * ctx.decode_counts.get("all_in_launches", 0)


def over_send_penalty(ctx: RewardContext) -> float:
    """target 별 Σships - required 의 양수 초과분 합 × coef. 음수 반환."""
    return -ctx.over_send_penalty_coef * ctx.decode_counts.get("over_send_excess_sum", 0)


def under_invested_penalty(ctx: RewardContext) -> float:
    """src.ships < required 발사 시도 수 × coef. 음수 반환."""
    return -ctx.under_invested_penalty_coef * ctx.decode_counts.get("under_invested_count", 0)


def launch_cost_penalty(ctx: RewardContext) -> float:
    """max(0, req/src - 0.5) 의 step 합계 × coef. 음수 반환."""
    return -ctx.launch_cost_penalty_coef * ctx.decode_counts.get("launch_cost_excess_sum", 0.0)


def terminal_reward(ctx: RewardContext) -> float:
    """에피소드 종료 시 ±terminal_win_reward / 0. 진행 중이면 항상 0.

    raw_terminal 은 env.state[player].reward (∈ {-1, 0, 1}) 그대로.
    """
    if not ctx.ep_done:
        return 0.0
    if ctx.raw_terminal == 1:
        return ctx.terminal_win_reward
    if ctx.raw_terminal == -1:
        return -ctx.terminal_win_reward
    return 0.0


def neutral_capture_bonus(ctx: RewardContext) -> float:
    """중립 행성 점령 보너스 (step-level 누적).

    prev_map: env.step() 전 {pid: (owner, prod)} 스냅샷.
    curr_obs: env.step() 이후 raw observation.

    early_boost (multiplicative, neutral GAIN 한정):
      cap_early_multiplier ≥ 1.0 일 때만 의미 있음.
      early_boost = 1.0 + (mult - 1.0) × max(0, 1 - turn_norm)
      → turn_norm=0 (초반) 일 때 multiplier 그대로, turn_norm=1 (말기) 일 때 1.0.
      base bonus 를 대체하지 않고 곱해서 amplify 만 함.
      enemy capture / own loss 는 boost 안 적용 — 초반 중립 race 만 가속.
    """
    tn = float(max(0.0, min(1.0, ctx.turn_norm)))
    early_boost = 1.0 + (ctx.cap_early_multiplier - 1.0) * max(0.0, 1.0 - tn)

    if isinstance(ctx.curr_obs, dict):
        curr_planets = ctx.curr_obs.get("planets", [])
    elif ctx.curr_obs is None:
        curr_planets = []
    else:
        curr_planets = getattr(ctx.curr_obs, "planets", [])

    bonus = 0.0
    for p in curr_planets:
        pid   = p[0] if isinstance(p, (list, tuple)) else p.id
        owner = p[1] if isinstance(p, (list, tuple)) else p.owner
        prev_owner, prod = ctx.prev_map.get(pid, (-1, 0))
        if prev_owner == -1 and owner == ctx.player:        # 중립 → 내 것
            bonus += prod * ctx.cap_gain_coef * early_boost
        elif prev_owner == ctx.player and owner != ctx.player:  # 내 것 → 잃음
            bonus -= prod * ctx.cap_loss_coef
    return bonus


# ── orchestration ─────────────────────────────────────────────────────────────

COMPONENTS = (
    dense_reward,
    neutral_capture_bonus,
    all_in_penalty,
    over_send_penalty,
    under_invested_penalty,
    launch_cost_penalty,
    terminal_reward,
)
COMPONENT_NAMES = tuple(c.__name__ for c in COMPONENTS)


def compose_rewards(ctx: RewardContext) -> RewardBreakdown:
    """모든 component 를 호출해서 RewardBreakdown 으로 묶는다.

    필드명 매핑 (component 함수명 → breakdown 필드):
      dense_reward            → dense
      neutral_capture_bonus   → cap_bonus
      terminal_reward         → terminal
      나머지는 동일 이름.
    """
    return RewardBreakdown(
        dense=dense_reward(ctx),
        cap_bonus=neutral_capture_bonus(ctx),
        all_in_penalty=all_in_penalty(ctx),
        over_send_penalty=over_send_penalty(ctx),
        under_invested_penalty=under_invested_penalty(ctx),
        launch_cost_penalty=launch_cost_penalty(ctx),
        terminal=terminal_reward(ctx),
    )
