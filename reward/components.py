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
    dense_reward             train.py:990 inline
    neutral_capture_bonus    train.py:181-218 (gain branch — 중립→내것)
    own_planet_loss_penalty  train.py:181-218 (loss branch — 내것→잃음)
    all_in_penalty           train.py:995 inline
    over_send_penalty        train.py:998 inline
    under_invested_penalty   train.py:1000 inline
    launch_cost_penalty      train.py:1003 inline
    terminal_reward          train.py:1009-1014 inline
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


def _curr_planets(curr_obs):
    """curr_obs (dict 또는 attr 객체 또는 None) → planets 시퀀스 추출."""
    if isinstance(curr_obs, dict):
        return curr_obs.get("planets", [])
    if curr_obs is None:
        return []
    return getattr(curr_obs, "planets", [])


def neutral_capture_bonus(ctx: RewardContext) -> float:
    """중립 → 내 것 점령 보너스 (gain only, step-level 누적).

    게임 의미:
        초반 중립 행성 확장을 강화. own loss 는 own_planet_loss_penalty 가 처리.
    적용:
        prev_owner == -1 AND curr_owner == player 인 행성에만.
        bonus = prod × cap_gain_coef × early_boost.
    반환:
        양수 (또는 0).
    책임:
        결과 평가만. 어떤 target 이 점령 가능한지는 mask/decoder 판단.
    주의:
        early_boost (multiplicative) 는 이 함수에만 적용 — own loss 와 비대칭.
        early_boost = 1.0 + (cap_early_multiplier - 1.0) × max(0, 1 - turn_norm)
        → turn_norm=0 초반엔 multiplier 그대로, turn_norm=1 말기엔 1.0 (off).
    """
    tn = float(max(0.0, min(1.0, ctx.turn_norm)))
    early_boost = 1.0 + (ctx.cap_early_multiplier - 1.0) * max(0.0, 1.0 - tn)

    bonus = 0.0
    for p in _curr_planets(ctx.curr_obs):
        pid   = p[0] if isinstance(p, (list, tuple)) else p.id
        owner = p[1] if isinstance(p, (list, tuple)) else p.owner
        prev_owner, prod = ctx.prev_map.get(pid, (-1, 0))
        if prev_owner == -1 and owner == ctx.player:        # 중립 → 내 것
            bonus += prod * ctx.cap_gain_coef * early_boost
    return bonus


def own_planet_loss_penalty(ctx: RewardContext) -> float:
    """내 행성 → 적/중립 전환 페널티 (loss only, step-level 누적).

    게임 의미:
        점령지/기존 행성을 잃는 행동을 벌한다. neutral capture gain 과 비대칭.
    적용:
        prev_owner == player AND curr_owner != player 인 행성에만.
        penalty = -prod × cap_loss_coef.
    반환:
        음수 (또는 0).
    책임:
        결과 평가만. 방어/지원 가능 여부는 mask/decoder 판단.
    주의:
        early_boost 적용 안 함 — phase 무관 일정 페널티 (capture gain 과 비대칭).
    """
    penalty = 0.0
    for p in _curr_planets(ctx.curr_obs):
        pid   = p[0] if isinstance(p, (list, tuple)) else p.id
        owner = p[1] if isinstance(p, (list, tuple)) else p.owner
        prev_owner, prod = ctx.prev_map.get(pid, (-1, 0))
        if prev_owner == ctx.player and owner != ctx.player:  # 내 것 → 잃음
            penalty -= prod * ctx.cap_loss_coef
    return penalty


def early_close_neutral_capture_bonus(ctx: RewardContext) -> float:
    """초반에 가깝고 저비용인 중립 점령에 추가 보너스 (capture_events 소비, stateless).

    게임 의미:
        "초반 중립 race 에서 가까운 저비용 중립을 선점" 행동을 강화. 기존
        neutral_capture_bonus (모든 중립 capture 일률) 위에 "초반 + 가까움 + 저비용"
        조건을 모두 만족할 때만 추가 가산 — 효율적 확장 패턴 amplify.
    적용:
        ctx.capture_events 의 (-1 → player) capture 중, linked_launches 안에
        아래 3 조건 모두 만족하는 launch 가 1개라도 있으면 가산:
            - launch.turn_norm_at_launch < ctx.early_close_threshold_turn_norm
            - 0 < launch.nearest_neutral_rank ≤ ctx.early_close_max_nearest_rank
            - launch.req_over_src ≤ ctx.early_close_max_req_over_src
        가산량 = ctx.early_close_neutral_capture_bonus_coef × event.target_prod.
        (target_prod 가중 — 작은 행성보다 큰 행성 점령에 더 큰 가산.)
    반환:
        양수 (또는 0). qualifying capture 마다 합산.
    책임:
        결과 평가만. nearest_rank / req_over_src / turn_norm 자체는 decoder/analyze
        가 launch 시점에 계산해 LaunchMetadata 에 저장 — reward 가 다시 계산 안 함.
    주의:
        ctx.capture_events 가 비었거나 linked_launches 가 비면 attribution 불가 → 0.
        coef=0 이면 항상 0 (kill switch). C 단계 도입 후 default 0.0 으로 시작.
        nearest_neutral_rank == 0 은 "계산 안 됨" sentinel (decoder fallback) — 가산 안 함.
    """
    if not ctx.capture_events:
        return 0.0
    coef = ctx.early_close_neutral_capture_bonus_coef
    if coef == 0.0:
        return 0.0

    threshold_tn = ctx.early_close_threshold_turn_norm
    max_rank     = ctx.early_close_max_nearest_rank
    max_cost     = ctx.early_close_max_req_over_src

    bonus = 0.0
    for ev in ctx.capture_events:
        if ev.prev_owner != -1 or ev.new_owner != ctx.player:
            continue  # 중립 → 내것 만 대상
        for launch in ev.linked_launches:
            if (0 < launch.nearest_neutral_rank <= max_rank
                    and launch.req_over_src <= max_cost
                    and launch.turn_norm_at_launch < threshold_tn):
                bonus += coef * ev.target_prod
                break  # 같은 event 에서 다중 launch 매칭해도 1번만 가산
    return bonus


# ── orchestration ─────────────────────────────────────────────────────────────

COMPONENTS = (
    dense_reward,
    neutral_capture_bonus,
    own_planet_loss_penalty,
    early_close_neutral_capture_bonus,
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
      terminal_reward         → terminal
      나머지는 동일 이름 (neutral_capture_bonus, own_planet_loss_penalty,
                       early_close_neutral_capture_bonus, all_in_penalty,
                       over_send_penalty, under_invested_penalty,
                       launch_cost_penalty).
    """
    return RewardBreakdown(
        dense=dense_reward(ctx),
        neutral_capture_bonus=neutral_capture_bonus(ctx),
        own_planet_loss_penalty=own_planet_loss_penalty(ctx),
        early_close_neutral_capture_bonus=early_close_neutral_capture_bonus(ctx),
        all_in_penalty=all_in_penalty(ctx),
        over_send_penalty=over_send_penalty(ctx),
        under_invested_penalty=under_invested_penalty(ctx),
        launch_cost_penalty=launch_cost_penalty(ctx),
        terminal=terminal_reward(ctx),
    )
