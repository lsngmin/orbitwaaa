"""Reward 합성 패키지 (training reward 전용).

────────────────────────────────────────────────────────────────────────────
설계 원칙

1. Reward 는 action 가능 여부를 판단하지 않는다.
       Reward 는 이미 실행된 행동과 그 결과를 평가한다.
       행동을 열고 닫는 책임은 mask/decoder/action space 에 있다.
       금지: target valid 판단, path/sun 판단, support target 허용 여부 판단,
             ships_to_send 재계산, action 무효화.
       허용: 실행된 launch 결과 평가, capture/loss/hold/reloss 평가,
             over-send/all-in/launch cost 같은 비용 평가, terminal win/loss 평가.

2. 책임 분리.
       mask     : 가능한가?
       decoder  : 실제 move 로 어떻게 변환되는가?
       reward   : 그 결과가 좋은가?
       예: support 를 보낼 수 있는지 = mask/decoder, support 가 방어에 도움 됐는지 = reward.

3. Component 는 단일 책임이어야 한다.
       한 함수는 하나의 보상 의미만 계산한다.
       금지: battle_reward / strategic_bonus / mixed_capture_and_support_bonus
             / total_game_reward 같은 묶음.

4. Stateless component 가 기본.
       def component(ctx: RewardContext) -> float
       한 step 의 정보만으로 계산 가능한 reward 는 components.py 에 둔다.

5. Stateful reward 는 tracker / event 로 분리한다.
       여러 step 을 봐야 하는 reward (capture hold, post-capture reloss,
       support defense 등) 는 component 에 억지로 넣지 않고 별도 파일로 옮긴다:
           reward/events.py     # extract_<name>_events(ctx, tracker) -> events
           reward/trackers.py   # <Thing>Tracker.update(events)
       호출 흐름:
           events    = extract_reward_events(ctx, tracker)
           breakdown = compose_rewards(ctx, events)
           tracker.update(events)
       RewardContext 에 cross-step 임시 필드를 우겨넣지 않는다.

6. RewardContext 는 데이터 컨테이너다.
       reward 계산에 필요한 관측값 / 집계값 / coef 만 담는다.
       금지: mask 판정, decoder 판정, action 후보 생성, env 상태 변경,
             tracker 상태 변경, env 객체 전체 주입.
       env 전체를 넣으면 reward 가 env 내부를 뒤져 책임이 무너진다 — 필요한 값만 뽑는다.

7. RewardBreakdown 은 component 별 값을 보존한다.
       total 만 남기지 않는다. 이유:
           - 어떤 reward 가 policy 를 움직이는지 확인 가능
           - reward scale 폭주 감지 가능
           - 새 reward 추가 시 ablation 가능

8. COMPONENTS 튜플이 합성 순서의 source of truth.
       reward/components.py 에 모든 component 를 등록한다.
       train.py 는 RewardContext 만 만들고 compose_rewards(ctx) 로 호출한다 —
       개별 component 직접 호출 금지.

9. Reward scale 은 기존 component 와 비교 가능한 크기로 둔다.
       새 reward 가 한 번에 ±0.1 이상이면 PPO 를 지배할 가능성이 크다.
       처음에는 작은 계수로 시작하고 mean magnitude 를 비교한다.

10. Reward 는 한 번에 하나씩 추가한다.
        동시에 여러 reward 를 추가하면 원인 분리가 불가능하다.

11. Reward 변경은 반드시 기대 지표와 연결되어야 한다.
        새 reward 추가 시 기대 지표를 PR 설명에 명시한다.

12. Reward 는 action space 부족을 대신 해결하면 안 된다.
        표현할 수 없는 행동은 reward 로 학습시킬 수 없다.
        먼저 action space/mask/decoder 가 행동을 표현하고, 그 다음 reward 가 평가한다.

────────────────────────────────────────────────────────────────────────────
파일 구성

reward/
    __init__.py        # RewardContext, RewardBreakdown, 규약 (이 파일)
    components.py      # stateless components + COMPONENTS 튜플 + compose_rewards
    events.py          # stateful reward 도입 시 추가
    trackers.py        # stateful reward 도입 시 추가

────────────────────────────────────────────────────────────────────────────
함수 / 변수 네이밍

함수
    <대상>_<종류>:
        *_penalty, *_bonus
        의미가 명확할 때만 *_reward (dense_reward, terminal_reward)
    합성:    compose_rewards
    이벤트:  extract_<event>_events
    트래커:  <Thing>Tracker

좋은 이름:
    dense_reward, terminal_reward, neutral_capture_bonus,
    early_neutral_capture_bonus, capture_hold_bonus, post_capture_reloss_penalty,
    launch_cost_penalty, all_in_penalty, over_send_penalty,
    under_invested_penalty, support_defense_bonus, support_wasted_penalty

나쁜 이름:
    reward1, battle_reward, strategic_bonus, calc_reward, custom_reward,
    final_reward, good_action_reward, bad_action_penalty
        — 게임 의미가 안 보임.

변수
    component 결과값  : r_<name>           (예: r_dense, r_cap_bonus)
    계수             : <name>_coef, <name>_multiplier
    누적합            : <name>_sum
    카운트            : <name>_count
    평균              : mean_<name>
    비율              : <name>_rate
    턴 기준           : <name>_turn(s) / <name>_turn_norm
    K턴 윈도우        : <name>_k

────────────────────────────────────────────────────────────────────────────
RewardContext 필드 규약

허용 필드 (현재):
    state    prev_score, curr_score, prev_map, curr_obs, decode_counts,
             turn_norm, ep_done, raw_terminal, player
    coefs    dense_coef, all_in_penalty_coef, over_send_penalty_coef,
             under_invested_penalty_coef, launch_cost_penalty_coef,
             cap_gain_coef, cap_loss_coef, cap_early_multiplier,
             terminal_win_reward

피해야 할 필드:
    mask, target_logits, action_logits, policy, model, optimizer, env
    → reward 가 env 내부를 뒤져 책임이 무너진다. 필요한 값만 뽑아 ctx 에 담는다.

새 필드 추가 시:
    - 정말로 component 가 step-local 결과값으로만 쓰는지 확인.
    - cross-step state 가 필요하면 ctx 가 아니라 tracker (5번 원칙).

────────────────────────────────────────────────────────────────────────────
RewardBreakdown 필드 규약

CSV / log 이름과 최대한 일치시킨다 (utils/logger.py).
    breakdown.dense                  ↔ mean_dense_rew
    breakdown.cap_bonus              ↔ mean_cap_bonus
    breakdown.terminal               ↔ mean_terminal_rew
    breakdown.all_in_penalty         ↔ mean_all_in_penalty
    breakdown.over_send_penalty      ↔ mean_over_send_penalty
    breakdown.launch_cost_penalty    ↔ mean_launch_cost_penalty
    breakdown.under_invested_penalty ↔ mean_under_invested_penalty

새 component 추가 시 동기화 필수:
    1. components.py: 함수 정의 + COMPONENTS 튜플 등록
    2. compose_rewards: RewardBreakdown 의 새 필드 매핑
    3. RewardBreakdown: 새 필드 + .total 합산에 포함
    4. utils/logger.py: mean_<name> CSV 컬럼 추가
    5. train.py: sum_<name> 누적기 추가 + breakdown 에서 읽기

────────────────────────────────────────────────────────────────────────────
Component docstring 규약

```
def post_capture_reloss_penalty(ctx: RewardContext) -> float:
    \"\"\"방금 점령한 행성을 너무 빨리 잃는 행동을 벌한다.

    게임 의미:
        점령 직후 방어/지원 없이 다시 빼앗기면 나쁜 확장이다.
    적용:
        내가 점령한 행성이 K턴 안에 다시 적/중립으로 넘어간 경우.
    반환:
        음수 penalty.
    책임:
        결과 평가만. support 허용 여부나 move 생성은 판단하지 않는다.
    주의:
        (필요한 경우 — stateful 의존성, 특정 phase 한정 등)
    \"\"\"
```

기본 항목 (누락 시 N/A):
    게임 의미 / 적용 / 반환 / 책임 / 주의

"tensor 에서 무엇을 거른다" 가 아니라 "게임에서 무슨 상황을 평가한다" 관점.

────────────────────────────────────────────────────────────────────────────
금지 규칙 (component 함수 안에서)

1. mask 를 만들지 않는다.
2. decoder 결과를 바꾸지 않는다.
3. ships_to_send 를 재계산하지 않는다.
4. action 을 invalid 로 만들지 않는다.
5. env.step 을 호출하지 않는다.
6. global config T 를 직접 읽지 않는다 — coef 는 ctx 통해 주입한다.
7. logger 를 직접 호출하지 않는다.
8. tracker 상태를 몰래 수정하지 않는다 — 명시적 update 만.

트래커 업데이트가 필요하면 명시적으로:
    events    = extract_reward_events(ctx, tracker)
    breakdown = compose_rewards(ctx, events)
    tracker.update(events)

────────────────────────────────────────────────────────────────────────────
새 reward 추가 승인 기준

새 reward 를 넣기 전 아래 항목을 PR / 커밋 메시지에 명시:
    1. 이름
    2. 게임 의미
    3. 수식
    4. 필요한 입력 (ctx 필드 / tracker / events)
    5. stateless / stateful
    6. 기대 지표
    7. 예상 부작용
    8. 초기 계수
    9. 끄는 config flag

────────────────────────────────────────────────────────────────────────────
현재 components (reward/components.py, COMPONENTS 순서)

    dense_reward             # Δstate_score × dense_coef
    neutral_capture_bonus    # 중립 → 내것 (early_boost) / 내것 → 잃음
    all_in_penalty           # source 80%+ 비우는 발사
    over_send_penalty        # target 별 Σships - required 초과분
    under_invested_penalty   # src.ships < required 발사 시도
    launch_cost_penalty      # max(0, req/src - 0.5) 합계 (Phase B continuous)
    terminal_reward          # ±terminal_win_reward / 0

순서 자체는 reward 합산에 영향 없음 (덧셈 교환). 진단 / breakdown 출력 순서 용도.

────────────────────────────────────────────────────────────────────────────
최종 규칙 한 줄

Reward 는 "행동 가능성" 을 판단하지 않는다.
Reward 는 "실행된 행동의 결과" 만 평가한다.

이 규칙만 지키면 mask, decoder, reward 가 다시 섞이지 않는다.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewardContext:
    """Step 단위 reward 계산 입력. component 함수가 모두 이걸로 받는다.

    필드 분류:
      state     prev/curr_score, prev_map, curr_obs, decode_counts
                turn_norm, ep_done, raw_terminal, player
      coefs     dense_coef ∼ terminal_win_reward (config 에서 한 번 주입)

    필드 추가 / 금지 규칙은 모듈 docstring 의 "RewardContext 필드 규약" 참조.
    """
    # ── episode/step state ────────────────────────────────────────────────
    prev_score: float = 0.0
    curr_score: float = 0.0
    prev_map: dict = field(default_factory=dict)        # _snapshot_planet_owners 결과
    curr_obs: Any = None                                 # env.step() 이후 raw_obs
    decode_counts: dict = field(default_factory=dict)   # decode_action_to_moves 결과
    turn_norm: float = 0.0
    ep_done: bool = False
    raw_terminal: int = 0                                # env.state[player].reward, ep_done=True 일 때만 의미
    player: int = 0
    # ── coefs (training config 에서 주입, episode 내 불변) ──────────────────
    dense_coef: float = 0.0
    all_in_penalty_coef: float = 0.0
    over_send_penalty_coef: float = 0.0
    under_invested_penalty_coef: float = 0.0
    launch_cost_penalty_coef: float = 0.0
    cap_gain_coef: float = 0.0
    cap_loss_coef: float = 0.0
    cap_early_multiplier: float = 1.0
    terminal_win_reward: float = 1.0


@dataclass
class RewardBreakdown:
    """Step 단위 reward component 분해. .total 이 PPO 가 받는 단일 스칼라.

    필드명은 component 함수명을 의미적으로 축약 (dense_reward → dense,
    neutral_capture_bonus → cap_bonus, terminal_reward → terminal).

    CSV 컬럼과의 매핑 / 동기화 규약은 모듈 docstring 의
    "RewardBreakdown 필드 규약" 참조.
    """
    dense: float = 0.0
    cap_bonus: float = 0.0
    all_in_penalty: float = 0.0
    over_send_penalty: float = 0.0
    under_invested_penalty: float = 0.0
    launch_cost_penalty: float = 0.0
    terminal: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.dense
            + self.cap_bonus
            + self.all_in_penalty
            + self.over_send_penalty
            + self.under_invested_penalty
            + self.launch_cost_penalty
            + self.terminal
        )


from reward.components import (
    dense_reward,
    neutral_capture_bonus,
    all_in_penalty,
    over_send_penalty,
    under_invested_penalty,
    launch_cost_penalty,
    terminal_reward,
    COMPONENTS,
    COMPONENT_NAMES,
    compose_rewards,
)

__all__ = [
    "RewardContext",
    "RewardBreakdown",
    "dense_reward",
    "neutral_capture_bonus",
    "all_in_penalty",
    "over_send_penalty",
    "under_invested_penalty",
    "launch_cost_penalty",
    "terminal_reward",
    "COMPONENTS",
    "COMPONENT_NAMES",
    "compose_rewards",
]
