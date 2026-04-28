# orbitwaaa — 강화학습 시스템 리팩토링 설계 (장기 청사진)

작성일: 2026-04-28
범위: 단기 구현 가능성·현 코드와의 호환성 무시. **표현력·일반화·확장성이 충분한 시스템의 완전한 설계**를 기록한다. 우선순위와 단계는 12장에만 모아둔다.

---

## 1. 문제 정의

현재 시스템이 부딪히는 천장은 두 개의 본질적 한계에서 온다.

1. **정책 표면이 좁다.**
   "초반 중립 먹기 / 점령 유지 / 지원 / 낭비 억제" 정도의 최소 생존 정책만 표현 가능하며, 상위권 봇이 보이는 *지역 네트워크 형성, 연합 공격, 거부(denial), 템포(tempo), king-maker* 같은 정책 축은 mask·feature·reward·league 어느 층에서도 명시적으로 다뤄지지 않는다.

2. **시간 표현과 멀티플레이어 표현이 약하다.**
   planet/fleet 을 *time-stamped 점*으로만 다룬다. 게임은 본질적으로 *궤적*이다. 또한 4인전은 2인전을 단순 확장한 것이 아니라 **별개의 게임 (general-sum, ranking-based, king-maker)** 인데, 같은 모델 가정으로 학습하고 있다.

이 문서는 위 두 천장을 모두 부수기 위한 시스템 전체 청사진이다.

---

## 2. 게임의 본질 — 자주 누락되는 사실

### 2.1 2인전과 4인전은 같은 게임이 아니다

```text
2인전 (zero-sum):
  내 손해 = 상대 이득. 정확히 zero-sum.
  의사결정: 나 vs 상대. 1차원.
  균형: minimax / Nash 가 잘 정의됨.
  reward: terminal +1/-1, 중간은 (나 - 상대) advantage 안전.
  정책 표면: aggressive ↔ defensive 의 명확한 trade-off.

4인전 (general-sum, semi-cooperative):
  내 손해 ≠ 다른 한 명의 이득. 나머지 두 명에게도 영향.
  "1등 가르기" 가 핵심. 2등이 1등을 끌어내리는 게 게임 이론적으로 정당.
  king-maker problem: 4등이 누구를 공격하느냐가 우승자를 결정.
  Nash 가 여럿, 메타게임 의존.
  reward: ranking 기반 ([+3,+1,-1,-3] 또는 1st-only).
  중간 보상의 baseline 을 "어느 상대로" 잡을지 자체가 비자명.
  정책 표면: "누구를 적으로 삼느냐" 가 1차 결정 변수.
```

이 차이는 단순 환경 변경이 아니라 **모델 구조·reward·league 모두 player-count-aware** 여야 함을 뜻한다.

### 2.2 게임 룰에서 파생되는 숨은 메커니즘

기존 정책 8축이 명시하지 않는 룰 디테일.

#### 2.2.1 Production-time 회계 (opportunity cost)
모든 ship 은 발사 순간 source 의 production-time 을 *선결제*한 자원이다.
```
source_opportunity_cost = ships_sent × src_prod_rate × ETA × src_threat_index
```
쌀수록 좋은 send (안전한 src 에서 가까운 target 으로). 비쌀수록 나쁜 send (위협받는 src 를 비워서 먼 target 으로).

#### 2.2.2 도착 동시성 (alpha strike)
같은 turn 에 여러 source 에서 도착하는 fleet 은 stack 되어 한 번에 적용. 분산 launch 보다 동시 도착이 강하다 (defense 가 사이에 끼어들 틈이 없음). 현재 sequential decoder 는 한 턴에 한 source 만 launch 한다고 가정 → alpha strike 표현력 부족.

#### 2.2.3 태양 / 경로 비대칭성
sun 이 있으면 src→tgt 와 tgt→src 의 reachable 여부가 다를 수 있다.
- 어떤 target 은 "내가 칠 수는 있지만 enemy 가 다시 못 치는" *safe 자산*.
- 반대로 "내가 가도 enemy 의 다른 source 가 더 빠른" *trap 자산*.
```
target_strategic_safety = min over enemy.src of ETA(enemy.src → target)
```

#### 2.2.4 점령 직후 vacuum
방금 capture 된 행성은 ship ≈ 0 → enemy reloss 표적 1순위. 능동 reinforcement (support) 가 필요.

#### 2.2.5 도착 시점 ownership 변화
fleet 이 비행 중 target 의 owner 가 바뀌면 fleet 의 의미가 변한다. 적 행성 → 도착 직전 아군화 → friendly stack 또는 낭비. 이 race condition 을 모델이 *예상*해야 한다.

---

## 3. 정책 지도 (16축)

기존 8축 + 2인전 전용 4축 + 4인전 전용 4축. **정책은 16개로 보지만, reward component 로는 절대 16개를 만들지 않는다.** 책임은 4장에서 분리.

### 3.1 기본 8축
```
1. 초반 확장              경제·공간
2. 지역 네트워크 형성     공간·구조
3. 병력 보존 / source 안정성   비용
4. 방어 / 지원            방어·비용
5. 점령 유지              경제·시간
6. 공격 / 연합 공격       공격·조정
7. 위협 회피 / 경로 안정성  비용·공간
8. 장기 경제 우위         경제·시간
```

### 3.2 2인전에서 빠져있는 4축
```
9.  템포 (initiative) — 누가 행동할 차례인가, 상대를 reactive 모드로 묶기
10. Race / arrival timing — 도착 우위 활용 (mask 가 아닌 reward 에서 race 판단)
11. 거부 (denial / spoiler) — 내가 못 먹어도 적도 못 먹게
12. 상대 모델링 (opponent prediction) — aux head 로 representation 강화
```

### 3.3 4인전 전용 4축
```
13. 위협 우선순위 (threat ranking) — 매 턴 누가 1등인지 동적 판단
14. King-maker 회피/활용 — 4등의 표적이 우승자를 결정
15. 삼자 견제 (triangulation) — 두 상대가 서로 싸우게 만들기
16. 랭킹 보존 (rank preservation) — 1등 못 할 거 같으면 2등 굳히기
```

---

## 4. 책임 분리 — 어느 층에서 어느 정책을 다루는가

각 정책은 보통 **여러 층의 협업**으로 구현된다. 한 곳에 다 박지 않는다.

| 정책 | mask/action | feature/attn | reward | league/eval |
|---|---|---|---|---|
| 1 초반 확장 |  | nearest_rank, ETA, req/src, prod | early_close_neutral_capture_bonus | rule-bot: greedy_radius |
| 2 지역 네트워크 |  | local ally/neutral density, neighbor graph, local graph attention |  | (간접 hold 결과로) |
| 3 source 안정성 | hard mask 최소 | req/src, remaining/source, incoming_threat_to_src | weak all_in_penalty, launch_cost_penalty |  |
| 4 방어/지원 | support mode 열기 | enemy incoming map | support_defense_bonus, own_loss_penalty |  |
| 5 점령 유지 |  | captured_at_turn, age | capture_hold_bonus, post_capture_reloss_penalty |  |
| 6 연합 공격 | multi-source decoder, Q-style joint | same-target linked launches, target deficit | synchronized_arrival_capture_bonus, weak over_send_penalty |  |
| 7 위협 회피 | sun/path hard invalid | ETA advantage, enemy arrival race | weak filtered/no-op penalty |  |
| 8 장기 경제 |  | production share, planet share, ship share | dense advantage, terminal win |  |
| 9 템포 |  | enemy_idle_ratio, my_pressure_index | (메트릭만, 또는 episodic pressure_sustained) |  |
| 10 race |  | eta_lead_vs_nearest_enemy, enemy_reachable_force_at_my_eta, target_owner_at_eta | race_won_capture, weak race_lost_attempt |  |
| 11 denial | enemy-incoming neutral 열기 | target_enemy_incoming_strength, target_capture_imminence | denial_bonus |  |
| 12 opponent modeling |  | shared encoder | aux head: predict_opponent_next_target |  |
| 13 threat ranking |  | per_opponent_share, per_opponent_threat_to_me, per_opponent_rank, my_rank | (feature 만) |  |
| 14 king-maker |  | per_opponent_share | rank_progress_bonus, rank_decay_penalty | role-asymmetric 4p match |
| 15 triangulation |  | opponent_pred (12) | (간접) | role-asymmetric pool |
| 16 rank 보존 |  | my_rank | terminal ranking weights | curriculum (1st-only ↔ ordinal) |

**핵심 원칙**:
- *mask 는 hard invalid 만* (sun/path/물리). 의사결정 가능한 행동은 막지 않는다.
- *reward 는 결과 평가만*. "행동 가능성" 판단 금지.
- *feature 는 판단 재료*. 정책 학습 자체는 head 가 한다.
- *league 는 일반화*. 같은 reward 라도 상대 분포가 좁으면 corner 정책에 갇힌다.

---

## 5. State 표현 — Fleet/Planet 의 시간축 통합

이 장이 사용자 강조점의 핵심이다. 현재 모델이 약한 가장 큰 이유는 fleet/planet 을 *시점 t의 점*으로만 보기 때문이다. 게임은 *궤적*이다.

### 5.1 Per-entity encoder

```text
planet_p:
  identity:    pos(x, y), prod_rate, max_ships, sun_distance
  state(t):    owner_onehot{neutral, me, opp1, opp2, opp3}, ships, age_since_capture
  derived:     production_at_capture_minus_now (capture 후 누적 생산)

fleet_f:
  identity:    src_idx, tgt_idx, launch_turn, eta, total_distance
  state(t):    owner_onehot, ships, alpha = (t - launch)/(eta - launch) ∈ [0,1]
  derived:     remaining_turns = eta - t, velocity_vector
```

owner_onehot 은 player_count 에 따라 길이가 다르다. **고정 5칸 (neutral + 최대 4 player) 으로 잡고 unused 슬롯은 0 mask** 가 가장 단순하다.

### 5.2 Cross-entity attention

planet 과 fleet 은 다음 관계를 명시적으로 attend 한다.

```text
planet ← fleet (arrival incoming):
  각 planet 이 자기에게 도착 예정인 fleet 들에 attend.
  → planet 의 vector 가 "미래 arrival schedule" 을 압축해서 가짐.

fleet ← fleet (same-target):
  같은 target 으로 가는 fleet 끼리 attend.
  → race / alpha-strike 패턴 인지.

fleet ← fleet (same-source):
  같은 source 에서 출발한 fleet 끼리 attend.
  → source depletion / coordinated launch 인지.

planet ← planet (k-NN spatial):
  공간적으로 가까운 planet 끼리 attend (local cluster).
  → 정책 2 (지역 네트워크) 의 토대.

planet ← planet (same-owner):
  같은 owner 의 planet 끼리 attend.
  → 전체 세력 형태 인지.
```

graph attention 의 edge 종류를 위처럼 분리해 multi-head/multi-relation 으로 짠다.

### 5.3 Arrival schedule 내장 feature

각 planet 은 다음 K 턴 (K=8 권장) 의 *결정론적 시뮬* 결과를 가진다.

```text
for each planet p, for each t in [now, now+K]:
  incoming_arrivals[t] = list of (owner, ships) 도착 예정
  expected_owner[t]    = simulate(arrivals, current_owner, prod_rate)
  expected_ships[t]    = "
  expected_threat[t]   = sum(enemy_ships arriving at t)
```

이걸 planet feature 에 concat 하면 모델이 *"이 target 은 turn+3 에 capture 확정. 추가로 보낼 필요 없음"* 같은 패턴을 학습 데이터 효율성 좋게 잡는다.

또한 **arrival bucket** 표현:
```text
arrival_bucket[target][bucket_t] = {
  my_ships_at_t,
  per_opponent_ships_at_t,
  net_ownership_change_at_t,
  capture_probability_at_t
}
bucket size = 1 turn, horizon = 8 turns.
```

### 5.4 Future rollout (deterministic, no-grad)

"지금 launch 안 하면 K턴 뒤 각 planet 의 owner/ships 가 어떨까" 를 **gradient 안 흘리고** 시뮬해 추가 feature 로 concat. 이것은 internal world model 의 가장 단순한 형태 (fixed dynamics, no learning).

이 시뮬에 자기 자신의 launch 후보 일부를 적용해보는 *imagined branch* 까지 가면 MCTS 의 1-step lookahead 와 비슷. 단계적 도입.

### 5.5 Player-count embedding

```text
입력에 player_count {2, 4} onehot 주입.
encoder weight 는 공유, 마지막 head 만 분기 (head_2p, head_4p).
또는 FiLM-style conditioning 으로 encoder 일부 layer 만 변조.
```

이게 있어야 같은 표현 학습이 두 모드의 의사결정 차이로 갈라진다.

### 5.6 Per-opponent encoder (4인전 전용)

4인전에선 **opponent identity 가 의미를 가진다**. opp1/opp2/opp3 를 구별해야 king-maker, threat ranking 정책이 동작.

```text
per_opponent_features:
  - production_share, planet_share, ship_share
  - threat_to_me (incoming fleet 합)
  - rank, my_rank
  - recent_target_pattern (지난 N턴에 누구를 친 분포)
opponent attention pool:
  3개 opponent feature → 별도 attention head → "지금 가장 위협적인 상대" softmax
```

---

## 6. Action space 설계

### 6.1 현재 가정
- **source 별 독립 launch decision** (sequential decoder, source 마다 자신의 (target, amount) 를 독립 sampling).
- 여러 source 가 우연히 같은 target 을 고를 수는 있으나, 같은 target 에 대한 **synchronized multi-source set 을 하나의 joint action** 으로 평가하지는 못한다 (이 가정 깨는 게 6.2.1 의 목표).
- (src, tgt, amount) tuple per source.
- target type: enemy / neutral / own (support).

### 6.2 확장
1. **Multi-source coordinated decoder**
   같은 target 에 대해 여러 src 가 한 번에 결정. autoregressive over source set.
   ```
   π(a | s) = ∏_i π(src_i, frac_i | tgt, prev_srcs)
   ```
   같은 turn 에 동일 target 을 여러 source 에서 치는 alpha strike 직접 표현.

2. **Q-style joint candidate evaluation**
   상위 후보 launch 조합을 enumerate 한 후 Q(s, a_set) 로 평가.
   비용 ↑ 지만 4인전·연합 공격에서 결정적.

3. **Macro / option action**
   ```
   "행성 P 를 hold 하라" (M턴짜리 macro)
   "이 target 을 K턴에 같이 도착시켜 alpha strike" (synchronized macro)
   ```
   option-critic 또는 hierarchical RL.

4. **Partial launch (fraction-aware)**
   현재 amount 가 카테고리라면, 연속 fraction (0,1) head 를 둠. denial / spoiler 의 minimum-ship send 정밀도 향상.

### 6.3 mask 정책
**장기 원칙**: mask 는 hard invalid (sun/path/owner physical impossibility) 만. 의사결정 가능한 모든 행동은 reward/feature 로 처리.

**Support mask 의 철학** (현 구현·plan 모두 정확히 다음 의미):
- *action 자체* (target_type=support) 는 항상 존재. policy head 에서 support 결정 자체는 막히지 않음.
- *target 선택* 은 필요성 기반 guard 가 붙는다 (현 구현: enemy_incoming > 0 OR required > 0, 35% cap). 이건 hard invalid 가 아니라 **action-space guard** — "본진 비우기·support 난사·source depletion" 방지.
- 즉 "own planet support 는 항상 열림" 이라는 표현은 정확하지 않다. 정확한 표현: **support action 은 모드로서 항상 존재하지만, target 은 방어 필요성 신호가 있을 때만 열린다.**

장기적으로 reward (own_planet_loss_penalty, support_defense_bonus) 가 충분히 학습되면 guard 강도를 낮추는 게 정공법. 단 가드 0 으로 가지 않고 (support 난사 risk), reward 신호가 axis metric 으로 검증된 후 단계적 완화.

---

## 7. Reward 시스템

### 7.1 2인전 reward

```text
terminal:
  +1 win, -1 loss, 0 draw

dense (per-step):
  Δ(my_share - opp_share) advantage     # 안전한 zero-sum
  capture / loss events
  + component bonus/penalty (아래 7.3)
```

### 7.2 4인전 reward

```text
terminal (curriculum):
  Phase A: 1st-only ([+1, 0, 0, 0])           # 공격 학습
  Phase B: ordinal ([+3, +1, -1, -3])         # ranking 보존 학습
  Phase C: mix or 1st-only (결정적 행동 회복)

dense:
  baseline 두 가지 weighted:
    Δ(my_share - mean(opp_shares))    # 평균 baseline (안정)
    Δ(my_share - max(opp_shares))     # top 상대 baseline (king-maker 유도)
  rank_progress reward (정책 16):
    +α 내 rank 가 좋아질 때
    -α 내 rank 가 나빠질 때
  component bonus/penalty (7.3 동일)
```

advantage normalization 은 **mode 별 별도 running stats** (2p / 4p episode 분산이 매우 다름).

### 7.3 Reward component 카탈로그

상태 분류:
- **[현]** 현재 `reward/components.py` 의 COMPONENTS 튜플에 실재.
- **[단]** 단기 후보 (Phase 2 예정, plan 8 component 직후 다음 wave).
- **[장]** 장기 후보 (Phase 4·5 이후).
- **[메트릭]** reward 화 금지 또는 매우 약하게만, logger 컬럼만.
- **[aux]** RL reward 아닌 supervised loss.

```text
[현] 현재 COMPONENTS 튜플 (총 9개, 2026-04-28 기준)
  ── stateless (8개) — components.py 한 함수당 한 step 입력만 사용
  dense_reward                           정책 8 (production share advantage)
  neutral_capture_bonus                  정책 1 (gain — neutral → me)
  own_planet_loss_penalty                정책 4 (loss — me → other)
  all_in_penalty (weak)                  정책 3
  over_send_penalty (weak)               정책 3 / 6
  under_invested_penalty (weak)          정책 3
  launch_cost_penalty (weak)             정책 3
  terminal_reward                        승리

  ── event-consuming (1개) — ctx.capture_events 소비, trackers 가 launch→capture 매칭
  early_close_neutral_capture_bonus      정책 1 (early 0.25 + nearest_rank ≤ 1
                                          + req/src ≤ 0.5 자격 launch 가 capture 로
                                          이어졌을 때 prod-가중 보너스, coef 0.007 활성)

[단] 단기 후보 (Phase 2 — 현재 병목과 직결, 모두 event-consuming 으로 구현 권장)
  post_capture_reloss_penalty            정책 5 (own_loss 의 "갓 잡은 행성" 가중판
                                          — capture_event 의 age 기반)
  capture_hold_bonus                     정책 5 (capture 후 K턴 hold 성공)
  support_defense_bonus                  정책 4 (support launch 가 실제 enemy
                                          incoming 을 막은 결과 — support→defense 매칭 필요)
  reinforce_young_planet_bonus           정책 4 / B4 vacuum 대응

[장] 장기 후보 (Phase 4·5 이후)
  race_won_capture_bonus                 정책 10
  weak race_lost_attempt_penalty         정책 10 (균형)
  denial_bonus                           정책 11
  synchronized_arrival_capture_bonus     정책 6 / B2 alpha
  coordinated_capture_bonus              정책 6 (multi-src 활성 후)
  rank_progress_bonus                    정책 16 (4p 활성 후)
  rank_decay_penalty                     정책 16

[메트릭] reward 아님 (logger 만)
  source_opportunity_cost                정책 3 보조
  pressure_index                         정책 9
  enemy_idle_ratio                       정책 9
  filtered_path_rate                     정책 7 (※ reward 화 신중)
  sun_crash_rate                         정책 7 (※ reward 화 신중)

[aux] RL reward 아닌 supervised loss
  opponent_pred_ce                       정책 12 (encoder 표현 강화, weight 0.1~0.3)
```

**weak filtered_path / weak race_lost_attempt 의 위치**: CLAUDE.md "reward 는 행동 가능성을 판단하지 않는다" 와 충돌 risk 가 있어 위 카탈로그에서 *기본은 메트릭*. 메트릭이 baseline 대비 유의 악화할 때만 reward 화 검토 (도입 단계는 15.2.7 참조).

가중치 원칙:
- terminal 을 **신호의 backbone** 으로. dense 는 항상 terminal 보다 작게.
- 새 component 추가 시 기존과의 *cross-correlation* 측정. 0.7 이상이면 redundant.
- weak penalty 는 모두 한 자리수 비율로만 (모델이 "send 자체를 회피" 하지 않게).

---

## 8. Architecture 청사진

### 8.1 전체 구조

```text
INPUTS
├── planet_features (N_p, F_p)
├── fleet_features  (N_f, F_f)
├── arrival_schedule (N_p, K, F_a)
├── per_opponent_features (3, F_o)        # 4인전만, 2인전은 zero-pad
└── player_count_onehot (2,)

ENCODER (shared across modes)
├── per-entity MLP
├── multi-relation graph attention (planet/fleet, same-tgt, same-src, k-NN, same-owner)
├── temporal cross-attention (planet ↔ arrival_schedule)
└── opponent attention pool                # 4인전만 활성

HEADS
├── policy_head_2p (target categorical, source categorical, amount)
├── policy_head_4p (위 + per-opponent attention 사용)
├── value_head_2p
├── value_head_4p
└── aux_opponent_pred_head (정책 12)

MODE GATE
└── player_count_onehot → head selector (FiLM 또는 직접 분기)
```

### 8.2 Multi-source coordinated decoder

```text
target 결정 → autoregressive over (src_i, frac_i)
  state ← state + selected (src_i, frac_i)
  stop token 으로 종료
training: teacher-forcing on collected trajectories
inference: beam over top-k coordinated sets
```

연합 공격 직접 표현. Q-style joint 와 비교해 implementation 단순.

### 8.3 Opponent prediction aux head

```text
shared encoder output → small MLP → softmax over opponent's next target
loss: cross-entropy with realized opponent action (per opponent in 4p)
weight: 0.1 ~ 0.3 (메인 RL loss 보다 작게)
```

학습 비용 거의 안 들고 representation 이 *어디가 위협인가* 를 인코딩하게 됨.

### 8.4 Parameter sharing 정책

```text
shared:
  encoder, opponent_pred_head
mode-specific:
  policy_head, value_head, advantage normalizer
4-player-only:
  per_opponent_attention, rank head feature
```

처음에는 encoder 만 공유, 나중에 충분히 안정되면 layer norm 까지 공유 확장.

---

## 9. League / Self-play 구조

### 9.1 2인 pool 구성

```text
- self snapshots (4 ~ 8개, 다양한 학습 단계)
- sibling reward seed 1 ~ 2개 (다른 weight 로 학습된 자기)
- rule-bot 4종:
    greedy_radius_K            (가까운 neutral rush)
    turtle_threshold_X         (수비 + 고생산 우선)
    rush_target_nearest        (early all-in)
    sync_attacker              (multi-source 동시 도착)
- 외부 supplied bot (대회/리더보드 sample) — 가능하면
샘플링: 70% snapshot, 20% rule-bot, 10% sibling.
```

### 9.2 4인 pool 구성

```text
한 매치 = 4명 동시. 의도적 역할 비대칭 편성.

매치 편성 템플릿:
  - 1× 최신 모델 + 1× rusher + 1× turtle + 1× rule-bot
  - 1× 최신 + 3× sibling-of-different-seed
  - 4× 최신 (mirror)
  - 1× 최신 + 1× snapshot_old + 1× rule-bot + 1× rule-bot

이렇게 해야 king-maker / triangulation 시나리오가 자연 발생.
mirror 만 돌리면 자기 정책 corner 에 갇힘.
```

### 9.3 샘플링 비율 curriculum

```text
초기 (encoder 안정화):
  2인전 80% / 4인전 20%
중기 (4인전 head 학습):
  50% / 50%
후기 (실배포 분포):
  실제 평가 분포 (예: 60/40)
```

### 9.4 Reward seed 다양성

같은 환경에서 *다른 reward weight* 로 학습된 sibling 을 pool 에 의도적으로 넣는다.
- aggressive seed (race / denial weight 높음)
- defensive seed (support / hold weight 높음)
- balanced seed
이렇게 하면 main 모델이 *다양한 정책 표면*을 만나면서 일반화.

---

## 10. 상대 archetype 별 counter

| Archetype | 취약점 | Counter 정책 | League 의무 편성 |
|---|---|---|---|
| Greedy expander | 갓 먹은 행성 다수, 분산 source | post-capture snipe (5+11) | greedy_radius bot |
| Turtler | 전선 형성 못함, 후반 경제 우위 못만듦 | economy outpace, 점진 확장 (1+8) | turtle_threshold bot |
| Rusher | source 비면 본진 노출 | rush 감지 → 본진 방어 + counter on empty src (4+11) | rush_nearest bot |
| Coordinated multi-attacker | 도착 전 intercept 가능 | race intercept (10), idle src 카운터 | sync_attacker bot |
| Mirror | novelty 부족 | sibling reward seed 정책 | sibling-of-other-seed |
| 4p king-maker | rank 인식 약함 | rank_progress, per-opponent threat | role-asymmetric 4p match |

**rule-bot 4종은 league pool 에 의무 편성**. 그렇지 않으면 self-play 가 *자기끼리만 이기는 corner case* 로 수렴.

---

## 11. 메트릭 / 평가

### 11.1 항상 metric (reward 화 금지 또는 매우 약하게만)

```text
- source_opportunity_cost
- pressure_index, enemy_idle_ratio
- target_strategic_safety (sun-asymmetry)
- per_opponent_target_distribution (4p)
- linked_launches_per_capture
- over_send_target_rate
- filtered_path_rate, sun_crash_rate
- send_fraction_of_src
- all_in_launch_rate
```

### 11.2 정책별 검증 metric

```text
정책 1: early_neutral_captured_per_episode, early_neutral_nearest_rank_mean
정책 2: local_cluster_capture_rate, support_reachable_planets
정책 4: support_launches, post_capture_reloss_rate_k
정책 5: capture_hold_k_rate, post_capture_reloss_rate_k
정책 6: linked_launches_per_capture, coordinated_capture_rate
정책 7: filtered_path_rate, sun_crash_rate, eta_advantage
정책 8: production_share, planet_count_share, win_rate
정책 9: pressure_index, enemy_idle_ratio
정책 10: race_attempt_rate, race_win_rate
정책 11: denial_attempt_rate, denial_success_rate, ships_per_denial
정책 12: opponent_pred_top1_acc, opponent_pred_kl
정책 13: per_opponent_target_distribution
정책 14: king_maker_rate (4등이 1등을 친 비율)
정책 16: rank_distribution, rank_change_per_episode
```

### 11.3 학습 안정성 metric

```text
- loss, kl_divergence, entropy (정책별)
- advantage running stats (mode 별 별도)
- aux_opponent_pred_loss
- gradient norm
- per-component reward magnitude (overflow 감지)
- mask block rates per gate (현 mask_block_* 연장)
```

### 11.4 Eval 매트릭스

```text
2p:
  vs each rule-bot (4): win_rate, avg_episode_length
  vs sibling: win_rate
  vs snapshot_-N: win_rate (자기 발전 곡선)

4p:
  rank distribution vs (3 rule-bots)
  rank distribution vs (3 siblings)
  rank distribution vs (3 snapshots_-N)
  king_maker_rate
  rank_progress_mean
```

평가는 **fix seed 풀**로 해야 단계 간 비교 가능.

---

## 12. 단계별 로드맵

이 장만 우선순위·구현순서. 다른 장은 청사진.

각 phase 의 **게임 성능** 항목은 그 단계가 완료됐을 때 모델이 *게임 안에서 어떻게 보이는지* 를 게임 플레이 용어로 적은 것이다. metric 이 아니라 행동 묘사 — 관전자가 봤을 때 "아 이런 식으로 두는구나" 라고 느낄 변화.

### Phase -1 — 현재 변경 안정화 (5~10 generation 검증)

15.2.x 의 보완을 시작하기 전에, 직전 리팩토링 (5게이트 mask + support-mode + reward gain/loss 분리) 이 정말 정착했는지 확인. 새 설계로 넘어가기 전에 *지금 고친 게 깨지지 않았는지* 가 우선.

```
[완료된 항목 — 진행 중 검증]
  · 5게이트 hard mask + support-mode 가드 정착
  · cap_bonus → neutral_capture_bonus / own_planet_loss_penalty 분리
  · reward/events.py + reward/trackers.py 파이프 (LaunchMetadata,
    CaptureEvent, LaunchCaptureTracker — 30턴 window, target_id 매칭)
  · early_close_neutral_capture_bonus (event-consuming) 0.007 활성
    — capture_events 의 첫 소비자, 파이프 동작 검증대 역할

[잔여 검증 — 5~10 gen 안정성 확인]
  · mean_filtered_invalid_target ≈ 0 유지
  · support_launches_per_step > 0 (방어 행동 자체는 일어남)
  · 분리 전후 win_rate ±1σ 이내 (gain/loss + early_close 추가의 parity)
  · capture_hold_k_rate 추세 ↑
  · post_capture_reloss_rate_k 추세 ↓
  · linked_launches_per_capture_neutral / _enemy 의 분포 sanity
    (window 30턴 이 너무 길/짧으면 multiplicity 가 비정상)

  ── 초반 확장 진단 (단일 metric 으로는 부족 — gen2 사례에서
       captured 0.75 로 떨어졌는데 nearest_rank 는 멀쩡했던 적 있음. 4-metric cluster 로 본다)
  · early_neutral_attempts_per_episode      "행동 시도 자체가 줄었나?"
  · early_neutral_launch_to_cap_rate         "시도가 capture 로 이어지나?"
  · early_neutral_captured_per_episode ≥ 1.0  최종 결과 (회복 지표)
  · early_close_trigger_rate
       너무 드물면 (≪ baseline) → 자격 조건 (turn_norm 0.25 / nearest_rank 1 /
         req_over_src 0.5) 중 하나가 too tight 이거나 LaunchMetadata 의
         nearest_rank 가 0 (not computed) 으로 떨어지는 metadata 문제
       너무 많으면 (≫ baseline) → coef 0.007 이 과다 또는 자격 조건 too loose,
         neutral_capture_bonus 와 신호 중복 risk
  · mean_early_close_neutral_capture_bonus  실제 보상 크기 (다른 component 비례 sanity)
- 위 11개가 모두 안정이면 Phase 0 진입
```

**게임 성능 (현재)**: "본진 안 비우는 안정형 그리디 + 초반 가까운 중립 우선 가산". 가까운 중립을 빠르게 확보 (early_close 가 그 행동을 prod-가중으로 보상), 본진에 적이 들어올 때만 짧게 support, 갓 잡은 행성을 즉시 잃는 빈도가 줄어든 *기본기 봇*. 동시 도착·연합 공격·king-maker 같은 상위 정책은 아직 없음.

### Phase 0 — 인프라 / 측정 / 환경 audit (코드 기반 다지기)

대형 변경을 시작하기 전에 *측정 가능성과 환경 가정* 을 확정. 이 phase 동안 모델은 변하지 않는다 (게임 성능 변화 없음). 그러나 이걸 안 깔면 Phase 1~5 의 회귀 검출이 불가능.

```
- env adapter audit (15.2.3)
  · obs 에서 fleet (src, tgt, launch_turn, eta, owner) 노출 여부 grep
  · 누락 필드는 obs delta 추론 가능성 검증 + plan 명시
- logger 16정책 컬럼 등록 (15.2.6, 단 NaN / available_flag 방식)
- fixed-seed eval harness 구축
  · seed 32~64 고정, opponent set = 현 baseline snapshot + greedy_radius rule-bot 1
  · 매 N=1k step 자동 실행 + axis metric diff 알람
- rule-bot greedy_radius 1개 우선 구현 (Phase 3 의 일부 선행)
- (병렬 0-track) 4p env wrapper 가능성 조사 — 자체 sim or Kaggle 4p 룰
```

**게임 성능**: 변화 없음. **회귀 검출 능력** 만 생긴다. (이 phase 의 산출물은 이후 모든 phase 의 안전벨트.)

### Phase 1 — 표현·관측 기반 (4 sub-phase 로 분할)

원래 plan 이 한 phase 로 묶었던 5개 변경을 4개 sub-phase 로 쪼갠다 (15.2.1). 각 sub-phase 끝마다 fixed-seed eval 회귀 검사 (≥ baseline - σ).

#### Phase 1a — arrival_schedule 내장 + future rollout (5.3, 5.4)

결정론적 시뮬, 모델 구조 변경 없음. feature 만 추가. 가장 안전한 선행 변경.

```
- 각 planet 에 다음 K=8 turn 의 (incoming_arrivals, expected_owner_t, expected_threat_t) feature concat
- gradient 안 흘리는 deterministic rollout 만, imagined branch 는 Phase 6
```

**게임 성능**: "이미 잡힐 행성에 추가 ship 안 보내는 절약형". turn t 에 도착 예정 fleet 으로 capture 가 확정된 target 에는 더 보내지 않음. over_send 자연 감소. 단순 알뜰함.

#### Phase 1b — player_count embedding + per-opponent feature pad (5.5, 5.6)

2p 환경에서도 입력 layout 을 4p 와 통일. opponent slot 은 0-pad. 모델 변경은 input dim 확장만.

```
- player_count_onehot {2,4} 입력 (현재는 [1,0] 고정)
- per_opponent_features (3, F_o) — 2p 면 opp1 만 채우고 나머지 2개는 0
- encoder 통과는 동일, 마지막 head 만 단일 분기 유지
```

**게임 성능**: 2p 한정 변화 거의 없음. **4p 환경이 0-track 에서 준비되면 즉시 합류 가능한 입력 형태** 가 갖춰진다는 게 핵심 (인프라 변경).

#### Phase 1c — multi-relation graph attention (5.2)

가장 큰 모델 구조 변경. 단독 sub-phase 로. 13장에 13.8 (encoder swap curriculum) 추가 후 진입 (15.2.8).

```
- 5종 relation: planet←fleet (arrival), fleet↔fleet (same-tgt), fleet↔fleet (same-src), planet↔planet (k-NN spatial), planet↔planet (same-owner)
- multi-head, multi-relation attention layer
- encoder swap curriculum: old encoder freeze + new linear-probe → 점진적 unfreeze
- win_rate 가 baseline - 2σ 이상 떨어지면 즉시 rollback
```

**게임 성능**: 이 단계부터 실제 *플레이 스타일* 이 바뀐다.
- **지역 거점 형성**: k-NN spatial relation 이 자기 행성 클러스터를 인지. 멀리 떨어진 행성을 무리하게 안 먹고 *연결된 영토* 를 만든다 (정책 2 — 지역 네트워크).
- **race / 동시 launch 인지**: same-target relation 이 적이 같은 target 에 ship 보내는 걸 본다. 도착 시점 비교가 표현에 들어감 (정책 10 의 토대).
- **본진 방어 회로**: same-owner + planet←fleet 결합으로 "내 행성에 들어오는 fleet 합" 이 한 번에 보임. 분산된 위협을 *통합된 위협 지도* 로.
- 관전자 인상: "더 이상 무계획 그리디가 아니라, 자기 영토를 인지하고 행동하는 봇".

#### Phase 1d — opponent_pred aux head (8.3)

1c 가 끝나야 의미. shared encoder 에 적의 다음 launch target 예측 supervised head 부착.

```
- 라벨링: self-play replay 는 직접 access, 외부 봇은 fleet delta 추론 (15.2.4)
- 라벨 신뢰도 < 0.7 step 은 mask out
- weight 0.1 ~ 0.3
```

**게임 성능**: "적의 다음 수를 미리 보는 카운터 플레이의 토대". reward 변화 없이도, encoder 표현이 *어디가 위협인가* 를 더 잘 인코딩 → 같은 reward 로도 위협 행성에 사전 cap 차단·source 비우기 패턴이 미세하게 빨라짐. 아직 *명시적* counter 행동은 안 나옴 (그건 Phase 2 의 denial/race reward 가 켜져야).

### Phase 2 — Reward 확장 (action space 동일, 순서 재배치)

현재 병목 (초반 확장 / hold / reloss / support) 이 직접 신호이고, denial/race 는 그 다음. 따라서 plan 원안의 순서를 **단기 catalog → 장기 catalog** 로 바꾼다 (사용자 지적 반영).

`early_close_neutral_capture_bonus` 는 Phase -1 단계에서 이미 도입·활성 (0.007). Phase 2.1 의 잔여 4개는 모두 같은 events/trackers 파이프를 재사용 (post_capture_reloss·capture_hold 는 capture_event.age 기반, support_defense·reinforce_young 은 별도 SupportDefenseTracker 가 필요할 수 있음).

```
[Phase -1 단계에서 이미 도입]
  ✓ early_close_neutral_capture_bonus (정책 1, 활성 0.007)
  ✓ neutral_capture / own_planet_loss 분리

2.1 단기 [단] catalog (잔여 4개 — Phase 1 의 표현이 차이를 만든다는 가정 위에)
  - post_capture_reloss_penalty (정책 5, capture_event.age ≤ N 일 때 own_loss 가중)
  - capture_hold_bonus (정책 5, capture 후 K턴 hold 성공 시)
  - support_defense_bonus (정책 4, support launch 가 실제 적 incoming 막은 결과만
    — 새 SupportDefenseTracker 또는 LaunchCaptureTracker 의 변형 필요)
  - reinforce_young_planet_bonus (정책 4 / B4 vacuum 대응)

2.2 장기 [장] catalog (단기 axis metric 안정 후)
  - denial_bonus (정책 11)
  - race_won_capture_bonus + weak race_lost_attempt_penalty (정책 10)
```

각 component 도입 게이트 (15.2.5): cross-corr < 0.5, 부재 시 axis metric 정체 ablation 증거, log-grid weight sweep, mode 별 별도. 통과 못 하면 메트릭만.

**게임 성능 (2.1 후)**: "초반 확장 정확도 ↑ + 점령 행성 hold ↑ + 본진 방어 결과 인지". 갓 잡은 행성을 다시 뺏기는 빈도 추가 감소, support 가 *적 fleet 을 실제로 막을 때만* 강화 → support 난사 줄어들고 정확한 방어. 게임 흐름 안정도 상승.
**게임 성능 (2.2 후)**: "거부 (denial) 와 race 의 등장". 적이 곧 잡을 중립을 *내가 못 먹어도 minimum-ship 으로 막아둠*. 같은 target 에 적보다 빠르면 race 시도, 늦으면 포기 (race_lost weak penalty). 관전자 인상: "이제 적의 흐름을 끊는다". 단, 4p 정책 (king-maker, ranking) 은 아직 없음.

### Phase 3 — League 다양성

Phase 2 의 reward 가 *한 가지 적 분포에서만* 효과가 있는 corner 가 아닌지 깨는 단계.

```
- rule-bot 4종 (greedy_radius, turtle_threshold, rush_target_nearest, sync_attacker)
- 의무 편성: 70% snapshot / 20% rule-bot / 10% sibling
- sibling reward seed: 별도 full run 1개 (main 안정화 후, 15.2.10) — 그 외엔 multi-head reward 또는 curriculum-weight snapshot 으로 의사-sibling
- 평가 매트릭스 (11.4) 운영 시작
```

**게임 성능**: "다양한 archetype 대응". turtle 상대로는 경제 우위 → 점진 확장으로 압박, rush 상대로는 본진 방어 + 빈 source 카운터, sync_attacker 상대로는 race intercept. 한 가지 메타에 갇히지 않고 *상대 보고 두는 봇*.

### Phase 4 — 4인전 학습 (0-track 의 4p env 가 준비된 후)

```
- 4인 head + per_opponent_attention 활성 (8.4, 1b 의 0-pad slot 활성화)
- ranking-based terminal reward (7.2): Phase A (1st-only) → Phase B (ordinal [+3,+1,-1,-3]) → Phase C (mix)
- rank_progress_bonus / rank_decay_penalty (정책 16)
- role-asymmetric 4인 매치 템플릿 (9.2)
- 2p/4p 비율 curriculum: 80/20 → 50/50 → 60/40
- mode 별 advantage normalizer 별도 (13.3)
```

**게임 성능**:
- **Phase A (1st-only)**: 공격적. "1등 못 하면 의미 없으니 일단 친다". king-maker 행동 자주 등장 (4등이 1등 침).
- **Phase B (ordinal)**: 신중. 1등 못 할 거 같으면 *2등 굳히기*, 4등 회피. 위협 ranking 인지 — 매 턴 누가 1등인지 동적 판단해 그 상대에 우선 압박.
- **Phase C (mix)**: 결정적 행동 회복 + ranking 인식. 게임 후반 *삼자 견제 (triangulation)* — 두 상대를 서로 싸우게 만들고 본인은 영토 굳히기 시도.
- 관전자 인상: 4p 에서 *어느 상대를 적으로 삼느냐* 를 매 턴 다시 평가하는 봇.

### Phase 5 — Action space 확장 (2 sub-phase 로 분할)

#### Phase 5a — list-aware action infrastructure (single-source 강제 유지)

PPO buffer / advantage / log_prob 모두 list-aware 로 리팩터, 단 max_sources=1 강제 (15.2.9). 동작은 single-source 와 동등해야 함 (회귀 검사).

```
- step 당 (s, [a_1...a_K], [log_prob_1...K], [adv_1...K]) shape 로 변경
- max_sources=1 cap → 기존 single-source 동작 보존
- fixed-seed eval 통과 = baseline 동등 ± σ
```

**게임 성능**: 변화 없음. 5b 진입 안전성만 확보.

#### Phase 5b — multi-source 활성 + synchronized reward + fraction head

```
- max_sources 점진적 증가 (1→2→3)
- synchronized_arrival_capture_bonus (정책 6, B2 alpha)
- partial launch fraction head (6.2.4) — denial/spoiler minimum-ship 정밀도
- coordinated_capture_bonus 활성
```

**게임 성능**: 게임 외관이 가장 크게 바뀌는 단계.
- **Alpha strike**: 같은 turn 에 여러 source 에서 동일 target 으로 동시 도착. defense 가 사이에 끼어들 틈 없이 한 번에 capture.
- **연합 공격**: 본진 방어와 공격을 한 turn 에 병행. multi-source set 이 자연스러움.
- **Spoiler 정밀**: minimum-ship 으로 적 capture 만 차단, 잉여 전력은 다른 target.
- 관전자 인상: "이 봇 한 턴에 movement 가 여러 개네" — 이제 *플레이어 같다*.

### Phase 6 — 고급 / 선택

```
- Q-style joint candidate evaluation (6.2.2): 4인전·연합 공격에서 결정적
- macro / option (6.2.3): "행성 P 를 hold", "K턴에 동시도착" 매크로
- imagined-branch rollout (5.4 확장): MCTS 1-step lookahead
```

**게임 성능**: "장기 plan 이 보인다". M턴 hold 매크로 — capture 직후 vacuum 단계에서 K턴 동안 자동 reinforce. K턴 동시도착 매크로 — 미리 송출해놓고 도착 시점에 stack. imagined branch — *지금 launch 안 하면 K턴 뒤가 어떨까* 를 1-step look-ahead 로 비교, 그 결과를 정책에 반영.

---

## 13. 위험 요소와 회피 원칙

### 13.1 Reward over-engineering
새 component 가 16개 정책에 비례해 늘어나면 학습 신호가 충돌·redundant. 추가 시:
- 기존 reward 와 cross-correlation > 0.7 이면 합치거나 폐기.
- weak penalty 가 모델을 *행동 회피* 로 몰면 즉시 약화.
- terminal 이 dense 보다 항상 큰 신호 비율 유지.

### 13.2 Mask 과잉
mask 가 *판단 가능한 행동* 까지 막으면 정책이 학습 자체를 못 한다. 원칙:
- mask 는 sun/path/물리 invalid 만.
- "비효율적인 행동" 은 reward 로 표현, mask 가 아님.
- own planet support / enemy-incoming neutral / race-disadvantage target 모두 *모드로서 열어둘 것* (action 자체를 막지 않음).
- 단 **action-space guard** (target filter, send-fraction cap 등) 는 학습 안정화 단계에선 허용. reward 신호가 axis metric 으로 검증된 후 단계적 완화 (6.3 참조).

### 13.3 4인전이 2인전 정책을 망가뜨림
4인전 noise 가 큼. shared encoder 가 4p episode 의 분산에 끌려가면 2p 정책이 흔들림.
- mode 별 advantage normalizer 별도.
- 초기 2p 80% 로 안정화 후 4p 도입.
- aux opponent_pred 는 mode 무관하게 도움됨.

### 13.4 League corner 수렴
self-play 만으로는 corner 정책 강해짐. rule-bot 의무 편성 + sibling seed 다양성 강제.

### 13.5 Action space 확장 시 정책 붕괴
multi-source decoder 도입 직후 기존 single-source 정책이 한 번 무너졌다 다시 학습됨. curriculum:
- multi-source decoder 도입 시 첫 K step 은 *single-source 모드 강제* (max 1 source).
- 점진적으로 max source 늘림.

### 13.6 Aux loss 지배
opponent_pred aux 가 RL loss 를 압도하지 않도록 weight 0.1~0.3.

### 13.7 메트릭과 reward 혼동
*항상 metric* 으로 분류된 것 (source_opportunity_cost, pressure_index, target_strategic_safety) 을 reward 화 하면 모델이 그 신호로 정책이 왜곡됨. 명시적 카탈로그 (7.3, 11.1) 분리 유지.

---

## 14. 단일 문장 요약

> **정책은 16축으로 보되 reward 는 그중 결과로 검증 가능한 것만 표현하고, fleet/planet 을 시점이 아닌 궤적으로 다루는 temporal-aware encoder 를 토대로 2인전·4인전을 구분된 head 와 ranking-aware reward 로 학습시키며, league 다양성과 rule-bot 의무 편성으로 self-play corner 를 깨고, action space 는 multi-source coordinated decoder 까지 확장한다.**

이 한 줄을 시스템 전체의 북극성으로 둔다.

---

## 15. 본 청사진의 단점 / 한계 — 현 코드 기준 진단 + 보완 (2026-04-28 추가)

### 15.1 현 코드 진단 (Plan 대비 위치)

현재 `main` 코드를 plan 의 각 장과 비교하면 **Phase A 기준선 + Phase -1 의 일부 진척** (events/trackers 파이프 + 첫 event-consuming component 활성). 단 Phase 1 ~ 6 의 본격 항목들 (encoder 변경, multi-source, 4p 등) 은 모두 미착수.

| 영역 | 현 상태 | Plan 목표 | 갭 |
|---|---|---|---|
| Mask (`mask/`) | 5게이트 hard mask 정렬됨 (`target_owner_allowed, flight_path_clear, projected_arrival_state, attack_still_needed, capacity_sufficient`). attack/support parity 정렬, invalid target 차단. 단 support target 은 *enemy_incoming OR required + 35% cap* guard — hard invalid 가 아닌 **action-space guard**. | 장기적으로 hard-invalid-only + own_loss / support_defense reward 가 가드를 대체 | **구조는 안정, 철학 일부 불일치** (작음) |
| Reward (`reward/components.py`) | **9 component** — stateless 8개 (`dense, neutral_capture, own_planet_loss, all_in, over_send, under_invested, launch_cost, terminal`) + **event-consuming 1개** (`early_close_neutral_capture_bonus`, coef 0.007 활성). gain/loss 분리 + capture_events 첫 소비자 정착. | ~20+ component, ranking/race/denial/coordinated 추가, 4p 모드별 분리 | **큼** (단, 단기 catalog 의 첫 한 칸은 채워짐) |
| Feature/Encoder (`model.py`) | snapshot + HISTORY=20 temporal, planet/fleet MLP+attention | arrival_schedule, multi-relation graph attn (5종), per-opponent encoder, player_count embed, future rollout sim | **매우 큼** |
| Architecture (`model.py`) | single-source decoder, 단일 critic, cost-bias MLP (Phase A) | multi-source coord decoder, aux opponent_pred head, 2p/4p 분기 head, FiLM mode gate | **매우 큼** |
| Action space | one-launch-per-turn, 이산 ship multiplier | multi-source set per turn, 연속 fraction head | **큼** |
| League (`train.py LeaguePool`) | snapshot rotation only | rule-bot 4종 의무 편성, sibling reward seed, 2p/4p curriculum, role-asymmetric 4p match | **매우 큼** |
| Logger (`utils/logger.py`) | ~140+ 컬럼, 정책 1/4/5/6 부분 검증. 신규: `mean_neutral_capture_bonus`, `mean_own_planet_loss_penalty`, `mean_early_close_neutral_capture_bonus`, `linked_launches_per_capture_{neutral,enemy}` (정책 6 multiplicity). | 정책 1~16 전체 검증 metric, 4p rank dist | **중간** (절반, 정책 6 multiplicity 는 새로 활성) |
| Eval | 2p win_rate 기반, fixed-seed pool 없음 | fixed-seed eval matrix (rule-bot/sibling/snapshot-N), 4p rank distribution | **큼** |
| Player-count 분기 | 없음 (Kaggle 2p 전제 단일 코드) | 2p/4p 별 head·normalizer·env·league·reward·eval 모두 분리 | **결정적** (4p 인프라 0%) |
| Stateful event infra | **활성화됨**. `reward/events.py` (LaunchMetadata, CaptureEvent — `linked_launches` 포함), `reward/trackers.py` (LaunchCaptureTracker, 30턴 window, owner-change 감지 + target_id 매칭). `early_close_neutral_capture_bonus` 가 `ctx.capture_events` 첫 소비자. 별도 `HitRateTracker` 는 진단 전용 그대로. | 다수의 event-consuming component (post_capture_reloss, capture_hold, support_defense …) 가 같은 파이프 사용 | **작음** (파이프 정착, 추가 소비자만 늘리면 됨) |

요약: **mask 정렬 + reward 파이프 (events/trackers) 정착 + 첫 event-consuming component (early_close) 활성**. 그 외 7개 영역 (encoder, architecture, action, league, eval, player-count, 단기 reward catalog 잔여 4개) 은 미착수. 4인전은 환경부터 0%.

### 15.2 청사진의 단점 + 보완

#### 15.2.1 단점 — Phase 1 자체가 거대해서 단일 phase 로 다룰 수 없음
Phase 1 은 (a) arrival_schedule (b) multi-relation graph attention (5종 관계) (c) per-opponent encoder (d) player_count embedding (e) opponent_pred aux head — 다섯 개의 독립 변경을 한 phase 로 묶었다. 각각이 1~2주 작업이고 디버깅까지 합치면 phase 1 만 1~3개월. 이 동안 reward·league 쪽 진척이 0 인 risk.
**보완**: Phase 1 을 1a~1d 로 쪼갠다.
- **1a**: arrival_schedule 내장 + future rollout (5.3, 5.4) — 결정론적 시뮬, 모델 변경 없음. 가장 안전. 먼저.
- **1b**: player_count embedding + per-opponent feature pad (4p 0-pad 으로 2p 도 입력 통일) — feature dim 만 변동.
- **1c**: multi-relation graph attention — 가장 큰 모델 변경. 단독 phase 로.
- **1d**: opponent_pred aux head — 1c 끝나야 의미 있음.
각 단계 끝마다 기존 win_rate 회귀 검사 (≥ baseline - σ).

#### 15.2.2 단점 — 4인전 환경이 0%인데 plan 의 30% 가 4p 전제
5.6 (per-opponent encoder), 7.2 (4p reward), 8 (4p head, FiLM gate), 9.2 (4p league), 11.4 (4p eval) — 모두 4p env 가 있어야 검증 가능. 현재 코드는 Kaggle 2p 환경만 사용. plan 에 "4p env 어떻게 만드나" 가 없다.
**보완**: Phase 0 ("4p env trakck") 를 main track 과 **병렬**로 분리 신설.
- 0-track: 4p sim 환경 wrapper (Kaggle 4p 룰 또는 자체 sim) + 4p match runner + 4p eval harness — main 트랙과 독립적으로 진행, **2p main 트랙은 plan Phase 1~3 까지 4p 무관하게 진행**.
- Phase 4 진입 시점에 0-track 산출물이 준비되어 있어야 main 트랙이 4p 로 합류.
- 만약 0-track 이 늦으면 Phase 4·일부 5 를 보류하고 Phase 5 (multi-source) 를 2p 안에서 먼저.

#### 15.2.3 단점 — Plan 이 env observability 를 무검증 가정
arrival_schedule (5.3), opponent_pred aux (8.3), per-opponent target distribution (11.2) 모두 fleet 의 (src, tgt, launch_turn, eta) 와 상대 launch event 가 obs 로 노출된다고 전제. Kaggle 환경이 이걸 모두 노출하는지 plan 은 검증하지 않음.
**보완**: Phase 1a 진입 전에 **env adapter audit** 1주 작업 추가.
- 현 obs 에서 fleet 별 (src, tgt, launch_turn, eta, owner) 가 직접 노출되는지 grep.
- 누락된 필드는 obs delta 로 inference 가능한지 (예: 새 fleet 등장 = launch event) 검증.
- inference 도 불가능한 필드 (예: opponent 의 ship 의도, 미래 launch 계획) 는 plan 에서 명시적으로 제외.

#### 15.2.4 단점 — Aux opponent_pred 의 ground-truth 정의 모호
8.3 은 "predict opponent's next target, CE with realized action" 이라고 쓰지만, Kaggle obs 에서 opponent action 이 atomic 하게 노출되지 않을 수 있다. fleet delta 로 추론 시 noise 가 들어가고, multi-source 환경에선 어느 src 의 어느 tgt 인지 라벨이 모호.
**보완**: 라벨링 layer 명시.
- self-play replay 에서는 opponent action 직접 access (그라운드트루스 100%).
- 외부 봇 매치는 fleet delta 로 inferred target — 다중 launch 면 single-target softmax 대신 multi-label BCE.
- 라벨 신뢰도 < 0.7 인 step 은 aux loss 에서 mask out.

#### 15.2.5 단점 — Reward catalog 폭증 → cross-correlation/weight 탐색 비용 폭발
7.3 catalog 는 component 13개+, 4p 전용 4개 추가. 각각 weight 가 필요하고 mode 별 (2p/4p) 별도. 단순 grid 로도 (3 weight per comp)^17 = 천문학적. plan 은 13.1 에 "corr > 0.7 이면 합치거나 폐기" 만 쓰고 *추가 시점 게이트* 가 약함.
**보완**: 새 component 추가 게이트 강화.
- (a) 기존 component 와의 cross-corr < 0.5 (0.7 → 0.5).
- (b) **해당 component 없을 때** 정책 metric 이 정체된다는 ablation 증거 (사후가 아닌 사전).
- (c) weight 는 단일 점이 아닌 logarithmic 3-point grid (×0.3, ×1, ×3) 로만 sweep, mode 별 별도.
- 둘 다 통과 못 하면 component 화 보류, **metric 으로만 logger 추가**.

#### 15.2.6 단점 — Logger 가 16정책 중 4정책만 검증 가능 → 측정 없는 학습 risk
Phase 1·2 변화의 효과를 axis 별로 측정 못 하면 "encoder 가 정책 11/13 을 실제로 강화했는지" 판단 불가. plan 11.2 가 정의는 했지만 *언제 logger 에 추가하나* 가 phase 에 없음.
**보완**: Phase 0 에 **metric infra** 명시 추가 — Phase 1 시작 *전에* logger 에 정책 2/9/10/11/12/13~16 컬럼 추가. 단 **값을 0 으로 채우지 말 것** (0 이 "측정했는데 0" 인지 "미구현" 인지 모호). 두 가지 방식 중 택일:
- **방식 A**: 미구현 metric 은 NaN / blank 로 두고, 동일 행에 `metric_available_<name> ∈ {0, 1}` flag 컬럼 동반.
- **방식 B (더 단순)**: 해당 정책의 **phase 진입 시점에만** 그 metric 컬럼을 추가 (phase 0 에선 컬럼 schema 만 코드에 등록, 실제 CSV 는 phase 별 활성).
어느 쪽이든 **컬럼 없는 reward component 도입 금지** 룰은 유지.

#### 15.2.7 단점 — Plan 7.3 의 "weak filtered_path_penalty / weak race_lost_attempt_penalty" 와 CLAUDE.md "reward = 결과만 평가" 충돌 의심
filtered_path 는 mask 가 거른 행동을 다시 reward 로 평가하는 모양새고, race_lost_attempt 는 "도착이 늦었으니 시도 자체가 잘못" 이라는 행동가능성 판정에 가깝다. CLAUDE.md 최종 규칙 ("Reward 는 행동 가능성을 판단하지 않는다") 와 마찰.
**보완**: 이런 보더라인 component 는 도입 단계 명시.
- 1단계: **metric only** (logger 에 rate 만 기록, reward 화 X).
- 2단계: 정책 metric (race_attempt_rate, filtered_path_rate) 이 baseline 대비 유의 악화 시에만 reward 화 검토.
- 3단계: 도입 시 weight 0.05 미만, 1만 step 후 axis metric 회복 못 하면 즉시 0.
이렇게 하면 "결과 평가" 원칙과 "약한 nudge" 의 타협점이 측정 기반.

#### 15.2.8 단점 — 13.5 는 action space curriculum 만 언급, encoder 변경 시 정책 붕괴 risk 미언급
Phase 1c (multi-relation graph attention) 처럼 encoder 가 통째로 바뀌면 기존 학습 정책의 표현이 mismatch 되어 win_rate 가 한 번 무너진다. plan 13.5 는 action space 만 다루지 encoder swap curriculum 은 없음.
**보완**: 13장에 **13.8 encoder swap curriculum** 항목 추가 (별도 보완은 아래 코드 변경엔 포함 안 함, 본 단점 노트로만 — 추후 phase 1c 직전에 13.8 작성).
- old encoder 는 freeze, new encoder 만 학습 → linear probe 로 동일 task 만 → 점진적 unfreeze.
- 또는 dual-stream + gating: gate 가 점진적으로 new encoder 쪽으로 이동.
- win_rate 가 baseline - 2σ 이상 떨어지면 즉시 rollback.

#### 15.2.9 단점 — Phase 5 multi-source decoder 는 "head 추가" 가 아닌 RL pipeline 전면 리팩터
PPO buffer 는 step 당 (s, a, log_prob, advantage) 가정. multi-source 면 step 당 (s, [a_1...a_K], [log_prob_1...K], [adv_1...K]) 으로 텐서 shape, advantage 분배, KL 계산 모두 변경. 단순 model 변경이 아니라 train loop 변경이라서 plan Phase 5 의 작업량이 크게 과소평가됨.
**보완**: Phase 5 를 5a/5b 로 분리.
- **5a (infra)**: action representation 을 list 로 리팩터 (단, max_sources=1 강제 → 동작은 single-source 와 동일). PPO buffer/advantage/log_prob 모두 list-aware. 회귀 검사 = baseline 동등.
- **5b (활성)**: max_sources 점진적 증가 (1→2→3). 5a 가 통과해야 5b 진입.

#### 15.2.10 단점 — Sibling reward seed 의 GPU·시간 비용 미반영
9.4 는 "다른 weight 로 학습된 sibling 을 pool 에 합류" 하는데, sibling 1개 = full 학습 run 1개 = main 학습과 동일 비용. 2~3개 sibling 이면 GPU 시간 3~4배. plan 은 이 비용을 다루지 않음.
**보완**: sibling 을 별도 full run 으로 만들지 말고,
- (a) **multi-head reward**: 하나의 학습에서 head 별로 다른 reward weight 로 학습 (encoder 공유). 의사-sibling 효과.
- (b) **single-run snapshot diversity**: 학습 도중 reward weight 를 주기적으로 변동 (curriculum 형태) → 그때 snapshot 을 sibling 처럼 풀에 합류.
- (c) 진짜 별도 sibling run 은 main 모델이 한 번 충분히 안정화된 *후* 1개만 학습 — 처음부터 병렬 N개는 ROI 부족.

#### 15.2.11 단점 — Fixed-seed eval matrix 부재 (전 phase 의 회귀 검출 불가)
11.4 는 "fix seed 풀" 을 명시하지만 phase 1 의 어디에도 *언제* 만드는지 없음. 회귀 검사 (15.2.1, 15.2.8 의 보완) 가 모두 fixed eval 에 의존하는데 정작 그게 없음.
**보완**: Phase 0 (15.2.6 의 metric infra 와 같이 묶음) 에 **fixed-seed eval harness** 도 포함.
- seed 32~64개 고정, opponent set = {현 baseline snapshot, 단순 rule-bot 1개 (greedy_radius 만이라도 먼저)}.
- 매 N=1k step 자동 평가, win_rate + axis metric 차분 알람.
- Phase 1~5 의 모든 회귀 검사가 이 harness 의 출력으로만 판단.

### 15.3 종합 — Plan 본문 반영 현황 (이번 turn 에 완료)

15.2 의 단점들을 반영해 **plan 본문을 다음과 같이 수정 완료**:

```text
[적용됨]
- 6.1: "one launch per turn" → "source 별 독립 launch decision, joint action 미평가" 로 정확화
- 6.3: support mask 철학 명시 — "action 자체는 항상 존재 / target 은 필요성 기반 guard" (15.2 의 사용자 지적)
- 7.3: reward catalog 를 [현]/[단]/[장]/[메트릭]/[aux] 5분류로 재작성. 현재 구현 8개 component 명시
- 12장 (로드맵):
    Phase -1 (현재 변경 안정화) 신설
    Phase 0 (인프라/측정/환경 audit) 신설
    Phase 1 → 1a (arrival_schedule) / 1b (player_count + per-opponent pad) / 1c (graph attention) / 1d (opponent_pred aux) 로 분할
    Phase 2 의 component 도입 순서를 [단] → [장] 으로 재배치
    Phase 5 → 5a (list-aware infra, max_sources=1 강제) / 5b (multi-source 활성) 로 분할
    각 phase 에 게임 성능 (관전자 시점 행동 묘사) 추가
- 13.2: support mask 표현 부드럽게 — "모드로서 열림 / guard 는 단계적 완화"
- 15.1: mask 진단을 "구조 정렬 / 철학 일부 불일치" 로 재분류
- 15.2.6: logger 미구현 metric 처리 — NaN+available_flag 또는 phase 진입 시점 컬럼 등록 (값 0 채우지 않음)

[보류 — 별도 turn]
- 13.8 encoder swap curriculum 본문 추가는 Phase 1c 직전에 작성 (지금은 15.2.8 에 의도만 기록)
- 2.2.1 source_opportunity_cost 식 정정 — 사용자 지적 (현 식의 ships_sent × prod_rate 가 의미 모호) 은 별도 turn 에 정리

[코드↔doc 재정렬 (2회차) — 2026-04-28 후반]
main 에 `0271820 → dae0af4 → 2a20954 → 3bca09e → 5cb59cc` 5개 커밋이 들어왔고
(reward/ 패키지 + cap_bonus 분리 + events.py/trackers.py + early_close_neutral_capture_bonus + 활성화 0.007),
이를 doc 에 반영:
  - 7.3 [현] 카탈로그를 8 → 9 로 갱신, stateless 8 + event-consuming 1 분류 신설
  - 15.1 reward / event infra / logger 행 갱신 — events/trackers 파이프 활성화 + early_close 가 첫 소비자
  - Phase -1 의 "완료된 항목" 명시 (events 파이프 정착 + early_close 활성)
  - Phase 2.1 단기 catalog 에서 early_close 빼고 잔여 4개로 축소
```

이 변경은 **plan 의 야심을 줄이지 않으면서 phase 단위 회귀 위험을 제어** 하기 위한 것이다.
