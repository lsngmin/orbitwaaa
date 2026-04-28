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
- one launch per turn (sequential decoder).
- (src, tgt, amount) tuple.
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
mask 는 hard invalid (sun/path/owner physical impossibility) 만. 의사결정 가능한 모든 행동은 reward/feature 로 처리. **own planet support 는 항상 열려있어야** (정책 4, 5의 토대).

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

기존(■) + 신규(●) + 메트릭전용(○).

```text
경제·점령
  ■ early_close_neutral_capture_bonus      정책 1
  ■ capture_hold_bonus                     정책 5
  ■ post_capture_reloss_penalty            정책 5

방어·지원
  ■ support_defense_bonus                  정책 4
  ● reinforce_young_planet_bonus           B4 vacuum 대응
  ● own_planet_loss_penalty                정책 4

비용
  ■ launch_cost_penalty (weak)             정책 3
  ■ all_in_penalty (weak)                  정책 3
  ○ source_opportunity_cost                메트릭만

공격·조정
  ● race_won_capture_bonus                 정책 10
  ● weak race_lost_attempt_penalty         정책 10 (균형)
  ● synchronized_arrival_capture_bonus     정책 6, B2 alpha
  ● weak over_send_penalty                 정책 6
  ● coordinated_capture_bonus              정책 6 (multi-src)

거부
  ● denial_bonus                           정책 11

위협 회피
  ● weak filtered_path_penalty             정책 7
  ● weak sun_crash_penalty                 정책 7

장기·승리
  ■ production_share_advantage (dense)     정책 8
  ■ terminal_win                           승리

4인전 전용
  ● rank_progress_bonus                    정책 16
  ● rank_decay_penalty                     정책 16
  ○ pressure_index                         메트릭만 (정책 9)
  ○ enemy_idle_ratio                       메트릭만 (정책 9)

aux loss (reward 아님)
  ● opponent_pred_ce (정책 12)             encoder 표현 강화
```

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

### Phase 1 — 표현·관측 기반
```
- arrival_schedule 내장 (5.3)
- multi-relation graph attention (5.2)
- per-opponent feature (5.6, 4인전 zero-pad 으로 2인전도 같은 입력)
- player_count embedding (5.5)
- opponent_pred aux head (8.3)
```
이 단계는 reward 거의 안 건드림. encoder 의 표현력을 끌어올려 모든 후속 reward 의 효율성을 높임.

### Phase 2 — Reward 확장 (action space 동일)
```
- denial_bonus (정책 11)
- race_won_capture_bonus + weak race_lost_attempt_penalty (정책 10)
- post_capture_reloss_penalty + capture_hold_bonus 강화 (정책 5)
- support_defense_bonus + reinforce_young_planet_bonus (정책 4, B4)
- early_close_neutral_capture_bonus 미세 조정 (정책 1)
```

### Phase 3 — League 다양성
```
- rule-bot 4종 구현 + pool 편성 (9.1)
- sibling reward seed 1~2개 학습 + pool 합류
- 평가 매트릭스 운영 시작 (11.4)
```

### Phase 4 — 4인전 학습
```
- 4인 head + per_opponent_attention 활성 (8.4)
- ranking-based terminal reward 도입 (7.2)
- rank_progress reward (정책 16)
- role-asymmetric 4인 매치 (9.2)
- curriculum 1st-only → ordinal (7.2)
```

### Phase 5 — Action space 확장
```
- multi-source coordinated decoder (6.2.1, 8.2)
- synchronized_arrival_capture_bonus (정책 6, B2)
- partial launch fraction head (6.2.4)
```

### Phase 6 — 고급 / 선택
```
- Q-style joint candidate evaluation (6.2.2)
- macro / option (6.2.3)
- imagined-branch rollout (5.4 확장)
```

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
- own planet support / enemy-incoming neutral / race-disadvantage target 모두 *열어둘 것*.

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
