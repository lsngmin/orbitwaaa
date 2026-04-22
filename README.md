# Orbit Wars Agent

Kaggle Orbit Wars 대회 참가 에이전트.
Rule-based + GA 가중치 튜닝, PPO + League Training 구조로 구성.

## 게임 소개

100×100 2D 우주 맵에서 행성을 점령해 배(ships)를 가장 많이 모으는 실시간 전략 게임.

- **맵**: 100×100 연속 공간, 중앙 (50,50)에 태양 (반경 10)
- **행성**: 소유하면 매 턴 ships 생산 (production 1~5). 궤도 행성은 태양 주위를 회전
- **fleet**: 각도 지정해서 직선 발사. ships 수에 따라 속도가 달라짐 (1척=1/턴, ~1000척=6/턴)
- **전투**: 공격 ships > 방어 garrison 이면 점령. 동수면 전멸
- **코멧**: 턴 50/150/250/350/450에 타원 궤도로 등장, 맵 밖으로 나가면 탑승 ships 소멸
- **승리**: 500턴 종료 시 보유 ships 합계 (행성 + fleet) 최다 or 상대 전멸

## 구조

```
orbitwaaa/
├── main.py            # 제출용 진입점 (strategy.decide() 호출)
├── strategy.py        # 중앙 조율자 전략 (가중치 기반 target 선택)
├── prediction.py      # 궤도 예측, 태양 충돌 체크, fleet 속도 계산
├── config.yaml        # 모델/학습 하이퍼파라미터
│
├── model.py           # Hierarchical Transformer Policy (PPO용)
├── env_wrapper.py     # Gymnasium 환경 wrapper
├── train.py           # PPO + League Training 학습 루프
├── tune.py            # GA 가중치 자동 튜닝
│
├── utils/
│   ├── logger.py      # CSV 학습 로그
│   └── checkpoint.py  # 학습 재개용 체크포인트
│
├── checkpoints/       # 학습된 모델 저장
├── logs/              # 학습 로그 CSV
├── best_weights.json  # GA 최적 가중치
│
├── GAME_RULES.md      # 대회 공식 룰
├── agents.md          # 에이전트 작성 가이드
├── setup.sh           # 환경 자동 셋업
└── kaggle_init.sh     # Kaggle API 인증 설정
```

## 전략 구조

### Phase 1 — Rule-based + GA (현재 제출)

중앙 조율자가 전체 게임 상태를 보고 target을 선택:

```
score = W_PRODUCTION * production
      - W_SHIPS      * ships
      - W_DISTANCE   * distance
      + W_ENEMY      * is_enemy
      - comet_penalty
```

가중치는 유전 알고리즘(GA) 토너먼트로 자동 최적화:
- 64개체 × 20세대 병렬 자가대전
- 개체끼리 직접 대전해서 승점 평가
- 서버 64코어 병렬 처리

**prediction.py 주요 기능:**
- `aim()`: 궤도 행성 도착 시점 위치 수렴 계산 (10회 반복)
- `crosses_sun()`: 선분-원 교차 판정으로 태양 충돌 체크 (버퍼 반경 13)
- `fleet_speed()`: ships 수 기반 로그 스케일 속도 계산

### Phase 2 — PPO + League Training (학습 중)

#### 모델 구조

```
입력 토큰 (행성 + fleet, 과거 20턴)
    ↓
Temporal Attention (2층)  — 시간 패턴 학습
    ↓
Local Attention    (2층)  — fleet ↔ 행성 관계
    ↓
Global Attention   (4층)  — 전체 전략 판단
    ↓
Actor  — 행성별 발사여부 + ships비율 + 타겟 선택
Critic — 상태 가치 추정
```

| 하이퍼파라미터 | 값 |
|---|---|
| embed_dim | 128 |
| num_heads | 8 |
| temporal_window | 20턴 |
| learning_rate | 3e-4 |
| batch_size | 256 |

#### League Training

AlphaStar 방식의 self-play:

```
Main agent      — 지속 학습, self-play(50%) + league(50%)
Main exploiter  — Main 약점만 공략, 5세대마다 리셋
LeaguePool      — 과거 버전 5개 유지, 승률 55% 이상이면 추가
```

## 결과 기록

| 버전 | 날짜 | 전략 | 리더보드 점수 | 비고 |
|------|------|------|-------------|------|
| orbitwaaa-v1 | 2026-04-22 | Rule-based + GA | 대기 중 | 첫 제출 |

### GA 튜닝 결과 (2026-04-22)

```json
{
  "W_PRODUCTION": 4.632,
  "W_SHIPS":      0.444,
  "W_DISTANCE":   0.419,
  "W_ENEMY":      4.055
}
```
- 64개체 × 20세대 토너먼트
- 최종 best score: 1.000 (토너먼트 기준)

### PPO 학습 현황

| 날짜 | steps | p_loss | v_loss | league_size | 비고 |
|------|-------|--------|--------|-------------|------|
| 2026-04-22 | 학습 중 | - | - | 1 | A40 × 1 |

## 셋업

```bash
bash setup.sh        # Python 3.11 venv + 패키지 설치
bash kaggle_init.sh  # Kaggle API 인증
```

## 로컬 테스트

```bash
source .venv/bin/activate

# rule-based vs random
python -c "
from kaggle_environments import make
env = make('orbit_wars', debug=True)
env.run(['main.py', 'random'])
final = env.steps[-1]
print(f'Player 0: {final[0].reward}, Player 1: {final[1].reward}')
"

# Jupyter 시각화
jupyter notebook
```

## GA 가중치 튜닝

```bash
python tune.py
# → best_weights.json 저장
```

## PPO 학습

```bash
python train.py
# → checkpoints/main_latest.pt 저장
# → logs/train_*.csv 학습 로그
# 서버 재시작 시 자동 재개 (checkpoints/resume.pt)
```

## 제출

**Rule-based 버전:**
```bash
kaggle competitions submit orbit-wars -f main.py -m "orbitwaaa-v1"
```

**PPO 버전 (학습 완료 후):**
```bash
tar -czf submission.tar.gz \
  main.py model.py env_wrapper.py prediction.py \
  config.yaml checkpoints/main_final.pt
kaggle competitions submit orbit-wars -f submission.tar.gz -m "orbitwaaa-v2"
```

## 리더보드 확인

```bash
kaggle competitions leaderboard orbit-wars -s
kaggle competitions submissions orbit-wars
```
