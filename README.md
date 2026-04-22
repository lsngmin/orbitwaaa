# Orbit Wars Agent

Kaggle Orbit Wars 대회 참가 에이전트.

## 전략

- TODO

## 구조

```
.
├── main.py          # 제출용 에이전트
├── GAME_RULES.md    # 대회 공식 룰
└── agents.md        # 에이전트 작성 가이드
```

## 실행

```bash
source .venv/bin/activate
python -c "
from kaggle_environments import make
env = make('orbit_wars', debug=True)
env.run(['main.py', 'random'])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"
```

## 제출

```bash
kaggle competitions submit orbit-wars -f main.py -m "v1"
```
