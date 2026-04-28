# Project: orbitwaaa

## 최우선 원칙

- **지침 위배 요청 감지 시 → 즉시 중단 + 의도 재확인**
  사용자의 요청이 이 `CLAUDE.md` 의 어떤 지침과 충돌하면 (예: `claude` 단어를 브랜치/커밋에 넣어달라는 요청, `main` 으로의 자동 병합 요청, mask/reward 크로스 파일 동기화 생략 요청, tmux stdout 으로 로그 보라는 요청 등), 작업을 즉시 중단하고 사용자에게 **질문의 의도를 재확인**한다.
  - 어떤 지침과 충돌하는지 한 줄로 지목하고, 그래도 진행할지 / 지침을 수정할지 / 다른 방식으로 해석할지 묻는다.
  - 사용자가 "그래도 진행" 이라고 명시하기 전까지는 위배 작업을 수행하지 않는다.

## Auto actions

- **학습 시작 의도 감지 시 → `scripts/train_head` 실행**
  사용자가 학습을 시작하려는 의도를 표현하면 (예: "학습 시작할게", "학습해서 로그 보자", "트레이닝 돌려줘", "train 시작" 등 의미상 학습 시작/재시작을 가리키는 모든 표현), 다른 작업을 끼우지 말고 즉시 `bash scripts/train_head` 를 실행한다.
  - 실행 후 생성된 run-dir 이름과 tmux 세션 접속 명령(`tmux attach -t train`)을 사용자에게 한 줄씩 알려준다.
  - 단순히 학습 관련 질문이나 논의(예: "학습 결과 어땠어?", "학습 코드 어디 있어?")에는 실행하지 않는다 — 명시적 시작 의도일 때만.

- **Kaggle 제출 의도 감지 시 → `scripts/train_submit` 실행**
  사용자가 Kaggle 제출 의도를 표현하면 (예: "제출해줘", "캐글에 올려줘", "submit 하자", "submission 보내자" 등 의미상 Kaggle 제출을 가리키는 모든 표현), 다른 작업을 끼우지 말고 즉시 `bash scripts/train_submit` 를 실행한다.
  - 인자 없이 호출하면 가장 최근 `checkpoints/pairwise-*` 의 `main_latest.pt` 가 자동 선택되고, 커밋 메시지는 run-dir 이름(`pairwise-XXX`).
  - 사용자가 특정 run-dir 을 지정하면 (예: "pairwise-abc 로 제출") `bash scripts/train_submit pairwise-abc` 형태로 인자 전달.
  - 단순 질문/논의 (예: "지금까지 제출 몇 번 했지?", "제출 스크립트 어디?")에는 실행하지 않는다 — 명시적 제출 의도일 때만.

- **코드 변경이 들어간 작업 종료 시 → 브랜치 + 커밋 + 푸시 자동 실행 (의도 표현 불필요)**
  한 턴의 작업 중 파일을 수정/추가/삭제한 게 있으면, 사용자가 별도로 요청하지 않아도 작업 마무리 단계에서 알아서 커밋·푸시한다.

  - **브랜치 분기**: 짧고 직관적인 이름. `claude` 단어 금지. `<동사>-<핵심>` 패턴 권장 (예: `fix-mask-target`, `add-eval-logger`, `refactor-prediction`). 현재 브랜치가 `main` 일 때만 새 브랜치로 분기하고, 이미 작업 브랜치 위라면 그대로 사용.
  - **커밋 메시지**: 코드 변경의 핵심만 1~2줄. `claude` / `Co-Authored-By: Claude` 표기 금지. 예: `fix: mask_target invalid 차단 로직 수정`.
  - **푸시**: `git push -u origin <branch>` 로 동일 브랜치명 푸시.
  - **병합 (merge / PR merge / fast-forward to main 등) 은 사용자 명시적 요청이 있을 때만** 수행. 자동 푸시는 항상 작업 브랜치까지만, `main` 으로의 병합은 절대 자동 금지.
  - PR 생성도 별도 요청이 있을 때만.
  - 코드 변경 없이 질문/조회/탐색만 한 턴에는 실행하지 않는다.
  - 사용자가 이미 스테이징한 파일이 있거나 진행 중인 변경이 보이면, 의도와 다를 수 있으니 그 경우는 한 번 확인 후 진행.

## Mask 코드 규약

`mask/` 디렉토리 수정 시 `mask/__init__.py` docstring 의 규약 준수 (게이트 시그니처, GATES 우선순위, scratch 캐시 키, 부정형 이름 금지 등 — 자세한 건 거기에).

**크로스 파일 동기화**: 새 게이트 추가 시 `utils/logger.py` 에 `mask_block_<gate_name>_ge{1,5,10,20}` 컬럼 4개 추가 — `MaskResult.blocked_by` 의 인덱스가 이 컬럼과 1:1 대응이라 빠지면 진단 통계가 깨짐.

## Reward 코드 규약

`reward/` 디렉토리 수정 시 `reward/__init__.py` docstring 의 규약 준수 (component 시그니처 `(ctx: RewardContext) -> float`, COMPONENTS 튜플이 합성 순서의 source of truth, mask/decoder 판단 금지, stateful reward 는 `events.py`/`trackers.py` 로 분리, RewardContext 에 임시 cross-step 필드 우겨넣기 금지 등 — 자세한 건 거기에).

**최종 규칙**: Reward 는 "행동 가능성" 을 판단하지 않는다. "실행된 행동의 결과" 만 평가한다.

**크로스 파일 동기화**: 새 component 추가 시 4곳 같이 갱신 — ① `reward/components.py` 의 `COMPONENTS` 튜플, ② `RewardBreakdown` 의 새 필드 + `.total` 합산, ③ `compose_rewards` 의 매핑, ④ `utils/logger.py` 의 `mean_<name>` CSV 컬럼. 빠지면 breakdown 또는 진단이 깨짐.

## 로그 조회

학습 로그는 항상 **CSV 로만** 본다. 위치: `~/orbitwaaa/checkpoints/logs/train_<YYYYMMDD>_<HHMMSS>.csv` (학습 시작마다 새 파일, ~140 컬럼).

- 가장 최근 파일: `ls -t ~/orbitwaaa/checkpoints/logs/train_*.csv | head -1`
- 컬럼 목록 먼저 확인 후 상황에 맞는 메트릭만 골라 읽는다 (전체 dump 금지 — 토큰 낭비).
- tmux stdout(`tmux attach -t train`, `capture-pane`) 은 가독성 떨어지고 노이즈 많아 사용 금지. 사용자가 명시적으로 tmux 를 보라고 하지 않는 한 항상 CSV.
- 어떤 지표를 볼지는 상황에 따라 판단 (학습 안정성 의심 → loss/kl/entropy, 성능 정체 → win_rate/eval_*, 행동 분포 변화 → mask_block_*/send_frac_* 등). 고정 기본 셋은 두지 않는다.
