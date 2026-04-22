#!/bin/bash
set -e

echo "=== Python 버전 확인 ==="
PYTHON=$(command -v python3.11 || command -v python3 || command -v python)
$PYTHON --version
echo "사용할 Python: $PYTHON"

echo "=== 가상환경 생성 ==="
$PYTHON -m venv .venv
source .venv/bin/activate

echo "=== pip 업그레이드 ==="
pip install --upgrade pip

echo "=== 패키지 설치 ==="
pip install kaggle-environments kagglehub kaggle jupyter

echo "=== kaggle-environments 소스 설치 ==="
if [ ! -d "$HOME/kaggle-environments" ]; then
    git clone https://github.com/Kaggle/kaggle-environments.git "$HOME/kaggle-environments"
fi
pip install -e "$HOME/kaggle-environments" --no-deps

echo "=== 환경 확인 ==="
python -c "from kaggle_environments import make; env = make('orbit_wars'); print('orbit_wars 환경 OK')"

echo ""
echo "완료. 이후 실행:"
echo "  source .venv/bin/activate"
