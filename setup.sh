#!/bin/bash
set -e

echo "=== Python 버전 확인 ==="
python3.11 --version || { echo "Python 3.11 필요. brew install python@3.11"; exit 1; }

echo "=== 가상환경 생성 ==="
python3.11 -m venv .venv
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
