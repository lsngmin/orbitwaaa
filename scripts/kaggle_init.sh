#!/bin/bash
set -e

echo "=== Kaggle 인증 설정 ==="

read -p "Kaggle username: " KAGGLE_USERNAME
read -p "Kaggle API key: " KAGGLE_KEY

mkdir -p ~/.kaggle
echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

echo "=== 인증 확인 ==="
kaggle competitions list --group entered 2>&1 | grep -E "orbit-wars|Unauthorized" || true

echo ""
echo "orbit-wars 가 목록에 있으면 성공."
echo "없으면 https://www.kaggle.com/competitions/orbit-wars/rules 에서 규칙 동의 필요."
