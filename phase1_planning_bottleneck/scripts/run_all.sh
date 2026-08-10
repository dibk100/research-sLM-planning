#!/usr/bin/env bash
# Usage:
# chmod +x scripts/run_all.sh
# ./scripts/run_all.sh

set -euo pipefail

echo "============================================================"
echo "Phase1 Planning Bottleneck (500 problems)"
echo "============================================================"

echo
echo "[1/3] Running Direct..."
python -m scripts.run_direct \
    --config configs/direct.yaml

echo
echo "[2/3] Running Self-Plan..."
python -m scripts.run_self_plan \
    --config configs/self_plan.yaml

echo
echo "[3/3] Running Teacher-Plan..."
python -m scripts.run_teacher_plan \
    --config configs/teacher_plan.yaml

echo
echo "============================================================"
echo "All experiments completed."
echo "============================================================"