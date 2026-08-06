#!/usr/bin/env bash

set -euo pipefail

echo "============================================================"
echo "Phase1 Planning Bottleneck (50 problems)"
echo "============================================================"

echo
echo "[1/2] Running Direct..."
python -m scripts.run_direct --config configs/direct.yaml

echo
echo "[2/2] Running Self-Plan..."
python -m scripts.run_self_plan --config configs/self_plan.yaml

echo
echo "============================================================"
echo "All experiments completed."
echo "============================================================"
