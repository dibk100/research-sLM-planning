#!/usr/bin/env bash

# Usage:
# chmod +x scripts/run_all.sh
# ./scripts/run_all.sh

set -euo pipefail

echo "============================================================"
echo "Phase 2 Replanning Bottleneck Experiments"
echo "============================================================"

echo
echo "[1/3] Running Feedback-based Regeneration..."
PYTHONPATH=. python -m scripts.run_experiment \
  --config configs/feedback_regeneration.yaml

echo
echo "[2/3] Running Self-Replanning Regeneration..."
PYTHONPATH=. python -m scripts.run_experiment \
  --config configs/self_replan.yaml

echo
echo "[3/3] Running Teacher-Replanning Regeneration..."
PYTHONPATH=. python -m scripts.run_experiment \
  --config configs/teacher_replan.yaml

echo
echo "============================================================"
echo "All experiments completed."
echo "============================================================"