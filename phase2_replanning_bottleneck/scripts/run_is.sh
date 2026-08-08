#!/usr/bin/env bash

# Usage:
# chmod +x scripts/run_is.sh
# ./scripts/run_is.sh

set -euo pipefail

echo "============================================================"
echo "Phase 2 Replanning Bottleneck Experiments"
echo "============================================================"

echo
echo "[1/2] Running Feedback-based Regeneration..."
PYTHONPATH=. python -m scripts.run_experiment \
  --config configs/feedback_regeneration.yaml

echo
echo "[2/2] Running Self-Replanning Regeneration..."
PYTHONPATH=. python -m scripts.run_experiment \
  --config configs/self_replan.yaml

echo
echo "============================================================"
echo "All experiments completed."
echo "============================================================"