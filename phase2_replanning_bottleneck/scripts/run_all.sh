#!/usr/bin/env bash
# Phase 2 세 전략을 순차 실행한다.
#
# 세 전략은 동일한 Phase 1 실패 trajectory에서 출발하므로
# 실행 순서는 결과에 영향을 주지 않는다.
#
# Usage:
#   bash scripts/run_all.sh              # config의 전체 대상
#   bash scripts/run_all.sh 10           # pilot: 앞 10건만
#
# 사전 조건:
#   - conda activate slm  (혹은 /mnt/hdd/conda_envs/slm/bin/python 사용)
#   - PYTHONPATH=. 로 실행
#   - teacher_replan 은 teacher replan JSONL이 먼저 준비되어 있어야 한다
#     (scripts/build_teacher_replans.py 참고)

set -euo pipefail

LIMIT="${1:-}"
LIMIT_ARG=""
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARG="--limit ${LIMIT}"
fi

export PYTHONPATH=.

for STRATEGY in feedback_repair self_replan teacher_replan; do
  echo "================================================================"
  echo "[RUN] ${STRATEGY}"
  echo "================================================================"

  python -m scripts.run_experiment \
    --config "configs/${STRATEGY}.yaml" \
    ${LIMIT_ARG}
done

echo "[DONE] all phase2 strategies finished."
