#!/usr/bin/env bash
# chmod +x phase4_method_discovery/vanilla_planning_rlvr/scripts/run_grpo_pilot.sh
# bash phase4_method_discovery/vanilla_planning_rlvr/scripts/run_grpo_pilot.sh

set -euo pipefail

# ==============================================================================
# Vanilla Planning-RLVR
# verl GRPO Pilot test
# ==============================================================================

# ------------------------------------------------------------------------------
# Fixed paths
# ------------------------------------------------------------------------------

PROJECT_ROOT="${HOME}/workspace/project_sLM_planning"
VERL_ROOT="${HOME}/workspace/verl"

CONDA_ENV="/mnt/hdd/conda_envs/planning_rlvr"

SMOKE_CONFIG_DIR="${PROJECT_ROOT}/phase4_method_discovery/vanilla_planning_rlvr/configs"
SMOKE_CONFIG_NAME="verl_grpo_pilot_50step"

RESEARCH_CONFIG="${SMOKE_CONFIG_DIR}/vanilla_planning_rlvr_qwen25coder3b.yaml"

TRAIN_PARQUET="/mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/vanilla_planning_rlvr/train.parquet"
VAL_PARQUET="/mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/vanilla_planning_rlvr/val.parquet"

REWARD_FILE="${PROJECT_ROOT}/phase4_method_discovery/vanilla_planning_rlvr/reward/planning_execution_reward.py"
REWARD_MANAGER_FILE="${PROJECT_ROOT}/phase4_method_discovery/vanilla_planning_rlvr/reward/planning_reward_manager.py"
FROZEN_CODER_FILE="${PROJECT_ROOT}/phase4_method_discovery/vanilla_planning_rlvr/workers/frozen_coder_worker.py"

EXPECTED_VERL_BRANCH="release/v0.8.0"
EXPECTED_VERL_COMMIT="3e4edb6e3d6872ad8aa21af83e98ee6e1bea19a2"


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

section() {
    echo
    echo "=========================================================================================="
    echo "$1"
    echo "=========================================================================================="
}


die() {
    echo
    echo "[ERROR] $1" >&2
    exit 1
}


require_file() {
    local path="$1"

    if [[ ! -f "${path}" ]]; then
        die "Required file not found: ${path}"
    fi
}


# ==============================================================================
# 1. Environment validation
# ==============================================================================

section "Vanilla Planning-RLVR GRPO Pilot Test"


if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
    die "Python not found in training environment: ${CONDA_ENV}"
fi


PYTHON_BIN="${CONDA_ENV}/bin/python"


echo "project root       : ${PROJECT_ROOT}"
echo "verl root          : ${VERL_ROOT}"
echo "python             : ${PYTHON_BIN}"
echo "pilot config       : ${SMOKE_CONFIG_DIR}/${SMOKE_CONFIG_NAME}.yaml"
echo "research config    : ${RESEARCH_CONFIG}"
echo "train parquet      : ${TRAIN_PARQUET}"
echo "val parquet        : ${VAL_PARQUET}"


# ==============================================================================
# 2. File validation
# ==============================================================================

section "Validate Required Files"


require_file "${SMOKE_CONFIG_DIR}/${SMOKE_CONFIG_NAME}.yaml"
require_file "${RESEARCH_CONFIG}"

require_file "${TRAIN_PARQUET}"
require_file "${VAL_PARQUET}"

require_file "${REWARD_FILE}"
require_file "${REWARD_MANAGER_FILE}"
require_file "${FROZEN_CODER_FILE}"

require_file "${VERL_ROOT}/verl/trainer/main_ppo_sync.py"
require_file "${VERL_ROOT}/verl/experimental/reward_loop/reward_loop.py"


echo "[OK] required files exist."


# ==============================================================================
# 3. verl revision validation
# ==============================================================================

section "Validate verl Revision"


cd "${VERL_ROOT}"


CURRENT_BRANCH="$(git branch --show-current)"
CURRENT_COMMIT="$(git rev-parse HEAD)"


echo "branch             : ${CURRENT_BRANCH}"
echo "commit             : ${CURRENT_COMMIT}"


if [[ "${CURRENT_BRANCH}" != "${EXPECTED_VERL_BRANCH}" ]]; then
    echo "[WARN] Expected branch '${EXPECTED_VERL_BRANCH}', got '${CURRENT_BRANCH}'."
fi


if [[ "${CURRENT_COMMIT}" != "${EXPECTED_VERL_COMMIT}" ]]; then
    echo "[WARN] Expected base commit:"
    echo "       ${EXPECTED_VERL_COMMIT}"
    echo "       current:"
    echo "       ${CURRENT_COMMIT}"
    echo
    echo "This is acceptable if the difference is only the local Planning-RLVR patch."
fi


echo
echo "git status:"
git status --short


# ==============================================================================
# 4. Python/runtime validation
# ==============================================================================

section "Validate Python Runtime"


"${PYTHON_BIN}" - <<'PY'
import sys

import torch
import ray
import tensordict
import transformers
import vllm

print("python            :", sys.executable)

print("torch             :", torch.__version__)
print("torch cuda        :", torch.version.cuda)
print("cuda available    :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("gpu               :", torch.cuda.get_device_name(0))
print(
    "gpu memory GB     :",
    round(
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3),
        2,
    ),
)

print("ray               :", ray.__version__)
print("tensordict        :", tensordict.__version__)
print("transformers      :", transformers.__version__)
print("vllm              :", vllm.__version__)
PY


# ==============================================================================
# 5. Import validation
# ==============================================================================

section "Validate Planning-RLVR Imports"


export PYTHONPATH="${PROJECT_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"


"${PYTHON_BIN}" - <<'PY'
from verl import DataProto

from phase4_method_discovery.vanilla_planning_rlvr.reward.planning_execution_reward import (
    compute_score,
)
from phase4_method_discovery.vanilla_planning_rlvr.reward.planning_reward_manager import (
    PlanningRewardManager,
)
from phase4_method_discovery.vanilla_planning_rlvr.workers.frozen_coder_worker import (
    FrozenCoderWorker,
)

print("[OK] verl.DataProto")
print("[OK] PlanningRewardManager")
print("[OK] FrozenCoderWorker")
print("[OK] planning_execution_reward.compute_score")
PY


# ==============================================================================
# 6. Compile modified files
# ==============================================================================

section "Compile Modified Python Files"


"${PYTHON_BIN}" -m py_compile \
    "${VERL_ROOT}/verl/trainer/main_ppo_sync.py" \
    "${VERL_ROOT}/verl/experimental/reward_loop/reward_loop.py" \
    "${REWARD_FILE}" \
    "${REWARD_MANAGER_FILE}" \
    "${FROZEN_CODER_FILE}"


echo "[OK] py_compile passed."


# ==============================================================================
# 7. Clear stale Ray state
# ==============================================================================

section "Clear Stale Ray Runtime"


"${PYTHON_BIN}" -m ray stop --force >/dev/null 2>&1 || true


echo "[OK] stale Ray processes cleared."


# ==============================================================================
# 8. Runtime environment
# ==============================================================================

section "Configure Runtime Environment"


# Prefer HF_HOME over deprecated TRANSFORMERS_CACHE.
if [[ -z "${HF_HOME:-}" ]]; then
    export HF_HOME="/mnt/hdd/hf_cache"
fi

# Avoid tokenizer fork noise.
export TOKENIZERS_PARALLELISM=false
# Useful while debugging the 1-step integration test.
export HYDRA_FULL_ERROR=1
# Make Python logs immediately visible.
export PYTHONUNBUFFERED=1

unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF

export CUDA_HOME=/mnt/hdd/conda_envs/planning_rlvr
export PATH="${CUDA_HOME}/bin:${PATH}"

export VLLM_ATTENTION_BACKEND=FLASHINFER


echo "HF_HOME            : ${HF_HOME}"
echo "PYTHONPATH         : ${PYTHONPATH}"
echo "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM}"
echo "HYDRA_FULL_ERROR   : ${HYDRA_FULL_ERROR}"
echo "VLLM_ATTENTION_BACKEND : ${VLLM_ATTENTION_BACKEND}"


# ==============================================================================
# 9. Memory monitoring
# ==============================================================================

# section "Start Memory Monitor"

# MEM_LOG="${PROJECT_ROOT}/phase4_method_discovery/vanilla_planning_rlvr/outputs/memory_pilot_$(date +%Y%m%d_%H%M%S).log"

# mkdir -p "$(dirname "${MEM_LOG}")"

# echo "[INFO] Memory log: ${MEM_LOG}"

# (
#     # Do not let monitoring commands terminate this loop under
#     # the parent shell's `set -euo pipefail`.
#     set +e

#     # Heavy mapping snapshots are throttled to at most once per second.
#     LAST_HIGH_SNAPSHOT=0

#     while true; do
#         echo "================================================================================"
#         date '+%Y-%m-%d %H:%M:%S.%3N'

#         # ----------------------------------------------------------------------
#         # System memory
#         # ----------------------------------------------------------------------

#         echo "[MEM]"
#         free -h

#         # ----------------------------------------------------------------------
#         # Top RSS processes
#         # ----------------------------------------------------------------------

#         echo
#         echo "[TOP RSS]"

#         ps -eo pid,ppid,rss,vsz,%mem,comm,args --sort=-rss \
#             | head -n 25 \
#             || true

#         # ----------------------------------------------------------------------
#         # GPU process memory
#         # ----------------------------------------------------------------------

#         echo
#         echo "[GPU]"

#         nvidia-smi \
#             --query-compute-apps=pid,process_name,used_memory \
#             --format=csv,noheader \
#             || true

#         # ----------------------------------------------------------------------
#         # Detailed memory accounting for top RSS processes
#         #
#         # This runs every 0.2 sec and is relatively lightweight compared with
#         # dumping full process mappings.
#         # ----------------------------------------------------------------------

#         echo
#         echo "[TOP PROCESS MEMORY DETAIL]"

#         ps -eo pid=,rss=,args= --sort=-rss \
#             | head -n 15 \
#             | while read -r PID RSS ARGS; do

#                 if [[ -z "${PID}" || ! -r "/proc/${PID}/status" ]]; then
#                     continue
#                 fi

#                 echo "------------------------------------------------------------"
#                 echo "PID=${PID} RSS_KB=${RSS}"
#                 echo "ARGS=${ARGS}"

#                 echo "[status]"

#                 grep -E \
#                     'VmPeak|VmSize|VmRSS|RssAnon|RssFile|RssShmem|VmSwap' \
#                     "/proc/${PID}/status" \
#                     || true

#                 if [[ -r "/proc/${PID}/smaps_rollup" ]]; then
#                     echo "[smaps_rollup]"

#                     grep -E \
#                         '^(Rss|Pss|Pss_Anon|Pss_File|Pss_Shmem|Anonymous|Swap):' \
#                         "/proc/${PID}/smaps_rollup" \
#                         || true
#                 fi
#             done

#         # ----------------------------------------------------------------------
#         # Heavy mapping snapshot near memory pressure only
#         #
#         # The previous OOMs occurred around 59-60 GiB node usage.
#         # Start collecting detailed mappings once used memory reaches 50 GiB.
#         #
#         # `pmap` and /proc/<pid>/maps inspection are throttled to once per second
#         # so the monitor itself does not materially increase memory/CPU pressure.
#         # ----------------------------------------------------------------------

#         MEM_USED_GIB="$(
#             free -b \
#                 | awk '/^Mem:/ {printf "%.0f", $3 / 1024 / 1024 / 1024}'
#         )"

#         NOW_SEC="$(date +%s)"

#         if [[ "${MEM_USED_GIB}" -ge 50 ]] && \
#            (( NOW_SEC - LAST_HIGH_SNAPSHOT >= 1 )); then

#             LAST_HIGH_SNAPSHOT="${NOW_SEC}"

#             echo
#             echo "[HIGH MEMORY MAPPING SNAPSHOT] used=${MEM_USED_GIB}GiB"

#             for TARGET in "ray::WorkerDict" "VLLM::Worker"; do

#                 # Search the process command line rather than relying on `comm`,
#                 # because Ray process titles can be truncated in `comm`.
#                 PID="$(
#                     ps -eo pid=,args= \
#                         | awk -v target="${TARGET}" \
#                             'index($0, target) {print $1; exit}'
#                 )"

#                 if [[ -z "${PID}" ]]; then
#                     echo "${TARGET}: not found"
#                     continue
#                 fi

#                 if [[ ! -r "/proc/${PID}/status" ]]; then
#                     echo "${TARGET}: PID=${PID} disappeared"
#                     continue
#                 fi

#                 echo
#                 echo "============================================================"
#                 echo "TARGET=${TARGET}"
#                 echo "PID=${PID}"

#                 echo
#                 echo "[status]"

#                 grep -E \
#                     'VmPeak|VmSize|VmRSS|RssAnon|RssFile|RssShmem|VmSwap' \
#                     "/proc/${PID}/status" \
#                     || true

#                 if [[ -r "/proc/${PID}/smaps_rollup" ]]; then
#                     echo
#                     echo "[smaps_rollup]"

#                     grep -E \
#                         '^(Rss|Pss|Pss_Anon|Pss_File|Pss_Shmem|Anonymous|Swap):' \
#                         "/proc/${PID}/smaps_rollup" \
#                         || true
#                 fi

#                 if [[ -r "/proc/${PID}/maps" ]]; then
#                     echo
#                     echo "[interesting mappings]"

#                     grep -Ei \
#                         'shm|memfd|cuda|nvidia|ipc|deleted' \
#                         "/proc/${PID}/maps" \
#                         | tail -n 150 \
#                         || true
#                 fi

#                 echo
#                 echo "[pmap largest mappings]"

#                 pmap -x "${PID}" 2>/dev/null \
#                     | awk '
#                         NR > 2 && $3 ~ /^[0-9]+$/ {
#                             print
#                         }
#                     ' \
#                     | sort -k3 -nr \
#                     | head -n 40 \
#                     || true
#             done
#         fi

#         echo
#         sleep 0.2
#     done

# ) >> "${MEM_LOG}" 2>&1 &

# MEM_MONITOR_PID=$!

# ==============================================================================
# Memory monitor cleanup
# ==============================================================================

# cleanup() {
#     if kill -0 "${MEM_MONITOR_PID}" 2>/dev/null; then
#         echo "[INFO] Stopping memory monitor PID=${MEM_MONITOR_PID}"
#         kill "${MEM_MONITOR_PID}" 2>/dev/null || true
#         wait "${MEM_MONITOR_PID}" 2>/dev/null || true
#     fi

#     echo "[INFO] Memory log saved to: ${MEM_LOG}"
# }

# trap cleanup EXIT INT TERM

# echo "[OK] memory monitor started."
# echo "PID                 : ${MEM_MONITOR_PID}"
# echo "log                 : ${MEM_LOG}"

# ==============================================================================
# 9. GPU status before training
# ==============================================================================

section "GPU Status Before Training"


nvidia-smi


# ==============================================================================
# 10. Launch 50-step GRPO pilot
# ==============================================================================

section "Launch verl GRPO 50-Step Pilot Test"

cd "${VERL_ROOT}"

echo "Entry point:"
echo "  verl.trainer.main_ppo_sync"
echo
echo "Hydra config:"
echo "  ${SMOKE_CONFIG_DIR}/${SMOKE_CONFIG_NAME}.yaml"
echo
echo "verl base config:"
echo "  ${VERL_ROOT}/verl/trainer/config/ppo_trainer.yaml"
echo
echo "Expected trajectory:"
echo "  prompt"
echo "    -> planner rollout x16"
echo "    -> PlanningRewardManager"
echo "    -> FrozenCoderWorker RPC"
echo "    -> TACO execution"
echo "    -> binary reward"
echo "    -> GRPO advantage"
echo "    -> actor update"
echo
echo "Starting..."
echo

TRAIN_LOG="${PROJECT_ROOT}/phase4_method_discovery/vanilla_planning_rlvr/outputs/training_pilot50_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$(dirname "${TRAIN_LOG}")"

echo "[INFO] Training log: ${TRAIN_LOG}"

"${PYTHON_BIN}" -m verl.trainer.main_ppo_sync \
    --config-path "${SMOKE_CONFIG_DIR}" \
    --config-name "${SMOKE_CONFIG_NAME}" \
    "hydra.searchpath=[file://${VERL_ROOT}/verl/trainer/config]" \
    2>&1 | tee "${TRAIN_LOG}"

# ==============================================================================
# 11. Success
# ==============================================================================

section "GRPO Pilot Test Finished"


echo "[PASS] verl main_ppo_sync.py exited successfully."
echo
echo "Verify the logs contain:"
echo "  - actor/rollout initialization"
echo "  - frozen coder initialization"
echo "  - reward-loop initialization"
echo "  - planner rollout"
echo "  - execution rewards"
echo "  - GRPO advantage / actor update"
echo "  - global step 50 completion"
echo "  - checkpoint at step 25"
echo "  - checkpoint at step 50"