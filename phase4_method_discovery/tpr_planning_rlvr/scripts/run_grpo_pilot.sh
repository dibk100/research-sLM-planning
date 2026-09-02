#!/usr/bin/env bash
# chmod +x phase4_method_discovery/tpr_planning_rlvr/scripts/run_grpo_pilot.sh
# bash phase4_method_discovery/tpr_planning_rlvr/scripts/run_grpo_pilot.sh

set -euo pipefail

# ==============================================================================
# TPR Planning-RLVR
# verl GRPO Pilot / Smoke Test
# ==============================================================================

# ------------------------------------------------------------------------------
# Fixed paths
# ------------------------------------------------------------------------------

PROJECT_ROOT="${HOME}/workspace/project_sLM_planning"
VERL_ROOT="${HOME}/workspace/verl"

CONDA_ENV="/mnt/hdd/conda_envs/planning_rlvr"

TPR_ROOT="${PROJECT_ROOT}/phase4_method_discovery/tpr_planning_rlvr"
VANILLA_ROOT="${PROJECT_ROOT}/phase4_method_discovery/vanilla_planning_rlvr"

SMOKE_CONFIG_DIR="${TPR_ROOT}/configs"
SMOKE_CONFIG_NAME="verl_grpo_pilot_50step"

TRAIN_PARQUET="/mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/vanilla_planning_rlvr/train.parquet"
VAL_PARQUET="/mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/vanilla_planning_rlvr/val.parquet"

REWARD_FILE="${TPR_ROOT}/reward/planning_tpr_reward.py"
REWARD_MANAGER_FILE="${TPR_ROOT}/reward/planning_tpr_reward_manager.py"

# Reuse the already validated frozen coder implementation.
# Vanilla baseline itself remains frozen.
FROZEN_CODER_FILE="${VANILLA_ROOT}/workers/frozen_coder_worker.py"

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

section "TPR Planning-RLVR GRPO Pilot / Smoke Test"


if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
    die "Python not found in training environment: ${CONDA_ENV}"
fi


PYTHON_BIN="${CONDA_ENV}/bin/python"


echo "project root       : ${PROJECT_ROOT}"
echo "verl root          : ${VERL_ROOT}"
echo "python             : ${PYTHON_BIN}"
echo "pilot config       : ${SMOKE_CONFIG_DIR}/${SMOKE_CONFIG_NAME}.yaml"
echo "train parquet      : ${TRAIN_PARQUET}"
echo "val parquet        : ${VAL_PARQUET}"
echo "reward             : ${REWARD_FILE}"
echo "reward manager     : ${REWARD_MANAGER_FILE}"
echo "frozen coder       : ${FROZEN_CODER_FILE}"


# ==============================================================================
# 2. File validation
# ==============================================================================

section "Validate Required Files"


require_file "${SMOKE_CONFIG_DIR}/${SMOKE_CONFIG_NAME}.yaml"

require_file "${TRAIN_PARQUET}"
require_file "${VAL_PARQUET}"

require_file "${REWARD_FILE}"
require_file "${REWARD_MANAGER_FILE}"
require_file "${FROZEN_CODER_FILE}"

require_file "${VERL_ROOT}/verl/trainer/main_ppo_sync.py"
require_file "${VERL_ROOT}/verl/trainer/config/ppo_trainer.yaml"
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

import ray
import tensordict
import torch
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

section "Validate TPR Planning-RLVR Imports"


export PYTHONPATH="${PROJECT_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"


"${PYTHON_BIN}" - <<'PY'
from verl import DataProto

from phase4_method_discovery.tpr_planning_rlvr.reward.planning_tpr_reward import (
    compute_score,
)
from phase4_method_discovery.tpr_planning_rlvr.reward.planning_tpr_reward_manager import (
    PlanningTPRRewardManager,
)
from phase4_method_discovery.vanilla_planning_rlvr.workers.frozen_coder_worker import (
    FrozenCoderWorker,
)

print("[OK] verl.DataProto")
print("[OK] PlanningTPRRewardManager")
print("[OK] FrozenCoderWorker")
print("[OK] planning_tpr_reward.compute_score")
PY


# ==============================================================================
# 6. Compile relevant Python files
# ==============================================================================

section "Compile Relevant Python Files"


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

# Full Hydra traceback is useful during the integration smoke test.
export HYDRA_FULL_ERROR=1

# Make Python/Ray logs immediately visible.
export PYTHONUNBUFFERED=1

# vLLM sleep mode is incompatible with expandable_segments.
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF

export CUDA_HOME="${CONDA_ENV}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# RTX 5090 setup validated with FlashInfer.
export VLLM_ATTENTION_BACKEND=FLASHINFER


echo "HF_HOME                 : ${HF_HOME}"
echo "PYTHONPATH              : ${PYTHONPATH}"
echo "TOKENIZERS_PARALLELISM  : ${TOKENIZERS_PARALLELISM}"
echo "HYDRA_FULL_ERROR        : ${HYDRA_FULL_ERROR}"
echo "VLLM_ATTENTION_BACKEND  : ${VLLM_ATTENTION_BACKEND}"


# ==============================================================================
# 9. GPU status before training
# ==============================================================================

section "GPU Status Before Training"


nvidia-smi


# ==============================================================================
# 10. Launch TPR Planning-RLVR
# ==============================================================================

section "Launch TPR Planning-RLVR GRPO Pilot / Smoke Test"


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
echo "    -> PlanningTPRRewardManager"
echo "    -> FrozenCoderWorker RPC"
echo "    -> generated code"
echo "    -> selected TACO reward tests"
echo "    -> non-fail-fast execution"
echo "    -> TPR reward [0, 1]"
echo "    -> GRPO advantage"
echo "    -> actor update"
echo
echo "Starting..."
echo


TRAIN_LOG="${TPR_ROOT}/outputs/training_smoke_$(date +%Y%m%d_%H%M%S).log"

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

section "TPR Planning-RLVR Pilot / Smoke Test Finished"


echo "[PASS] verl main_ppo_sync.py exited successfully."
echo
echo "Training log:"
echo "  ${TRAIN_LOG}"
echo
echo "Verify the logs contain:"
echo "  - actor/rollout initialization"
echo "  - frozen coder initialization"
echo "  - reward-loop initialization"
echo "  - PlanningTPRRewardManager"
echo "  - planner rollout x16"
echo "  - fractional TPR rewards when partial tests pass"
echo "  - GRPO advantage"
echo "  - actor update"
echo
echo "For the current 1-step smoke configuration:"
echo "  - global step 1 should complete"
echo "  - checkpoint creation is not required"