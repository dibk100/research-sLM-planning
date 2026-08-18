#!/usr/bin/env bash

# Usage examples:
#
# chmod +x phase1_planning_bottleneck/scripts/run_all.sh
#
# ./phase1_planning_bottleneck/scripts/run_all.sh \
#   direct self_plan teacher_plan \
#   --config-name qwen25Coder3b
#
# ./phase1_planning_bottleneck/scripts/run_all.sh \
#   teacher_plan \
#   --config-name qwen253b
#
# ./phase1_planning_bottleneck/scripts/run_all.sh \
#   direct self_plan teacher_plan \
#   --config-name phi3
#
# ./phase1_planning_bottleneck/scripts/run_all.sh \
#   direct self_plan \
#   --config-name qwen253b \
#   --limit 10


set -euo pipefail


PROJECT_ROOT="$HOME/workspace/project_sLM_planning"
LCB_ROOT="$HOME/workspace/LiveCodeBench"

PHASE1_DIR="$PROJECT_ROOT/phase1_planning_bottleneck"
CONFIG_DIR="$PHASE1_DIR/configs"

export PYTHONPATH="$PROJECT_ROOT:$LCB_ROOT"

cd "$PROJECT_ROOT"


LIMIT=""
CONFIG_NAME=""

RUN_DIRECT=false
RUN_SELF_PLAN=false
RUN_TEACHER_PLAN=false


usage() {
    echo "Usage:"
    echo "  $0 [direct] [self_plan] [teacher_plan] --config-name NAME [--limit N]"
    echo
    echo "Config naming convention:"
    echo "  direct_NAME.yaml"
    echo "  self_plan_NAME.yaml"
    echo "  teacher_plan_NAME.yaml"
    echo
    echo "Examples:"
    echo "  $0 direct --config-name qwen25Coder3b"
    echo "  $0 self_plan --config-name qwen253b"
    echo "  $0 teacher_plan --config-name phi3"
    echo "  $0 direct self_plan teacher_plan --config-name qwen253b"
    echo "  $0 direct self_plan --config-name qwen253b --limit 10"
}


if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi


while [[ $# -gt 0 ]]; do
    case "$1" in
        direct)
            RUN_DIRECT=true
            shift
            ;;

        self_plan)
            RUN_SELF_PLAN=true
            shift
            ;;

        teacher_plan)
            RUN_TEACHER_PLAN=true
            shift
            ;;

        --config-name)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --config-name requires a value."
                exit 1
            fi

            CONFIG_NAME="$2"
            shift 2
            ;;

        --limit)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --limit requires a value."
                exit 1
            fi

            LIMIT="$2"
            shift 2
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        *)
            echo "[ERROR] Unknown argument: $1"
            echo
            usage
            exit 1
            ;;
    esac
done


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

if [[ -z "$CONFIG_NAME" ]]; then
    echo "[ERROR] --config-name is required."
    echo
    usage
    exit 1
fi


if [[ "$RUN_DIRECT" == false ]] \
    && [[ "$RUN_SELF_PLAN" == false ]] \
    && [[ "$RUN_TEACHER_PLAN" == false ]]; then

    echo "[ERROR] No strategy selected."
    echo
    usage
    exit 1
fi


DIRECT_CONFIG="$CONFIG_DIR/direct_${CONFIG_NAME}.yaml"
SELF_PLAN_CONFIG="$CONFIG_DIR/self_plan_${CONFIG_NAME}.yaml"
TEACHER_PLAN_CONFIG="$CONFIG_DIR/teacher_plan_${CONFIG_NAME}.yaml"


if [[ "$RUN_DIRECT" == true ]] \
    && [[ ! -f "$DIRECT_CONFIG" ]]; then

    echo "[ERROR] Direct config not found:"
    echo "        $DIRECT_CONFIG"
    exit 1
fi


if [[ "$RUN_SELF_PLAN" == true ]] \
    && [[ ! -f "$SELF_PLAN_CONFIG" ]]; then

    echo "[ERROR] Self-Plan config not found:"
    echo "        $SELF_PLAN_CONFIG"
    exit 1
fi


if [[ "$RUN_TEACHER_PLAN" == true ]] \
    && [[ ! -f "$TEACHER_PLAN_CONFIG" ]]; then

    echo "[ERROR] Teacher-Plan config not found:"
    echo "        $TEACHER_PLAN_CONFIG"
    exit 1
fi


# ------------------------------------------------------------------
# Common arguments
# ------------------------------------------------------------------

COMMON_ARGS=()

if [[ -n "$LIMIT" ]]; then
    COMMON_ARGS+=(--limit "$LIMIT")
fi


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

echo "================================================================================"
echo "Phase 1 Planning Bottleneck Experiments"
echo "================================================================================"
echo "Project root : $PROJECT_ROOT"
echo "Config name  : $CONFIG_NAME"

if [[ -n "$LIMIT" ]]; then
    echo "Limit        : $LIMIT"
else
    echo "Limit        : config default"
fi

echo


# ------------------------------------------------------------------
# Direct
# ------------------------------------------------------------------

if [[ "$RUN_DIRECT" == true ]]; then
    echo "================================================================================"
    echo "Direct"
    echo "================================================================================"
    echo "Config : $DIRECT_CONFIG"
    echo

    python phase1_planning_bottleneck/scripts/run_direct.py \
        --config "$DIRECT_CONFIG" \
        "${COMMON_ARGS[@]}"

    echo
fi


# ------------------------------------------------------------------
# Self-Plan
# ------------------------------------------------------------------

if [[ "$RUN_SELF_PLAN" == true ]]; then
    echo "================================================================================"
    echo "Self-Planning"
    echo "================================================================================"
    echo "Config : $SELF_PLAN_CONFIG"
    echo

    python phase1_planning_bottleneck/scripts/run_self_plan.py \
        --config "$SELF_PLAN_CONFIG" \
        "${COMMON_ARGS[@]}"

    echo
fi


# ------------------------------------------------------------------
# Teacher-Plan
# ------------------------------------------------------------------

if [[ "$RUN_TEACHER_PLAN" == true ]]; then
    echo "================================================================================"
    echo "Teacher-Planning"
    echo "================================================================================"
    echo "Config : $TEACHER_PLAN_CONFIG"
    echo

    python phase1_planning_bottleneck/scripts/run_teacher_plan.py \
        --config "$TEACHER_PLAN_CONFIG" \
        "${COMMON_ARGS[@]}"

    echo
fi


echo "================================================================================"
echo "Selected Phase 1 Experiments Completed"
echo "================================================================================"