#!/usr/bin/env bash

"""
chmod +x phase1_planning_bottleneck/scripts/run_all.sh

./phase1_planning_bottleneck/scripts/run_all.sh all

./phase1_planning_bottleneck/scripts/run_all.sh direct self_plan
./phase1_planning_bottleneck/scripts/run_all.sh teacher_plan

"""

set -euo pipefail

PROJECT_ROOT="$HOME/workspace/project_sLM_planning"
LCB_ROOT="$HOME/workspace/LiveCodeBench"

PHASE1_DIR="$PROJECT_ROOT/phase1_planning_bottleneck"
CONFIG_DIR="$PHASE1_DIR/configs"

export PYTHONPATH="$PROJECT_ROOT:$LCB_ROOT"

cd "$PROJECT_ROOT"

LIMIT=""
RUN_DIRECT=false
RUN_SELF_PLAN=false
RUN_TEACHER_PLAN=false


usage() {
    echo "Usage:"
    echo "  $0 [direct] [self_plan] [teacher_plan] [--limit N]"
    echo
    echo "Examples:"
    echo "  $0 direct"
    echo "  $0 self_plan"
    echo "  $0 teacher_plan"
    echo "  $0 direct self_plan"
    echo "  $0 direct self_plan teacher_plan"
    echo "  $0 direct self_plan --limit 10"
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


COMMON_ARGS=()

if [[ -n "$LIMIT" ]]; then
    COMMON_ARGS+=(--limit "$LIMIT")
fi


echo "================================================================================"
echo "Phase 1 Planning Bottleneck Experiments"
echo "================================================================================"
echo "Project root : $PROJECT_ROOT"

if [[ -n "$LIMIT" ]]; then
    echo "Limit        : $LIMIT"
else
    echo "Limit        : config default"
fi

echo


if [[ "$RUN_DIRECT" == true ]]; then
    echo "================================================================================"
    echo "Direct"
    echo "================================================================================"

    python phase1_planning_bottleneck/scripts/run_direct.py \
        --config "$CONFIG_DIR/direct.yaml" \
        "${COMMON_ARGS[@]}"

    echo
fi


if [[ "$RUN_SELF_PLAN" == true ]]; then
    echo "================================================================================"
    echo "Self-Planning"
    echo "================================================================================"

    python phase1_planning_bottleneck/scripts/run_self_plan.py \
        --config "$CONFIG_DIR/self_plan.yaml" \
        "${COMMON_ARGS[@]}"

    echo
fi


if [[ "$RUN_TEACHER_PLAN" == true ]]; then
    echo "================================================================================"
    echo "Teacher-Planning"
    echo "================================================================================"

    python phase1_planning_bottleneck/scripts/run_teacher_plan.py \
        --config "$CONFIG_DIR/teacher_plan.yaml" \
        "${COMMON_ARGS[@]}"

    echo
fi


echo "================================================================================"
echo "Selected Phase 1 Experiments Completed"
echo "================================================================================"