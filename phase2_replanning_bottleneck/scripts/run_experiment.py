"""
Phase 2 refinement 통합 실행 스크립트.

Phase 1은 전략별로 스크립트가 나뉘어 있었으나(run_direct / run_self_plan /
run_teacher_plan), Phase 2는 세 전략이 동일한 입력(FailureCase)과 동일한
평가 경로를 공유하므로 하나의 엔트리포인트에서 --strategy 로 분기한다.

Usage:

python -m scripts.run_experiment \
  --config configs/feedback_repair.yaml

python -m scripts.run_experiment \
  --config configs/self_replan.yaml \
  --limit 10

python -m scripts.run_experiment \
  --config configs/teacher_replan.yaml \
  --output-path outputs/pilot/teacher_replan/results.jsonl

결과 확인:

wc -l /mnt/hdd/project_sLM_planning/output/phase2/feedback_repair_500/results.jsonl

파이프라인
----------
1. Phase1FailureLoader   : phase1 direct results.jsonl -> FailureCase 목록
2. Strategy.run          : refinement 1회 (전략별 분기)
3. CodeExtractor.extract : 응답에서 코드 추출
4. Evaluator.evaluate    : 복원된 테스트로 재평가
5. build_refinement_record -> JSONLLogger.append

TODO(구현)
----------
- [ ] build_strategy : config -> 전략 객체
- [ ] main : Phase 1 run_teacher_plan.py 의 루프 구조를 따라 구현
             (resume / skip / 예외 처리 / 요약 출력)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets.phase1_failure_loader import (
    Phase1FailureLoader,
)
from src.execution.code_extractor import (
    CodeExtractionError,
    CodeExtractor,
)
from src.execution.evaluator import Evaluator
from src.models.generator import ModelGenerator
from src.plans.teacher_replan_store import (
    TeacherReplanStore,
)
from src.schemas import EvaluationResult
from src.strategies import (
    FeedbackRepairStrategy,
    SelfReplanStrategy,
    TeacherReplanStrategy,
)
from src.utils.config import load_config
from src.utils.jsonl_logger import JSONLLogger
from src.utils.record_builder import (
    build_refinement_record,
)
from src.utils.run_metadata import (
    save_run_config,
    save_run_metadata,
)
from src.utils.seed import set_seed

STRATEGY_NAMES = (
    "feedback_repair",
    "self_replan",
    "teacher_replan",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Phase 2 refinement strategy on "
            "Phase 1 failure trajectories."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--strategy",
        choices=STRATEGY_NAMES,
        default=None,
        help=(
            "Override strategy.name from config."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Override the number of failure cases."
        ),
    )

    parser.add_argument(
        "--output-path",
        default=None,
        help="Override output JSONL path.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume.",
    )

    return parser.parse_args()


def build_failure_evaluation(
    *,
    status: str,
    error: Exception,
) -> EvaluationResult:
    """실행 불가능한 결과를 실패로 변환한다."""
    return EvaluationResult(
        passed=False,
        status=status,
        passed_tests=0,
        total_tests=0,
        execution_time=0.0,
        test_results=[],
        error_message=str(error),
    )


def build_strategy(
    *,
    strategy_name: str,
    generator: ModelGenerator,
    strategy_config: dict,
    generation_config: dict,
):
    """config에 따라 refinement 전략 객체를 생성한다."""
    raise NotImplementedError(
        "TODO: strategy_name 별 객체 생성 "
        "(teacher_replan 은 TeacherReplanStore 주입)"
    )


def main() -> None:
    raise NotImplementedError(
        "TODO: Phase1 run_teacher_plan.py 의 실행 루프를 "
        "FailureCase 기준으로 옮겨 구현"
    )


if __name__ == "__main__":
    main()
