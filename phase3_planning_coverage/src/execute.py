"""
코드 추출 및 테스트 실행.

Phase 1 `scripts/run_self_plan.py`의 추출/평가/예외 처리 흐름을 그대로 옮겼다.
- CodeExtractor  : Phase 1과 동일 (src/common/execution/code_extractor.py)
- Evaluator      : Phase 1과 동일 (src/common/execution/evaluator.py)
- timeout        : config로 주입, Phase 1과 동일 값(5.0s) 사용
- 실패 status 명 : EXTRACTION_ERROR / UNSUPPORTED_TEST_TYPE / EVALUATION_ERROR

Phase 3에서 추가되는 것은 candidate 단위 test_pass_ratio 뿐이다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.common.execution.code_extractor import (
    CodeExtractionError,
    CodeExtractor,
)
from src.common.execution.evaluator import Evaluator
from src.common.schemas import (
    EvaluationResult,
    ProblemExample,
)


@dataclass
class ExecutionOutcome:
    """하나의 candidate에 대한 실행 결과."""

    code: str

    passed: bool
    status: str
    passed_tests: int
    total_tests: int
    test_pass_ratio: float
    execution_time: float

    error_message: str | None = None
    test_results: list[dict[str, Any]] = field(
        default_factory=list
    )


def build_failure_evaluation(
    *,
    status: str,
    error: Exception,
) -> EvaluationResult:
    """평가 불가능한 candidate를 실패 결과로 변환한다.

    Phase 1 run_self_plan.py의 동명 함수와 동일하다.
    """
    return EvaluationResult(
        passed=False,
        status=status,
        passed_tests=0,
        total_tests=0,
        execution_time=0.0,
        test_results=[],
        error_message=str(error),
    )


def compute_test_pass_ratio(
    *,
    passed_tests: int,
    total_tests: int,
) -> float:
    """통과한 테스트 비율. 테스트가 없으면 0.0."""
    if total_tests <= 0:
        return 0.0

    return passed_tests / total_tests


class CandidateExecutor:
    """raw_output -> 코드 추출 -> 테스트 실행 -> 결과 요약."""

    def __init__(
        self,
        extractor: CodeExtractor,
        evaluator: Evaluator,
        *,
        store_test_results: bool = False,
    ) -> None:
        self.extractor = extractor
        self.evaluator = evaluator
        self.store_test_results = store_test_results

    def run(
        self,
        *,
        example: ProblemExample,
        raw_output: str,
    ) -> ExecutionOutcome:
        try:
            extracted_code = self.extractor.extract(
                raw_output
            )

        except CodeExtractionError as error:
            extracted_code = ""

            evaluation = build_failure_evaluation(
                status="EXTRACTION_ERROR",
                error=error,
            )

        else:
            try:
                evaluation = self.evaluator.evaluate(
                    example=example,
                    code=extracted_code,
                )

            except ValueError as error:
                if "Unsupported test type" in str(error):
                    evaluation = build_failure_evaluation(
                        status="UNSUPPORTED_TEST_TYPE",
                        error=error,
                    )
                else:
                    evaluation = build_failure_evaluation(
                        status="EVALUATION_ERROR",
                        error=error,
                    )

            except Exception as error:
                evaluation = build_failure_evaluation(
                    status="EVALUATION_ERROR",
                    error=error,
                )

        return ExecutionOutcome(
            code=extracted_code,
            passed=evaluation.passed,
            status=evaluation.status,
            passed_tests=evaluation.passed_tests,
            total_tests=evaluation.total_tests,
            test_pass_ratio=compute_test_pass_ratio(
                passed_tests=evaluation.passed_tests,
                total_tests=evaluation.total_tests,
            ),
            execution_time=evaluation.execution_time,
            error_message=evaluation.error_message,
            test_results=(
                [
                    asdict(test_result)
                    for test_result
                    in evaluation.test_results
                ]
                if self.store_test_results
                else []
            ),
        )

    def skipped(
        self,
        *,
        status: str,
        error_message: str,
    ) -> ExecutionOutcome:
        """생성 자체가 실패해 실행하지 않은 candidate."""
        return ExecutionOutcome(
            code="",
            passed=False,
            status=status,
            passed_tests=0,
            total_tests=0,
            test_pass_ratio=0.0,
            execution_time=0.0,
            error_message=error_message,
        )
