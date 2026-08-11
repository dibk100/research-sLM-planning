"""
candidate 코드 실행 및 채점.

Phase 3-A와 동일한 실행 경로를 쓴다.
(공용 Evaluator를 그대로 사용하여 Phase 1/3-A/3-B 채점 기준을 일치시킨다.)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.common.execution.code_extractor import (
    CodeExtractionError,
    CodeExtractor,
)
from src.common.execution.evaluator import Evaluator
from src.common.schemas import ProblemExample

@dataclass
class ExecutionOutcome:
    """candidate 하나의 실행 결과 요약."""

    extracted_code: str

    passed: bool
    status: str

    num_tests: int
    num_passed: int
    test_pass_ratio: float

    error_message: str | None = None

    test_results: list[dict[str, Any]] = field(
        default_factory=list
    )

    execution_seconds: float = 0.0


def build_failure_evaluation(
    status: str,
    message: str,
    num_tests: int = 0,
) -> ExecutionOutcome:
    """
    코드 추출 실패 등 실행 이전 단계의 실패를
    표준 ExecutionOutcome 형태로 만든다.
    """
    return ExecutionOutcome(
        extracted_code="",
        passed=False,
        status=status,
        num_tests=num_tests,
        num_passed=0,
        test_pass_ratio=0.0,
        error_message=message,
        test_results=[],
        execution_seconds=0.0,
    )


def compute_test_pass_ratio(
    num_passed: int,
    num_tests: int,
) -> float:
    """
    통과한 test 비율.

    평가 가능한 test가 없으면 0.0으로 둔다.
    """
    if num_tests <= 0:
        return 0.0

    return num_passed / num_tests


class CandidateExecutor:
    """생성 코드 -> 추출 -> 실행 -> ExecutionOutcome."""

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
        example: ProblemExample,
        raw_output: str,
    ) -> ExecutionOutcome:
        """
        하나의 raw model output을 평가한다.

        흐름:
            raw output
                -> CodeExtractor
                -> evaluator
                -> standardized ExecutionOutcome
        """

        # --------------------------------------------------------------
        # 1. Code extraction
        # --------------------------------------------------------------

        try:
            extracted_code = self.extractor.extract(
                raw_output
            )

        except CodeExtractionError as error:
            return build_failure_evaluation(
                status="EXTRACTION_ERROR",
                message=str(error),
                num_tests=0,
            )

        except Exception as error:
            return build_failure_evaluation(
                status="EXTRACTION_ERROR",
                message=str(error),
                num_tests=0,
            )

        if not extracted_code.strip():
            return build_failure_evaluation(
                status="EXTRACTION_ERROR",
                message="Extracted code is empty.",
                num_tests=0,
            )

        # --------------------------------------------------------------
        # 2. Evaluation
        # --------------------------------------------------------------

        try:
            evaluation = self.evaluator.evaluate(
                example=example,
                code=extracted_code,
            )

        except ValueError as error:
            message = str(error)

            if "Unsupported test type" in message:
                return build_failure_evaluation(
                    status="UNSUPPORTED_TEST_TYPE",
                    message=message,
                    num_tests=0,
                )

            return build_failure_evaluation(
                status="EVALUATION_ERROR",
                message=message,
                num_tests=0,
            )

        except Exception as error:
            return build_failure_evaluation(
                status="EVALUATION_ERROR",
                message=str(error),
                num_tests=0,
            )

        # --------------------------------------------------------------
        # 3. Normalize evaluator result
        # --------------------------------------------------------------

        num_tests = int(
            evaluation.total_tests
        )

        num_passed = int(
            evaluation.passed_tests
        )

        ratio = compute_test_pass_ratio(
            num_passed=num_passed,
            num_tests=num_tests,
        )

        if self.store_test_results:
            serialized_test_results = []

            for result in evaluation.test_results:
                if hasattr(
                    result,
                    "__dataclass_fields__",
                ):
                    serialized_test_results.append(
                        asdict(result)
                    )

                elif isinstance(result, dict):
                    serialized_test_results.append(
                        result
                    )

                else:
                    serialized_test_results.append(
                        {
                            "value": str(result)
                        }
                    )
        else:
            serialized_test_results = []

        return ExecutionOutcome(
            extracted_code=extracted_code,
            passed=bool(evaluation.passed),
            status=str(evaluation.status),
            num_tests=num_tests,
            num_passed=num_passed,
            test_pass_ratio=ratio,
            error_message=evaluation.error_message,
            test_results=serialized_test_results,
            execution_seconds=float(
                evaluation.execution_time
            ),
        )

    def skipped(
        self,
        *,
        status: str,
        error_message: str,
    ) -> ExecutionOutcome:
        return build_failure_evaluation(
            status=status,
            message=error_message,
            num_tests=0,
        )