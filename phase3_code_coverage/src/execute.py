"""candidate 코드 실행 및 채점.

Phase 3-A와 동일한 실행 경로를 쓴다.
(공용 Evaluator를 그대로 사용하여 Phase 1/3-A/3-B 채점 기준을 일치시킨다.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.common.execution.code_extractor import extract_python_code
from src.common.execution.evaluator import Evaluator
from src.common.schemas import ProblemExample


@dataclass
class ExecutionOutcome:
    """candidate 하나의 실행 결과 요약."""

    passed: bool
    status: str
    num_tests: int
    num_passed: int
    test_pass_ratio: float
    error_message: str | None = None
    test_results: list[dict[str, Any]] = field(default_factory=list)
    execution_seconds: float = 0.0


def build_failure_evaluation(status: str, message: str, num_tests: int) -> ExecutionOutcome:
    """코드 추출 실패 등 실행 이전 단계의 실패를 표준 형태로 만든다."""
    raise NotImplementedError


def compute_test_pass_ratio(num_passed: int, num_tests: int) -> float:
    raise NotImplementedError


class CandidateExecutor:
    """생성 코드 -> 추출 -> 실행 -> ExecutionOutcome."""

    def __init__(
        self,
        evaluator: Evaluator,
        store_test_results: bool = False,
    ) -> None:
        raise NotImplementedError

    def run(self, example: ProblemExample, raw_output: str) -> ExecutionOutcome:
        raise NotImplementedError
