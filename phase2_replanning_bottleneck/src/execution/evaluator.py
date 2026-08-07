"""
테스트 실행 및 평가 결과 산출.

선택한 벤치마크의 공식 평가 코드를 감싸는 wrapper 형태로 구현할 것.
생성된 코드를 현재 Python 프로세스에서 직접 exec()하는 방식은 피하는 것이 좋다. 
다음 중 하나를 사용해야 한다.

- 벤치마크 공식 evaluator
- subprocess 기반 격리
- Docker sandbox
- EvalPlus 공식 evaluation pipeline

초기 MVP에서는 공식 evaluator wrapper를 사용하는 것이 가장 안정적이라고 함.

"""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from src.schemas import (
    EvaluationResult,
    ProblemExample,
    TestCaseResult,
)


class Evaluator:
    """Python 코드를 stdin/stdout 테스트 케이스로 평가한다."""

    VALID_TEST_TYPES = {
        "stdin",
    }

    def __init__(
        self,
        *,
        timeout_seconds: float = 5.0,
        python_executable: str | None = None,
        include_public_tests: bool = True,
        include_private_tests: bool = True,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than 0."
            )

        if not include_public_tests and not include_private_tests:
            raise ValueError(
                "At least one test group must be enabled."
            )

        self.timeout_seconds = timeout_seconds
        self.python_executable = (
            python_executable or sys.executable
        )
        self.include_public_tests = include_public_tests
        self.include_private_tests = include_private_tests

    def evaluate(
        self,
        example: ProblemExample,
        code: str,
    ) -> EvaluationResult:
        """문제 하나의 모든 테스트 케이스로 코드를 평가한다."""
        if not isinstance(code, str):
            raise TypeError(
                f"code must be a string, "
                f"got {type(code).__name__}."
            )

        if not code.strip():
            return EvaluationResult(
                passed=False,
                status="EMPTY_CODE",
                passed_tests=0,
                total_tests=0,
                execution_time=0.0,
                error_message="Code is empty.",
            )

        syntax_error = self._check_syntax(code)

        if syntax_error is not None:
            return EvaluationResult(
                passed=False,
                status="SYNTAX_ERROR",
                passed_tests=0,
                total_tests=0,
                execution_time=0.0,
                error_message=syntax_error,
            )

        test_cases = self._collect_test_cases(example)

        if not test_cases:
            return EvaluationResult(
                passed=False,
                status="NO_TESTS",
                passed_tests=0,
                total_tests=0,
                execution_time=0.0,
                error_message="No enabled test cases.",
            )

        start_time = time.perf_counter()

        with tempfile.TemporaryDirectory(
            prefix="phase1_eval_"
        ) as temp_dir:
            code_path = Path(temp_dir) / "solution.py"
            code_path.write_text(
                code,
                encoding="utf-8",
            )

            test_results: list[TestCaseResult] = []

            for index, test_case in enumerate(test_cases):
                result = self._run_test_case(
                    code_path=code_path,
                    test_case=test_case,
                    test_index=index,
                )
                test_results.append(result)

        execution_time = (
            time.perf_counter() - start_time
        )

        passed_tests = sum(
            result.passed
            for result in test_results
        )
        total_tests = len(test_results)

        status = self._determine_overall_status(
            test_results
        )

        return EvaluationResult(
            passed=passed_tests == total_tests,
            status=status,
            passed_tests=passed_tests,
            total_tests=total_tests,
            execution_time=execution_time,
            test_results=test_results,
            error_message=self._get_first_error(
                test_results
            ),
        )

    def _collect_test_cases(
        self,
        example: ProblemExample,
    ) -> list[dict[str, Any]]:
        test_cases: list[dict[str, Any]] = []

        if self.include_public_tests:
            test_cases.extend(example.public_tests)

        if self.include_private_tests:
            test_cases.extend(example.private_tests)

        for index, test_case in enumerate(test_cases):
            self._validate_test_case(
                test_case=test_case,
                test_index=index,
                problem_id=example.problem_id,
            )

        return test_cases

    def _run_test_case(
        self,
        *,
        code_path: Path,
        test_case: dict[str, Any],
        test_index: int,
    ) -> TestCaseResult:
        input_text = test_case["input"]
        expected_output = test_case["output"]

        start_time = time.perf_counter()

        try:
            completed = subprocess.run(
                [
                    self.python_executable,
                    str(code_path),
                ],
                input=input_text,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )

        except subprocess.TimeoutExpired as error:
            execution_time = (
                time.perf_counter() - start_time
            )

            stdout = self._to_text(error.stdout)
            stderr = self._to_text(error.stderr)

            return TestCaseResult(
                test_index=test_index,
                passed=False,
                status="TIMEOUT",
                input_text=input_text,
                expected_output=expected_output,
                actual_output=stdout,
                execution_time=execution_time,
                return_code=None,
                stderr=stderr,
            )

        execution_time = (
            time.perf_counter() - start_time
        )

        actual_output = completed.stdout
        stderr = completed.stderr

        if completed.returncode != 0:
            return TestCaseResult(
                test_index=test_index,
                passed=False,
                status="RUNTIME_ERROR",
                input_text=input_text,
                expected_output=expected_output,
                actual_output=actual_output,
                execution_time=execution_time,
                return_code=completed.returncode,
                stderr=stderr,
            )

        passed = self._outputs_match(
            expected_output=expected_output,
            actual_output=actual_output,
        )

        return TestCaseResult(
            test_index=test_index,
            passed=passed,
            status=(
                "PASS"
                if passed
                else "WRONG_ANSWER"
            ),
            input_text=input_text,
            expected_output=expected_output,
            actual_output=actual_output,
            execution_time=execution_time,
            return_code=completed.returncode,
            stderr=stderr,
        )

    @staticmethod
    def _check_syntax(
        code: str,
    ) -> str | None:
        try:
            ast.parse(code)
        except SyntaxError as error:
            return (
                f"{error.msg} "
                f"(line={error.lineno}, "
                f"offset={error.offset})"
            )

        return None

    @staticmethod
    def _outputs_match(
        *,
        expected_output: str,
        actual_output: str,
    ) -> bool:
        expected_tokens = expected_output.split()
        actual_tokens = actual_output.split()

        return expected_tokens == actual_tokens

    @staticmethod
    def _determine_overall_status(
        results: list[TestCaseResult],
    ) -> str:
        if all(result.passed for result in results):
            return "PASS"

        status_priority = (
            "TIMEOUT",
            "RUNTIME_ERROR",
            "WRONG_ANSWER",
        )

        observed_statuses = {
            result.status
            for result in results
        }

        for status in status_priority:
            if status in observed_statuses:
                return status

        return "FAILED"

    @staticmethod
    def _get_first_error(
        results: list[TestCaseResult],
    ) -> str | None:
        for result in results:
            if result.passed:
                continue

            if result.status == "WRONG_ANSWER":
                return (
                    f"Test {result.test_index}: "
                    f"expected={result.expected_output!r}, "
                    f"actual={result.actual_output!r}"
                )

            if result.stderr:
                return (
                    f"Test {result.test_index}: "
                    f"{result.stderr.strip()}"
                )

            return (
                f"Test {result.test_index}: "
                f"{result.status}"
            )

        return None

    @classmethod
    def _validate_test_case(
        cls,
        *,
        test_case: dict[str, Any],
        test_index: int,
        problem_id: str,
    ) -> None:
        if not isinstance(test_case, dict):
            raise TypeError(
                f"Test case must be dict: "
                f"{problem_id}, index={test_index}"
            )

        for field_name in ("input", "output"):
            if field_name not in test_case:
                raise ValueError(
                    f"Missing {field_name}: "
                    f"{problem_id}, index={test_index}"
                )

            if not isinstance(
                test_case[field_name],
                str,
            ):
                raise TypeError(
                    f"{field_name} must be str: "
                    f"{problem_id}, index={test_index}"
                )

        test_type = test_case.get(
            "testtype",
            "stdin",
        )

        if test_type not in cls.VALID_TEST_TYPES:
            raise ValueError(
                f"Unsupported test type: {test_type} "
                f"({problem_id}, index={test_index})"
            )

    @staticmethod
    def _to_text(
        value: str | bytes | None,
    ) -> str:
        if value is None:
            return ""

        if isinstance(value, bytes):
            return value.decode(
                "utf-8",
                errors="replace",
            )

        return value