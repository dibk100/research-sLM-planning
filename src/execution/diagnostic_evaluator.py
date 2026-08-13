"""
공식 LiveCodeBench 평가 로직을 그대로 재사용하면서, test case를 하나씩 독립 평가하는 방식
"""

# src/execution/diagnostic_evaluator.py

from __future__ import annotations

import json
import time
from typing import Any

from lcb_runner.evaluation.compute_code_generation_metrics import (
    check_correctness,
)

from src.schemas import (
    EvaluationResult,
    ProblemExample,
    TestCaseResult,
)


class DiagnosticEvaluator:
    """
    Diagnostic evaluator for detailed per-test analysis.

    Important:
    - This evaluator is NOT the final correctness authority.
    - Final PASS/FAIL should be determined by LiveCodeBenchEvaluator.
    - This evaluator runs each test independently using the official
      LiveCodeBench check_correctness() logic so that all test outcomes
      can be observed.

    Primary use cases:
    - test pass ratio
    - per-test failure analysis
    - refinement improvement/degradation analysis
    - recovery dynamics
    """

    def __init__(
        self,
        *,
        timeout_seconds: int = 6,
        debug: bool = False,
        include_public_tests: bool = True,
        include_private_tests: bool = True,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than 0."
            )

        if (
            not include_public_tests
            and not include_private_tests
        ):
            raise ValueError(
                "At least one test group must be enabled."
            )

        self.timeout_seconds = timeout_seconds
        self.debug = debug

        self.include_public_tests = (
            include_public_tests
        )
        self.include_private_tests = (
            include_private_tests
        )

    def evaluate(
        self,
        problem: ProblemExample,
        code: str,
    ) -> EvaluationResult:
        """
        Run every enabled test case independently.

        Unlike the official problem-level evaluator,
        evaluation does not stop after the first failed test.
        """

        self._validate_problem(problem)

        if not isinstance(code, str):
            raise TypeError(
                f"code must be str, got "
                f"{type(code).__name__}"
            )

        if not code.strip():
            return EvaluationResult(
                passed=False,
                status="EMPTY_CODE",
                passed_tests=0,
                total_tests=0,
                execution_time=0.0,
                test_results=[],
                error_message="Code is empty.",
            )

        test_cases = self._collect_test_cases(
            problem
        )

        if not test_cases:
            return EvaluationResult(
                passed=False,
                status="NO_TESTS",
                passed_tests=0,
                total_tests=0,
                execution_time=0.0,
                test_results=[],
                error_message="No enabled test cases.",
            )

        test_results: list[TestCaseResult] = []

        total_start_time = time.perf_counter()

        for test_index, test_case in enumerate(
            test_cases
        ):
            result = self._evaluate_single_test(
                problem=problem,
                code=code,
                test_case=test_case,
                test_index=test_index,
            )

            test_results.append(result)

        wall_clock_time = (
            time.perf_counter()
            - total_start_time
        )

        passed_tests = sum(
            result.passed
            for result in test_results
        )

        total_tests = len(test_results)

        passed = (
            total_tests > 0
            and passed_tests == total_tests
        )

        status = (
            "PASS"
            if passed
            else self._determine_overall_status(
                test_results
            )
        )

        return EvaluationResult(
            passed=passed,
            status=status,
            passed_tests=passed_tests,
            total_tests=total_tests,
            execution_time=wall_clock_time,
            test_results=test_results,
            error_message=self._get_first_error(
                test_results
            ),
        )

    def _evaluate_single_test(
        self,
        *,
        problem: ProblemExample,
        code: str,
        test_case: dict[str, Any],
        test_index: int,
    ) -> TestCaseResult:
        """
        Evaluate exactly one test using the official
        LiveCodeBench check_correctness() pipeline.
        """

        sample = self._build_single_test_sample(
            problem=problem,
            test_case=test_case,
        )

        start_time = time.perf_counter()

        try:
            results, metadata = check_correctness(
                sample=sample,
                generation=code,
                timeout=self.timeout_seconds,
                debug=self.debug,
            )

        except Exception as error:
            execution_time = (
                time.perf_counter()
                - start_time
            )

            return TestCaseResult(
                test_index=test_index,
                passed=False,
                status="TEST_RUNNER_ERROR",
                input_text=test_case["input"],
                expected_output=test_case["output"],
                actual_output="",
                execution_time=execution_time,
                return_code=None,
                stderr=repr(error),
            )

        execution_time = (
            time.perf_counter()
            - start_time
        )

        status, passed = self._interpret_result(
            results=results,
            metadata=metadata,
        )

        actual_output = self._extract_actual_output(
            metadata
        )

        error_text = self._extract_error_text(
            metadata
        )

        return TestCaseResult(
            test_index=test_index,
            passed=passed,
            status=status,
            input_text=test_case["input"],
            expected_output=test_case["output"],
            actual_output=actual_output,
            execution_time=execution_time,
            return_code=None,
            stderr=error_text,
        )

    @staticmethod
    def _build_single_test_sample(
        *,
        problem: ProblemExample,
        test_case: dict[str, Any],
    ) -> dict[str, str]:
        """
        Build the sample format expected by the official
        LiveCodeBench evaluator, but containing only one test.
        """

        input_output: dict[str, Any] = {
            "inputs": [
                test_case["input"]
            ],
            "outputs": [
                test_case["output"]
            ],
        }

        if (
            problem.evaluation_type
            == "functional"
        ):
            if not problem.function_name:
                raise ValueError(
                    "Missing function_name for "
                    f"functional problem: "
                    f"{problem.problem_id}"
                )

            input_output["fn_name"] = (
                problem.function_name
            )

        return {
            "input_output": json.dumps(
                input_output,
                ensure_ascii=False,
            )
        }

    @staticmethod
    def _interpret_result(
        *,
        results: list[Any],
        metadata: dict[str, Any],
    ) -> tuple[str, bool]:
        """
        Convert LiveCodeBench result codes to local statuses.

        LiveCodeBench result codes:
            True  -> PASS
            False / -2 -> WRONG_ANSWER
            -3 -> TIME_LIMIT_EXCEEDED
            -4 -> RUNTIME_ERROR

        Test framework failures are additionally inferred
        from metadata.
        """

        if not results:
            error_code = metadata.get(
                "error_code"
            )

            if error_code == -5:
                return (
                    "TEST_RUNNER_ERROR",
                    False,
                )

            return "FAILED", False

        result = results[0]

        # bool is a subclass of int:
        # check True/False before integer codes.
        if result is True:
            return "PASS", True

        if result is False or result == -2:
            return "WRONG_ANSWER", False

        if result == -3:
            return (
                "TIME_LIMIT_EXCEEDED",
                False,
            )

        if result == -4:
            return "RUNTIME_ERROR", False

        if result == -5:
            return (
                "TEST_RUNNER_ERROR",
                False,
            )

        error_code = metadata.get(
            "error_code"
        )

        if error_code == -5:
            return (
                "TEST_RUNNER_ERROR",
                False,
            )

        return "FAILED", False

    def _collect_test_cases(
        self,
        problem: ProblemExample,
    ) -> list[dict[str, Any]]:
        test_cases: list[
            dict[str, Any]
        ] = []

        if self.include_public_tests:
            test_cases.extend(
                problem.public_tests
            )

        if self.include_private_tests:
            test_cases.extend(
                problem.private_tests
            )

        for index, test_case in enumerate(
            test_cases
        ):
            self._validate_test_case(
                test_case=test_case,
                test_index=index,
                problem_id=problem.problem_id,
            )

        return test_cases

    @staticmethod
    def _determine_overall_status(
        test_results: list[
            TestCaseResult
        ],
    ) -> str:
        statuses = {
            result.status
            for result in test_results
            if not result.passed
        }

        priority = (
            "TEST_RUNNER_ERROR",
            "TIME_LIMIT_EXCEEDED",
            "RUNTIME_ERROR",
            "WRONG_ANSWER",
            "FAILED",
        )

        for status in priority:
            if status in statuses:
                return status

        return "FAILED"

    @staticmethod
    def _get_first_error(
        test_results: list[
            TestCaseResult
        ],
    ) -> str | None:
        for result in test_results:
            if result.passed:
                continue

            if (
                result.status
                == "WRONG_ANSWER"
            ):
                return (
                    f"Test {result.test_index}: "
                    f"expected="
                    f"{result.expected_output!r}, "
                    f"actual="
                    f"{result.actual_output!r}"
                )

            if result.stderr:
                return (
                    f"Test {result.test_index}: "
                    f"{result.stderr}"
                )

            return (
                f"Test {result.test_index}: "
                f"{result.status}"
            )

        return None

    @staticmethod
    def _extract_actual_output(
        metadata: dict[str, Any],
    ) -> str:
        """
        LiveCodeBench metadata stores the failed prediction
        under 'output' for many Wrong Answer cases.
        """

        value = metadata.get("output")

        if value is None:
            return ""

        return str(value)

    @staticmethod
    def _extract_error_text(
        metadata: dict[str, Any],
    ) -> str:
        value = (
            metadata.get("error")
            or metadata.get(
                "error_message"
            )
        )

        if value is None:
            return ""

        return str(value)

    @staticmethod
    def _validate_problem(
        problem: ProblemExample,
    ) -> None:
        if (
            problem.dataset
            != "livecodebench_v6"
        ):
            raise ValueError(
                "DiagnosticEvaluator currently "
                "supports only "
                "livecodebench_v6, got "
                f"{problem.dataset}"
            )

        if problem.evaluation_type not in {
            "stdin",
            "functional",
        }:
            raise ValueError(
                "Unsupported "
                "evaluation_type: "
                f"{problem.evaluation_type}"
            )

        if (
            problem.evaluation_type
            == "functional"
            and not problem.function_name
        ):
            raise ValueError(
                "Missing function_name for "
                "functional problem: "
                f"{problem.problem_id}"
            )

    @staticmethod
    def _validate_test_case(
        *,
        test_case: dict[str, Any],
        test_index: int,
        problem_id: str,
    ) -> None:
        if not isinstance(
            test_case,
            dict,
        ):
            raise TypeError(
                "Test case must be dict: "
                f"{problem_id}, "
                f"index={test_index}"
            )

        for field_name in (
            "input",
            "output",
        ):
            if field_name not in test_case:
                raise ValueError(
                    f"Missing {field_name}: "
                    f"{problem_id}, "
                    f"index={test_index}"
                )

            if not isinstance(
                test_case[field_name],
                str,
            ):
                raise TypeError(
                    f"{field_name} must "
                    f"be str: "
                    f"{problem_id}, "
                    f"index={test_index}"
                )

    @staticmethod
    def test_pass_ratio(
        result: EvaluationResult,
    ) -> float:
        """
        Convenience helper for diagnostic analysis.
        """

        if result.total_tests == 0:
            return 0.0

        return (
            result.passed_tests
            / result.total_tests
        )