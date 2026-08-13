# src/execution/livecodebench_evaluator.py

from __future__ import annotations

import json
from typing import Any

from lcb_runner.evaluation.compute_code_generation_metrics import (
    check_correctness,
)

from src.schemas import (
    EvaluationResult,
    ProblemExample,
    TestCaseResult,
)


class LiveCodeBenchEvaluator:
    """
    Thin wrapper around the official LiveCodeBench code-generation evaluator.

    Final correctness is determined only by the official LiveCodeBench
    evaluation logic.
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

        if not include_public_tests and not include_private_tests:
            raise ValueError(
                "At least one test group must be enabled."
            )

        self.timeout_seconds = timeout_seconds
        self.debug = debug
        self.include_public_tests = include_public_tests
        self.include_private_tests = include_private_tests

    def evaluate(
        self,
        problem: ProblemExample,
        code: str,
    ) -> EvaluationResult:
        """
        Evaluate one generated solution using the official LiveCodeBench
        code-generation evaluator.
        """

        self._validate_problem(problem)

        if not isinstance(code, str):
            raise TypeError(
                f"code must be str, got {type(code).__name__}"
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

        test_cases = self._collect_test_cases(problem)

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

        sample = self._build_official_sample(
            problem=problem,
            test_cases=test_cases,
        )

        results, metadata = check_correctness(
            sample=sample,
            generation=code,
            timeout=self.timeout_seconds,
            debug=self.debug,
        )

        return self._convert_result(
            results=results,
            metadata=metadata,
            total_tests=len(test_cases),
        )

    def _collect_test_cases(
        self,
        problem: ProblemExample,
    ) -> list[dict[str, Any]]:
        test_cases: list[dict[str, Any]] = []

        if self.include_public_tests:
            test_cases.extend(problem.public_tests)

        if self.include_private_tests:
            test_cases.extend(problem.private_tests)

        for index, test_case in enumerate(test_cases):
            self._validate_test_case(
                test_case=test_case,
                test_index=index,
                problem_id=problem.problem_id,
            )

        return test_cases

    @staticmethod
    def _build_official_sample(
        *,
        problem: ProblemExample,
        test_cases: list[dict[str, Any]],
    ) -> dict[str, str]:
        """
        Convert ProblemExample into the input format expected by
        LiveCodeBench's official run_test/check_correctness pipeline.
        """

        input_output: dict[str, Any] = {
            "inputs": [
                test_case["input"]
                for test_case in test_cases
            ],
            "outputs": [
                test_case["output"]
                for test_case in test_cases
            ],
        }

        if problem.evaluation_type == "functional":
            if not problem.function_name:
                raise ValueError(
                    f"Missing function_name for functional problem: "
                    f"{problem.problem_id}"
                )

            input_output["fn_name"] = problem.function_name

        return {
            "input_output": json.dumps(
                input_output,
                ensure_ascii=False,
            )
        }

    @staticmethod
    def _convert_result(
        *,
        results: list[Any],
        metadata: dict[str, Any],
        total_tests: int,
    ) -> EvaluationResult:
        """
        Convert LiveCodeBench's official result codes into the local schema.

        Official result codes observed in LiveCodeBench:
            True  -> PASS
            False / -2 -> WRONG_ANSWER
            -3 -> TIME_LIMIT_EXCEEDED
            -4 -> RUNTIME_ERROR

        LiveCodeBench is fail-fast, so len(results) can be smaller than
        total_tests.
        """

        test_results: list[TestCaseResult] = []

        for index, result in enumerate(results):
            passed, status = (
                LiveCodeBenchEvaluator._map_status(result)
            )

            test_results.append(
                TestCaseResult(
                    test_index=index,
                    passed=passed,
                    status=status,
                    input_text="",
                    expected_output="",
                    actual_output="",
                    execution_time=0.0,
                    return_code=None,
                    stderr="",
                )
            )

        passed_tests = sum(
            test_result.passed
            for test_result in test_results
        )

        official_passed = (
            total_tests > 0
            and len(results) == total_tests
            and all(
                test_result.passed
                for test_result in test_results
            )
        )

        status = (
            "PASS"
            if official_passed
            else LiveCodeBenchEvaluator._determine_overall_status(
                test_results=test_results,
                metadata=metadata,
            )
        )

        execution_time = LiveCodeBenchEvaluator._extract_execution_time(
            metadata
        )

        error_message = LiveCodeBenchEvaluator._extract_error_message(
            metadata
        )

        return EvaluationResult(
            passed=official_passed,
            status=status,
            passed_tests=passed_tests,
            total_tests=total_tests,
            execution_time=execution_time,
            test_results=test_results,
            error_message=error_message,
        )

    @staticmethod
    def _map_status(
        result: Any,
    ) -> tuple[bool, str]:
        # bool is a subclass of int in Python, so check True first.
        if result is True:
            return True, "PASS"

        if result is False or result == -2:
            return False, "WRONG_ANSWER"

        if result == -3:
            return False, "TIME_LIMIT_EXCEEDED"

        if result == -4:
            return False, "RUNTIME_ERROR"

        if result == -5:
            return False, "TEST_RUNNER_ERROR"

        return False, "FAILED"

    @staticmethod
    def _determine_overall_status(
        *,
        test_results: list[TestCaseResult],
        metadata: dict[str, Any],
    ) -> str:
        metadata_error_code = metadata.get("error_code")

        if metadata_error_code == -5:
            return "TEST_RUNNER_ERROR"

        statuses = {
            result.status
            for result in test_results
            if not result.passed
        }

        priority = (
            "TIME_LIMIT_EXCEEDED",
            "RUNTIME_ERROR",
            "WRONG_ANSWER",
            "TEST_RUNNER_ERROR",
            "FAILED",
        )

        for status in priority:
            if status in statuses:
                return status

        metadata_message = str(
            metadata.get("error_message", "")
        ).lower()

        if "time limit" in metadata_message:
            return "TIME_LIMIT_EXCEEDED"

        if "runtime" in metadata_message:
            return "RUNTIME_ERROR"

        if "wrong answer" in metadata_message:
            return "WRONG_ANSWER"

        return "FAILED"

    @staticmethod
    def _extract_execution_time(
        metadata: dict[str, Any],
    ) -> float:
        value = metadata.get("execution time", 0.0)

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _extract_error_message(
        metadata: dict[str, Any],
    ) -> str | None:
        message = (
            metadata.get("error_message")
            or metadata.get("error")
        )

        if message is None:
            return None

        return str(message)

    @staticmethod
    def _validate_problem(
        problem: ProblemExample,
    ) -> None:
        if problem.dataset != "livecodebench_v6":
            raise ValueError(
                f"LiveCodeBenchEvaluator received unsupported dataset: "
                f"{problem.dataset}"
            )

        if problem.evaluation_type not in {
            "stdin",
            "functional",
        }:
            raise ValueError(
                f"Unsupported evaluation_type: "
                f"{problem.evaluation_type}"
            )

        if (
            problem.evaluation_type == "functional"
            and not problem.function_name
        ):
            raise ValueError(
                f"Missing function_name for functional problem: "
                f"{problem.problem_id}"
            )

    @staticmethod
    def _validate_test_case(
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