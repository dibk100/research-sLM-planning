from __future__ import annotations

import json
import multiprocessing
from typing import Any

from src.execution.deepcoder.livecodebench import (
    run_test as deepcoder_lcb_run_test,
)
from src.schemas import (
    EvaluationResult,
    ProblemExample,
    TestCaseResult,
)


class TACOEvaluator:
    """
    DeepCoder/rLLM-compatible evaluator for the DeepCoder TACO dataset.

    Phase 4 reward path:

        ProblemExample.private_tests
            ->
        TACO dict-of-lists
            ->
        taco_to_lcb_format()
            ->
        lcb_check_correctness_v2()
            ->
        vendored DeepCoder/rLLM LiveCodeBench runner
            ->
        binary all-tests-pass result

    The vendored evaluator originates from:

        agentica-project/rllm
        commit:
        7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4

    Initial Phase 4 scope:
        - dataset = deepcoder_taco
        - evaluation_type = stdin

    Phase 1-3 LiveCodeBench evaluation remains completely separate.
    """

    DATASET_NAME = "deepcoder_taco"

    def __init__(
        self,
        *,
        timeout_seconds: int = 6,
        debug: bool = False,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than 0."
            )

        self.timeout_seconds = timeout_seconds
        self.debug = debug

    # ==================================================================
    # Public API
    # ==================================================================

    def evaluate(
        self,
        problem: ProblemExample,
        code: str,
    ) -> EvaluationResult:
        """
        Evaluate one generated code solution.

        Reward semantics:

            passed=True
                iff every test executed by the DeepCoder/rLLM
                backend passes.

            passed=False
                otherwise.
        """

        self._validate_problem(problem)

        if not isinstance(code, str):
            raise TypeError(
                "code must be str, "
                f"got {type(code).__name__}"
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

        if not problem.private_tests:
            return EvaluationResult(
                passed=False,
                status="NO_TESTS",
                passed_tests=0,
                total_tests=0,
                execution_time=0.0,
                test_results=[],
                error_message="No TACO tests available.",
            )

        # --------------------------------------------------------------
        # 1. Restore original TACO ground-truth representation.
        # --------------------------------------------------------------

        taco_tests = self._build_taco_tests(
            problem
        )

        # --------------------------------------------------------------
        # 2. Same conversion used by rLLM RewardCodeFn for TACO.
        # --------------------------------------------------------------

        lcb_tests = self.taco_to_lcb_format(
            taco_tests
        )

        # --------------------------------------------------------------
        # 3. DeepCoder/rLLM-style correctness evaluation.
        # --------------------------------------------------------------

        try:
            is_correct, details = (
                self.lcb_check_correctness_v2(
                    sample=lcb_tests,
                    generation=code,
                    timeout=self.timeout_seconds,
                    debug=self.debug,
                )
            )

        except Exception as exc:
            return EvaluationResult(
                passed=False,
                status="EVALUATION_ERROR",
                passed_tests=0,
                total_tests=0,
                execution_time=0.0,
                test_results=[],
                error_message=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        # --------------------------------------------------------------
        # 4. Convert rLLM result to local project schema.
        # --------------------------------------------------------------

        return self._convert_result(
            is_correct=bool(is_correct),
            details=details,
        )

    # ==================================================================
    # Local ProblemExample -> TACO format
    # ==================================================================

    @staticmethod
    def _build_taco_tests(
        problem: ProblemExample,
    ) -> dict[str, list[str]]:
        """
        Convert local private_tests back into TACO format.

        Local:

            [
                {
                    "input": "...",
                    "output": "..."
                },
                ...
            ]

        TACO:

            {
                "inputs": [...],
                "outputs": [...]
            }
        """

        inputs: list[str] = []
        outputs: list[str] = []

        for test_index, test_case in enumerate(
            problem.private_tests
        ):
            if not isinstance(
                test_case,
                dict,
            ):
                raise TypeError(
                    "private_tests entries must be dict, "
                    f"problem={problem.problem_id}, "
                    f"test_index={test_index}"
                )

            if "input" not in test_case:
                raise KeyError(
                    f"Missing test input: "
                    f"problem={problem.problem_id}, "
                    f"test_index={test_index}"
                )

            if "output" not in test_case:
                raise KeyError(
                    f"Missing test output: "
                    f"problem={problem.problem_id}, "
                    f"test_index={test_index}"
                )

            test_input = test_case["input"]
            test_output = test_case["output"]

            if not isinstance(
                test_input,
                str,
            ):
                raise TypeError(
                    "TACO stdin test input must be str, "
                    f"problem={problem.problem_id}, "
                    f"test_index={test_index}, "
                    f"got={type(test_input).__name__}"
                )

            if not isinstance(
                test_output,
                str,
            ):
                raise TypeError(
                    "TACO stdin test output must be str, "
                    f"problem={problem.problem_id}, "
                    f"test_index={test_index}, "
                    f"got={type(test_output).__name__}"
                )

            inputs.append(
                test_input
            )

            outputs.append(
                test_output
            )

        if len(inputs) != len(outputs):
            raise RuntimeError(
                "Internal TACO input/output length mismatch."
            )

        return {
            "inputs": inputs,
            "outputs": outputs,
        }

    # ==================================================================
    # rLLM: taco_to_lcb_format
    # ==================================================================

    @staticmethod
    def taco_to_lcb_format(
        tests: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Reproduce rLLM's taco_to_lcb_format().

        This is intentionally kept behaviorally identical to the
        implementation in rLLM code_reward.py.
        """

        inputs = tests.get(
            "inputs",
            [],
        )

        outputs = tests.get(
            "outputs",
            [],
        )

        n = max(
            len(inputs),
            len(outputs),
        )

        test_cases: list[
            dict[str, Any]
        ] = []

        for i in range(n):
            inp = (
                inputs[i]
                if i < len(inputs)
                else (
                    inputs[0]
                    if inputs
                    else ""
                )
            )

            out = (
                outputs[i]
                if i < len(outputs)
                else (
                    outputs[0]
                    if outputs
                    else ""
                )
            )

            # Same behavior as rLLM.
            if isinstance(out, list):
                out = (
                    out[0]
                    if out
                    else ""
                )

            test_case: dict[
                str,
                Any,
            ] = {
                "input": inp,
                "output": out,
                "metadata": {},
            }

            if "fn_name" in tests:
                test_case[
                    "testtype"
                ] = "functional"

                test_case[
                    "metadata"
                ][
                    "func_name"
                ] = tests[
                    "fn_name"
                ]

            test_cases.append(
                test_case
            )

        return test_cases

    # ==================================================================
    # rLLM: postprocess_lcb_sample
    # ==================================================================

    @staticmethod
    def postprocess_lcb_sample(
        sample: list[
            dict[str, Any]
        ],
    ) -> dict[str, str]:
        """
        Reproduce rLLM's postprocess_lcb_sample().
        """

        if not sample:
            raise ValueError(
                "sample must contain at least one test case."
            )

        sample_inputs = [
            item["input"]
            for item in sample
        ]

        sample_outputs = [
            item["output"]
            for item in sample
        ]

        sample_dict: dict[
            str,
            Any,
        ] = {
            "inputs": sample_inputs,
            "outputs": sample_outputs,
        }

        if (
            sample[0].get(
                "testtype"
            )
            == "functional"
        ):
            metadata = (
                sample[0].get(
                    "metadata",
                    {},
                )
            )

            fn_name = metadata.get(
                "func_name"
            )

            if fn_name is None:
                raise ValueError(
                    "Functional TACO sample "
                    "is missing func_name."
                )

            sample_dict[
                "fn_name"
            ] = fn_name

        return {
            "input_output": json.dumps(
                sample_dict,
                ensure_ascii=False,
            )
        }

    # ==================================================================
    # rLLM: multiprocessing wrapper
    # ==================================================================

    @staticmethod
    def _temp_run(
        sample: dict[str, str],
        generation: str,
        debug: bool,
        result: Any,
        metadata_list: Any,
        timeout: int,
    ) -> None:
        """
        Same role as rLLM's _temp_run().
        """

        try:
            res, metadata = (
                deepcoder_lcb_run_test(
                    sample,
                    test=generation,
                    debug=debug,
                    timeout=timeout,
                )
            )

            result.append(
                res
            )

            metadata_list.append(
                metadata
            )

        except Exception as exc:
            if debug:
                print(
                    "[TACOEvaluator] "
                    "deepcoder_lcb_run_test "
                    "raised exception: "
                    f"{type(exc).__name__}: {exc}"
                )

    # ==================================================================
    # rLLM: lcb_check_correctness_v2
    # ==================================================================

    @classmethod
    def lcb_check_correctness_v2(
        cls,
        *,
        sample: list[
            dict[str, Any]
        ],
        generation: str,
        timeout: int = 6,
        debug: bool = False,
    ) -> tuple[
        bool,
        dict[str, Any],
    ]:
        """
        Reproduce rLLM's lcb_check_correctness_v2() using the vendored
        DeepCoder/rLLM LiveCodeBench runner.

        Note:
        The underlying runner is fail-fast. Therefore the number of
        returned test results can be smaller than the number of
        available tests.
        """

        if not sample:
            raise ValueError(
                "Sample must contain at least one test case."
            )

        processed_sample = (
            cls.postprocess_lcb_sample(
                sample
            )
        )

        manager = (
            multiprocessing.Manager()
        )

        result = manager.list()
        metadata_list = (
            manager.list()
        )

        process = (
            multiprocessing.Process(
                target=cls._temp_run,
                args=(
                    processed_sample,
                    generation,
                    debug,
                    result,
                    metadata_list,
                    timeout,
                ),
            )
        )

        process.start()

        in_outs = json.loads(
            processed_sample[
                "input_output"
            ]
        )

        num_available_tests = len(
            in_outs["inputs"]
        )

        # Same global timeout formula as rLLM.
        global_timeout = (
            (timeout + 1)
            * num_available_tests
            + 5
        )

        process.join(
            timeout=global_timeout
        )

        detailed_results: dict[
            str,
            Any,
        ] = {
            "all_passed": False,
            "test_results": [],
            "total_tests": 0,
            "passed_tests": 0,
            "available_tests": (
                num_available_tests
            ),
        }

        # --------------------------------------------------------------
        # Global process timeout
        # --------------------------------------------------------------

        if process.is_alive():
            process.kill()
            process.join()

            detailed_results[
                "total_tests"
            ] = num_available_tests

            detailed_results[
                "test_results"
            ] = [
                {
                    "input": inp,
                    "expected": out,
                    "passed": False,
                    "error": (
                        "global timeout"
                    ),
                    "error_message": (
                        "Global timeout"
                    ),
                    "output": None,
                }
                for inp, out
                in zip(
                    in_outs[
                        "inputs"
                    ],
                    in_outs[
                        "outputs"
                    ],
                )
            ]

            detailed_results[
                "error"
            ] = "global timeout"

            return (
                False,
                detailed_results,
            )

        # --------------------------------------------------------------
        # No result returned from child process
        # --------------------------------------------------------------

        if not result:
            detailed_results[
                "total_tests"
            ] = num_available_tests

            detailed_results[
                "error"
            ] = (
                "No result returned "
                "from DeepCoder LCB runner."
            )

            return (
                False,
                detailed_results,
            )

        raw_results = list(
            result[0]
        )

        if metadata_list:
            raw_metadata = (
                metadata_list[0]
            )

            try:
                metadata = dict(
                    raw_metadata
                )
            except Exception:
                metadata = {}

        else:
            metadata = {}

        # --------------------------------------------------------------
        # Build detailed results
        # --------------------------------------------------------------

        test_results: list[
            dict[str, Any]
        ] = []

        for inp, out, raw_result in zip(
            in_outs["inputs"],
            in_outs["outputs"],
            raw_results,
        ):
            passed = (
                raw_result is True
            )

            test_results.append(
                {
                    "input": inp,
                    "expected": out,
                    "passed": passed,

                    "error": metadata.get(
                        "error"
                    ),

                    "error_message": (
                        metadata.get(
                            "error_message"
                        )
                    ),

                    "output": metadata.get(
                        "output"
                    ),

                    "raw_result": (
                        raw_result
                    ),
                }
            )

        passed_tests = sum(
            1
            for raw_result in raw_results
            if raw_result is True
        )

        all_passed = all(
            raw_result is True
            for raw_result in raw_results
        )

        detailed_results[
            "test_results"
        ] = test_results

        detailed_results[
            "total_tests"
        ] = len(
            raw_results
        )

        detailed_results[
            "passed_tests"
        ] = passed_tests

        detailed_results[
            "all_passed"
        ] = all_passed

        detailed_results[
            "metadata"
        ] = metadata

        return (
            all_passed,
            detailed_results,
        )

    # ==================================================================
    # rLLM result -> local EvaluationResult
    # ==================================================================

    @classmethod
    def _convert_result(
        cls,
        *,
        is_correct: bool,
        details: dict[
            str,
            Any,
        ],
    ) -> EvaluationResult:
        raw_results = details.get(
            "test_results",
            [],
        )

        if not isinstance(
            raw_results,
            list,
        ):
            raw_results = []

        test_results: list[
            TestCaseResult
        ] = []

        for test_index, raw in enumerate(
            raw_results
        ):
            if not isinstance(
                raw,
                dict,
            ):
                raw = {
                    "passed": False,
                    "error": str(raw),
                }

            passed = bool(
                raw.get(
                    "passed",
                    False,
                )
            )

            if passed:
                status = "PASS"
            else:
                status = (
                    cls._infer_failure_status(
                        raw
                    )
                )

            actual_output = raw.get(
                "output"
            )

            error_message = raw.get(
                "error_message"
            )

            test_results.append(
                TestCaseResult(
                    test_index=test_index,

                    passed=passed,
                    status=status,

                    input_text=str(
                        raw.get(
                            "input",
                            "",
                        )
                    ),

                    expected_output=str(
                        raw.get(
                            "expected",
                            "",
                        )
                    ),

                    actual_output=(
                        ""
                        if actual_output is None
                        else str(actual_output)
                    ),

                    # rLLM lcb_check_correctness_v2 does not expose
                    # per-test timing.
                    execution_time=0.0,

                    return_code=None,

                    stderr=(
                        ""
                        if error_message is None
                        else str(
                            error_message
                        )
                    ),

                    metadata={
                        "raw_result": (
                            raw.get(
                                "raw_result"
                            )
                        ),
                        "error": (
                            raw.get(
                                "error"
                            )
                        ),
                        "error_message": (
                            raw.get(
                                "error_message"
                            )
                        ),
                    },
                )
            )

        passed_tests = cls._safe_int(
            details.get(
                "passed_tests",
                0,
            )
        )

        total_tests = cls._safe_int(
            details.get(
                "total_tests",
                len(test_results),
            )
        )

        if is_correct:
            overall_status = "PASS"

        elif test_results:
            overall_status = (
                cls._infer_overall_status(
                    test_results
                )
            )

        else:
            overall_status = (
                cls._status_from_details(
                    details
                )
            )

        error_message = (
            cls._extract_error_message(
                details=details,
                test_results=test_results,
            )
        )

        return EvaluationResult(
            passed=is_correct,
            status=overall_status,

            passed_tests=passed_tests,
            total_tests=total_tests,

            # Current rLLM correctness wrapper does not expose
            # aggregate execution time directly.
            execution_time=0.0,

            test_results=test_results,

            error_message=error_message,
        )

    # ==================================================================
    # Failure status helpers
    # ==================================================================

    @staticmethod
    def _infer_failure_status(
        raw: dict[str, Any],
    ) -> str:
        raw_result = raw.get(
            "raw_result"
        )

        # DeepCoder vendored LCB result codes.
        if raw_result == -3:
            return (
                "TIME_LIMIT_EXCEEDED"
            )

        if raw_result == -4:
            return (
                "RUNTIME_ERROR"
            )

        if (
            raw_result is False
            or raw_result == -2
        ):
            return (
                "WRONG_ANSWER"
            )

        error_text = " ".join(
            [
                str(
                    raw.get(
                        "error",
                        "",
                    )
                    or ""
                ),
                str(
                    raw.get(
                        "error_message",
                        "",
                    )
                    or ""
                ),
            ]
        ).lower()

        if (
            "time limit exceeded"
            in error_text
            or "timeout"
            in error_text
            or "timeoutexception"
            in error_text
            or "global timeout"
            in error_text
        ):
            return (
                "TIME_LIMIT_EXCEEDED"
            )

        if (
            "runtime error"
            in error_text
            or "runtimeerror"
            in error_text
            or "exception"
            in error_text
        ):
            return (
                "RUNTIME_ERROR"
            )

        if (
            "wrong answer"
            in error_text
            or "mismatched output"
            in error_text
        ):
            return (
                "WRONG_ANSWER"
            )

        return "FAILED"

    @staticmethod
    def _infer_overall_status(
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
    def _status_from_details(
        details: dict[str, Any],
    ) -> str:
        text = " ".join(
            [
                str(
                    details.get(
                        "error",
                        "",
                    )
                    or ""
                ),
                str(
                    details.get(
                        "error_message",
                        "",
                    )
                    or ""
                ),
            ]
        ).lower()

        if "timeout" in text:
            return (
                "TIME_LIMIT_EXCEEDED"
            )

        if (
            "runtime"
            in text
            or "exception"
            in text
        ):
            return (
                "RUNTIME_ERROR"
            )

        return "FAILED"

    # ==================================================================
    # Error helpers
    # ==================================================================

    @staticmethod
    def _extract_error_message(
        *,
        details: dict[str, Any],
        test_results: list[
            TestCaseResult
        ],
    ) -> str | None:
        direct_error = (
            details.get(
                "error_message"
            )
            or details.get(
                "error"
            )
        )

        if direct_error:
            return str(
                direct_error
            )

        metadata = details.get(
            "metadata"
        )

        if isinstance(
            metadata,
            dict,
        ):
            metadata_error = (
                metadata.get(
                    "error_message"
                )
                or metadata.get(
                    "error"
                )
            )

            if metadata_error:
                return str(
                    metadata_error
                )

        for result in test_results:
            if result.passed:
                continue

            if result.stderr:
                return result.stderr

            metadata_error = (
                result.metadata.get(
                    "error_message"
                )
                or result.metadata.get(
                    "error"
                )
            )

            if metadata_error:
                return str(
                    metadata_error
                )

        return None

    # ==================================================================
    # Validation
    # ==================================================================

    @classmethod
    def _validate_problem(
        cls,
        problem: ProblemExample,
    ) -> None:
        if not isinstance(
            problem,
            ProblemExample,
        ):
            raise TypeError(
                "problem must be ProblemExample, "
                f"got {type(problem).__name__}"
            )

        if (
            problem.dataset
            != cls.DATASET_NAME
        ):
            raise ValueError(
                "TACOEvaluator received "
                "unsupported dataset: "
                f"{problem.dataset!r}. "
                f"Expected "
                f"{cls.DATASET_NAME!r}."
            )

        if (
            problem.evaluation_type
            != "stdin"
        ):
            raise ValueError(
                "Initial Phase 4 "
                "TACOEvaluator supports "
                "stdin only, got "
                f"{problem.evaluation_type!r}."
            )

    @staticmethod
    def _safe_int(
        value: Any,
    ) -> int:
        try:
            return int(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0