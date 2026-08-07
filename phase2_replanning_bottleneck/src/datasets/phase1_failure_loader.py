"""
Phase 1 실패 trajectory 로더.

Phase 2는 LiveCodeBench를 다시 읽고 initial code를 다시 생성하지 않는다.
Phase 1 Direct 실행 결과(results.jsonl)에서 refinement 가능한 실패 레코드만 골라
FailureCase로 변환한다.

Flow:
    /mnt/hdd/project_sLM_planning/output/direct_500_stdin/results.jsonl
        -> Phase1FailureLoader
        -> FailureCase
            - ProblemExample
            - initial code / generation info
            - ExecutionFeedback

Phase 1 results.jsonl은 매우 크기 때문에 전체 파일을 메모리에 올리지 않고
한 줄씩 streaming 한다.

주의:
    - Phase 1 레코드에는 public/private test 구분이 남아 있지 않다.
      복원된 ProblemExample은 모든 테스트를 public_tests에 넣는다.
      Phase 2 Evaluator는 include_public_tests=True 사용을 전제로 한다.

    - starter_code는 Phase 1 결과에 남아 있지 않다.
      현재 실험은 stdin 문제만 사용하므로 빈 문자열로 복원한다.

    - feedback은 첫 번째 실패 테스트를 기준으로 생성한다.

Usage:
    python -m src.datasets.phase1_failure_loader
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from src.schemas import (
    ExecutionFeedback,
    FailureCase,
    ProblemExample,
)


# ---------------------------------------------------------------------------
# Refinement 대상에서 제외할 상태
# ---------------------------------------------------------------------------

EXCLUDED_STATUSES = {
    "EXTRACTION_ERROR",
    "UNSUPPORTED_TEST_TYPE",
    "NO_TESTS",
}


class Phase1FailureLoader:
    """
    Phase 1 Direct results.jsonl에서 refinement 가능한 실패 trajectory를 읽는다.
    """

    def __init__(
        self,
        results_path: str | Path,
        *,
        limit: int | None = None,
        difficulties: list[str] | None = None,
        max_feedback_chars: int = 2000,
        include_statuses: set[str] | None = None,
    ) -> None:
        self.results_path = Path(results_path)
        self.limit = limit
        self.difficulties = difficulties
        self.max_feedback_chars = max_feedback_chars
        self.include_statuses = include_statuses

        self._validate_path()

        if self.limit is not None and self.limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        if self.max_feedback_chars <= 0:
            raise ValueError(
                "max_feedback_chars must be greater than 0."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Iterator[FailureCase]:
        """
        FailureCase를 하나씩 streaming한다.
        """

        yielded = 0

        for record in self.iter_records():
            if not self._is_failure(record):
                continue

            example = self._build_example(record)
            feedback = self._build_feedback(record)

            yield FailureCase(
                example=example,

                initial_code=str(
                    record.get("extracted_code") or ""
                ),
                initial_raw_output=str(
                    record.get("raw_output") or ""
                ),

                initial_status=str(
                    record.get("status") or "UNKNOWN"
                ),
                initial_passed_tests=int(
                    record.get("passed_tests") or 0
                ),
                initial_total_tests=int(
                    record.get("total_tests") or 0
                ),

                feedback=feedback,

                initial_prompt_tokens=int(
                    record.get("prompt_tokens") or 0
                ),
                initial_completion_tokens=int(
                    record.get("completion_tokens") or 0
                ),
                initial_generation_time=float(
                    record.get("generation_time") or 0.0
                ),
            )

            yielded += 1

            if (
                self.limit is not None
                and yielded >= self.limit
            ):
                break

    def iter_records(
        self,
    ) -> Iterator[dict[str, Any]]:
        """
        results.jsonl을 한 줄씩 streaming한다.
        """

        with self.results_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Invalid JSON at line "
                        f"{line_number}: "
                        f"{self.results_path}"
                    ) from error

                self._validate_record(
                    record,
                    line_number,
                )

                yield record

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_path(self) -> None:
        if not self.results_path.exists():
            raise FileNotFoundError(
                "Phase 1 results file not found: "
                f"{self.results_path}"
            )

        if not self.results_path.is_file():
            raise ValueError(
                "Phase 1 results path is not a file: "
                f"{self.results_path}"
            )

    @staticmethod
    def _validate_record(
        record: dict[str, Any],
        line_number: int,
    ) -> None:
        """
        Phase 2에 필요한 최소 필드만 검증한다.
        """

        required_fields = (
            "problem_id",
            "passed",
            "status",
            "extracted_code",
            "test_results",
        )

        missing = [
            field
            for field in required_fields
            if field not in record
        ]

        if missing:
            raise ValueError(
                f"Missing required fields at line "
                f"{line_number}: {missing}"
            )

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _is_failure(
        self,
        record: dict[str, Any],
    ) -> bool:
        """
        refinement 대상인지 판정한다.
        """

        if record.get("passed") is True:
            return False

        status = str(
            record.get("status") or ""
        )

        if status in EXCLUDED_STATUSES:
            return False

        if (
            self.include_statuses is not None
            and status not in self.include_statuses
        ):
            return False

        if self.difficulties is not None:
            difficulty = record.get(
                "difficulty"
            )

            if difficulty not in self.difficulties:
                return False

        return True

    # ------------------------------------------------------------------
    # Phase 1 record -> ProblemExample
    # ------------------------------------------------------------------

    def _build_example(
        self,
        record: dict[str, Any],
    ) -> ProblemExample:
        """
        Phase 1 record에서 evaluator용 ProblemExample을 복원한다.
        """

        public_tests: list[dict[str, Any]] = []

        for test_result in (
            record.get("test_results") or []
        ):
            input_text = test_result.get(
                "input_text"
            )

            expected_output = test_result.get(
                "expected_output"
            )

            if (
                input_text is None
                or expected_output is None
            ):
                continue

            public_tests.append(
                {
                    "input": input_text,
                    "output": expected_output,
                }
            )

        if not public_tests:
            raise ValueError(
                "Could not reconstruct tests for "
                f"problem_id={record.get('problem_id')}"
            )

        return ProblemExample(
            problem_id=str(
                record["problem_id"]
            ),
            title=str(
                record.get("title") or ""
            ),

            # Phase 1 formatted_prompt를 문제 prompt로 재사용.
            # Phase 2에서는 Phase 1의 generation용 formatted_prompt가 아니라
            # 원본 problem statement를 사용한다.
            prompt=str(
                record.get("problem") or ""
            ),

            platform=str(
                record.get("platform") or ""
            ),
            contest_id=str(
                record.get("contest_id") or ""
            ),
            contest_date=str(
                record.get("contest_date") or ""
            ),
            difficulty=str(
                record.get("difficulty") or ""
            ),

            starter_code="",

            public_tests=public_tests,
            private_tests=[],

            metadata={
                "phase1_source": str(
                    self.results_path
                ),
                "phase1_status": record.get(
                    "status"
                ),
            },

            test_type=str(
                record.get("test_type") or "stdin"
            ),
            source=str(
                record.get("dataset") or "livecodebench_v6"
            ),
        )

    # ------------------------------------------------------------------
    # Phase 1 record -> ExecutionFeedback
    # ------------------------------------------------------------------

    def _build_feedback(
        self,
        record: dict[str, Any],
    ) -> ExecutionFeedback:
        """
        Phase 1 실행 결과에서 refinement용 feedback을 생성한다.

        정책:
            - 첫 번째 실패 테스트 1개 사용
            - input / expected / actual 각각 길이 제한
            - runtime stderr / error_message 포함
        """

        test_results = (
            record.get("test_results") or []
        )

        failed_test = self._find_first_failed_test(
            test_results
        )

        status = str(
            record.get("status") or "UNKNOWN"
        )

        passed_tests = int(
            record.get("passed_tests") or 0
        )

        total_tests = int(
            record.get("total_tests") or 0
        )

        error_message = self._optional_str(
            record.get("error_message")
        )

        # --------------------------------------------------------------
        # 개별 실패 테스트가 존재하지 않는 경우
        # --------------------------------------------------------------

        if failed_test is None:
            stderr = self._extract_stderr(
                test_results
            )

            feedback_text = self._format_feedback_text(
                status=status,
                failed_test_index=None,
                failed_input=None,
                expected_output=None,
                actual_output=None,
                error_message=error_message,
                stderr=stderr,
            )

            return ExecutionFeedback(
                status=status,
                passed_tests=passed_tests,
                total_tests=total_tests,
                feedback_text=feedback_text,

                failed_test_index=None,
                failed_input=None,
                expected_output=None,
                actual_output=None,

                error_message=error_message,
                stderr=stderr,
            )

        # --------------------------------------------------------------
        # 대표 실패 테스트 존재
        # --------------------------------------------------------------

        failed_test_index = self._optional_int(
            failed_test.get("test_index")
        )

        failed_input = self._truncate(
            failed_test.get("input_text")
        )

        expected_output = self._truncate(
            failed_test.get("expected_output")
        )

        actual_output = self._truncate(
            failed_test.get("actual_output")
        )

        stderr = self._truncate(
            failed_test.get("stderr")
        ) or ""

        feedback_text = self._format_feedback_text(
            status=status,
            failed_test_index=failed_test_index,
            failed_input=failed_input,
            expected_output=expected_output,
            actual_output=actual_output,
            error_message=error_message,
            stderr=stderr,
        )

        return ExecutionFeedback(
            status=status,
            passed_tests=passed_tests,
            total_tests=total_tests,

            feedback_text=feedback_text,

            failed_test_index=failed_test_index,
            failed_input=failed_input,
            expected_output=expected_output,
            actual_output=actual_output,

            error_message=error_message,
            stderr=stderr,
        )

    # ------------------------------------------------------------------
    # Feedback helpers
    # ------------------------------------------------------------------

    def _find_first_failed_test(
        self,
        test_results: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        첫 번째 실패 test_result를 찾는다.
        """

        for test_result in test_results:
            if not self._test_passed(
                test_result
            ):
                return test_result

        return None

    @staticmethod
    def _test_passed(
        test_result: dict[str, Any],
    ) -> bool:
        """
        개별 테스트의 성공 여부를 판단한다.

        explicit passed 필드가 있으면 이를 우선 사용한다.
        """

        if "passed" in test_result:
            return bool(
                test_result["passed"]
            )

        expected = test_result.get(
            "expected_output"
        )

        actual = test_result.get(
            "actual_output"
        )

        if expected is None or actual is None:
            return False

        return (
            str(expected).strip()
            == str(actual).strip()
        )

    def _format_feedback_text(
        self,
        *,
        status: str,
        failed_test_index: int | None,
        failed_input: str | None,
        expected_output: str | None,
        actual_output: str | None,
        error_message: str | None,
        stderr: str,
    ) -> str:
        """
        Feedback-based Regeneration prompt에 삽입할 텍스트를 만든다.
        """

        lines: list[str] = [
            f"Execution Status: {status}",
        ]

        if failed_test_index is not None:
            lines.extend(
                [
                    "",
                    (
                        "Failed Test Index: "
                        f"{failed_test_index}"
                    ),
                ]
            )

        if failed_input is not None:
            lines.extend(
                [
                    "",
                    "Input:",
                    failed_input,
                ]
            )

        if expected_output is not None:
            lines.extend(
                [
                    "",
                    "Expected Output:",
                    expected_output,
                ]
            )

        if actual_output is not None:
            lines.extend(
                [
                    "",
                    "Actual Output:",
                    actual_output,
                ]
            )

        if error_message:
            lines.extend(
                [
                    "",
                    "Error Message:",
                    self._truncate(
                        error_message
                    ) or "",
                ]
            )

        if stderr:
            lines.extend(
                [
                    "",
                    "stderr:",
                    stderr,
                ]
            )

        return "\n".join(lines).strip()

    def _truncate(
        self,
        value: Any,
    ) -> str | None:
        """
        feedback field 길이를 제한한다.
        """

        if value is None:
            return None

        text = str(value)

        if len(text) <= self.max_feedback_chars:
            return text

        return (
            text[: self.max_feedback_chars]
            + "\n...[truncated]"
        )

    @staticmethod
    def _extract_stderr(
        test_results: list[dict[str, Any]],
    ) -> str:
        for test_result in test_results:
            stderr = test_result.get(
                "stderr"
            )

            if stderr:
                return str(stderr)

        return ""

    @staticmethod
    def _optional_str(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        return str(value)

    @staticmethod
    def _optional_int(
        value: Any,
    ) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    RESULTS_PATH = (
        "/mnt/hdd/project_sLM_planning/"
        "output/direct_500_stdin/results.jsonl"
    )

    loader = Phase1FailureLoader(
        RESULTS_PATH,
        limit=3,
    )

    print(
        "Inspect first 3 eligible Phase 1 failures"
    )
    print("=" * 80)

    for index, failure_case in enumerate(
        loader.load(),
        start=1,
    ):
        print()
        print(f"[FailureCase {index}]")
        print("-" * 80)

        print(
            "problem_id:",
            failure_case.example.problem_id,
        )

        print(
            "difficulty:",
            failure_case.example.difficulty,
        )

        print(
            "status:",
            failure_case.initial_status,
        )

        print(
            "tests:",
            (
                f"{failure_case.initial_passed_tests}/"
                f"{failure_case.initial_total_tests}"
            ),
        )

        print(
            "reconstructed_tests:",
            len(
                failure_case.example.public_tests
            ),
        )

        print(
            "initial_code_length:",
            len(
                failure_case.initial_code
            ),
        )

        print()
        print("[feedback_text]")
        print(
            failure_case.feedback.feedback_text
        )

        print()
        print("=" * 80)