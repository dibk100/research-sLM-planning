"""
Phase2 : 코드 실패 정보(오류 메세지) 기반으로 plan을 생성해서 코드를 구현할 수 있는지 검사하는 작업.

Phase 1 Direct results.jsonl에서 실패 문제만 선택하여 phase2의 활용 데이터로 사용.
input_text, expected_output, actual_output, stderr는 전체 test 결과가 아니라 첫 번째 실패 테스트(passed == False)에서 추출한다.

입력:
Phase 1 Direct results.jsonl

출력:
Phase1FailureRecord
    problem_id
    problem
    difficulty

    failed_code

    first_failed_test:
        test_index
        input_text
        expected_output
        actual_output
        evaluator_message

    original_status
    passed_tests
    total_tests
    test_pass_ratio

"""
"""Load refinable failure cases from Phase 1 Direct results."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REFINABLE_STATUSES = {
    "WRONG_ANSWER",
    "RUNTIME_ERROR",
    "TIME_LIMIT_EXCEEDED",
}

@dataclass(frozen=True)
class Phase1FailureRecord:
    """A refinable failure trajectory from Phase 1 Direct."""

    # Problem information
    problem_id: str
    problem: str
    difficulty: str | None
    starter_code : str
    
    # Failed generation
    extracted_code: str

    # Overall Phase 1 evaluation
    status: str
    passed_tests: int
    total_tests: int
    test_pass_ratio: float

    # First failing test feedback
    test_index: int
    input_text: str
    expected_output: str
    actual_output: str
    stderr: str


def _load_json_record(
    line: str,
    *,
    line_number: int,
    path: Path,
) -> dict[str, Any]:
    """Parse one JSONL record."""

    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON at line {line_number}: {path}"
        ) from error

    if not isinstance(record, dict):
        raise TypeError(
            "Phase 1 result record must be a JSON object "
            f"at line {line_number}: {path}"
        )

    return record


def _find_first_failed_test(
    test_results: list[dict[str, Any]],
    *,
    problem_id: str,
) -> dict[str, Any]:
    """Return the first test case whose evaluation failed."""

    for test_result in test_results:
        if test_result.get("passed") is False:
            return test_result

    raise ValueError(
        "No failed test found for failed Phase 1 record: "
        f"{problem_id}"
    )


def _build_failure_record(
    record: dict[str, Any],
) -> Phase1FailureRecord:
    """Convert a Phase 1 result into a Phase 2 failure record."""

    problem_id = str(
        record.get("problem_id", "")
    ).strip()

    if not problem_id:
        raise ValueError(
            "Phase 1 result has empty problem_id."
        )

    problem = str(
        record.get("problem", "")
    ).strip()

    if not problem:
        raise ValueError(
            f"Problem text is empty: {problem_id}"
        )

    extracted_code = str(
        record.get("extracted_code", "")
    ).strip()

    if not extracted_code:
        raise ValueError(
            f"Extracted code is empty: {problem_id}"
        )

    test_results = record.get(
        "test_results",
        [],
    )

    if not isinstance(test_results, list):
        raise TypeError(
            "test_results must be a list: "
            f"{problem_id}"
        )

    if not test_results:
        raise ValueError(
            f"test_results is empty: {problem_id}"
        )

    first_failed = _find_first_failed_test(
        test_results,
        problem_id=problem_id,
    )

    return Phase1FailureRecord(
        problem_id=problem_id,
        problem=problem,
        difficulty=record.get("difficulty"),
        
        starter_code=str(
            record.get(
                "starter_code",
                "",
            )
        ).strip(),

        extracted_code=extracted_code,

        status=str(
            record.get("status", "")
        ),
        passed_tests=int(
            record.get("passed_tests", 0)
        ),
        total_tests=int(
            record.get("total_tests", 0)
        ),
        test_pass_ratio=float(
            record.get("test_pass_ratio", 0.0)
        ),

        test_index=int(
            first_failed.get(
                "test_index",
                -1,
            )
        ),
        input_text=str(
            first_failed.get(
                "input_text",
                "",
            )
        ),
        expected_output=str(
            first_failed.get(
                "expected_output",
                "",
            )
        ),
        actual_output=str(
            first_failed.get(
                "actual_output",
                "",
            )
        ),
        stderr=str(
            first_failed.get(
                "stderr",
                "",
            )
        ),
    )


def load_phase1_failures(
    result_path: str | Path,
    *,
    limit: int | None = None,
) -> list[Phase1FailureRecord]:
    """

Load refinable failures from a Phase 1 Direct results.jsonl.

A record is considered refinable when:
- strategy == "direct"
- passed == False
- status is one of:
    WRONG_ANSWER
    RUNTIME_ERROR
    TIME_LIMIT_EXCEEDED
- parse_status == "SUCCESS"
- extracted_code is non-empty
- test_results contains at least one failed test

TEST_RUNNER_ERROR is excluded because it represents
an evaluation/infrastructure failure rather than a valid
model-level execution outcome.

The first failed test is used as execution feedback for Phase 2.

    """

    path = Path(result_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Phase 1 result file not found: {path}"
        )

    if limit is not None and limit <= 0:
        raise ValueError(
            "limit must be greater than 0."
        )

    failures: list[Phase1FailureRecord] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            record = _load_json_record(
                line,
                line_number=line_number,
                path=path,
            )

            # --------------------------------------------------
            # Phase 2 uses only Phase 1 Direct trajectories.
            # --------------------------------------------------

            strategy = str(
                record.get("strategy", "")
            ).strip()

            if strategy != "direct":
                raise ValueError(
                    "Phase 1 failure loader expects "
                    "Direct results only. "
                    f"Found strategy={strategy!r} "
                    f"at line {line_number}."
                )

            # --------------------------------------------------
            # Successful problems do not need refinement.
            # --------------------------------------------------

            if record.get("passed") is True:
                continue


            # --------------------------------------------------
            # Exclude evaluation/infrastructure failures.
            # Phase 2 targets model-level executable failures.
            # --------------------------------------------------

            status = str(
                record.get("status", "")
            ).strip()

            if status not in REFINABLE_STATUSES:
                continue

            # --------------------------------------------------
            # Phase 2 currently targets executable code
            # failures, not parsing failures.
            # --------------------------------------------------

            if record.get("parse_status") != "SUCCESS":
                continue

            extracted_code = str(
                record.get(
                    "extracted_code",
                    "",
                )
            ).strip()

            if not extracted_code:
                continue

            # --------------------------------------------------
            # At least one failed test must exist because
            # Phase 2 feedback is constructed from the first
            # failing test case.
            # --------------------------------------------------

            test_results = record.get(
                "test_results",
                [],
            )

            if not isinstance(test_results, list):
                raise TypeError(
                    "test_results must be a list "
                    f"at line {line_number}."
                )

            first_failed_exists = any(
                isinstance(test_result, dict)
                and test_result.get("passed") is False
                for test_result in test_results
            )

            if not first_failed_exists:
                continue

            failure = _build_failure_record(
                record
            )

            failures.append(
                failure
            )

            if (
                limit is not None
                and len(failures) >= limit
            ):
                break

    return failures