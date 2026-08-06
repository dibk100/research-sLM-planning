"""
Teacher-Plan 실패 사례를 추출하고 수작업 라벨링 파일을 생성한다.

분석 대상:
    Teacher-Plan PASS == False

Usage:

python -m archive.analyze_teacher_failures \
  --direct-path /mnt/hdd/project_sLM_planning/output/direct_50_stdin/results.jsonl \
  --self-plan-path /mnt/hdd/project_sLM_planning/output/self_plan_50_stdin/results.jsonl \
  --teacher-plan-path /mnt/hdd/project_sLM_planning/output/teacher_plan_50_stdin/results.jsonl \
  --output-dir ./archive/teacher_failures_50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DIRECT_PATH = Path(
    "/mnt/hdd/project_sLM_planning/output/"
    "direct_50_stdin/results.jsonl"
)

DEFAULT_SELF_PLAN_PATH = Path(
    "/mnt/hdd/project_sLM_planning/output/"
    "self_plan_50_stdin/results.jsonl"
)

DEFAULT_TEACHER_PLAN_PATH = Path(
    "/mnt/hdd/project_sLM_planning/output/"
    "teacher_plan_50_stdin/results.jsonl"
)

DEFAULT_OUTPUT_DIR = Path(
    "./archive/teacher_failures_50"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and analyze Teacher-Plan "
            "failure cases."
        )
    )

    parser.add_argument(
        "--direct-path",
        type=Path,
        default=DEFAULT_DIRECT_PATH,
    )

    parser.add_argument(
        "--self-plan-path",
        type=Path,
        default=DEFAULT_SELF_PLAN_PATH,
    )

    parser.add_argument(
        "--teacher-plan-path",
        type=Path,
        default=DEFAULT_TEACHER_PLAN_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--expected-count",
        type=int,
        default=26,
        help=(
            "Expected number of Teacher-Plan failures. "
            "Use -1 to disable this check."
        ),
    )

    return parser.parse_args()


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Result file not found: {path.resolve()}"
        )

    records: list[dict[str, Any]] = []

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

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}: {path}"
                ) from error

            if not isinstance(record, dict):
                raise TypeError(
                    f"Record at line {line_number} "
                    "is not a JSON object."
                )

            records.append(record)

    if not records:
        raise ValueError(
            f"No records found: {path}"
        )

    problem_ids = [
        str(record.get("problem_id", "")).strip()
        for record in records
    ]

    if any(not problem_id for problem_id in problem_ids):
        raise ValueError(
            f"Missing problem_id in {path}"
        )

    duplicated = (
        pd.Series(problem_ids)[
            pd.Series(problem_ids).duplicated(
                keep=False
            )
        ]
        .unique()
        .tolist()
    )

    if duplicated:
        raise ValueError(
            f"Duplicated problem IDs in {path}: "
            f"{sorted(duplicated)}"
        )

    return records


def index_by_problem_id(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(record["problem_id"]): record
        for record in records
    }


def normalize_passed(
    value: Any,
) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value == 1

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
            "true",
            "1",
            "pass",
            "passed",
        }:
            return True

        if normalized in {
            "false",
            "0",
            "fail",
            "failed",
        }:
            return False

    return False


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        if value is None:
            return default

        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_test_pass_ratio(
    record: dict[str, Any],
) -> float:
    passed_tests = safe_float(
        record.get("passed_tests")
    )

    total_tests = safe_float(
        record.get("total_tests")
    )

    if total_tests <= 0:
        return 0.0

    return passed_tests / total_tests


def extract_code(
    record: dict[str, Any],
) -> str:
    value = record.get(
        "extracted_code",
        "",
    )

    if isinstance(value, str):
        return value.strip()

    return ""


def extract_teacher_plan(
    record: dict[str, Any],
) -> str:
    value = record.get(
        "teacher_plan",
        "",
    )

    if isinstance(value, str):
        return value.strip()

    return ""


def extract_trace_output(
    record: dict[str, Any],
    candidate_names: set[str],
) -> str:
    trace = record.get(
        "strategy_trace",
        [],
    )

    if not isinstance(trace, list):
        return ""

    for step in trace:
        if not isinstance(step, dict):
            continue

        step_name = str(
            step.get("name", "")
        ).strip().lower()

        if step_name not in candidate_names:
            continue

        output = step.get(
            "raw_output",
            step.get("output", ""),
        )

        if isinstance(output, str):
            return output.strip()

    return ""


def extract_self_plan(
    record: dict[str, Any],
) -> str:
    candidate_fields = [
        "generated_plan",
        "plan",
        "self_plan",
        "implementation_plan",
    ]

    for field in candidate_fields:
        value = record.get(field)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return extract_trace_output(
        record,
        {
            "plan_generation",
            "planning",
            "generate_plan",
            "plan",
        },
    )


def validate_same_problem_set(
    direct_records: list[dict[str, Any]],
    self_records: list[dict[str, Any]],
    teacher_records: list[dict[str, Any]],
) -> None:
    direct_ids = {
        str(record["problem_id"])
        for record in direct_records
    }

    self_ids = {
        str(record["problem_id"])
        for record in self_records
    }

    teacher_ids = {
        str(record["problem_id"])
        for record in teacher_records
    }

    if direct_ids != self_ids:
        raise ValueError(
            "Direct and Self-Plan problem sets differ."
        )

    if direct_ids != teacher_ids:
        raise ValueError(
            "Direct and Teacher-Plan problem sets differ."
        )

    print(
        f"[PASS] Same problem set: "
        f"{len(direct_ids)} problems"
    )


def make_pattern(
    direct_passed: bool,
    self_passed: bool,
    teacher_passed: bool,
) -> str:
    return "-".join(
        [
            "P" if direct_passed else "F",
            "P" if self_passed else "F",
            "P" if teacher_passed else "F",
        ]
    )


def build_failure_rows(
    direct_index: dict[str, dict[str, Any]],
    self_index: dict[str, dict[str, Any]],
    teacher_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for problem_id in sorted(teacher_index):
        direct_record = direct_index[problem_id]
        self_record = self_index[problem_id]
        teacher_record = teacher_index[problem_id]

        teacher_passed = normalize_passed(
            teacher_record.get("passed")
        )

        if teacher_passed:
            continue

        direct_passed = normalize_passed(
            direct_record.get("passed")
        )

        self_passed = normalize_passed(
            self_record.get("passed")
        )

        teacher_ratio = calculate_test_pass_ratio(
            teacher_record
        )

        self_ratio = calculate_test_pass_ratio(
            self_record
        )

        row = {
            "problem_id": problem_id,
            "title": teacher_record.get(
                "title",
                self_record.get("title", ""),
            ),
            "difficulty": teacher_record.get(
                "difficulty",
                self_record.get("difficulty", ""),
            ),
            "three_strategy_pattern": make_pattern(
                direct_passed,
                self_passed,
                teacher_passed,
            ),
            "direct_passed": direct_passed,
            "self_plan_passed": self_passed,
            "teacher_plan_passed": teacher_passed,
            "direct_status": direct_record.get(
                "status",
                "",
            ),
            "self_status": self_record.get(
                "status",
                "",
            ),
            "teacher_status": teacher_record.get(
                "status",
                "",
            ),
            "teacher_passed_tests": (
                teacher_record.get(
                    "passed_tests",
                    0,
                )
            ),
            "teacher_total_tests": (
                teacher_record.get(
                    "total_tests",
                    0,
                )
            ),
            "teacher_test_pass_ratio": (
                teacher_ratio
            ),
            "self_test_pass_ratio": (
                self_ratio
            ),
            "teacher_minus_self_ratio": (
                teacher_ratio - self_ratio
            ),
            "teacher_prompt_tokens": (
                teacher_record.get(
                    "prompt_tokens",
                    0,
                )
            ),
            "teacher_completion_tokens": (
                teacher_record.get(
                    "completion_tokens",
                    0,
                )
            ),
            "teacher_total_tokens": (
                safe_float(
                    teacher_record.get(
                        "prompt_tokens"
                    )
                )
                + safe_float(
                    teacher_record.get(
                        "completion_tokens"
                    )
                )
            ),
            "teacher_generation_time": (
                teacher_record.get(
                    "generation_time",
                    0.0,
                )
            ),
            "teacher_execution_time": (
                teacher_record.get(
                    "execution_time",
                    0.0,
                )
            ),
            "teacher_error_message": (
                teacher_record.get(
                    "error_message",
                    "",
                )
            ),
            "teacher_plan": (
                extract_teacher_plan(
                    teacher_record
                )
            ),
            "teacher_extracted_code": (
                extract_code(
                    teacher_record
                )
            ),
            "self_generated_plan": (
                extract_self_plan(
                    self_record
                )
            ),
            "self_extracted_code": (
                extract_code(
                    self_record
                )
            ),

            # 수작업 라벨링 필드
            "teacher_plan_correctness": "",
            "implementation_fidelity": "",
            "primary_bottleneck": "",
            "failure_type": "",
            "algorithm_category": "",
            "missing_or_wrong_information": "",
            "analysis_note": "",
        }

        rows.append(row)

    return rows


def build_status_summary(
    failure_df: pd.DataFrame,
) -> pd.DataFrame:
    return (
        failure_df["teacher_status"]
        .value_counts(dropna=False)
        .rename_axis("teacher_status")
        .reset_index(name="count")
    )


def build_difficulty_summary(
    failure_df: pd.DataFrame,
) -> pd.DataFrame:
    result = (
        failure_df.groupby(
            "difficulty",
            dropna=False,
        )
        .agg(
            num_failures=(
                "problem_id",
                "count",
            ),
            mean_test_pass_ratio=(
                "teacher_test_pass_ratio",
                "mean",
            ),
            mean_total_tokens=(
                "teacher_total_tokens",
                "mean",
            ),
            mean_generation_time=(
                "teacher_generation_time",
                "mean",
            ),
        )
        .reset_index()
    )

    difficulty_order = {
        "easy": 0,
        "medium": 1,
        "hard": 2,
    }

    result["_order"] = (
        result["difficulty"]
        .map(difficulty_order)
        .fillna(99)
    )

    return (
        result.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def serialize_test_results(
    record: dict[str, Any],
) -> str:
    test_results = record.get(
        "test_results",
        [],
    )

    try:
        return json.dumps(
            test_results,
            ensure_ascii=False,
            indent=2,
        )
    except TypeError:
        return str(test_results)


def write_problem_report(
    output_path: Path,
    direct_record: dict[str, Any],
    self_record: dict[str, Any],
    teacher_record: dict[str, Any],
) -> None:
    problem_id = str(
        teacher_record["problem_id"]
    )

    title = str(
        teacher_record.get("title", "")
    )

    difficulty = str(
        teacher_record.get(
            "difficulty",
            "",
        )
    )

    sections = [
        "=" * 100,
        (
            f"{problem_id} | {title} | "
            f"{difficulty}"
        ),
        "=" * 100,
        "",
        "[Strategy Outcomes]",
        (
            f"Direct       : "
            f"{direct_record.get('status', '')}"
        ),
        (
            f"Self-Plan    : "
            f"{self_record.get('status', '')}"
        ),
        (
            f"Teacher-Plan : "
            f"{teacher_record.get('status', '')}"
        ),
        "",
        "[Problem]",
        str(
            teacher_record.get(
                "prompt",
                self_record.get("prompt", ""),
            )
        ),
        "",
        "[Teacher Plan]",
        extract_teacher_plan(
            teacher_record
        ) or "(Teacher plan not found)",
        "",
        "[Teacher-Plan Extracted Code]",
        extract_code(
            teacher_record
        ) or "(Code not found)",
        "",
        "[Teacher-Plan Evaluation]",
        (
            f"Status       : "
            f"{teacher_record.get('status', '')}"
        ),
        (
            f"Passed tests : "
            f"{teacher_record.get('passed_tests', 0)}"
            f"/"
            f"{teacher_record.get('total_tests', 0)}"
        ),
        (
            f"Error        : "
            f"{teacher_record.get('error_message', '')}"
        ),
        "",
        "[Teacher-Plan Test Results]",
        serialize_test_results(
            teacher_record
        ),
        "",
        "[Self-Generated Plan - Reference]",
        extract_self_plan(
            self_record
        ) or "(Self plan not found)",
        "",
        "[Self-Plan Extracted Code - Reference]",
        extract_code(
            self_record
        ) or "(Code not found)",
        "",
        "[Manual Analysis]",
        "Teacher plan correctness       : ",
        "Implementation fidelity         : ",
        "Primary bottleneck              : ",
        "Failure type                    : ",
        "Algorithm category              : ",
        "Missing or wrong information    : ",
        "Analysis note                   : ",
        "",
        "[Label Guide]",
        (
            "Teacher plan correctness: "
            "Correct / Partially Correct / Incorrect"
        ),
        (
            "Implementation fidelity: "
            "High / Medium / Low"
        ),
        (
            "Primary bottleneck: "
            "Planning / Implementation / Both"
        ),
        (
            "Failure type: Algorithm Error / Logic Error / "
            "Edge Case / Runtime / Timeout / Input-Output / "
            "Complexity / Other"
        ),
        "",
    ]

    output_path.write_text(
        "\n".join(sections),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    direct_records = load_jsonl(
        args.direct_path
    )

    self_records = load_jsonl(
        args.self_plan_path
    )

    teacher_records = load_jsonl(
        args.teacher_plan_path
    )

    validate_same_problem_set(
        direct_records,
        self_records,
        teacher_records,
    )

    direct_index = index_by_problem_id(
        direct_records
    )

    self_index = index_by_problem_id(
        self_records
    )

    teacher_index = index_by_problem_id(
        teacher_records
    )

    failure_rows = build_failure_rows(
        direct_index,
        self_index,
        teacher_index,
    )

    failure_df = pd.DataFrame(
        failure_rows
    )

    if failure_df.empty:
        raise ValueError(
            "No Teacher-Plan failure cases found."
        )

    if (
        args.expected_count >= 0
        and len(failure_df) != args.expected_count
    ):
        raise ValueError(
            "Unexpected Teacher-Plan failure count: "
            f"expected={args.expected_count}, "
            f"actual={len(failure_df)}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir = (
        args.output_dir
        / "problem_reports"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    failure_csv_path = (
        args.output_dir
        / "teacher_failure_cases.csv"
    )

    failure_df.to_csv(
        failure_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    status_summary = build_status_summary(
        failure_df
    )

    status_summary_path = (
        args.output_dir
        / "failure_status_summary.csv"
    )

    status_summary.to_csv(
        status_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    difficulty_summary = (
        build_difficulty_summary(
            failure_df
        )
    )

    difficulty_summary_path = (
        args.output_dir
        / "failure_difficulty_summary.csv"
    )

    difficulty_summary.to_csv(
        difficulty_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    for problem_id in failure_df[
        "problem_id"
    ]:
        write_problem_report(
            output_path=(
                reports_dir
                / f"{problem_id}.txt"
            ),
            direct_record=direct_index[
                problem_id
            ],
            self_record=self_index[
                problem_id
            ],
            teacher_record=teacher_index[
                problem_id
            ],
        )

    display_columns = [
        "problem_id",
        "title",
        "difficulty",
        "three_strategy_pattern",
        "teacher_status",
        "teacher_passed_tests",
        "teacher_total_tests",
        "teacher_test_pass_ratio",
    ]

    print()
    print("=" * 100)
    print("Teacher-Plan Failure Cases")
    print("=" * 100)
    print(
        failure_df[
            display_columns
        ].to_string(index=False)
    )

    print()
    print("=" * 100)
    print("Failure Status Summary")
    print("=" * 100)
    print(
        status_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("Failure Difficulty Summary")
    print("=" * 100)
    print(
        difficulty_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("Output Files")
    print("=" * 100)
    print(f"[SAVED] {failure_csv_path}")
    print(f"[SAVED] {status_summary_path}")
    print(f"[SAVED] {difficulty_summary_path}")
    print(f"[SAVED] {reports_dir}")

    print()
    print(
        f"[DONE] Extracted "
        f"{len(failure_df)} "
        "Teacher-Plan failure cases."
    )


if __name__ == "__main__":
    main()