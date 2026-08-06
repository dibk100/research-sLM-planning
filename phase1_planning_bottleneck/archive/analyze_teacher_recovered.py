"""
Teacher-Plan Successfully Recovered Problems 분석.

분석 대상:
    Self-Plan FAIL -> Teacher-Plan PASS

Usage:

python -m archive.analyze_teacher_recovered \
  --direct-path /mnt/hdd/project_sLM_planning/output/direct_50_stdin/results.jsonl \
  --self-plan-path /mnt/hdd/project_sLM_planning/output/self_plan_50_stdin/results.jsonl \
  --teacher-plan-path /mnt/hdd/project_sLM_planning/output/teacher_plan_50_stdin/results.jsonl \
  --output-dir ./archive/teacher_recovered_50
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
    "./archive/teacher_recovered_50"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze problems recovered by Teacher-Plan "
            "from Self-Plan failures."
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

    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Result file not found: {path.resolve()}"
        )

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {path}"
                ) from error

            if not isinstance(record, dict):
                raise TypeError(
                    f"Record at line {line_number} is not an object."
                )

            records.append(record)

    if not records:
        raise ValueError(f"No records found: {path}")

    problem_ids = [
        str(record.get("problem_id", ""))
        for record in records
    ]

    if len(problem_ids) != len(set(problem_ids)):
        duplicated = sorted(
            {
                problem_id
                for problem_id in problem_ids
                if problem_ids.count(problem_id) > 1
            }
        )

        raise ValueError(
            f"Duplicated problem IDs in {path}: {duplicated}"
        )

    return records


def index_by_problem_id(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(record["problem_id"]): record
        for record in records
    }


def normalize_passed(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value == 1

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"true", "1", "pass", "passed"}:
            return True

        if normalized in {"false", "0", "fail", "failed"}:
            return False

    return False


def safe_number(
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
    passed_tests = safe_number(
        record.get("passed_tests")
    )
    total_tests = safe_number(
        record.get("total_tests")
    )

    if total_tests <= 0:
        return 0.0

    return passed_tests / total_tests


def extract_trace_output(
    record: dict[str, Any],
    candidate_names: set[str],
) -> str:
    """
    strategy_trace에서 지정된 단계의 raw_output을 찾는다.

    Self-Plan 구현에 따라 단계 이름이
    plan_generation, planning, generate_plan 등일 수 있으므로
    여러 후보를 지원한다.
    """
    strategy_trace = record.get("strategy_trace", [])

    if not isinstance(strategy_trace, list):
        return ""

    for step in strategy_trace:
        if not isinstance(step, dict):
            continue

        step_name = str(
            step.get("name", "")
        ).strip().lower()

        if step_name in candidate_names:
            return str(
                step.get(
                    "raw_output",
                    step.get("output", ""),
                )
            ).strip()

    return ""


def extract_self_plan(
    record: dict[str, Any],
) -> str:
    """
    Self-Plan 결과에서 생성된 계획을 추출한다.

    프로젝트 버전에 따라 필드명이 다를 수 있어
    여러 후보 필드를 순차적으로 확인한다.
    """
    direct_candidates = [
        "generated_plan",
        "plan",
        "self_plan",
        "implementation_plan",
    ]

    for key in direct_candidates:
        value = record.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    trace_plan = extract_trace_output(
        record,
        {
            "plan_generation",
            "planning",
            "generate_plan",
            "plan",
        },
    )

    return trace_plan


def extract_teacher_plan(
    record: dict[str, Any],
) -> str:
    value = record.get("teacher_plan", "")

    if isinstance(value, str):
        return value.strip()

    return ""


def extract_code(
    record: dict[str, Any],
) -> str:
    extracted_code = record.get(
        "extracted_code",
        "",
    )

    if isinstance(extracted_code, str):
        return extracted_code.strip()

    return ""


def extract_raw_output(
    record: dict[str, Any],
) -> str:
    raw_output = record.get("raw_output", "")

    if isinstance(raw_output, str):
        return raw_output.strip()

    return ""


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

    if not (
        direct_ids == self_ids == teacher_ids
    ):
        raise ValueError(
            "The three result files do not contain "
            "the same problem set."
        )

    print(
        f"[PASS] Same problem set: "
        f"{len(direct_ids)} problems"
    )


def build_recovered_rows(
    direct_index: dict[str, dict[str, Any]],
    self_index: dict[str, dict[str, Any]],
    teacher_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for problem_id in sorted(self_index):
        direct_record = direct_index[problem_id]
        self_record = self_index[problem_id]
        teacher_record = teacher_index[problem_id]

        self_passed = normalize_passed(
            self_record.get("passed")
        )

        teacher_passed = normalize_passed(
            teacher_record.get("passed")
        )

        if self_passed or not teacher_passed:
            continue

        direct_passed = normalize_passed(
            direct_record.get("passed")
        )

        self_plan = extract_self_plan(
            self_record
        )

        teacher_plan = extract_teacher_plan(
            teacher_record
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
            "direct_passed": direct_passed,
            "self_plan_passed": self_passed,
            "teacher_plan_passed": teacher_passed,
            "three_strategy_pattern": (
                f"{'P' if direct_passed else 'F'}-"
                f"{'P' if self_passed else 'F'}-"
                f"{'P' if teacher_passed else 'F'}"
            ),
            "self_status": self_record.get(
                "status",
                "",
            ),
            "teacher_status": teacher_record.get(
                "status",
                "",
            ),
            "self_passed_tests": self_record.get(
                "passed_tests",
                0,
            ),
            "self_total_tests": self_record.get(
                "total_tests",
                0,
            ),
            "teacher_passed_tests": teacher_record.get(
                "passed_tests",
                0,
            ),
            "teacher_total_tests": teacher_record.get(
                "total_tests",
                0,
            ),
            "self_test_pass_ratio": (
                calculate_test_pass_ratio(
                    self_record
                )
            ),
            "teacher_test_pass_ratio": (
                calculate_test_pass_ratio(
                    teacher_record
                )
            ),
            "test_pass_ratio_delta": (
                calculate_test_pass_ratio(
                    teacher_record
                )
                - calculate_test_pass_ratio(
                    self_record
                )
            ),
            "self_prompt_tokens": self_record.get(
                "prompt_tokens",
                0,
            ),
            "self_completion_tokens": self_record.get(
                "completion_tokens",
                0,
            ),
            "self_total_tokens": (
                safe_number(
                    self_record.get(
                        "prompt_tokens"
                    )
                )
                + safe_number(
                    self_record.get(
                        "completion_tokens"
                    )
                )
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
                safe_number(
                    teacher_record.get(
                        "prompt_tokens"
                    )
                )
                + safe_number(
                    teacher_record.get(
                        "completion_tokens"
                    )
                )
            ),
            "self_generation_time": (
                self_record.get(
                    "generation_time",
                    0.0,
                )
            ),
            "teacher_generation_time": (
                teacher_record.get(
                    "generation_time",
                    0.0,
                )
            ),
            "self_error_message": (
                self_record.get(
                    "error_message",
                    "",
                )
            ),
            "self_generated_plan": self_plan,
            "teacher_plan": teacher_plan,
            "self_extracted_code": extract_code(
                self_record
            ),
            "teacher_extracted_code": extract_code(
                teacher_record
            ),
            # 아래 필드는 수작업 분석용
            "self_plan_correctness": "",
            "teacher_plan_correctness": "",
            "plan_difference_category": "",
            "recovery_mechanism": "",
            "algorithm_category": "",
            "key_teacher_information": "",
            "analysis_note": "",
        }

        rows.append(row)

    return rows


def build_summary(
    recovered_df: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []

    summary_rows.append(
        {
            "metric": "num_recovered",
            "value": len(recovered_df),
        }
    )

    summary_rows.append(
        {
            "metric": "mean_self_test_pass_ratio",
            "value": (
                recovered_df[
                    "self_test_pass_ratio"
                ].mean()
            ),
        }
    )

    summary_rows.append(
        {
            "metric": "mean_teacher_test_pass_ratio",
            "value": (
                recovered_df[
                    "teacher_test_pass_ratio"
                ].mean()
            ),
        }
    )

    summary_rows.append(
        {
            "metric": "mean_test_pass_ratio_delta",
            "value": (
                recovered_df[
                    "test_pass_ratio_delta"
                ].mean()
            ),
        }
    )

    summary_rows.append(
        {
            "metric": "mean_self_total_tokens",
            "value": (
                recovered_df[
                    "self_total_tokens"
                ].mean()
            ),
        }
    )

    summary_rows.append(
        {
            "metric": "mean_teacher_total_tokens",
            "value": (
                recovered_df[
                    "teacher_total_tokens"
                ].mean()
            ),
        }
    )

    return pd.DataFrame(summary_rows)


def build_difficulty_summary(
    recovered_df: pd.DataFrame,
) -> pd.DataFrame:
    result = (
        recovered_df
        .groupby(
            "difficulty",
            dropna=False,
        )
        .agg(
            num_recovered=(
                "problem_id",
                "count",
            ),
            mean_self_test_pass_ratio=(
                "self_test_pass_ratio",
                "mean",
            ),
            mean_teacher_test_pass_ratio=(
                "teacher_test_pass_ratio",
                "mean",
            ),
            mean_test_pass_ratio_delta=(
                "test_pass_ratio_delta",
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
        result
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


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

    self_plan = extract_self_plan(
        self_record
    )

    teacher_plan = extract_teacher_plan(
        teacher_record
    )

    sections = [
        "=" * 100,
        (
            f"{problem_id} | {title} | "
            f"{difficulty}"
        ),
        "=" * 100,
        "",
        "[Recovery Pattern]",
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
        "[Self-Generated Plan]",
        self_plan or "(Plan not found)",
        "",
        "[Teacher Plan]",
        teacher_plan or "(Plan not found)",
        "",
        "[Self-Plan Extracted Code]",
        extract_code(self_record)
        or "(Code not found)",
        "",
        "[Teacher-Plan Extracted Code]",
        extract_code(teacher_record)
        or "(Code not found)",
        "",
        "[Self-Plan Evaluation]",
        (
            f"Status       : "
            f"{self_record.get('status', '')}"
        ),
        (
            f"Passed tests : "
            f"{self_record.get('passed_tests', 0)}"
            f"/{self_record.get('total_tests', 0)}"
        ),
        (
            f"Error        : "
            f"{self_record.get('error_message', '')}"
        ),
        "",
        "[Teacher-Plan Evaluation]",
        (
            f"Status       : "
            f"{teacher_record.get('status', '')}"
        ),
        (
            f"Passed tests : "
            f"{teacher_record.get('passed_tests', 0)}"
            f"/{teacher_record.get('total_tests', 0)}"
        ),
        "",
        "[Manual Analysis]",
        "Self plan correctness      : ",
        "Teacher plan correctness   : ",
        "Plan difference category   : ",
        "Recovery mechanism         : ",
        "Algorithm category         : ",
        "Key teacher information    : ",
        "Analysis note              : ",
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

    recovered_rows = build_recovered_rows(
        direct_index,
        self_index,
        teacher_index,
    )

    recovered_df = pd.DataFrame(
        recovered_rows
    )

    if recovered_df.empty:
        raise ValueError(
            "No Self-Plan FAIL -> "
            "Teacher-Plan PASS cases found."
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

    recovered_csv_path = (
        args.output_dir
        / "teacher_recovered_cases.csv"
    )

    recovered_df.to_csv(
        recovered_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = build_summary(
        recovered_df
    )

    summary_path = (
        args.output_dir
        / "recovered_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    difficulty_summary_df = (
        build_difficulty_summary(
            recovered_df
        )
    )

    difficulty_summary_path = (
        args.output_dir
        / "recovered_difficulty_summary.csv"
    )

    difficulty_summary_df.to_csv(
        difficulty_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    for problem_id in recovered_df[
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
        "self_status",
        "teacher_status",
        "self_test_pass_ratio",
        "teacher_test_pass_ratio",
        "test_pass_ratio_delta",
    ]

    print()
    print("=" * 100)
    print(
        "Successfully Recovered Problems "
        "(Self FAIL -> Teacher PASS)"
    )
    print("=" * 100)
    print(
        recovered_df[
            display_columns
        ].to_string(index=False)
    )

    print()
    print("=" * 100)
    print("Difficulty Summary")
    print("=" * 100)
    print(
        difficulty_summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("Output Files")
    print("=" * 100)
    print(f"[SAVED] {recovered_csv_path}")
    print(f"[SAVED] {summary_path}")
    print(
        f"[SAVED] "
        f"{difficulty_summary_path}"
    )
    print(f"[SAVED] {reports_dir}")

    print()
    print(
        f"[DONE] Extracted "
        f"{len(recovered_df)} recovered cases."
    )


if __name__ == "__main__":
    main()