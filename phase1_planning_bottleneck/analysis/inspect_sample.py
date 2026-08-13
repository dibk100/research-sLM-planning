"""
130번 문제 확인

PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase1_planning_bottleneck/analysis/inspect_sample.py \
  --index 130 \
  --direct /mnt/hdd/project_sLM_planning/phase1/qwen25Coder3b/livecodebench_v6_stdin/direct/results.jsonl \
  --self-plan /mnt/hdd/project_sLM_planning/phase1/qwen25Coder3b/livecodebench_v6_stdin/self_plan/results.jsonl \
  --teacher-plan /mnt/hdd/project_sLM_planning/phase1/qwen25Coder3b/livecodebench_v6_stdin/teacher_plan/results.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Direct, Self-Plan, and Teacher-Plan "
            "records for the same problem."
        )
    )

    parser.add_argument(
        "--index",
        type=int,
        default=130,
        help="1-based record index to inspect.",
    )

    parser.add_argument(
        "--direct",
        required=True,
        help="Path to Direct results.jsonl.",
    )

    parser.add_argument(
        "--self-plan",
        required=True,
        help="Path to Self-Plan results.jsonl.",
    )

    parser.add_argument(
        "--teacher-plan",
        required=True,
        help="Path to Teacher-Plan results.jsonl.",
    )

    return parser.parse_args()


def load_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at "
                    f"{path}:{line_number}"
                ) from error

            records.append(record)

    return records


def get_record(
    records: list[dict[str, Any]],
    index: int,
    name: str,
) -> dict[str, Any]:

    if index < 1:
        raise ValueError(
            "--index must be >= 1"
        )

    if index > len(records):
        raise IndexError(
            f"{name} has only {len(records)} records, "
            f"but index={index} was requested."
        )

    return records[index - 1]


def extract_self_plan(
    record: dict[str, Any],
) -> str | None:

    for step in record.get(
        "strategy_trace",
        [],
    ):
        if step.get("name") == "plan_generation":
            return step.get("raw_output")

    return None


def print_separator(
    title: str,
) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:
    args = parse_args()

    direct_records = load_jsonl(
        args.direct
    )

    self_records = load_jsonl(
        args.self_plan
    )

    teacher_records = load_jsonl(
        args.teacher_plan
    )

    direct = get_record(
        direct_records,
        args.index,
        "Direct",
    )

    self_plan = get_record(
        self_records,
        args.index,
        "Self-Plan",
    )

    teacher = get_record(
        teacher_records,
        args.index,
        "Teacher-Plan",
    )

    # --------------------------------------------------------------
    # Make sure all three records refer to the same problem
    # --------------------------------------------------------------

    ids = {
        direct.get("problem_id"),
        self_plan.get("problem_id"),
        teacher.get("problem_id"),
    }

    if len(ids) != 1:
        raise ValueError(
            "Problem ID mismatch at "
            f"index {args.index}:\n"
            f"Direct      = {direct.get('problem_id')}\n"
            f"Self-Plan   = {self_plan.get('problem_id')}\n"
            f"Teacher-Plan= {teacher.get('problem_id')}"
        )

    problem_id = direct["problem_id"]

    # --------------------------------------------------------------
    # Extract plans
    # --------------------------------------------------------------

    self_generated_plan = extract_self_plan(
        self_plan
    )

    teacher_generated_plan = teacher.get(
        "teacher_plan"
    )

    # --------------------------------------------------------------
    # Problem
    # --------------------------------------------------------------

    print_separator(
        f"Sample #{args.index}"
    )

    print(
        f"Problem ID : {problem_id}"
    )

    print(
        f"Title      : "
        f"{direct.get('title', '')}"
    )

    print(
        f"Difficulty : "
        f"{direct.get('difficulty')}"
    )

    print_separator(
        "PROBLEM"
    )

    print(
        direct.get(
            "problem",
            "",
        )
    )

    # --------------------------------------------------------------
    # Direct
    # --------------------------------------------------------------

    print_separator(
        "DIRECT"
    )

    print(
        f"Status     : "
        f"{direct.get('status')}"
    )
    print(
        f"Passed     : "
        f"{direct.get('passed')}"
    )

    if (
        direct.get("diagnostic_total_tests")
        is not None
    ):
        print(
            "Tests      : "
            f"{direct.get('diagnostic_passed_tests')}/"
            f"{direct.get('diagnostic_total_tests')} "
            f"("
            f"{direct.get('diagnostic_test_pass_ratio'):.4f}"
            f")"
        )

    print()
    print("[Generated Code]")
    print(
        direct.get(
            "extracted_code",
            "",
        )
    )

    # --------------------------------------------------------------
    # Self-Plan
    # --------------------------------------------------------------

    print_separator(
        "SELF-PLAN"
    )

    print(
        f"Status     : "
        f"{self_plan.get('status')}"
    )
    print(
        f"Passed     : "
        f"{self_plan.get('passed')}"
    )

    if (
        self_plan.get("diagnostic_total_tests")
        is not None
    ):
        print(
            "Tests      : "
            f"{self_plan.get('diagnostic_passed_tests')}/"
            f"{self_plan.get('diagnostic_total_tests')} "
            f"("
            f"{self_plan.get('diagnostic_test_pass_ratio'):.4f}"
            f")"
        )

    print()
    print("[Self-Generated Plan]")

    if self_generated_plan:
        print(
            self_generated_plan
        )
    else:
        print(
            "<PLAN NOT FOUND>"
        )

    print()
    print("[Generated Code]")
    print(
        self_plan.get(
            "extracted_code",
            "",
        )
    )

    # --------------------------------------------------------------
    # Teacher-Plan
    # --------------------------------------------------------------

    print_separator(
        "TEACHER-PLAN"
    )

    print(
        f"Status     : "
        f"{teacher.get('status')}"
    )
    print(
        f"Passed     : "
        f"{teacher.get('passed')}"
    )

    if (
        teacher.get("diagnostic_total_tests")
        is not None
    ):
        print(
            "Tests      : "
            f"{teacher.get('diagnostic_passed_tests')}/"
            f"{teacher.get('diagnostic_total_tests')} "
            f"("
            f"{teacher.get('diagnostic_test_pass_ratio'):.4f}"
            f")"
        )

    print()
    print("[Teacher Plan]")

    if teacher_generated_plan:
        print(
            teacher_generated_plan
        )
    else:
        print(
            "<TEACHER PLAN NOT FOUND>"
        )

    print()
    print("[Generated Code]")
    print(
        teacher.get(
            "extracted_code",
            "",
        )
    )

    # --------------------------------------------------------------
    # Plan-only comparison
    # --------------------------------------------------------------

    print_separator(
        "PLAN COMPARISON"
    )

    print("[SELF-PLAN]")
    print(
        self_generated_plan
        or "<PLAN NOT FOUND>"
    )

    print()
    print("-" * 100)
    print()

    print("[TEACHER-PLAN]")
    print(
        teacher_generated_plan
        or "<TEACHER PLAN NOT FOUND>"
    )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()