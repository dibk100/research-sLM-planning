"""
작업 순서 3번

1. 300개 모두 존재
2. problem_id가 원본 300문제와 정확히 일치
3. 중복 problem_id 없음
4. teacher_plan이 비어 있지 않음

PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase1_planning_bottleneck/teacher_plan_generation/validate_teacher_plans.py \
  --config phase1_planning_bottleneck/configs/teacher_plan_make.yaml

"""

# phase1_planning_bottleneck/teacher_plan_generation/validate_teacher_plans.py

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.datasets.dataset_loader import load_dataset
from src.utils.config import load_config


REQUIRED_FIELDS = {
    "problem_id",
    "teacher_plan",
    "teacher_model",
    "plan_version",
    "verified",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate generated teacher-plan dataset."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to teacher-plan generation YAML config.",
    )

    parser.add_argument(
        "--plan-path",
        default=None,
        help=(
            "Optional override for teacher plan JSONL path. "
            "If omitted, use the path derived from config."
        ),
    )

    return parser.parse_args()


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"JSONL file not found: {path}"
        )

    records: list[dict[str, Any]] = []

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
                    f"Invalid JSON at line {line_number}: "
                    f"{error}"
                ) from error

            if not isinstance(
                record,
                dict,
            ):
                raise TypeError(
                    f"Line {line_number} must contain "
                    "a JSON object."
                )

            records.append(record)

    return records


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset_config = config["dataset"]
    teacher_config = config["teacher"]
    output_config = config["output"]

    # --------------------------------------------------------------
    # Load reference problems
    # --------------------------------------------------------------

    problems = load_dataset(
        dataset_name=dataset_config["name"],
        data_path=dataset_config["path"],
        limit=dataset_config.get("limit"),
    )

    if not problems:
        raise ValueError(
            "No reference problems were loaded."
        )

    expected_ids = [
        problem.problem_id
        for problem in problems
    ]

    expected_id_set = set(
        expected_ids
    )

    # --------------------------------------------------------------
    # Resolve teacher-plan path
    # --------------------------------------------------------------

    if args.plan_path is not None:
        plan_path = Path(
            args.plan_path
        )

    else:
        output_dir = Path(
            output_config["dir"]
        )

        expected_count = len(
            expected_ids
        )

        plan_path = (
            output_dir
            / f"teacher_plans_{expected_count}.jsonl"
        )

    # --------------------------------------------------------------
    # Load teacher plans
    # --------------------------------------------------------------

    records = load_jsonl(
        plan_path
    )

    # --------------------------------------------------------------
    # Basic record validation
    # --------------------------------------------------------------

    schema_errors: list[str] = []
    empty_plan_ids: list[str] = []
    invalid_problem_ids: list[str] = []

    actual_ids: list[str] = []

    for index, record in enumerate(
        records,
        start=1,
    ):
        missing_fields = (
            REQUIRED_FIELDS
            - set(record)
        )

        if missing_fields:
            schema_errors.append(
                f"record {index}: missing fields "
                f"{sorted(missing_fields)}"
            )

        problem_id = record.get(
            "problem_id"
        )

        if not isinstance(
            problem_id,
            str,
        ) or not problem_id.strip():
            invalid_problem_ids.append(
                f"record {index}"
            )
            continue

        actual_ids.append(
            problem_id
        )

        teacher_plan = record.get(
            "teacher_plan"
        )

        if (
            not isinstance(
                teacher_plan,
                str,
            )
            or not teacher_plan.strip()
        ):
            empty_plan_ids.append(
                problem_id
            )

        if (
            "plan_version" in record
            and record["plan_version"]
            != teacher_config.get(
                "plan_version",
                "v1",
            )
        ):
            schema_errors.append(
                f"{problem_id}: unexpected "
                f"plan_version="
                f"{record['plan_version']!r}"
            )

        if (
            "verified" in record
            and not isinstance(
                record["verified"],
                bool,
            )
        ):
            schema_errors.append(
                f"{problem_id}: verified must be bool"
            )

    # --------------------------------------------------------------
    # Count / duplicate validation
    # --------------------------------------------------------------

    actual_count = len(
        records
    )

    expected_count = len(
        expected_ids
    )

    id_counts = Counter(
        actual_ids
    )

    duplicate_ids = sorted(
        problem_id
        for problem_id, count
        in id_counts.items()
        if count > 1
    )

    actual_id_set = set(
        actual_ids
    )

    missing_ids = sorted(
        expected_id_set
        - actual_id_set
    )

    extra_ids = sorted(
        actual_id_set
        - expected_id_set
    )

    # --------------------------------------------------------------
    # Order validation
    # --------------------------------------------------------------

    order_matches = (
        actual_ids == expected_ids
    )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print("=" * 80)
    print("Teacher Plan Validation")
    print("=" * 80)

    print(
        f"Dataset          : "
        f"{dataset_config['name']}"
    )
    print(
        f"Plan file        : "
        f"{plan_path}"
    )
    print(
        f"Expected records : "
        f"{expected_count}"
    )
    print(
        f"Actual records   : "
        f"{actual_count}"
    )
    print(
        f"Unique IDs       : "
        f"{len(actual_id_set)}"
    )
    print(
        f"Missing IDs      : "
        f"{len(missing_ids)}"
    )
    print(
        f"Extra IDs        : "
        f"{len(extra_ids)}"
    )
    print(
        f"Duplicate IDs    : "
        f"{len(duplicate_ids)}"
    )
    print(
        f"Empty plans      : "
        f"{len(empty_plan_ids)}"
    )
    print(
        f"Schema errors    : "
        f"{len(schema_errors)}"
    )
    print(
        f"Order matches    : "
        f"{order_matches}"
    )

    # --------------------------------------------------------------
    # Detailed failures
    # --------------------------------------------------------------

    if missing_ids:
        print()
        print("Missing problem_ids:")

        for problem_id in missing_ids:
            print(
                f"  - {problem_id}"
            )

    if extra_ids:
        print()
        print("Extra problem_ids:")

        for problem_id in extra_ids:
            print(
                f"  - {problem_id}"
            )

    if duplicate_ids:
        print()
        print("Duplicate problem_ids:")

        for problem_id in duplicate_ids:
            print(
                f"  - {problem_id} "
                f"(count={id_counts[problem_id]})"
            )

    if empty_plan_ids:
        print()
        print("Empty teacher plans:")

        for problem_id in empty_plan_ids:
            print(
                f"  - {problem_id}"
            )

    if invalid_problem_ids:
        print()
        print("Invalid problem_id records:")

        for record_info in (
            invalid_problem_ids
        ):
            print(
                f"  - {record_info}"
            )

    if schema_errors:
        print()
        print("Schema errors:")

        for error in schema_errors:
            print(
                f"  - {error}"
            )

    # --------------------------------------------------------------
    # Final decision
    # --------------------------------------------------------------

    validation_failed = any(
        [
            actual_count
            != expected_count,

            bool(
                missing_ids
            ),

            bool(
                extra_ids
            ),

            bool(
                duplicate_ids
            ),

            bool(
                empty_plan_ids
            ),

            bool(
                invalid_problem_ids
            ),

            bool(
                schema_errors
            ),
        ]
    )

    if validation_failed:
        raise ValueError(
            "Teacher plan validation failed."
        )

    print()
    print(
        "[PASS] Teacher plan dataset is valid."
    )


if __name__ == "__main__":
    main()