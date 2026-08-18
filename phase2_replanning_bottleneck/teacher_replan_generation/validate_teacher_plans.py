"""
작업 순서 3번

1. 300개 모두 존재
2. problem_id가 원본 300문제와 정확히 일치
3. 중복 problem_id 없음
4. teacher_plan이 비어 있지 않음

PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase2_replanning_bottleneck/teacher_replan_generation/validate_teacher_replans.py \
  --config phase2_replanning_bottleneck/configs/teacher_replan_make.yaml

"""
# phase2_replanning_bottleneck/teacher_replan_generation/
# validate_teacher_replans.py

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.datasets.phase1_failure_loader import (
    load_phase1_failures,
)
from src.utils.config import load_config


REQUIRED_FIELDS = {
    "problem_id",
    "teacher_replan",
    "teacher_model",
    "replan_version",
    "verified",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a Teacher-Replan JSONL dataset "
            "against Phase 1 Direct refinable failures."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to teacher-replan validation "
            "YAML config."
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
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(
                    line
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON at "
                    f"{path}:{line_number}"
                ) from error

            if not isinstance(
                record,
                dict,
            ):
                raise TypeError(
                    "JSONL record must be an object "
                    f"at {path}:{line_number}"
                )

            records.append(
                record
            )

    return records


def validate_schema(
    records: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []

    for index, record in enumerate(
        records,
        start=1,
    ):
        missing = (
            REQUIRED_FIELDS
            - set(record)
        )

        if missing:
            errors.append(
                f"row={index}: "
                f"missing fields={sorted(missing)}"
            )
            continue

        problem_id = str(
            record.get(
                "problem_id",
                "",
            )
        ).strip()

        teacher_replan = str(
            record.get(
                "teacher_replan",
                "",
            )
        ).strip()

        teacher_model = str(
            record.get(
                "teacher_model",
                "",
            )
        ).strip()

        replan_version = str(
            record.get(
                "replan_version",
                "",
            )
        ).strip()

        verified = record.get(
            "verified"
        )

        if not problem_id:
            errors.append(
                f"row={index}: empty problem_id"
            )

        if not teacher_replan:
            errors.append(
                f"row={index}: empty teacher_replan"
            )

        if not teacher_model:
            errors.append(
                f"row={index}: empty teacher_model"
            )

        if not replan_version:
            errors.append(
                f"row={index}: empty replan_version"
            )

        if not isinstance(
            verified,
            bool,
        ):
            errors.append(
                f"row={index}, problem_id={problem_id}: "
                "verified must be bool"
            )

    return errors


def main() -> None:
    args = parse_args()
    config = load_config(
        args.config
    )

    dataset_config = config[
        "dataset"
    ]

    teacher_config = config[
        "teacher"
    ]

    output_config = config[
        "output"
    ]

    # --------------------------------------------------------------
    # Expected Phase 1 failure set
    # --------------------------------------------------------------

    phase1_result_path = Path(
        dataset_config[
            "path"
        ]
    )

    expected_failures = (
        load_phase1_failures(
            result_path=phase1_result_path,
            limit=dataset_config.get(
                "limit"
            ),
        )
    )

    expected_ids = [
        failure.problem_id
        for failure in expected_failures
    ]

    expected_id_set = set(
        expected_ids
    )

    # --------------------------------------------------------------
    # Teacher-Replan output
    # --------------------------------------------------------------

    replan_path = Path(
        output_config[
            "replan_file"
        ]
    )

    records = load_jsonl(
        replan_path
    )

    actual_ids = [
        str(
            record.get(
                "problem_id",
                "",
            )
        ).strip()
        for record in records
    ]

    actual_id_set = set(
        actual_ids
    )

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    id_counts = Counter(
        actual_ids
    )

    duplicate_ids = sorted(
        problem_id
        for problem_id, count
        in id_counts.items()
        if problem_id and count > 1
    )

    missing_ids = sorted(
        expected_id_set
        - actual_id_set
    )

    extra_ids = sorted(
        actual_id_set
        - expected_id_set
    )

    empty_replans = [
        str(
            record.get(
                "problem_id",
                "",
            )
        ).strip()
        for record in records
        if not str(
            record.get(
                "teacher_replan",
                "",
            )
        ).strip()
    ]

    schema_errors = (
        validate_schema(
            records
        )
    )

    order_matches = (
        actual_ids
        == expected_ids
    )

    expected_teacher_model = (
        teacher_config.get(
            "model"
        )
    )

    expected_replan_version = (
        teacher_config.get(
            "replan_version",
            "v1",
        )
    )

    teacher_model_mismatches = []

    replan_version_mismatches = []

    for record in records:
        problem_id = str(
            record.get(
                "problem_id",
                "",
            )
        ).strip()

        teacher_model = str(
            record.get(
                "teacher_model",
                "",
            )
        ).strip()

        replan_version = str(
            record.get(
                "replan_version",
                "",
            )
        ).strip()

        if (
            expected_teacher_model
            and teacher_model
            != expected_teacher_model
        ):
            teacher_model_mismatches.append(
                problem_id
            )

        if (
            expected_replan_version
            and replan_version
            != expected_replan_version
        ):
            replan_version_mismatches.append(
                problem_id
            )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print("=" * 80)
    print(
        "Teacher-Replan Validation"
    )
    print("=" * 80)

    print(
        f"Phase1 source     : "
        f"{phase1_result_path}"
    )

    print(
        f"Replan file       : "
        f"{replan_path}"
    )

    print(
        f"Expected records  : "
        f"{len(expected_ids)}"
    )

    print(
        f"Actual records    : "
        f"{len(records)}"
    )

    print(
        f"Unique IDs        : "
        f"{len(actual_id_set)}"
    )

    print(
        f"Missing IDs       : "
        f"{len(missing_ids)}"
    )

    print(
        f"Extra IDs         : "
        f"{len(extra_ids)}"
    )

    print(
        f"Duplicate IDs     : "
        f"{len(duplicate_ids)}"
    )

    print(
        f"Empty replans     : "
        f"{len(empty_replans)}"
    )

    print(
        f"Schema errors     : "
        f"{len(schema_errors)}"
    )

    print(
        f"Teacher mismatch  : "
        f"{len(teacher_model_mismatches)}"
    )

    print(
        f"Version mismatch  : "
        f"{len(replan_version_mismatches)}"
    )

    print(
        f"Order matches     : "
        f"{order_matches}"
    )

    # --------------------------------------------------------------
    # Details
    # --------------------------------------------------------------

    if missing_ids:
        print()
        print(
            "[Missing IDs]"
        )
        for problem_id in missing_ids[:20]:
            print(
                f"- {problem_id}"
            )

    if extra_ids:
        print()
        print(
            "[Extra IDs]"
        )
        for problem_id in extra_ids[:20]:
            print(
                f"- {problem_id}"
            )

    if duplicate_ids:
        print()
        print(
            "[Duplicate IDs]"
        )
        for problem_id in duplicate_ids[:20]:
            print(
                f"- {problem_id}"
            )

    if empty_replans:
        print()
        print(
            "[Empty Replans]"
        )
        for problem_id in empty_replans[:20]:
            print(
                f"- {problem_id}"
            )

    if schema_errors:
        print()
        print(
            "[Schema Errors]"
        )
        for error in schema_errors[:20]:
            print(
                f"- {error}"
            )

    if teacher_model_mismatches:
        print()
        print(
            "[Teacher Model Mismatches]"
        )
        for problem_id in (
            teacher_model_mismatches[:20]
        ):
            print(
                f"- {problem_id}"
            )

    if replan_version_mismatches:
        print()
        print(
            "[Replan Version Mismatches]"
        )
        for problem_id in (
            replan_version_mismatches[:20]
        ):
            print(
                f"- {problem_id}"
            )

    # --------------------------------------------------------------
    # Final decision
    # --------------------------------------------------------------

    passed = (
        len(records)
        == len(expected_ids)
        and len(actual_id_set)
        == len(expected_id_set)
        and not missing_ids
        and not extra_ids
        and not duplicate_ids
        and not empty_replans
        and not schema_errors
        and not teacher_model_mismatches
        and not replan_version_mismatches
    )

    print()

    if passed:
        print(
            "[PASS] Teacher-Replan dataset is valid."
        )
    else:
        print(
            "[FAIL] Teacher-Replan dataset validation failed."
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()