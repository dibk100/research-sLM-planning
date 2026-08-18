"""
Teacher plan batch builder / validator / merger.

teacher_plan은 Claude가 각 teacher_prompt를 직접 읽고 작성하며, 이 스크립트는
그 결과를 batch 단위 JSONL로 저장하고 검증/병합하는 보조 도구이다.
(API 호출 없음. build_teacher_plans.py와 별개의 수동 batch 파이프라인.)

사용법:

  # batch 1 저장 + 검증 (plans JSON은 Claude가 작성한 plan 목록)
  python phase1_planning_bottleneck/teacher_plan_generation/batch_teacher_plans.py \
      write --batch 1 --plans /path/to/batch_001_plans.json

  # 특정 batch 검증
  python .../batch_teacher_plans.py validate --batch 1

  # 전체 batch 상태 확인 (resume 대상 판별)
  python .../batch_teacher_plans.py status

  # 30개 batch 전부 통과 시 최종 병합 + 최종 검증
  python .../batch_teacher_plans.py merge

plans JSON 포맷:

  {
    "batch": 1,
    "teacher_model": "claude-opus-5",
    "plans": [
      {"problem_id": "abc400_a", "teacher_plan": "- ..."},
      ...
    ]
  }
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(
    "/mnt/hdd/project_sLM_planning/data/teacher_plans"
    "/livecodebench_v6/opus5_v1"
)

INPUT_PATH = ROOT / "teacher_inputs.jsonl"
BATCH_DIR = ROOT / "batches"
FINAL_PATH = ROOT / "teacher_plans_300.jsonl"

BATCH_SIZE = 10
NUM_BATCHES = 30
TOTAL = BATCH_SIZE * NUM_BATCHES

PLAN_VERSION = "v1"

REQUIRED_FIELDS = [
    "problem_id",
    "teacher_plan",
    "teacher_model",
    "plan_version",
    "verified",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()

            if not stripped:
                raise ValueError(
                    f"Blank line at {path}:{line_number}"
                )

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise TypeError(
                    f"Line {line_number} of {path} is not a JSON object."
                )

            records.append(record)

    return records


def load_inputs() -> list[dict[str, Any]]:
    inputs = load_jsonl(INPUT_PATH)

    if len(inputs) != TOTAL:
        raise ValueError(
            f"Expected {TOTAL} teacher inputs, found {len(inputs)}."
        )

    return inputs


def input_slice(
    inputs: list[dict[str, Any]],
    batch_index: int,
) -> list[dict[str, Any]]:
    if not 1 <= batch_index <= NUM_BATCHES:
        raise ValueError(
            f"batch index must be in [1, {NUM_BATCHES}]: {batch_index}"
        )

    start = (batch_index - 1) * BATCH_SIZE

    return inputs[start:start + BATCH_SIZE]


def batch_path(batch_index: int) -> Path:
    return BATCH_DIR / f"batch_{batch_index:03d}.jsonl"


def validate_records(
    records: list[dict[str, Any]],
    expected_ids: list[str],
    *,
    expected_count: int,
) -> list[str]:
    """Return a list of validation error messages (empty means pass)."""

    errors: list[str] = []

    if len(records) != expected_count:
        errors.append(
            f"record count is {len(records)}, expected {expected_count}"
        )

    actual_ids: list[str] = []

    for index, record in enumerate(records, start=1):
        missing = [
            field
            for field in REQUIRED_FIELDS
            if field not in record
        ]

        if missing:
            errors.append(
                f"record {index}: missing fields {missing}"
            )

        extra = sorted(set(record) - set(REQUIRED_FIELDS))

        if extra:
            errors.append(
                f"record {index}: unexpected fields {extra}"
            )

        problem_id = record.get("problem_id")

        if not isinstance(problem_id, str) or not problem_id.strip():
            errors.append(
                f"record {index}: problem_id must be a non-empty string"
            )
            actual_ids.append(f"<invalid@{index}>")
        else:
            actual_ids.append(problem_id)

        teacher_plan = record.get("teacher_plan")

        if not isinstance(teacher_plan, str) or not teacher_plan.strip():
            errors.append(
                f"record {index}: teacher_plan must be a non-empty string"
            )

        teacher_model = record.get("teacher_model")

        if not isinstance(teacher_model, str) or not teacher_model.strip():
            errors.append(
                f"record {index}: teacher_model must be a non-empty string"
            )

        if record.get("plan_version") != PLAN_VERSION:
            errors.append(
                f"record {index}: plan_version must be {PLAN_VERSION!r}, "
                f"got {record.get('plan_version')!r}"
            )

        verified = record.get("verified")

        if not isinstance(verified, bool) or verified is not False:
            errors.append(
                f"record {index}: verified must be boolean false, "
                f"got {verified!r}"
            )

    duplicates = sorted(
        problem_id
        for problem_id, count in Counter(actual_ids).items()
        if count > 1
    )

    if duplicates:
        errors.append(f"duplicate problem_ids: {duplicates}")

    missing_ids = sorted(set(expected_ids) - set(actual_ids))
    extra_ids = sorted(set(actual_ids) - set(expected_ids))

    if missing_ids:
        errors.append(f"missing problem_ids: {missing_ids}")

    if extra_ids:
        errors.append(f"extra problem_ids: {extra_ids}")

    if actual_ids != expected_ids:
        if not missing_ids and not extra_ids:
            errors.append(
                "problem_id order does not match the input order"
            )

    # Consistent teacher_model within the file.
    models = {
        record.get("teacher_model")
        for record in records
        if isinstance(record.get("teacher_model"), str)
    }

    if len(models) > 1:
        errors.append(
            f"inconsistent teacher_model values: {sorted(models)}"
        )

    return errors


def validate_batch(
    batch_index: int,
    inputs: list[dict[str, Any]],
) -> list[str]:
    path = batch_path(batch_index)

    if not path.exists():
        return [f"{path.name} does not exist"]

    try:
        records = load_jsonl(path)
    except (ValueError, TypeError) as error:
        return [str(error)]

    expected_ids = [
        record["problem_id"]
        for record in input_slice(inputs, batch_index)
    ]

    return validate_records(
        records,
        expected_ids,
        expected_count=BATCH_SIZE,
    )


def cmd_write(args: argparse.Namespace) -> None:
    inputs = load_inputs()
    batch_index = args.batch

    payload = json.loads(
        Path(args.plans).read_text(encoding="utf-8")
    )

    if payload.get("batch") != batch_index:
        raise ValueError(
            f"plans file declares batch {payload.get('batch')}, "
            f"but --batch {batch_index} was given"
        )

    teacher_model = payload["teacher_model"]

    if not isinstance(teacher_model, str) or not teacher_model.strip():
        raise ValueError("teacher_model must be a non-empty string")

    plans = payload["plans"]

    expected = input_slice(inputs, batch_index)
    expected_ids = [record["problem_id"] for record in expected]

    if len(plans) != len(expected_ids):
        raise ValueError(
            f"plans file has {len(plans)} plans, "
            f"expected {len(expected_ids)}"
        )

    records: list[dict[str, Any]] = []

    for plan, expected_id in zip(plans, expected_ids):
        if plan["problem_id"] != expected_id:
            raise ValueError(
                f"problem_id mismatch: got {plan['problem_id']!r}, "
                f"expected {expected_id!r}"
            )

        records.append(
            {
                "problem_id": expected_id,
                "teacher_plan": plan["teacher_plan"].strip(),
                "teacher_model": teacher_model,
                "plan_version": PLAN_VERSION,
                "verified": False,
            }
        )

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    path = batch_path(batch_index)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[WRITE] {path} ({len(records)} records)")

    errors = validate_batch(batch_index, inputs)

    if errors:
        print(f"[FAIL] batch_{batch_index:03d}")

        for error in errors:
            print(f"  - {error}")

        raise SystemExit(1)

    print(f"[PASS] batch_{batch_index:03d} validation")


def cmd_validate(args: argparse.Namespace) -> None:
    inputs = load_inputs()

    indices = (
        [args.batch]
        if args.batch is not None
        else list(range(1, NUM_BATCHES + 1))
    )

    failed = 0

    for batch_index in indices:
        errors = validate_batch(batch_index, inputs)

        if errors:
            failed += 1
            print(f"[FAIL] batch_{batch_index:03d}")

            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[PASS] batch_{batch_index:03d}")

    if failed:
        raise SystemExit(1)


def cmd_status(args: argparse.Namespace) -> None:
    inputs = load_inputs()

    done: list[int] = []
    todo: list[int] = []

    for batch_index in range(1, NUM_BATCHES + 1):
        if validate_batch(batch_index, inputs):
            todo.append(batch_index)
        else:
            done.append(batch_index)

    print(f"validated batches : {len(done)}/{NUM_BATCHES}")
    print(f"remaining batches : {todo}")


def cmd_merge(args: argparse.Namespace) -> None:
    inputs = load_inputs()

    failed = [
        batch_index
        for batch_index in range(1, NUM_BATCHES + 1)
        if validate_batch(batch_index, inputs)
    ]

    if failed:
        print(f"[ABORT] batches not validated: {failed}")
        raise SystemExit(1)

    merged: list[dict[str, Any]] = []

    for batch_index in range(1, NUM_BATCHES + 1):
        merged.extend(load_jsonl(batch_path(batch_index)))

    with FINAL_PATH.open("w", encoding="utf-8") as f:
        for record in merged:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[WRITE] {FINAL_PATH} ({len(merged)} records)")

    expected_ids = [record["problem_id"] for record in inputs]

    errors = validate_records(
        load_jsonl(FINAL_PATH),
        expected_ids,
        expected_count=TOTAL,
    )

    if errors:
        print("[FAIL] final validation")

        for error in errors:
            print(f"  - {error}")

        raise SystemExit(1)

    print(f"[PASS] final validation ({TOTAL} records, 100% coverage)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_write = sub.add_parser("write", help="Write and validate one batch.")
    p_write.add_argument("--batch", type=int, required=True)
    p_write.add_argument("--plans", required=True)
    p_write.set_defaults(func=cmd_write)

    p_validate = sub.add_parser("validate", help="Validate batches.")
    p_validate.add_argument("--batch", type=int, default=None)
    p_validate.set_defaults(func=cmd_validate)

    p_status = sub.add_parser("status", help="Show resume status.")
    p_status.set_defaults(func=cmd_status)

    p_merge = sub.add_parser("merge", help="Merge all batches.")
    p_merge.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
