"""
문제 ID와 순서가 동일하다는 것을 확인하는 코드.
Phase 1과 Phase 3에서 동일 problem의 unit tests도 정말 같은지 확인하기 위해
verify_phase1_phase3_tests.py를 별도로 작성했다.

Usage:

PYTHONPATH=. python -m scripts.freeze_problem_ids

PYTHONPATH=. python -m scripts.freeze_problem_ids \
  --limit 10 \
  --output data/livecodebench_pilot_10.jsonl

phase1 결과 위치
/mnt/hdd/project_sLM_planning/output/direct_500_stdin/results.jsonl
/mnt/hdd/project_sLM_planning/output/self_plan_500_stdin/results.jsonl
/mnt/hdd/project_sLM_planning/output/teacher_plan_500_stdin/results.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_DIRECT = (
    "/mnt/hdd/project_sLM_planning/output/"
    "direct_500_stdin/results.jsonl"
)

DEFAULT_SELF_PLAN = (
    "/mnt/hdd/project_sLM_planning/output/"
    "self_plan_500_stdin/results.jsonl"
)

DEFAULT_TEACHER_PLAN = (
    "/mnt/hdd/project_sLM_planning/output/"
    "teacher_plan_500_stdin/results.jsonl"
)

DEFAULT_OUTPUT = "data/pilot_livecodebench_10.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze Phase 1 LiveCodeBench problem IDs "
            "for Phase 3 reproducibility."
        )
    )

    parser.add_argument(
        "--direct",
        default=DEFAULT_DIRECT,
        help="Phase 1 Direct results.jsonl.",
    )

    parser.add_argument(
        "--self-plan",
        default=DEFAULT_SELF_PLAN,
        help="Phase 1 Self-Plan results.jsonl.",
    )

    parser.add_argument(
        "--teacher-plan",
        default=DEFAULT_TEACHER_PLAN,
        help="Phase 1 Teacher-Plan results.jsonl.",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output frozen manifest JSONL.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing manifest.",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of Phase 1 problems to freeze.",
    )

    return parser.parse_args()


def load_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at "
                    f"{path}:{line_number}"
                ) from error

            if "problem_id" not in record:
                raise ValueError(
                    f"Missing problem_id at "
                    f"{path}:{line_number}"
                )

            records.append(record)

    if not records:
        raise ValueError(
            f"No records found: {path}"
        )

    return records


def assert_unique_problem_ids(
    records: list[dict[str, Any]],
    *,
    name: str,
) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []

    for record in records:
        problem_id = str(record["problem_id"])

        if problem_id in seen:
            duplicates.append(problem_id)

        seen.add(problem_id)

    if duplicates:
        raise ValueError(
            f"{name}: duplicate problem IDs found: "
            f"{sorted(set(duplicates))[:20]}"
        )


def assert_same_problem_sequence(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    reference_name: str,
    candidate_name: str,
) -> None:
    reference_ids = [
        str(record["problem_id"])
        for record in reference
    ]

    candidate_ids = [
        str(record["problem_id"])
        for record in candidate
    ]

    if reference_ids == candidate_ids:
        return

    reference_set = set(reference_ids)
    candidate_set = set(candidate_ids)

    missing = reference_set - candidate_set
    unexpected = candidate_set - reference_set

    messages = [
        f"{candidate_name} does not match "
        f"{reference_name}.",
        f"{reference_name}: {len(reference_ids)}",
        f"{candidate_name}: {len(candidate_ids)}",
    ]

    if missing:
        messages.append(
            f"Missing IDs: {sorted(missing)[:20]}"
        )

    if unexpected:
        messages.append(
            f"Unexpected IDs: "
            f"{sorted(unexpected)[:20]}"
        )

    if (
        not missing
        and not unexpected
        and len(reference_ids) == len(candidate_ids)
    ):
        for index, (
            reference_id,
            candidate_id,
        ) in enumerate(
            zip(reference_ids, candidate_ids)
        ):
            if reference_id != candidate_id:
                messages.append(
                    "First order mismatch at "
                    f"index={index}: "
                    f"{reference_name}={reference_id}, "
                    f"{candidate_name}={candidate_id}"
                )
                break

    raise ValueError("\n".join(messages))


def optional_field(
    record: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    value = record.get(key, default)

    if value is None:
        return default

    return value


def build_manifest_record(
    record: dict[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    """
    Phase 3에서 필요한 최소한의 고정 메타데이터만 저장한다.

    Phase 1 results schema에 해당 필드가 존재하는 경우에만
    추가 metadata를 보존한다.
    """
    manifest: dict[str, Any] = {
        "index": index,
        "problem_id": str(record["problem_id"]),
    }

    optional_keys = (
        "title",
        "platform",
        "contest_id",
        "contest_date",
        "difficulty",
    )

    for key in optional_keys:
        if key in record:
            manifest[key] = record[key]

    return manifest


def write_manifest(
    path: Path,
    records: list[dict[str, Any]],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest already exists: {path}\n"
            "Use --overwrite if replacement is intended."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            json.dump(
                record,
                file,
                ensure_ascii=False,
            )
            file.write("\n")


def main() -> None:
    args = parse_args()

    direct_path = Path(args.direct)
    self_plan_path = Path(args.self_plan)
    teacher_plan_path = Path(args.teacher_plan)
    output_path = Path(args.output)

    print("=" * 80)
    print("Freeze Phase 1 Problem Manifest")
    print("=" * 80)

    print(f"Direct       : {direct_path}")
    print(f"Self-Plan    : {self_plan_path}")
    print(f"Teacher-Plan : {teacher_plan_path}")
    print(f"Output       : {output_path}")
    print()

    direct = load_jsonl(direct_path)
    self_plan = load_jsonl(self_plan_path)
    teacher_plan = load_jsonl(
        teacher_plan_path
    )
    
    # Pilot에서는 Phase 1의 첫 N개 문제만 사용
    direct = direct[:args.limit]
    self_plan = self_plan[:args.limit]
    teacher_plan = teacher_plan[:args.limit]

    print(
        f"Loaded Direct       : "
        f"{len(direct)} problems"
    )
    print(
        f"Loaded Self-Plan    : "
        f"{len(self_plan)} problems"
    )
    print(
        f"Loaded Teacher-Plan : "
        f"{len(teacher_plan)} problems"
    )
    print()

    # ---------------------------------------------------------
    # 1. Duplicate ID check
    # ---------------------------------------------------------

    assert_unique_problem_ids(
        direct,
        name="Direct",
    )

    assert_unique_problem_ids(
        self_plan,
        name="Self-Plan",
    )

    assert_unique_problem_ids(
        teacher_plan,
        name="Teacher-Plan",
    )

    print("[OK] No duplicate problem IDs.")

    # ---------------------------------------------------------
    # 2. Same problem set + same order
    # ---------------------------------------------------------

    assert_same_problem_sequence(
        direct,
        self_plan,
        reference_name="Direct",
        candidate_name="Self-Plan",
    )

    assert_same_problem_sequence(
        direct,
        teacher_plan,
        reference_name="Direct",
        candidate_name="Teacher-Plan",
    )

    print(
        "[OK] Direct / Self-Plan / Teacher-Plan "
        "contain the same problem IDs in the same order."
    )

    # ---------------------------------------------------------
    # 3. Expected size
    # ---------------------------------------------------------


    print("[OK] Exactly 500 problems found.")

    # ---------------------------------------------------------
    # 4. Freeze manifest
    # ---------------------------------------------------------

    manifest = [
        build_manifest_record(
            record,
            index=index,
        )
        for index, record in enumerate(direct)
    ]

    write_manifest(
        output_path,
        manifest,
        overwrite=args.overwrite,
    )

    print()
    print("=" * 80)
    print("Manifest Summary")
    print("=" * 80)
    print(f"Problems : {len(manifest)}")
    print(
        f"First    : "
        f"{manifest[0]['problem_id']}"
    )
    print(
        f"Last     : "
        f"{manifest[-1]['problem_id']}"
    )
    print(f"Saved to : {output_path}")
    print()
    print("[DONE] Frozen problem manifest created.")


if __name__ == "__main__":
    main()