"""
Phase 1과 Phase 3가 동일한 problem ID뿐 아니라, 실제 evaluation test content까지 동일한지 검증한다.

PYTHONPATH=. python -m scripts.verify_phase1_phase3_tests

PYTHONPATH=. python -m scripts.verify_phase1_phase3_tests \
  --limit 10 \
  --output data/phase1_phase3_test_manifest_pilot10.jsonl


"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.common.datasets.dataset_loader import DatasetLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that Phase 1 and Phase 3 use identical "
            "LiveCodeBench problems and test cases."
        )
    )

    parser.add_argument(
        "--phase1-results",
        default=(
            "/mnt/hdd/project_sLM_planning/output/"
            "self_plan_500_stdin/results.jsonl"
        ),
        help=(
            "Phase 1 results.jsonl used to define "
            "the reference problem IDs."
        ),
    )

    parser.add_argument(
        "--dataset-name",
        default="livecodebench_v6",
    )

    parser.add_argument(
        "--split",
        default="test",
    )

    parser.add_argument(
        "--release-version",
        default="release_v6",
    )

    parser.add_argument(
        "--test-type",
        default="stdin",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--output",
        default="data/phase1_phase3_test_manifest.jsonl",
        help="Where to save the verification manifest.",
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

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL: {path}:{line_number}"
                ) from error

    return records


def canonical_json(value: Any) -> str:
    """Stable serialization for hashing."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_of_value(value: Any) -> str:
    payload = canonical_json(value).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def get_problem_id(example: Any) -> str:
    return str(example.problem_id)


def extract_tests(example: Any) -> dict[str, Any]:
    """
    Extract the exact test-related fields available in ProblemExample.

    Adjust the candidate field names below if your schema uses
    different attribute names.
    """
    possible_fields = (
        "public_tests",
        "private_tests",
        "generated_tests",
        "tests",
        "test",
    )

    extracted: dict[str, Any] = {}

    for field_name in possible_fields:
        if hasattr(example, field_name):
            value = getattr(example, field_name)

            if value is not None:
                extracted[field_name] = value

    if not extracted:
        raise AttributeError(
            "Could not find test fields on ProblemExample. "
            "Inspect src/common/schemas.py and update "
            "extract_tests() with the actual field names."
        )

    return extracted


def count_tests(test_bundle: dict[str, Any]) -> int:
    """
    Count test cases across the extracted test containers.

    This is only a descriptive count; the SHA-256 hash is the
    stronger equality check.
    """
    total = 0

    for value in test_bundle.values():
        if isinstance(value, list):
            total += len(value)

        elif isinstance(value, dict):
            # Common format:
            # {"input": [...], "output": [...]}
            lengths = [
                len(item)
                for item in value.values()
                if isinstance(item, list)
            ]

            if lengths:
                total += max(lengths)

    return total


def build_phase1_id_sequence(
    records: list[dict[str, Any]],
) -> list[str]:
    ids = [
        str(record["problem_id"])
        for record in records
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate problem IDs found in Phase 1 results."
        )

    return ids


def main() -> None:
    args = parse_args()

    phase1_path = Path(args.phase1_results)
    output_path = Path(args.output)

    print("=" * 80)
    print("Verify Phase 1 vs Phase 3 Tests")
    print("=" * 80)

    print(f"Phase 1 results : {phase1_path}")
    print(f"Dataset         : {args.dataset_name}")
    print(f"Split           : {args.split}")
    print(f"Release         : {args.release_version}")
    print(f"Test type       : {args.test_type}")
    print(f"Limit           : {args.limit}")
    print()

    # ------------------------------------------------------------------
    # 1. Load Phase 1 reference problem IDs
    # ------------------------------------------------------------------

    phase1_records = load_jsonl(
        phase1_path
    )
    
    # Pilot에서는 앞의 N개만 검증
    phase1_records = phase1_records[:args.limit]

    phase1_ids = build_phase1_id_sequence(
        phase1_records
    )


    if len(phase1_ids) != args.limit:
        raise ValueError(
            f"Phase 1 result count mismatch: "
            f"expected={args.limit}, "
            f"observed={len(phase1_ids)}"
        )

    print(
        f"[OK] Loaded {len(phase1_ids)} "
        "Phase 1 problem IDs."
    )

    # ------------------------------------------------------------------
    # 2. Reload LiveCodeBench exactly as Phase 3 does
    # ------------------------------------------------------------------

    loader = DatasetLoader(
        dataset_name=args.dataset_name,
        split=args.split,
        limit=args.limit,
        test_type=args.test_type,
        release_version=args.release_version,
    )

    examples = loader.load()

    phase3_ids = [
        get_problem_id(example)
        for example in examples
    ]

    if phase3_ids != phase1_ids:
        phase1_set = set(phase1_ids)
        phase3_set = set(phase3_ids)

        missing = phase1_set - phase3_set
        unexpected = phase3_set - phase1_set

        details = [
            "Problem IDs do not match.",
            f"Phase 1 count={len(phase1_ids)}",
            f"Phase 3 count={len(phase3_ids)}",
        ]

        if missing:
            details.append(
                f"Missing in Phase 3: "
                f"{sorted(missing)[:20]}"
            )

        if unexpected:
            details.append(
                f"Unexpected in Phase 3: "
                f"{sorted(unexpected)[:20]}"
            )

        if (
            not missing
            and not unexpected
            and len(phase1_ids) == len(phase3_ids)
        ):
            for index, (
                phase1_id,
                phase3_id,
            ) in enumerate(
                zip(phase1_ids, phase3_ids)
            ):
                if phase1_id != phase3_id:
                    details.append(
                        f"First ordering mismatch at "
                        f"index={index}: "
                        f"phase1={phase1_id}, "
                        f"phase3={phase3_id}"
                    )
                    break

        raise ValueError(
            "\n".join(details)
        )

    print(
        "[OK] Phase 1 and Phase 3 problem IDs "
        "match exactly in the same order."
    )

    # ------------------------------------------------------------------
    # 3. Extract test content and hash it
    # ------------------------------------------------------------------

    manifest: list[dict[str, Any]] = []

    total_tests = 0

    for index, example in enumerate(examples):
        problem_id = get_problem_id(example)

        tests = extract_tests(example)

        test_count = count_tests(tests)
        test_hash = sha256_of_value(tests)

        total_tests += test_count

        manifest.append(
            {
                "index": index,
                "problem_id": problem_id,
                "test_type": args.test_type,
                "num_tests": test_count,
                "test_sha256": test_hash,
            }
        )

    # ------------------------------------------------------------------
    # 4. Compare against Phase 1 recorded total_tests if available
    # ------------------------------------------------------------------

    count_mismatches: list[dict[str, Any]] = []

    for phase1_record, manifest_record in zip(
        phase1_records,
        manifest,
    ):
        if "total_tests" not in phase1_record:
            continue

        phase1_total = int(
            phase1_record["total_tests"]
        )

        current_total = int(
            manifest_record["num_tests"]
        )

        if phase1_total != current_total:
            count_mismatches.append(
                {
                    "problem_id": manifest_record[
                        "problem_id"
                    ],
                    "phase1_total_tests": phase1_total,
                    "phase3_loaded_tests": current_total,
                }
            )

    # ------------------------------------------------------------------
    # 5. Save manifest
    # ------------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in manifest:
            json.dump(
                record,
                file,
                ensure_ascii=False,
            )
            file.write("\n")

    # ------------------------------------------------------------------
    # 6. Report
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("Test Manifest Summary")
    print("=" * 80)

    print(f"Problems       : {len(manifest)}")
    print(f"Total tests    : {total_tests}")
    print(f"Manifest       : {output_path}")

    print()

    print("First 10 problems")
    print("-" * 80)

    for record in manifest[:10]:
        print(
            f"{record['problem_id']:<20} "
            f"tests={record['num_tests']:<4} "
            f"sha256={record['test_sha256'][:16]}..."
        )

    print()

    if count_mismatches:
        print(
            f"[FAIL] Found {len(count_mismatches)} "
            "test-count mismatches."
        )

        for mismatch in count_mismatches[:20]:
            print(
                f"  {mismatch['problem_id']}: "
                f"Phase1="
                f"{mismatch['phase1_total_tests']} "
                f"Phase3="
                f"{mismatch['phase3_loaded_tests']}"
            )

        raise SystemExit(1)

    print(
        "[OK] No Phase 1 vs Phase 3 test-count "
        "mismatches were detected."
    )

    print()
    print(
        "[IMPORTANT] test_sha256 records the exact "
        "currently loaded test content."
    )
    print(
        "To prove byte/content-level equality across "
        "future runs, compare this manifest's hashes "
        "against another run generated with the same script."
    )


if __name__ == "__main__":
    main()