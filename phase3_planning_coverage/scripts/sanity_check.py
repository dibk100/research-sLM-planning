"""
PYTHONPATH=. python -m scripts.run_best_of_n \
  --config configs/qwen25_coder_3b_pilot.yaml

"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils import (
    check_monotonicity,
    oracle_at_k,
    prefix_ks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity check Phase 3-A pilot results."
    )

    parser.add_argument(
        "--results",
        required=True,
        help="Path to Phase 3-A results.jsonl.",
    )

    parser.add_argument(
        "--expected-problems",
        type=int,
        default=10,
        help="Expected number of pilot problems.",
    )

    parser.add_argument(
        "--expected-samples",
        type=int,
        default=8,
        help="Expected number of candidates per problem.",
    )

    return parser.parse_args()


def load_results(
    path: Path,
) -> list[dict[str, Any]]:
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
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}"
                ) from error

    return records


def main() -> int:
    args = parse_args()

    results_path = Path(args.results)
    records = load_results(results_path)

    errors: list[str] = []
    warnings: list[str] = []

    print("=" * 80)
    print("Phase 3-A Pilot Sanity Check")
    print("=" * 80)
    print(f"Results            : {results_path}")
    print(f"Problems           : {len(records)}")
    print(f"Expected problems  : {args.expected_problems}")
    print(f"Expected samples   : {args.expected_samples}")
    print()

    # ------------------------------------------------------------------
    # 1. Problem count
    # ------------------------------------------------------------------

    if len(records) != args.expected_problems:
        errors.append(
            "Problem count mismatch: "
            f"expected={args.expected_problems}, "
            f"observed={len(records)}"
        )

    # ------------------------------------------------------------------
    # 2. Duplicate problem IDs
    # ------------------------------------------------------------------

    problem_ids = [
        record["problem_id"]
        for record in records
    ]

    duplicate_ids = [
        problem_id
        for problem_id, count
        in Counter(problem_ids).items()
        if count > 1
    ]

    if duplicate_ids:
        errors.append(
            f"Duplicate problem IDs: {duplicate_ids}"
        )

    # ------------------------------------------------------------------
    # Candidate-level checks
    # ------------------------------------------------------------------

    expected_sample_ids = list(
        range(args.expected_samples)
    )

    status_counter: Counter[str] = Counter()

    total_candidates = 0
    total_empty_plans = 0
    total_prompt_failures = 0

    problems_all_identical = 0

    for record in records:
        problem_id = record["problem_id"]
        candidates = record["candidates"]

        total_candidates += len(candidates)

        # --------------------------------------------------------------
        # 3. Candidate count
        # --------------------------------------------------------------

        if len(candidates) != args.expected_samples:
            errors.append(
                f"{problem_id}: candidate count "
                f"{len(candidates)} != "
                f"{args.expected_samples}"
            )

            continue

        # --------------------------------------------------------------
        # 4. sample_id sequence
        # --------------------------------------------------------------

        sample_ids = [
            int(candidate["sample_id"])
            for candidate in candidates
        ]

        if sample_ids != expected_sample_ids:
            errors.append(
                f"{problem_id}: invalid sample sequence "
                f"{sample_ids}"
            )

        # --------------------------------------------------------------
        # 5. Unique candidate seeds
        # --------------------------------------------------------------

        seeds = [
            int(candidate["sample_seed"])
            for candidate in candidates
        ]

        if len(set(seeds)) != len(seeds):
            errors.append(
                f"{problem_id}: duplicate candidate seeds"
            )

        # --------------------------------------------------------------
        # 6. Plan diversity
        # --------------------------------------------------------------

        plans = [
            candidate["plan"].strip()
            for candidate in candidates
        ]

        nonempty_plans = [
            plan for plan in plans if plan
        ]

        empty_count = (
            len(plans) - len(nonempty_plans)
        )

        total_empty_plans += empty_count

        if (
            nonempty_plans
            and len(set(nonempty_plans)) == 1
        ):
            problems_all_identical += 1

        # --------------------------------------------------------------
        # 7. Plan actually entered code prompt
        # --------------------------------------------------------------

        for candidate in candidates:
            status_counter[
                candidate["status"]
            ] += 1

            if candidate.get(
                "plan_empty",
                False,
            ):
                continue

            if not candidate.get(
                "plan_in_code_prompt",
                False,
            ):
                total_prompt_failures += 1

                errors.append(
                    f"{problem_id}/sample"
                    f"{candidate['sample_id']}: "
                    "plan missing from code prompt"
                )

        # --------------------------------------------------------------
        # 8. test_pass_ratio range
        # --------------------------------------------------------------

        for candidate in candidates:
            ratio = float(
                candidate["test_pass_ratio"]
            )

            if not 0.0 <= ratio <= 1.0:
                errors.append(
                    f"{problem_id}/sample"
                    f"{candidate['sample_id']}: "
                    f"invalid test_pass_ratio={ratio}"
                )

        # --------------------------------------------------------------
        # 9. Oracle prefix monotonicity
        # --------------------------------------------------------------

        ks = prefix_ks(len(candidates))

        oracle_values = [
            float(
                oracle_at_k(
                    candidates,
                    k,
                )
            )
            for k in ks
        ]

        violations = check_monotonicity(
            oracle_values
        )

        if violations:
            errors.append(
                f"{problem_id}: Oracle prefix "
                "monotonicity violation"
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    expected_total_candidates = (
        args.expected_problems
        * args.expected_samples
    )

    print("Integrity")
    print("-" * 80)
    print(
        f"Candidates          : "
        f"{total_candidates}/"
        f"{expected_total_candidates}"
    )
    print(
        f"Empty plans         : "
        f"{total_empty_plans}"
    )
    print(
        f"Prompt linkage fail : "
        f"{total_prompt_failures}"
    )
    print(
        f"All-identical plans : "
        f"{problems_all_identical}/"
        f"{len(records)}"
    )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    print()
    print("Candidate Status")
    print("-" * 80)

    for status, count in (
        status_counter.most_common()
    ):
        print(
            f"{status:<25} "
            f"{count:>5}"
        )

    # Infrastructure failures
    infrastructure_failures = (
        status_counter["EVALUATION_ERROR"]
        + status_counter["UNSUPPORTED_TEST_TYPE"]
    )

    if infrastructure_failures > 0:
        warnings.append(
            "Infrastructure-related evaluation failures "
            f"detected: {infrastructure_failures}"
        )

    # Sampling diversity warning
    if (
        len(records) > 0
        and problems_all_identical
        == len(records)
    ):
        errors.append(
            "All problems produced identical plans "
            "for all samples. Sampling may not be active."
        )

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    print()
    print("=" * 80)

    if warnings:
        print("WARNINGS")
        print("-" * 80)

        for warning in warnings:
            print(f"[WARN] {warning}")

        print()

    if errors:
        print("SANITY CHECK FAILED")
        print("-" * 80)

        for error in errors:
            print(f"[FAIL] {error}")

        return 1

    print("SANITY CHECK PASSED")
    print("-" * 80)
    print(
        "[OK] Candidate count and ordering are valid."
    )
    print(
        "[OK] Candidate seeds are distinct per problem."
    )
    print(
        "[OK] Plans are connected to their code prompts."
    )
    print(
        "[OK] Oracle prefix monotonicity holds."
    )
    print(
        "[OK] No structural problem detected."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())