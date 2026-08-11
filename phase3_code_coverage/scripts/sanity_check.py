"""
Phase 3-B Code Best-of-N pilot sanity check.

검증 목적:
1. Phase 3-B가 Phase 1 Self-Plan을 정확히 고정했는가?
2. 문제당 code candidate N개가 정상 생성됐는가?
3. stochastic code sampling이 실제 diversity를 만드는가?
4. candidate seed / ordering / evaluation 결과가 정상인가?
5. Code Coverage@1 <= @2 <= @4 <= @8이 성립하는가?

Phase 3-B:
    Phase 1 plan_generation.raw_output
        -> fixed plan
        -> stochastic code x N
        
        
PYTHONPATH=. python -m scripts.sanity_check \
  --results /mnt/hdd/project_sLM_planning/output_phase3b/qwen25_coder_3b/best_of_8_pilot/results.jsonl \
  --expected-problems 10 \
  --expected-samples 8 \
  --require-code-diversity
  
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_PHASE1_RESULTS = Path(
    "/mnt/hdd/project_sLM_planning/output/"
    "self_plan_500_stdin/results.jsonl"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity check Phase 3-B "
            "Fixed-Plan Code Best-of-N results."
        )
    )

    parser.add_argument(
        "--results",
        required=True,
        help="Phase 3-B results.jsonl path.",
    )

    parser.add_argument(
        "--phase1-results",
        default=str(DEFAULT_PHASE1_RESULTS),
        help=(
            "Phase 1 Self-Plan results.jsonl "
            "used as fixed-plan source."
        ),
    )

    parser.add_argument(
        "--expected-problems",
        type=int,
        default=10,
        help="Expected pilot problem count.",
    )

    parser.add_argument(
        "--expected-samples",
        type=int,
        default=8,
        help="Expected code samples per problem.",
    )

    parser.add_argument(
        "--require-code-diversity",
        action="store_true",
        help=(
            "Fail if stochastic code generation "
            "produces no diversity."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
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
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(
                    stripped
                )

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL: "
                    f"{path}:{line_number}"
                ) from error

            if not isinstance(
                record,
                dict,
            ):
                raise ValueError(
                    "JSONL record must be "
                    f"an object: {path}:{line_number}"
                )

            records.append(
                record
            )

    return records


# ---------------------------------------------------------------------------
# Phase 1 fixed-plan loading
# ---------------------------------------------------------------------------


def extract_phase1_plan(
    record: dict[str, Any],
    *,
    line_number: int,
) -> str:
    """
    Phase 1 Self-Plan에서 실제 생성된 plan:

        strategy_trace
          -> name == "plan_generation"
          -> raw_output
    """

    trace = record.get(
        "strategy_trace"
    )

    if not isinstance(trace, list):
        raise ValueError(
            "Invalid strategy_trace in "
            f"Phase 1 line {line_number}."
        )

    plan_steps = [
        step
        for step in trace
        if isinstance(step, dict)
        and step.get("name")
        == "plan_generation"
    ]

    if len(plan_steps) != 1:
        raise ValueError(
            "Expected exactly one "
            "plan_generation step at "
            f"Phase 1 line {line_number}, "
            f"found {len(plan_steps)}."
        )

    raw_output = plan_steps[0].get(
        "raw_output"
    )

    if not isinstance(
        raw_output,
        str,
    ):
        raise ValueError(
            "plan_generation.raw_output "
            f"is invalid at line {line_number}."
        )

    plan = raw_output.strip()

    if not plan:
        raise ValueError(
            "Empty Phase 1 plan at "
            f"line {line_number}."
        )

    return plan


def load_phase1_plans(
    path: Path,
) -> dict[str, str]:
    records = load_jsonl(
        path
    )

    plans: dict[str, str] = {}

    for line_number, record in enumerate(
        records,
        start=1,
    ):
        problem_id = str(
            record.get(
                "problem_id",
                "",
            )
        ).strip()

        if not problem_id:
            raise ValueError(
                "Missing problem_id in "
                f"Phase 1 line {line_number}."
            )

        if problem_id in plans:
            raise ValueError(
                "Duplicate Phase 1 problem_id: "
                f"{problem_id}"
            )

        strategy = str(
            record.get(
                "strategy",
                "",
            )
        ).strip()

        if strategy != "self_plan":
            raise ValueError(
                f"{problem_id}: expected "
                "strategy='self_plan', "
                f"got '{strategy}'."
            )

        plans[problem_id] = (
            extract_phase1_plan(
                record,
                line_number=line_number,
            )
        )

    return plans


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def prefix_ks(
    num_samples: int,
) -> list[int]:
    if num_samples <= 0:
        return []

    ks: list[int] = []

    k = 1

    while k <= num_samples:
        ks.append(k)
        k *= 2

    if ks[-1] != num_samples:
        ks.append(
            num_samples
        )

    return ks


def coverage_at_k(
    candidates: list[dict[str, Any]],
    k: int,
) -> int:
    return int(
        any(
            bool(candidate["passed"])
            for candidate
            in candidates[:k]
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    results_path = Path(
        args.results
    )

    phase1_path = Path(
        args.phase1_results
    )

    results = load_jsonl(
        results_path
    )

    phase1_plans = (
        load_phase1_plans(
            phase1_path
        )
    )

    errors: list[str] = []
    warnings: list[str] = []

    status_counter: Counter[str] = (
        Counter()
    )

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    print("=" * 80)
    print(
        "Phase 3-B Code Best-of-N "
        "Pilot Sanity Check"
    )
    print("=" * 80)

    print(
        f"Results            : "
        f"{results_path}"
    )

    print(
        f"Phase 1 source     : "
        f"{phase1_path}"
    )

    print(
        f"Phase 1 plans      : "
        f"{len(phase1_plans)}"
    )

    print(
        f"Observed problems  : "
        f"{len(results)}"
    )

    print(
        f"Expected problems  : "
        f"{args.expected_problems}"
    )

    print(
        f"Expected samples   : "
        f"{args.expected_samples}"
    )

    # ------------------------------------------------------------------
    # Problem count
    # ------------------------------------------------------------------

    if (
        len(results)
        != args.expected_problems
    ):
        errors.append(
            "Problem count mismatch: "
            f"expected={args.expected_problems}, "
            f"observed={len(results)}"
        )

    # ------------------------------------------------------------------
    # Duplicate problem IDs
    # ------------------------------------------------------------------

    problem_ids = [
        str(
            record.get(
                "problem_id",
                "",
            )
        )
        for record in results
    ]

    duplicate_ids = [
        problem_id
        for problem_id, count
        in Counter(
            problem_ids
        ).items()
        if count > 1
    ]

    if duplicate_ids:
        errors.append(
            "Duplicate Phase 3-B "
            f"problem IDs: {duplicate_ids[:20]}"
        )

    # ------------------------------------------------------------------
    # Aggregate counters
    # ------------------------------------------------------------------

    total_candidates = 0

    fixed_plan_mismatches = 0
    missing_phase1_plans = 0

    invalid_candidate_counts = 0
    invalid_sample_sequences = 0
    duplicate_seed_problems = 0
    missing_seeds = 0

    prompt_linkage_failures = 0

    empty_raw_outputs = 0
    empty_extracted_codes = 0

    all_identical_problems = 0
    fully_distinct_problems = 0

    invalid_ratios = 0
    pass_ratio_mismatches = 0

    expected_sample_ids = list(
        range(
            args.expected_samples
        )
    )

    # ------------------------------------------------------------------
    # Per-problem checks
    # ------------------------------------------------------------------

    for record in results:
        problem_id = str(
            record.get(
                "problem_id",
                "",
            )
        ).strip()

        if not problem_id:
            errors.append(
                "Result record with "
                "empty problem_id."
            )
            continue

        # --------------------------------------------------------------
        # Fixed plan source
        # --------------------------------------------------------------

        fixed_plan = str(
            record.get(
                "fixed_plan",
                "",
            )
        ).strip()

        if not fixed_plan:
            errors.append(
                f"{problem_id}: "
                "fixed_plan is empty."
            )

        if problem_id not in phase1_plans:
            missing_phase1_plans += 1

            errors.append(
                f"{problem_id}: "
                "not found in Phase 1 "
                "Self-Plan results."
            )

        else:
            expected_plan = (
                phase1_plans[
                    problem_id
                ].strip()
            )

            if (
                fixed_plan
                != expected_plan
            ):
                fixed_plan_mismatches += 1

                errors.append(
                    f"{problem_id}: "
                    "Phase 3-B fixed_plan "
                    "does not match "
                    "Phase 1 plan_generation."
                )

        # --------------------------------------------------------------
        # Candidate count
        # --------------------------------------------------------------

        candidates = record.get(
            "candidates",
            [],
        )

        if not isinstance(
            candidates,
            list,
        ):
            errors.append(
                f"{problem_id}: "
                "candidates is not a list."
            )
            continue

        total_candidates += len(
            candidates
        )

        if (
            len(candidates)
            != args.expected_samples
        ):
            invalid_candidate_counts += 1

            errors.append(
                f"{problem_id}: "
                f"expected "
                f"{args.expected_samples} "
                f"candidates, got "
                f"{len(candidates)}."
            )

            continue

        # --------------------------------------------------------------
        # sample_id ordering
        # --------------------------------------------------------------

        sample_ids = [
            int(
                candidate.get(
                    "sample_id",
                    -1,
                )
            )
            for candidate
            in candidates
        ]

        if (
            sample_ids
            != expected_sample_ids
        ):
            invalid_sample_sequences += 1

            errors.append(
                f"{problem_id}: invalid "
                f"sample sequence "
                f"{sample_ids}."
            )

        # --------------------------------------------------------------
        # Seeds
        # --------------------------------------------------------------

        seeds = [
            candidate.get(
                "sample_seed"
            )
            for candidate
            in candidates
        ]

        valid_seeds = [
            seed
            for seed in seeds
            if seed is not None
        ]

        if (
            len(valid_seeds)
            != len(candidates)
        ):
            missing_seeds += (
                len(candidates)
                - len(valid_seeds)
            )

            errors.append(
                f"{problem_id}: "
                "missing candidate seed."
            )

        elif (
            len(set(valid_seeds))
            != len(valid_seeds)
        ):
            duplicate_seed_problems += 1

            errors.append(
                f"{problem_id}: "
                "duplicate candidate seeds."
            )

        # --------------------------------------------------------------
        # Candidate structural checks
        # --------------------------------------------------------------

        codes: list[str] = []

        for candidate in candidates:
            sample_id = int(
                candidate[
                    "sample_id"
                ]
            )

            if not bool(
                candidate.get(
                    "plan_in_code_prompt",
                    False,
                )
            ):
                prompt_linkage_failures += 1

                errors.append(
                    f"{problem_id}/"
                    f"sample{sample_id}: "
                    "plan_in_code_prompt=False."
                )

            raw_output = str(
                candidate.get(
                    "raw_output",
                    "",
                )
            ).strip()

            if not raw_output:
                empty_raw_outputs += 1

            code = str(
                candidate.get(
                    "code",
                    "",
                )
            ).strip()

            if not code:
                empty_extracted_codes += 1

            else:
                codes.append(
                    code
                )

            ratio = float(
                candidate.get(
                    "test_pass_ratio",
                    0.0,
                )
            )

            if not (
                0.0 <= ratio <= 1.0
            ):
                invalid_ratios += 1

                errors.append(
                    f"{problem_id}/"
                    f"sample{sample_id}: "
                    "invalid "
                    f"test_pass_ratio={ratio}"
                )

            passed = bool(
                candidate.get(
                    "passed",
                    False,
                )
            )

            if (
                passed
                and ratio < 1.0
            ):
                pass_ratio_mismatches += 1

                errors.append(
                    f"{problem_id}/"
                    f"sample{sample_id}: "
                    "passed=True but "
                    f"ratio={ratio}"
                )

            status = str(
                candidate.get(
                    "status",
                    "UNKNOWN",
                )
            )

            status_counter[
                status
            ] += 1

        # --------------------------------------------------------------
        # Code diversity
        # --------------------------------------------------------------

        if codes:
            distinct_codes = len(
                set(codes)
            )

            if distinct_codes == 1:
                all_identical_problems += 1

            if (
                distinct_codes
                == len(codes)
            ):
                fully_distinct_problems += 1

        # --------------------------------------------------------------
        # Problem-level Oracle sanity
        # --------------------------------------------------------------

        summary = record.get(
            "summary",
            {},
        )

        calculated_num_passed = sum(
            bool(
                candidate[
                    "passed"
                ]
            )
            for candidate
            in candidates
        )

        calculated_oracle = (
            calculated_num_passed > 0
        )

        if isinstance(summary, dict):

            stored_num_passed = (
                summary.get(
                    "num_passed"
                )
            )

            if (
                stored_num_passed
                is not None
                and int(
                    stored_num_passed
                )
                != calculated_num_passed
            ):
                errors.append(
                    f"{problem_id}: "
                    "summary.num_passed "
                    "mismatch."
                )

            stored_oracle = (
                summary.get(
                    "oracle_passed"
                )
            )

            if (
                stored_oracle
                is not None
                and bool(
                    stored_oracle
                )
                != calculated_oracle
            ):
                errors.append(
                    f"{problem_id}: "
                    "summary.oracle_passed "
                    "mismatch."
                )

        # --------------------------------------------------------------
        # Prefix monotonicity
        # --------------------------------------------------------------

        ks = prefix_ks(
            len(candidates)
        )

        previous = -1

        for k in ks:
            current = (
                coverage_at_k(
                    candidates,
                    k,
                )
            )

            if current < previous:
                errors.append(
                    f"{problem_id}: "
                    "Code Coverage "
                    "monotonicity violation "
                    f"at k={k}."
                )

            previous = current

    # ------------------------------------------------------------------
    # Aggregate coverage
    # ------------------------------------------------------------------

    ks = prefix_ks(
        args.expected_samples
    )

    coverage_rows: list[
        tuple[int, int, float]
    ] = []

    valid_records = [
        record
        for record in results
        if isinstance(
            record.get(
                "candidates"
            ),
            list,
        )
        and len(
            record["candidates"]
        )
        == args.expected_samples
    ]

    for k in ks:
        solved = sum(
            coverage_at_k(
                record["candidates"],
                k,
            )
            for record
            in valid_records
        )

        rate = (
            solved
            / len(valid_records)
            if valid_records
            else 0.0
        )

        coverage_rows.append(
            (
                k,
                solved,
                rate,
            )
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    print()
    print("Integrity")
    print("-" * 80)

    expected_candidates = (
        args.expected_problems
        * args.expected_samples
    )

    print(
        f"Problems               : "
        f"{len(results)}/"
        f"{args.expected_problems}"
    )

    print(
        f"Candidates             : "
        f"{total_candidates}/"
        f"{expected_candidates}"
    )

    print(
        f"Fixed-plan mismatches  : "
        f"{fixed_plan_mismatches}"
    )

    print(
        f"Missing Phase1 plans   : "
        f"{missing_phase1_plans}"
    )

    print(
        f"Invalid candidate cnt  : "
        f"{invalid_candidate_counts}"
    )

    print(
        f"Invalid sample seq     : "
        f"{invalid_sample_sequences}"
    )

    print(
        f"Duplicate seed probs   : "
        f"{duplicate_seed_problems}"
    )

    print(
        f"Missing seeds          : "
        f"{missing_seeds}"
    )

    print(
        f"Prompt linkage failure : "
        f"{prompt_linkage_failures}"
    )

    print(
        f"Invalid ratios         : "
        f"{invalid_ratios}"
    )

    print(
        f"PASS/ratio mismatch    : "
        f"{pass_ratio_mismatches}"
    )

    # ------------------------------------------------------------------
    # Diversity
    # ------------------------------------------------------------------

    print()
    print("Code Diversity")
    print("-" * 80)

    print(
        f"All-identical problems : "
        f"{all_identical_problems}/"
        f"{len(valid_records)}"
    )

    print(
        f"Fully-distinct problems: "
        f"{fully_distinct_problems}/"
        f"{len(valid_records)}"
    )

    print(
        f"Empty raw outputs      : "
        f"{empty_raw_outputs}"
    )

    print(
        f"Empty extracted codes  : "
        f"{empty_extracted_codes}"
    )

    if (
        valid_records
        and all_identical_problems
        == len(valid_records)
    ):
        message = (
            "All problems produced "
            "identical code candidates. "
            "Stochastic code sampling "
            "may not be active."
        )

        if args.require_code_diversity:
            errors.append(
                message
            )

        else:
            warnings.append(
                message
            )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    print()
    print(
        "Candidate Status Distribution"
    )
    print("-" * 80)

    for status, count in (
        status_counter.most_common()
    ):
        ratio = (
            count
            / total_candidates
            if total_candidates
            else 0.0
        )

        print(
            f"{status:<24} "
            f"{count:>5} "
            f"{ratio:>8.4f}"
        )

    # Infrastructure failures

    infrastructure_failures = (
        status_counter[
            "EVALUATION_ERROR"
        ]
        + status_counter[
            "UNSUPPORTED_TEST_TYPE"
        ]
    )

    if infrastructure_failures:
        warnings.append(
            "Infrastructure evaluation "
            "failures detected: "
            f"{infrastructure_failures}"
        )

    generation_failures = (
        status_counter[
            "CODE_GENERATION_ERROR"
        ]
    )

    if generation_failures:
        warnings.append(
            "Code generation failures "
            f"detected: {generation_failures}"
        )

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------

    print()
    print("Pilot Code Coverage")
    print("-" * 80)

    print(
        f"{'N':>4} "
        f"{'Solved':>8} "
        f"{'Coverage':>10}"
    )

    previous_rate = -1.0

    for k, solved, rate in (
        coverage_rows
    ):
        print(
            f"{k:>4} "
            f"{solved:>8} "
            f"{rate:>10.4f}"
        )

        if rate < previous_rate:
            errors.append(
                "Aggregate Code Coverage "
                f"decreased at k={k}."
            )

        previous_rate = rate

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    print()
    print("=" * 80)

    if warnings:
        print("WARNINGS")
        print("-" * 80)

        for warning in warnings:
            print(
                f"[WARN] {warning}"
            )

        print()

    if errors:
        print(
            "SANITY CHECK FAILED"
        )
        print("-" * 80)

        for error in errors:
            print(
                f"[FAIL] {error}"
            )

        return 1

    print(
        "SANITY CHECK PASSED"
    )
    print("-" * 80)

    print(
        "[OK] Phase 1 fixed plans match exactly."
    )

    print(
        "[OK] Problem/candidate counts are valid."
    )

    print(
        "[OK] sample_id ordering is valid."
    )

    print(
        "[OK] Candidate seeds are unique."
    )

    print(
        "[OK] Fixed plans are linked to code prompts."
    )

    print(
        "[OK] Evaluation ratios are valid."
    )

    print(
        "[OK] Code Coverage@k is monotonic."
    )

    print(
        "[OK] No structural pipeline errors found."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )