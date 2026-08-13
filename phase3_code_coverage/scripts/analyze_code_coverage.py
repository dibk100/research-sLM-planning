"""
Phase 3-B Code Coverage Analysis.

Phase 3-A:
    sampled plan x N
        -> greedy code
        -> Planning Coverage@k

Phase 3-B:
    fixed Phase-1 Self-Plan
        -> sampled code x N
        -> Code Coverage@k

주요 분석:
- Code Coverage@1/@2/@4/@8
- unbiased pass@k
- mean best test-pass ratio
- difficulty breakdown
- per-candidate performance
- code diversity
- candidate status distribution
- Phase 3-A Planning Coverage와 직접 비교

CSV 저장:
    <현재 실행 위치>/archive/analysis/
    
    
PYTHONPATH=. python -m scripts.analyze_code_coverage \
  --results /mnt/hdd/project_sLM_planning/output_phase3/qwen25_coder_3b/coder_best_of_8/results.jsonl \
  --planning-results /mnt/hdd/project_sLM_planning/output_phase3/qwen25_coder_3b/best_of_8/results.jsonl \
  --phase1-pass-rate 0.168 \
  --teacher-pass-rate 0.340
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Phase 3-B "
            "Fixed-Plan Code Best-of-N coverage."
        )
    )

    parser.add_argument(
        "--results",
        required=True,
        help="Phase 3-B results.jsonl.",
    )

    parser.add_argument(
        "--planning-results",
        default=None,
        help=(
            "Optional Phase 3-A Planning Best-of-N results.jsonl. "
            "If provided, Planning vs Code coverage is compared directly."
        ),
    )

    parser.add_argument(
        "--phase1-pass-rate",
        type=float,
        default=0.168,
        help="Phase 1 greedy Self-Plan pass rate.",
    )

    parser.add_argument(
        "--teacher-pass-rate",
        type=float,
        default=0.340,
        help="Phase 1 Teacher-Plan pass rate.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Analysis directory. "
            "Default: ./archive/analysis"
        ),
    )

    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}"
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

            if not isinstance(record, dict):
                raise ValueError(
                    "JSONL record must be an object: "
                    f"{path}:{line_number}"
                )

            records.append(
                record
            )

    if not records:
        raise ValueError(
            f"No records found: {path}"
        )

    return records


def mean(
    values: Iterable[float],
) -> float:
    values = list(values)

    if not values:
        return 0.0

    return sum(values) / len(values)


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


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Candidate access
# ---------------------------------------------------------------------------


def resolve_num_samples(
    records: list[dict[str, Any]],
) -> int:
    counts = {
        len(record.get("candidates", []))
        for record in records
    }

    if len(counts) != 1:
        raise ValueError(
            "Inconsistent candidate counts: "
            f"{sorted(counts)}"
        )

    num_samples = counts.pop()

    if num_samples <= 0:
        raise ValueError(
            "No candidates found."
        )

    return num_samples


def oracle_at_k(
    candidates: list[dict[str, Any]],
    k: int,
) -> bool:
    return any(
        bool(candidate.get("passed", False))
        for candidate in candidates[:k]
    )


def best_ratio_at_k(
    candidates: list[dict[str, Any]],
    k: int,
) -> float:
    return max(
        (
            float(
                candidate.get(
                    "test_pass_ratio",
                    0.0,
                )
            )
            for candidate in candidates[:k]
        ),
        default=0.0,
    )


# ---------------------------------------------------------------------------
# Unbiased pass@k
# ---------------------------------------------------------------------------


def unbiased_pass_at_k(
    *,
    num_samples: int,
    num_correct: int,
    k: int,
) -> float:
    """
    Standard unbiased pass@k estimator:

        1 - C(n-c, k) / C(n, k)
    """

    if k <= 0:
        return 0.0

    if num_correct <= 0:
        return 0.0

    if num_samples - num_correct < k:
        return 1.0

    return 1.0 - (
        math.comb(
            num_samples - num_correct,
            k,
        )
        / math.comb(
            num_samples,
            k,
        )
    )


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


def validate_records(
    records: list[dict[str, Any]],
    num_samples: int,
) -> dict[str, int]:
    expected_ids = list(
        range(num_samples)
    )

    seen_ids: set[str] = set()

    duplicate_problem_ids = 0
    invalid_sequences = 0
    invalid_ratios = 0
    pass_ratio_mismatches = 0
    missing_extracted_codes = 0

    for record in records:
        problem_id = str(
            record["problem_id"]
        )

        if problem_id in seen_ids:
            duplicate_problem_ids += 1

        seen_ids.add(
            problem_id
        )

        candidates = record[
            "candidates"
        ]

        sample_ids = [
            int(
                candidate[
                    "sample_id"
                ]
            )
            for candidate in candidates
        ]

        if sample_ids != expected_ids:
            invalid_sequences += 1

        for candidate in candidates:
            ratio = float(
                candidate.get(
                    "test_pass_ratio",
                    0.0,
                )
            )

            if not 0.0 <= ratio <= 1.0:
                invalid_ratios += 1

            if (
                bool(
                    candidate.get(
                        "passed",
                        False,
                    )
                )
                and ratio < 1.0
            ):
                pass_ratio_mismatches += 1

            extracted_code = str(
                candidate.get(
                    "extracted_code",
                    "",
                )
            ).strip()

            if not extracted_code:
                missing_extracted_codes += 1

    return {
        "duplicate_problem_ids": (
            duplicate_problem_ids
        ),
        "invalid_sequences": (
            invalid_sequences
        ),
        "invalid_ratios": (
            invalid_ratios
        ),
        "pass_ratio_mismatches": (
            pass_ratio_mismatches
        ),
        "missing_extracted_codes": (
            missing_extracted_codes
        ),
    }


# ---------------------------------------------------------------------------
# Overall coverage
# ---------------------------------------------------------------------------


def coverage_rows(
    records: list[dict[str, Any]],
    ks: list[int],
    num_samples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for k in ks:
        solved_flags = [
            oracle_at_k(
                record["candidates"],
                k,
            )
            for record in records
        ]

        best_ratios = [
            best_ratio_at_k(
                record["candidates"],
                k,
            )
            for record in records
        ]

        unbiased_values: list[float] = []

        for record in records:
            num_correct = sum(
                bool(
                    candidate.get(
                        "passed",
                        False,
                    )
                )
                for candidate
                in record["candidates"]
            )

            unbiased_values.append(
                unbiased_pass_at_k(
                    num_samples=num_samples,
                    num_correct=num_correct,
                    k=k,
                )
            )

        rows.append(
            {
                "k": k,
                "num_problems": (
                    len(records)
                ),
                "solved": sum(
                    solved_flags
                ),
                "coverage": mean(
                    float(flag)
                    for flag
                    in solved_flags
                ),
                "unbiased_pass_at_k": mean(
                    unbiased_values
                ),
                "mean_best_test_pass_ratio": mean(
                    best_ratios
                ),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Difficulty
# ---------------------------------------------------------------------------


DIFFICULTY_ORDER = (
    "easy",
    "medium",
    "hard",
)


def difficulty_rows(
    records: list[dict[str, Any]],
    ks: list[int],
    num_samples: int,
) -> list[dict[str, Any]]:
    groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        difficulty = str(
            record.get(
                "difficulty",
                "unknown",
            )
        ).lower()

        groups[
            difficulty
        ].append(
            record
        )

    ordered = [
        difficulty
        for difficulty
        in DIFFICULTY_ORDER
        if difficulty in groups
    ]

    ordered += [
        difficulty
        for difficulty
        in sorted(groups)
        if difficulty
        not in DIFFICULTY_ORDER
    ]

    rows: list[dict[str, Any]] = []

    for difficulty in ordered:
        coverage = coverage_rows(
            groups[difficulty],
            ks,
            num_samples,
        )

        for row in coverage:
            rows.append(
                {
                    "difficulty": (
                        difficulty
                    ),
                    **row,
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Per-candidate performance
# ---------------------------------------------------------------------------


def per_candidate_rows(
    records: list[dict[str, Any]],
    num_samples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for sample_id in range(
        num_samples
    ):
        candidates = [
            record["candidates"][
                sample_id
            ]
            for record in records
        ]

        rows.append(
            {
                "sample_id": (
                    sample_id
                ),
                "num_problems": (
                    len(candidates)
                ),
                "solved": sum(
                    bool(
                        candidate.get(
                            "passed",
                            False,
                        )
                    )
                    for candidate
                    in candidates
                ),
                "pass_rate": mean(
                    float(
                        bool(
                            candidate.get(
                                "passed",
                                False,
                            )
                        )
                    )
                    for candidate
                    in candidates
                ),
                "mean_test_pass_ratio": mean(
                    float(
                        candidate.get(
                            "test_pass_ratio",
                            0.0,
                        )
                    )
                    for candidate
                    in candidates
                ),
                "mean_completion_tokens": mean(
                    float(
                        candidate.get(
                            "completion_tokens",
                            0,
                        )
                    )
                    for candidate
                    in candidates
                ),
                "mean_generation_time": mean(
                    float(
                        candidate.get(
                            "generation_time",
                            0.0,
                        )
                    )
                    for candidate
                    in candidates
                ),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Code diversity
# ---------------------------------------------------------------------------


def normalized_code_tokens(
    code: str,
) -> set[str]:
    return set(
        code.lower().split()
    )


def code_diversity_rows(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for record in records:
        codes = [
            str(
                candidate.get(
                    "extracted_code",
                    "",
                )
            ).strip()
            for candidate
            in record["candidates"]
        ]

        nonempty_codes = [
            code
            for code in codes
            if code
        ]

        token_sets = [
            normalized_code_tokens(
                code
            )
            for code
            in nonempty_codes
        ]

        similarities: list[
            float
        ] = []

        for left in range(
            len(token_sets)
        ):
            for right in range(
                left + 1,
                len(token_sets),
            ):
                union = (
                    token_sets[left]
                    | token_sets[right]
                )

                if not union:
                    continue

                intersection = (
                    token_sets[left]
                    & token_sets[right]
                )

                similarities.append(
                    len(intersection)
                    / len(union)
                )

        rows.append(
            {
                "problem_id": (
                    record["problem_id"]
                ),
                "difficulty": (
                    record.get(
                        "difficulty",
                        "unknown",
                    )
                ),
                "num_candidates": (
                    len(codes)
                ),
                "distinct_codes": (
                    len(
                        set(
                            nonempty_codes
                        )
                    )
                ),
                "empty_codes": (
                    len(codes)
                    - len(nonempty_codes)
                ),
                "mean_pairwise_jaccard": mean(
                    similarities
                ),
                "max_pairwise_jaccard": (
                    max(similarities)
                    if similarities
                    else 0.0
                ),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Problem-level coverage
# ---------------------------------------------------------------------------


def problem_rows(
    records: list[dict[str, Any]],
    ks: list[int],
) -> list[dict[str, Any]]:
    rows: list[
        dict[str, Any]
    ] = []

    for record in records:
        candidates = record[
            "candidates"
        ]

        first_pass = next(
            (
                int(
                    candidate["sample_id"]
                )
                for candidate
                in candidates
                if candidate.get(
                    "passed",
                    False,
                )
            ),
            None,
        )

        row: dict[str, Any] = {
            "problem_id": (
                record["problem_id"]
            ),
            "difficulty": (
                record.get(
                    "difficulty",
                    "unknown",
                )
            ),
            "num_passed": sum(
                bool(
                    candidate.get(
                        "passed",
                        False,
                    )
                )
                for candidate
                in candidates
            ),
            "first_pass_sample_id": (
                ""
                if first_pass is None
                else first_pass
            ),
            "distinct_codes": len(
                {
                    str(
                        candidate.get(
                            "extracted_code",
                            "",
                        )
                    ).strip()
                    for candidate
                    in candidates
                    if str(
                        candidate.get(
                            "extracted_code",
                            "",
                        )
                    ).strip()
                }
            ),
        }

        for k in ks:
            row[
                f"coverage_at_{k}"
            ] = int(
                oracle_at_k(
                    candidates,
                    k,
                )
            )

            row[
                f"best_test_ratio_at_{k}"
            ] = best_ratio_at_k(
                candidates,
                k,
            )

        rows.append(
            row
        )

    return rows


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def status_rows(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counter: Counter[str] = (
        Counter()
    )

    for record in records:
        for candidate in record[
            "candidates"
        ]:
            counter[
                str(
                    candidate.get(
                        "status",
                        "UNKNOWN",
                    )
                )
            ] += 1

    total = sum(
        counter.values()
    )

    return [
        {
            "status": status,
            "count": count,
            "ratio": (
                count / total
                if total
                else 0.0
            ),
        }
        for status, count
        in counter.most_common()
    ]


# ---------------------------------------------------------------------------
# Phase 3-A comparison
# ---------------------------------------------------------------------------


def planning_comparison_rows(
    code_rows: list[dict[str, Any]],
    planning_records: list[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    planning_n = (
        resolve_num_samples(
            planning_records
        )
    )

    ks = [
        int(
            row["k"]
        )
        for row
        in code_rows
        if int(row["k"])
        <= planning_n
    ]

    planning_rows = coverage_rows(
        planning_records,
        ks,
        planning_n,
    )

    planning_by_k = {
        int(row["k"]): row
        for row in planning_rows
    }

    rows: list[
        dict[str, Any]
    ] = []

    for code_row in code_rows:
        k = int(
            code_row["k"]
        )

        if k not in planning_by_k:
            continue

        plan_coverage = float(
            planning_by_k[k][
                "coverage"
            ]
        )

        code_coverage = float(
            code_row[
                "coverage"
            ]
        )

        rows.append(
            {
                "k": k,
                "planning_coverage": (
                    plan_coverage
                ),
                "code_coverage": (
                    code_coverage
                ),
                "planning_minus_code": (
                    plan_coverage
                    - code_coverage
                ),
                "code_minus_planning": (
                    code_coverage
                    - plan_coverage
                ),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    results_path = Path(
        args.results
    )

    records = load_jsonl(
        results_path
    )

    num_samples = (
        resolve_num_samples(
            records
        )
    )

    ks = prefix_ks(
        num_samples
    )

    integrity = (
        validate_records(
            records,
            num_samples,
        )
    )

    output_dir = Path(
        args.output_dir
        if args.output_dir
        else (
            Path.cwd()
            / "archive"
            / "analysis"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall = coverage_rows(
        records,
        ks,
        num_samples,
    )

    difficulty = difficulty_rows(
        records,
        ks,
        num_samples,
    )

    per_candidate = (
        per_candidate_rows(
            records,
            num_samples,
        )
    )

    diversity = (
        code_diversity_rows(
            records
        )
    )

    problems = problem_rows(
        records,
        ks,
    )

    statuses = status_rows(
        records
    )

    # ------------------------------------------------------------------
    # Phase 3-A comparison
    # ------------------------------------------------------------------

    comparison: list[
        dict[str, Any]
    ] = []

    if args.planning_results:
        planning_path = Path(
            args.planning_results
        )

        planning_records = (
            load_jsonl(
                planning_path
            )
        )

        comparison = (
            planning_comparison_rows(
                overall,
                planning_records,
            )
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    write_csv(
        output_dir
        / "code_coverage_overall.csv",
        overall,
    )

    write_csv(
        output_dir
        / "code_coverage_difficulty.csv",
        difficulty,
    )

    write_csv(
        output_dir
        / "code_per_candidate.csv",
        per_candidate,
    )

    write_csv(
        output_dir
        / "code_diversity.csv",
        diversity,
    )

    write_csv(
        output_dir
        / "code_problem_coverage.csv",
        problems,
    )

    write_csv(
        output_dir
        / "code_status_counts.csv",
        statuses,
    )

    if comparison:
        write_csv(
            output_dir
            / "planning_vs_code_coverage.csv",
            comparison,
        )

    # ------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------

    print("=" * 80)
    print(
        "Phase 3-B Code Coverage Analysis"
    )
    print("=" * 80)

    print(
        f"Results      : "
        f"{results_path}"
    )

    print(
        f"Problems     : "
        f"{len(records)}"
    )

    print(
        f"N (samples)  : "
        f"{num_samples}"
    )

    print(
        f"Candidates   : "
        f"{len(records) * num_samples}"
    )

    print(
        f"Analysis dir : "
        f"{output_dir}"
    )

    # ------------------------------------------------------------------
    # Overall
    # ------------------------------------------------------------------

    print()
    print("Overall Code Coverage")
    print("-" * 80)

    print(
        f"{'N':>4} "
        f"{'Solved':>8} "
        f"{'Coverage':>10} "
        f"{'Unbiased':>10} "
        f"{'BestRatio':>12}"
    )

    for row in overall:
        print(
            f"{row['k']:>4} "
            f"{row['solved']:>8} "
            f"{row['coverage']:>10.4f} "
            f"{row['unbiased_pass_at_k']:>10.4f} "
            f"{row['mean_best_test_pass_ratio']:>12.4f}"
        )

    # ------------------------------------------------------------------
    # Reference
    # ------------------------------------------------------------------

    sampling_at_1 = float(
        overall[0][
            "coverage"
        ]
    )

    mean_single_sample = mean(
        float(
            row["pass_rate"]
        )
        for row
        in per_candidate
    )

    print()
    print("Reference")
    print("-" * 80)

    print(
        f"Code Sampling Coverage@1 : "
        f"{sampling_at_1:.4f}"
    )

    print(
        f"Mean single-sample rate  : "
        f"{mean_single_sample:.4f}"
    )

    print(
        f"Phase 1 greedy Self-Plan : "
        f"{args.phase1_pass_rate:.4f}"
    )

    print(
        f"Phase 1 Teacher-Plan     : "
        f"{args.teacher_pass_rate:.4f}"
    )

    # ------------------------------------------------------------------
    # Difficulty
    # ------------------------------------------------------------------

    print()
    print("Difficulty Breakdown")
    print("-" * 80)

    print(
        f"{'Difficulty':>12} "
        f"{'N':>4} "
        f"{'Problems':>9} "
        f"{'Solved':>8} "
        f"{'Coverage':>10}"
    )

    for row in difficulty:
        print(
            f"{row['difficulty']:>12} "
            f"{row['k']:>4} "
            f"{row['num_problems']:>9} "
            f"{row['solved']:>8} "
            f"{row['coverage']:>10.4f}"
        )

    # ------------------------------------------------------------------
    # Candidate positions
    # ------------------------------------------------------------------

    print()
    print(
        "Per-Candidate Performance"
    )
    print("-" * 80)

    print(
        f"{'ID':>4} "
        f"{'Solved':>8} "
        f"{'PassRate':>10} "
        f"{'TestRatio':>10} "
        f"{'CodeTok':>10}"
    )

    for row in per_candidate:
        print(
            f"{row['sample_id']:>4} "
            f"{row['solved']:>8} "
            f"{row['pass_rate']:>10.4f} "
            f"{row['mean_test_pass_ratio']:>10.4f} "
            f"{row['mean_completion_tokens']:>10.1f}"
        )
    # ------------------------------------------------------------------
    # Diversity
    # ------------------------------------------------------------------

    print()
    print("Code Diversity")
    print("-" * 80)

    mean_distinct_codes = mean(
        float(row["distinct_codes"])
        for row in diversity
    )

    all_identical_count = sum(
        1
        for row in diversity
        if row["distinct_codes"] == 1
    )

    fully_distinct_count = sum(
        1
        for row in diversity
        if row["distinct_codes"] == num_samples
    )

    total_empty_codes = sum(
        int(row["empty_codes"])
        for row in diversity
    )

    mean_pairwise_jaccard = mean(
        float(row["mean_pairwise_jaccard"])
        for row in diversity
    )

    print(
        f"Mean distinct codes / {num_samples} : "
        f"{mean_distinct_codes:.2f}"
    )

    print(
        f"All-identical problems     : "
        f"{all_identical_count}"
    )

    print(
        f"Fully-distinct problems    : "
        f"{fully_distinct_count}"
    )

    print(
        f"Total empty codes          : "
        f"{total_empty_codes}"
    )

    print(
        f"Mean pairwise Jaccard      : "
        f"{mean_pairwise_jaccard:.4f}"
    )
    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    print()
    print(
        "Candidate Status Distribution"
    )
    print("-" * 80)

    for row in statuses:
        print(
            f"{row['status']:<24} "
            f"{row['count']:>6} "
            f"{row['ratio']:>8.4f}"
        )

    # ------------------------------------------------------------------
    # Planning vs Code
    # ------------------------------------------------------------------

    if comparison:
        print()
        print(
            "Planning vs Code Coverage"
        )
        print("-" * 80)

        print(
            f"{'N':>4} "
            f"{'Planning':>10} "
            f"{'Code':>10} "
            f"{'Plan-Code':>12}"
        )

        for row in comparison:
            print(
                f"{row['k']:>4} "
                f"{row['planning_coverage']:>10.4f} "
                f"{row['code_coverage']:>10.4f} "
                f"{row['planning_minus_code']:>12.4f}"
            )

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    print()
    print("Integrity")
    print("-" * 80)

    for key, value in (
        integrity.items()
    ):
        print(
            f"{key:<28}: "
            f"{value}"
        )

    structural_errors = (
        integrity[
            "duplicate_problem_ids"
        ]
        + integrity[
            "invalid_sequences"
        ]
        + integrity[
            "invalid_ratios"
        ]
        + integrity[
            "pass_ratio_mismatches"
        ]
    )

    # ------------------------------------------------------------------
    # Coverage monotonicity
    # ------------------------------------------------------------------

    coverage_values = [
        float(
            row["coverage"]
        )
        for row in overall
    ]

    monotonic = all(
        coverage_values[index]
        >= coverage_values[
            index - 1
        ]
        for index in range(
            1,
            len(coverage_values),
        )
    )

    print()

    if monotonic:
        curve = " <= ".join(
            f"@{row['k']}="
            f"{row['coverage']:.4f}"
            for row in overall
        )

        print(
            "[OK] Code Coverage monotonicity holds: "
            + curve
        )

    else:
        print(
            "[FAIL] Code Coverage monotonicity violated."
        )

        if args.fail_on_violation:
            return 1

    if structural_errors:
        print(
            "[WARN] Structural integrity issues detected."
        )
    else:
        print(
            "[OK] Structural integrity checks passed."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )