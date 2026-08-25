"""
Compare Phase 3-A Planning Coverage
vs Phase 3-B Code Coverage.

For each model and candidate budget k:

- Planning Coverage@k
- Code Coverage@k
- absolute difference
- paired recovery pattern
    * Both PASS
    * Planning-only PASS
    * Code-only PASS
    * Both FAIL
- McNemar exact test
- difficulty-wise paired comparison
- best test-pass-ratio comparison

Each result argument must use:

    LABEL=/path/to/results.jsonl

PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase3_coverage_analysis/analysis/compare_planning_vs_code.py \
  --planning-result qwen25Coder3b=/mnt/hdd/project_sLM_planning/phase3/livecodebench_v6_stdin/qwen25Coder3b/planning_coverage/results.jsonl \
  --planning-result qwen25_3b=/mnt/hdd/project_sLM_planning/phase3/livecodebench_v6_stdin/qwen253b/planning_coverage/results.jsonl \
  --planning-result phi3mini=/mnt/hdd/project_sLM_planning/phase3/livecodebench_v6_stdin/phi3/planning_coverage/results.jsonl \
  --code-result qwen25Coder3b=/mnt/hdd/project_sLM_planning/phase3/livecodebench_v6_stdin/qwen25Coder3b/code_coverage/results.jsonl \
  --code-result qwen25_3b=/mnt/hdd/project_sLM_planning/phase3/livecodebench_v6_stdin/qwen253b/code_coverage/results.jsonl \
  --code-result phi3mini=/mnt/hdd/project_sLM_planning/phase3/livecodebench_v6_stdin/phi3/code_coverage/results.jsonl \
  --expected-problems 300
"""

from __future__ import annotations

import argparse
import json
import math
import statistics

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelResult:
    label: str
    path: Path
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class PairCounts:
    both_pass: int
    planning_only: int
    code_only: int
    both_fail: int

    @property
    def total(self) -> int:
        return (
            self.both_pass
            + self.planning_only
            + self.code_only
            + self.both_fail
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Phase 3-A Planning Coverage "
            "with Phase 3-B Code Coverage."
        )
    )

    parser.add_argument(
        "--planning-result",
        action="append",
        required=True,
        help=(
            "Planning result in the form "
            "LABEL=/path/to/results.jsonl. "
            "Repeat for multiple models."
        ),
    )

    parser.add_argument(
        "--code-result",
        action="append",
        required=True,
        help=(
            "Code result in the form "
            "LABEL=/path/to/results.jsonl. "
            "Repeat for multiple models."
        ),
    )

    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[
            1,
            2,
            4,
            8,
            16,
        ],
        help=(
            "Candidate prefix sizes used "
            "for coverage comparison."
        ),
    )

    parser.add_argument(
        "--expected-problems",
        type=int,
        default=None,
        help=(
            "Optional expected problem count "
            "for integrity checking."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def parse_result_spec(
    value: str,
) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(
            "Result must use "
            "LABEL=/path/to/results.jsonl"
        )

    label, path_text = value.split(
        "=",
        1,
    )

    label = label.strip()
    path_text = path_text.strip()

    if not label:
        raise ValueError(
            "Result label must not be empty."
        )

    if not path_text:
        raise ValueError(
            "Result path must not be empty."
        )

    return (
        label,
        Path(path_text),
    )


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Result file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Result path is not a file: {path}"
        )

    records: list[
        dict[str, Any]
    ] = []

    seen_ids: set[str] = set()

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
                    "Invalid JSONL record: "
                    f"{path}:{line_number}"
                ) from error

            if not isinstance(
                record,
                dict,
            ):
                raise TypeError(
                    "Each JSONL record must "
                    "be an object: "
                    f"{path}:{line_number}"
                )

            problem_id = str(
                record.get(
                    "problem_id",
                    "",
                )
            ).strip()

            if not problem_id:
                raise ValueError(
                    "Missing problem_id: "
                    f"{path}:{line_number}"
                )

            if problem_id in seen_ids:
                raise ValueError(
                    "Duplicate problem_id="
                    f"{problem_id}: {path}"
                )

            candidates = record.get(
                "candidates"
            )

            if not isinstance(
                candidates,
                list,
            ):
                raise TypeError(
                    "candidates must be list: "
                    f"problem_id={problem_id}"
                )

            seen_ids.add(
                problem_id
            )

            records.append(
                record
            )

    if not records:
        raise ValueError(
            f"No records loaded: {path}"
        )

    return records


def load_results(
    specs: Sequence[str],
) -> dict[str, ModelResult]:
    results: dict[
        str,
        ModelResult,
    ] = {}

    for spec in specs:
        label, path = parse_result_spec(
            spec
        )

        if label in results:
            raise ValueError(
                f"Duplicate label: {label}"
            )

        results[
            label
        ] = ModelResult(
            label=label,
            path=path,
            records=load_jsonl(
                path
            ),
        )

    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_ks(
    ks: Sequence[int],
) -> list[int]:
    if not ks:
        raise ValueError(
            "ks must not be empty."
        )

    normalized = sorted(
        set(
            int(k)
            for k in ks
        )
    )

    if normalized[0] <= 0:
        raise ValueError(
            "All k values must be > 0."
        )

    return normalized


def validate_model_pair(
    planning: ModelResult,
    code: ModelResult,
    *,
    ks: Sequence[int],
    expected_problems: int | None,
) -> None:
    if (
        expected_problems is not None
        and len(planning.records)
        != expected_problems
    ):
        raise ValueError(
            f"{planning.label}: planning "
            f"expected {expected_problems}, "
            f"found {len(planning.records)}"
        )

    if (
        expected_problems is not None
        and len(code.records)
        != expected_problems
    ):
        raise ValueError(
            f"{code.label}: code "
            f"expected {expected_problems}, "
            f"found {len(code.records)}"
        )

    planning_ids = [
        str(
            record[
                "problem_id"
            ]
        )
        for record
        in planning.records
    ]

    code_ids = [
        str(
            record[
                "problem_id"
            ]
        )
        for record
        in code.records
    ]

    if planning_ids != code_ids:
        planning_set = set(
            planning_ids
        )

        code_set = set(
            code_ids
        )

        raise ValueError(
            "Planning / Code dataset mismatch.\n"
            f"Model      : {planning.label}\n"
            f"Planning n : {len(planning_ids)}\n"
            f"Code n     : {len(code_ids)}\n"
            f"Missing in code: "
            f"{sorted(planning_set - code_set)[:10]}\n"
            f"Missing in planning: "
            f"{sorted(code_set - planning_set)[:10]}"
        )

    max_k = max(
        ks
    )

    for planning_record, code_record in zip(
        planning.records,
        code.records,
    ):
        problem_id = str(
            planning_record[
                "problem_id"
            ]
        )

        planning_candidates = (
            planning_record[
                "candidates"
            ]
        )

        code_candidates = (
            code_record[
                "candidates"
            ]
        )

        if len(
            planning_candidates
        ) < max_k:
            raise ValueError(
                f"{planning.label}: planning "
                f"has fewer than {max_k} "
                f"candidates: {problem_id}"
            )

        if len(
            code_candidates
        ) < max_k:
            raise ValueError(
                f"{code.label}: code "
                f"has fewer than {max_k} "
                f"candidates: {problem_id}"
            )

        planning_sample_ids = [
            int(
                candidate.get(
                    "sample_id",
                    -1,
                )
            )
            for candidate
            in planning_candidates
        ]

        code_sample_ids = [
            int(
                candidate.get(
                    "sample_id",
                    -1,
                )
            )
            for candidate
            in code_candidates
        ]

        if planning_sample_ids != list(
            range(
                len(
                    planning_candidates
                )
            )
        ):
            raise ValueError(
                "Invalid planning candidate "
                f"sequence: {problem_id}"
            )

        if code_sample_ids != list(
            range(
                len(
                    code_candidates
                )
            )
        ):
            raise ValueError(
                "Invalid code candidate "
                f"sequence: {problem_id}"
            )


# ---------------------------------------------------------------------------
# Candidate helpers
# ---------------------------------------------------------------------------


def candidate_passed(
    candidate: dict[str, Any],
) -> bool:
    return (
        candidate.get(
            "passed"
        )
        is True
    )


def candidate_ratio(
    candidate: dict[str, Any],
) -> float:
    value = candidate.get(
        "test_pass_ratio",
        0.0,
    )

    if value is None:
        return 0.0

    try:
        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    if not math.isfinite(
        result
    ):
        return 0.0

    return result


def prefix_solved(
    record: dict[str, Any],
    k: int,
) -> bool:
    return any(
        candidate_passed(
            candidate
        )
        for candidate
        in record[
            "candidates"
        ][
            :k
        ]
    )


def prefix_best_ratio(
    record: dict[str, Any],
    k: int,
) -> float:
    candidates = record[
        "candidates"
    ][
        :k
    ]

    if not candidates:
        return 0.0

    return max(
        candidate_ratio(
            candidate
        )
        for candidate
        in candidates
    )


def difficulty_of(
    record: dict[str, Any],
) -> str:
    difficulty = record.get(
        "difficulty"
    )

    if difficulty is None:
        return "unknown"

    text = str(
        difficulty
    ).strip()

    if not text:
        return "unknown"

    return text.lower()


# ---------------------------------------------------------------------------
# McNemar exact test
# ---------------------------------------------------------------------------


def binomial_coefficient(
    n: int,
    k: int,
) -> int:
    return math.comb(
        n,
        k,
    )


def binomial_probability_half(
    n: int,
    k: int,
) -> float:
    return (
        binomial_coefficient(
            n,
            k,
        )
        * (0.5 ** n)
    )


def mcnemar_exact_pvalue(
    planning_only: int,
    code_only: int,
) -> float:
    """
    Two-sided exact McNemar test.

    Under H0:
        planning_only and code_only are equally likely.

    Conditional on discordant pairs:
        X ~ Binomial(n, 0.5)
    """

    b = int(
        planning_only
    )

    c = int(
        code_only
    )

    n = (
        b
        + c
    )

    if n == 0:
        return 1.0

    smaller = min(
        b,
        c,
    )

    lower_tail = sum(
        binomial_probability_half(
            n,
            k,
        )
        for k in range(
            smaller + 1
        )
    )

    return min(
        1.0,
        2.0
        * lower_tail,
    )


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------


def paired_counts(
    planning_records: Sequence[
        dict[str, Any]
    ],
    code_records: Sequence[
        dict[str, Any]
    ],
    *,
    k: int,
) -> PairCounts:
    both_pass = 0
    planning_only = 0
    code_only = 0
    both_fail = 0

    for planning_record, code_record in zip(
        planning_records,
        code_records,
    ):
        planning_pass = prefix_solved(
            planning_record,
            k,
        )

        code_pass = prefix_solved(
            code_record,
            k,
        )

        if (
            planning_pass
            and code_pass
        ):
            both_pass += 1

        elif (
            planning_pass
            and not code_pass
        ):
            planning_only += 1

        elif (
            not planning_pass
            and code_pass
        ):
            code_only += 1

        else:
            both_fail += 1

    return PairCounts(
        both_pass=both_pass,
        planning_only=planning_only,
        code_only=code_only,
        both_fail=both_fail,
    )


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def print_header(
    title: str,
) -> None:
    print()
    print("=" * 104)
    print(
        title
    )
    print("=" * 104)


def print_subheader(
    title: str,
) -> None:
    print()
    print(
        title
    )
    print("-" * 104)


# ---------------------------------------------------------------------------
# Overall coverage comparison
# ---------------------------------------------------------------------------


def print_overall_comparison(
    planning_results: dict[
        str,
        ModelResult,
    ],
    code_results: dict[
        str,
        ModelResult,
    ],
    *,
    ks: Sequence[int],
) -> None:
    print_header(
        "Planning Coverage vs Code Coverage"
    )

    for label in planning_results:
        planning = planning_results[
            label
        ]

        code = code_results[
            label
        ]

        print_subheader(
            label
        )

        print(
            f"{'k':>5}"
            f"{'Planning':>18}"
            f"{'Code':>18}"
            f"{'Delta':>14}"
        )

        print(
            "-" * 56
        )

        total = len(
            planning.records
        )

        for k in ks:
            planning_solved = sum(
                prefix_solved(
                    record,
                    k,
                )
                for record
                in planning.records
            )

            code_solved = sum(
                prefix_solved(
                    record,
                    k,
                )
                for record
                in code.records
            )

            planning_rate = (
                planning_solved
                / total
            )

            code_rate = (
                code_solved
                / total
            )

            delta = (
                planning_rate
                - code_rate
            )

            print(
                f"@{k:<4}"
                f"{planning_solved:4d}/"
                f"{total:<4d} "
                f"({planning_rate:6.2%})"
                f"{code_solved:4d}/"
                f"{total:<4d} "
                f"({code_rate:6.2%})"
                f"{delta:>+14.2%}"
            )


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------


def print_paired_comparison(
    planning_results: dict[
        str,
        ModelResult,
    ],
    code_results: dict[
        str,
        ModelResult,
    ],
    *,
    ks: Sequence[int],
) -> None:
    print_header(
        "Paired Planning-vs-Code Comparison"
    )

    for label in planning_results:
        planning = planning_results[
            label
        ]

        code = code_results[
            label
        ]

        print_subheader(
            label
        )

        print(
            f"{'k':>5}"
            f"{'Both':>10}"
            f"{'Plan only':>12}"
            f"{'Code only':>12}"
            f"{'Neither':>10}"
            f"{'McNemar p':>18}"
        )

        print(
            "-" * 72
        )

        for k in ks:
            counts = paired_counts(
                planning.records,
                code.records,
                k=k,
            )

            p_value = (
                mcnemar_exact_pvalue(
                    counts.planning_only,
                    counts.code_only,
                )
            )

            print(
                f"@{k:<4}"
                f"{counts.both_pass:>10}"
                f"{counts.planning_only:>12}"
                f"{counts.code_only:>12}"
                f"{counts.both_fail:>10}"
                f"{p_value:>18.10g}"
            )


# ---------------------------------------------------------------------------
# Difficulty-wise comparison
# ---------------------------------------------------------------------------


def group_indices_by_difficulty(
    records: Sequence[
        dict[str, Any]
    ],
) -> dict[
    str,
    list[int],
]:
    groups: dict[
        str,
        list[int],
    ] = defaultdict(
        list
    )

    for index, record in enumerate(
        records
    ):
        groups[
            difficulty_of(
                record
            )
        ].append(
            index
        )

    return groups


def print_difficulty_comparison(
    planning_results: dict[
        str,
        ModelResult,
    ],
    code_results: dict[
        str,
        ModelResult,
    ],
    *,
    ks: Sequence[int],
) -> None:
    print_header(
        "Planning vs Code Coverage by Difficulty"
    )

    preferred_order = [
        "easy",
        "medium",
        "hard",
    ]

    for label in planning_results:
        planning = planning_results[
            label
        ]

        code = code_results[
            label
        ]

        print_subheader(
            label
        )

        groups = (
            group_indices_by_difficulty(
                planning.records
            )
        )

        ordered = [
            difficulty
            for difficulty
            in preferred_order
            if difficulty in groups
        ]

        ordered.extend(
            sorted(
                difficulty
                for difficulty
                in groups
                if difficulty
                not in preferred_order
            )
        )

        for difficulty in ordered:
            indices = groups[
                difficulty
            ]

            print()
            print(
                f"[{difficulty.upper()}] "
                f"n={len(indices)}"
            )

            print(
                f"{'k':>5}"
                f"{'Planning':>14}"
                f"{'Code':>14}"
                f"{'Delta':>12}"
                f"{'P-only':>10}"
                f"{'C-only':>10}"
                f"{'p':>14}"
            )

            for k in ks:
                planning_subset = [
                    planning.records[
                        index
                    ]
                    for index
                    in indices
                ]

                code_subset = [
                    code.records[
                        index
                    ]
                    for index
                    in indices
                ]

                planning_solved = sum(
                    prefix_solved(
                        record,
                        k,
                    )
                    for record
                    in planning_subset
                )

                code_solved = sum(
                    prefix_solved(
                        record,
                        k,
                    )
                    for record
                    in code_subset
                )

                total = len(
                    indices
                )

                planning_rate = (
                    planning_solved
                    / total
                )

                code_rate = (
                    code_solved
                    / total
                )

                counts = paired_counts(
                    planning_subset,
                    code_subset,
                    k=k,
                )

                p_value = (
                    mcnemar_exact_pvalue(
                        counts.planning_only,
                        counts.code_only,
                    )
                )

                print(
                    f"@{k:<4}"
                    f"{planning_rate:>14.2%}"
                    f"{code_rate:>14.2%}"
                    f"{planning_rate - code_rate:>+12.2%}"
                    f"{counts.planning_only:>10}"
                    f"{counts.code_only:>10}"
                    f"{p_value:>14.6g}"
                )


# ---------------------------------------------------------------------------
# Best test-pass ratio comparison
# ---------------------------------------------------------------------------


def print_best_ratio_comparison(
    planning_results: dict[
        str,
        ModelResult,
    ],
    code_results: dict[
        str,
        ModelResult,
    ],
    *,
    ks: Sequence[int],
) -> None:
    print_header(
        "Best Test-Pass Ratio: Planning vs Code"
    )

    for label in planning_results:
        planning = planning_results[
            label
        ]

        code = code_results[
            label
        ]

        print_subheader(
            label
        )

        print(
            f"{'k':>5}"
            f"{'Planning':>14}"
            f"{'Code':>14}"
            f"{'Delta':>14}"
        )

        print(
            "-" * 48
        )

        for k in ks:
            planning_ratios = [
                prefix_best_ratio(
                    record,
                    k,
                )
                for record
                in planning.records
            ]

            code_ratios = [
                prefix_best_ratio(
                    record,
                    k,
                )
                for record
                in code.records
            ]

            planning_mean = (
                statistics.mean(
                    planning_ratios
                )
            )

            code_mean = (
                statistics.mean(
                    code_ratios
                )
            )

            print(
                f"@{k:<4}"
                f"{planning_mean:>14.4f}"
                f"{code_mean:>14.4f}"
                f"{planning_mean - code_mean:>+14.4f}"
            )


# ---------------------------------------------------------------------------
# @max-k problem pattern
# ---------------------------------------------------------------------------


def print_final_budget_patterns(
    planning_results: dict[
        str,
        ModelResult,
    ],
    code_results: dict[
        str,
        ModelResult,
    ],
    *,
    k: int,
) -> None:
    print_header(
        f"Problem-Level Patterns at @{k}"
    )

    for label in planning_results:
        planning = planning_results[
            label
        ]

        code = code_results[
            label
        ]

        planning_only_ids: list[
            str
        ] = []

        code_only_ids: list[
            str
        ] = []

        both_ids: list[
            str
        ] = []

        neither_ids: list[
            str
        ] = []

        for planning_record, code_record in zip(
            planning.records,
            code.records,
        ):
            problem_id = str(
                planning_record[
                    "problem_id"
                ]
            )

            planning_pass = prefix_solved(
                planning_record,
                k,
            )

            code_pass = prefix_solved(
                code_record,
                k,
            )

            if (
                planning_pass
                and code_pass
            ):
                both_ids.append(
                    problem_id
                )

            elif planning_pass:
                planning_only_ids.append(
                    problem_id
                )

            elif code_pass:
                code_only_ids.append(
                    problem_id
                )

            else:
                neither_ids.append(
                    problem_id
                )

        print_subheader(
            label
        )

        print(
            f"Both PASS     : "
            f"{len(both_ids)}"
        )

        print(
            f"Planning only : "
            f"{len(planning_only_ids)}"
        )

        print(
            f"Code only     : "
            f"{len(code_only_ids)}"
        )

        print(
            f"Neither       : "
            f"{len(neither_ids)}"
        )

        print()

        print(
            "Planning-only IDs:"
        )

        print(
            ", ".join(
                planning_only_ids
            )
            if planning_only_ids
            else "None"
        )

        print()

        print(
            "Code-only IDs:"
        )

        print(
            ", ".join(
                code_only_ids
            )
            if code_only_ids
            else "None"
        )


# ---------------------------------------------------------------------------
# Cross-model summary
# ---------------------------------------------------------------------------


def print_cross_model_summary(
    planning_results: dict[
        str,
        ModelResult,
    ],
    code_results: dict[
        str,
        ModelResult,
    ],
    *,
    k: int,
) -> None:
    print_header(
        f"Cross-Model Summary at @{k}"
    )

    print(
        f"{'Model':<28}"
        f"{'Planning':>14}"
        f"{'Code':>14}"
        f"{'Delta':>12}"
        f"{'P-only':>10}"
        f"{'C-only':>10}"
        f"{'McNemar p':>16}"
    )

    print(
        "-" * 104
    )

    for label in planning_results:
        planning = planning_results[
            label
        ]

        code = code_results[
            label
        ]

        total = len(
            planning.records
        )

        planning_solved = sum(
            prefix_solved(
                record,
                k,
            )
            for record
            in planning.records
        )

        code_solved = sum(
            prefix_solved(
                record,
                k,
            )
            for record
            in code.records
        )

        counts = paired_counts(
            planning.records,
            code.records,
            k=k,
        )

        p_value = (
            mcnemar_exact_pvalue(
                counts.planning_only,
                counts.code_only,
            )
        )

        planning_rate = (
            planning_solved
            / total
        )

        code_rate = (
            code_solved
            / total
        )

        print(
            f"{label:<28}"
            f"{planning_rate:>14.2%}"
            f"{code_rate:>14.2%}"
            f"{planning_rate - code_rate:>+12.2%}"
            f"{counts.planning_only:>10}"
            f"{counts.code_only:>10}"
            f"{p_value:>16.8g}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    ks = validate_ks(
        args.ks
    )

    planning_results = load_results(
        args.planning_result
    )

    code_results = load_results(
        args.code_result
    )

    planning_labels = set(
        planning_results
    )

    code_labels = set(
        code_results
    )

    if planning_labels != code_labels:
        raise ValueError(
            "Planning / Code model labels differ.\n"
            f"Planning labels: "
            f"{sorted(planning_labels)}\n"
            f"Code labels: "
            f"{sorted(code_labels)}"
        )

    # Preserve planning-result CLI order.
    for label in planning_results:
        validate_model_pair(
            planning_results[
                label
            ],
            code_results[
                label
            ],
            ks=ks,
            expected_problems=(
                args.expected_problems
            ),
        )

    print_overall_comparison(
        planning_results,
        code_results,
        ks=ks,
    )

    print_paired_comparison(
        planning_results,
        code_results,
        ks=ks,
    )

    print_difficulty_comparison(
        planning_results,
        code_results,
        ks=ks,
    )

    print_best_ratio_comparison(
        planning_results,
        code_results,
        ks=ks,
    )

    max_k = max(
        ks
    )

    print_final_budget_patterns(
        planning_results,
        code_results,
        k=max_k,
    )

    print_cross_model_summary(
        planning_results,
        code_results,
        k=max_k,
    )


if __name__ == "__main__":
    main()