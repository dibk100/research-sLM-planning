"""
Analyze Phase 3-A Planning Coverage results.

Main analyses:
- Planning Coverage@1/@2/@4/@8/@16
- difficulty-wise Coverage@k
- marginal coverage gain
- candidate success frequency
- best test-pass-ratio
- generation cost statistics
- multi-model comparison

Each --result argument must use:

    LABEL=/path/to/results.jsonl


PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase3_coverage_analysis/a_planning_coverage/analysis/analyze_planning_coverage.py \
  --result qwen25Coder3b=/mnt/hdd/project_sLM_planning/phase3/livecodebench_v6_stdin/qwen25Coder3b/planning_coverage/results.jsonl \
  --result qwen25_3b=/mnt/hdd/project_sLM_planning/phase3/livecodebench_v6_stdin/qwen253b/planning_coverage/results.jsonl \
  --result phi3mini=/mnt/hdd/project_sLM_planning/phase3/livecodebench_v6_stdin/phi3/planning_coverage/results.jsonl \
  --expected-problems 300
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
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
class CoverageStats:
    k: int
    solved: int
    total: int

    @property
    def rate(self) -> float:
        if self.total <= 0:
            return 0.0

        return self.solved / self.total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Phase 3-A Planning Coverage results."
        )
    )

    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help=(
            "Model result in the form "
            "LABEL=/path/to/results.jsonl. "
            "Repeat this argument for multiple models."
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
            "Candidate prefix sizes used for "
            "Planning Coverage@k."
        ),
    )

    parser.add_argument(
        "--expected-problems",
        type=int,
        default=None,
        help=(
            "Optional expected number of problems "
            "for dataset integrity checking."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def parse_result_spec(
    value: str,
) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(
            "--result must have the form "
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

    problem_ids: set[str] = set()

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
                    "Each result record must be "
                    "a JSON object: "
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

            if problem_id in problem_ids:
                raise ValueError(
                    "Duplicated problem_id="
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
                    "candidates must be a list: "
                    f"problem_id={problem_id}"
                )

            problem_ids.add(
                problem_id
            )

            records.append(
                record
            )

    if not records:
        raise ValueError(
            f"No result records loaded: {path}"
        )

    return records


def load_model_results(
    specs: Sequence[str],
) -> list[ModelResult]:
    results: list[
        ModelResult
    ] = []

    labels: set[str] = set()

    for spec in specs:
        label, path = parse_result_spec(
            spec
        )

        if label in labels:
            raise ValueError(
                f"Duplicated result label: {label}"
            )

        labels.add(
            label
        )

        results.append(
            ModelResult(
                label=label,
                path=path,
                records=load_jsonl(
                    path
                ),
            )
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
            "All k values must be greater than 0."
        )

    return normalized


def validate_model_result(
    result: ModelResult,
    *,
    ks: Sequence[int],
    expected_problems: int | None,
) -> None:
    if (
        expected_problems is not None
        and len(result.records)
        != expected_problems
    ):
        raise ValueError(
            f"{result.label}: expected "
            f"{expected_problems} problems, "
            f"found {len(result.records)}."
        )

    max_k = max(
        ks
    )

    expected_num_samples: int | None = None

    for record in result.records:
        problem_id = str(
            record["problem_id"]
        )

        candidates = record[
            "candidates"
        ]

        num_samples = int(
            record.get(
                "num_samples",
                len(candidates),
            )
        )

        if (
            expected_num_samples
            is None
        ):
            expected_num_samples = (
                num_samples
            )

        elif (
            num_samples
            != expected_num_samples
        ):
            raise ValueError(
                f"{result.label}: inconsistent "
                "num_samples. "
                f"problem_id={problem_id}, "
                f"expected={expected_num_samples}, "
                f"found={num_samples}"
            )

        if len(candidates) != num_samples:
            raise ValueError(
                f"{result.label}: candidate count "
                "does not match num_samples. "
                f"problem_id={problem_id}, "
                f"candidates={len(candidates)}, "
                f"num_samples={num_samples}"
            )

        if len(candidates) < max_k:
            raise ValueError(
                f"{result.label}: not enough "
                "candidates for Coverage@{max_k}. "
                f"problem_id={problem_id}, "
                f"candidates={len(candidates)}"
            )

        sample_ids = [
            int(
                candidate.get(
                    "sample_id",
                    -1,
                )
            )
            for candidate in candidates
        ]

        expected_ids = list(
            range(
                len(candidates)
            )
        )

        if sample_ids != expected_ids:
            raise ValueError(
                f"{result.label}: candidate sequence "
                "is not sample_id=0..N-1. "
                f"problem_id={problem_id}"
            )


def validate_cross_model_alignment(
    results: Sequence[
        ModelResult
    ],
) -> None:
    if len(results) <= 1:
        return

    reference = results[
        0
    ]

    reference_ids = [
        str(
            record[
                "problem_id"
            ]
        )
        for record
        in reference.records
    ]

    for result in results[
        1:
    ]:
        ids = [
            str(
                record[
                    "problem_id"
                ]
            )
            for record
            in result.records
        ]

        if ids == reference_ids:
            continue

        reference_set = set(
            reference_ids
        )

        result_set = set(
            ids
        )

        missing = sorted(
            reference_set
            - result_set
        )

        unexpected = sorted(
            result_set
            - reference_set
        )

        raise ValueError(
            "Cross-model dataset mismatch.\n"
            f"Reference : {reference.label}\n"
            f"Target    : {result.label}\n"
            f"Missing   : {missing[:10]}\n"
            f"Unexpected: {unexpected[:10]}"
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

    return float(
        value
    )


def prefix_solved(
    record: dict[str, Any],
    k: int,
) -> bool:
    candidates = record[
        "candidates"
    ]

    return any(
        candidate_passed(
            candidate
        )
        for candidate
        in candidates[:k]
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


def num_successful_candidates(
    record: dict[str, Any],
) -> int:
    return sum(
        candidate_passed(
            candidate
        )
        for candidate
        in record[
            "candidates"
        ]
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
# Coverage
# ---------------------------------------------------------------------------


def compute_coverage(
    records: Sequence[
        dict[str, Any]
    ],
    *,
    k: int,
) -> CoverageStats:
    solved = sum(
        prefix_solved(
            record,
            k,
        )
        for record in records
    )

    return CoverageStats(
        k=k,
        solved=solved,
        total=len(records),
    )


def compute_all_coverages(
    records: Sequence[
        dict[str, Any]
    ],
    *,
    ks: Sequence[int],
) -> list[CoverageStats]:
    return [
        compute_coverage(
            records,
            k=k,
        )
        for k in ks
    ]


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def print_header(
    title: str,
) -> None:
    print()
    print("=" * 96)
    print(
        title
    )
    print("=" * 96)


def print_subheader(
    title: str,
) -> None:
    print()
    print(
        title
    )
    print("-" * 96)


# ---------------------------------------------------------------------------
# Dataset integrity
# ---------------------------------------------------------------------------


def print_dataset_summary(
    results: Sequence[
        ModelResult
    ],
) -> None:
    print_header(
        "Phase 3-A Planning Coverage Analysis"
    )

    print(
        f"{'Model':<28}"
        f"{'Problems':>12}"
        f"{'N':>8}"
        f"{'Easy':>10}"
        f"{'Medium':>10}"
        f"{'Hard':>10}"
        f"{'Other':>10}"
    )

    print(
        "-" * 88
    )

    for result in results:
        difficulties = Counter(
            difficulty_of(
                record
            )
            for record
            in result.records
        )

        n = int(
            result.records[
                0
            ].get(
                "num_samples",
                len(
                    result.records[
                        0
                    ][
                        "candidates"
                    ]
                ),
            )
        )

        known = (
            difficulties.get(
                "easy",
                0,
            )
            + difficulties.get(
                "medium",
                0,
            )
            + difficulties.get(
                "hard",
                0,
            )
        )

        other = (
            len(
                result.records
            )
            - known
        )

        print(
            f"{result.label:<28}"
            f"{len(result.records):>12}"
            f"{n:>8}"
            f"{difficulties.get('easy', 0):>10}"
            f"{difficulties.get('medium', 0):>10}"
            f"{difficulties.get('hard', 0):>10}"
            f"{other:>10}"
        )


# ---------------------------------------------------------------------------
# Overall coverage
# ---------------------------------------------------------------------------


def print_overall_coverage(
    results: Sequence[
        ModelResult
    ],
    *,
    ks: Sequence[int],
) -> None:
    print_header(
        "Overall Planning Coverage"
    )

    header = (
        f"{'Model':<28}"
        + "".join(
            f"@{k:<4}".rjust(
                16
            )
            for k in ks
        )
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    for result in results:
        stats = compute_all_coverages(
            result.records,
            ks=ks,
        )

        row = (
            f"{result.label:<28}"
        )

        for stat in stats:
            value = (
                f"{stat.solved}/{stat.total} "
                f"({stat.rate:.2%})"
            )

            row += (
                f"{value:>16}"
            )

        print(
            row
        )


# ---------------------------------------------------------------------------
# Marginal gain
# ---------------------------------------------------------------------------


def print_marginal_gain(
    results: Sequence[
        ModelResult
    ],
    *,
    ks: Sequence[int],
) -> None:
    print_header(
        "Marginal Coverage Gain"
    )

    for result in results:
        print_subheader(
            result.label
        )

        stats = compute_all_coverages(
            result.records,
            ks=ks,
        )

        previous_solved = 0
        previous_rate = 0.0

        for stat in stats:
            gained = (
                stat.solved
                - previous_solved
            )

            rate_gain = (
                stat.rate
                - previous_rate
            )

            print(
                f"@{stat.k:<3} "
                f"solved={stat.solved:3d}/"
                f"{stat.total:<3d} "
                f"coverage={stat.rate:.2%} "
                f"new={gained:+3d} "
                f"delta={rate_gain:+.2%}"
            )

            previous_solved = (
                stat.solved
            )
            previous_rate = (
                stat.rate
            )


# ---------------------------------------------------------------------------
# Difficulty-wise coverage
# ---------------------------------------------------------------------------


def group_by_difficulty(
    records: Sequence[
        dict[str, Any]
    ],
) -> dict[
    str,
    list[dict[str, Any]],
]:
    groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(
        list
    )

    for record in records:
        groups[
            difficulty_of(
                record
            )
        ].append(
            record
        )

    return groups


def print_difficulty_coverage(
    results: Sequence[
        ModelResult
    ],
    *,
    ks: Sequence[int],
) -> None:
    print_header(
        "Planning Coverage by Difficulty"
    )

    preferred_order = [
        "easy",
        "medium",
        "hard",
    ]

    for result in results:
        print_subheader(
            result.label
        )

        groups = group_by_difficulty(
            result.records
        )

        ordered = [
            difficulty
            for difficulty
            in preferred_order
            if difficulty
            in groups
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
            records = groups[
                difficulty
            ]

            print()
            print(
                f"[{difficulty.upper()}] "
                f"n={len(records)}"
            )

            for k in ks:
                stat = compute_coverage(
                    records,
                    k=k,
                )

                print(
                    f"  @{k:<3} "
                    f"{stat.solved:3d}/"
                    f"{stat.total:<3d} "
                    f"({stat.rate:.2%})"
                )


# ---------------------------------------------------------------------------
# Successful-candidate frequency
# ---------------------------------------------------------------------------


def print_candidate_success_frequency(
    results: Sequence[
        ModelResult
    ],
) -> None:
    print_header(
        "Successful Candidate Frequency"
    )

    for result in results:
        counts = [
            num_successful_candidates(
                record
            )
            for record
            in result.records
        ]

        distribution = Counter(
            counts
        )

        total_candidate_passes = sum(
            counts
        )

        total_candidates = sum(
            len(
                record[
                    "candidates"
                ]
            )
            for record
            in result.records
        )

        mean_successes = (
            statistics.mean(
                counts
            )
        )

        problems_with_success = sum(
            count > 0
            for count in counts
        )

        print_subheader(
            result.label
        )

        print(
            "Problems with >=1 successful candidate "
            f": {problems_with_success}/"
            f"{len(counts)} "
            f"({problems_with_success / len(counts):.2%})"
        )

        print(
            "Total successful candidates            "
            f": {total_candidate_passes}/"
            f"{total_candidates} "
            f"({total_candidate_passes / total_candidates:.2%})"
        )

        print(
            "Mean successful candidates / problem   "
            f": {mean_successes:.4f}"
        )

        print()
        print(
            "Successful candidates per problem:"
        )

        for count in sorted(
            distribution
        ):
            frequency = (
                distribution[
                    count
                ]
            )

            print(
                f"  {count:2d} successful "
                f": {frequency:3d} problems "
                f"({frequency / len(counts):.2%})"
            )


# ---------------------------------------------------------------------------
# Best test-pass-ratio
# ---------------------------------------------------------------------------


def print_best_ratio_analysis(
    results: Sequence[
        ModelResult
    ],
    *,
    ks: Sequence[int],
) -> None:
    print_header(
        "Best Test-Pass Ratio by Candidate Budget"
    )

    print(
        f"{'Model':<28}"
        + "".join(
            f"@{k:<4}".rjust(
                14
            )
            for k in ks
        )
    )

    print(
        "-" * (
            28
            + 14
            * len(
                ks
            )
        )
    )

    for result in results:
        row = (
            f"{result.label:<28}"
        )

        for k in ks:
            ratios = [
                prefix_best_ratio(
                    record,
                    k,
                )
                for record
                in result.records
            ]

            value = (
                statistics.mean(
                    ratios
                )
            )

            row += (
                f"{value:>14.4f}"
            )

        print(
            row
        )


# ---------------------------------------------------------------------------
# Plan diversity
# ---------------------------------------------------------------------------


def normalized_plan(
    plan: Any,
) -> str:
    return str(
        plan
        if plan is not None
        else ""
    ).strip()


def print_plan_diversity(
    results: Sequence[
        ModelResult
    ],
) -> None:
    print_header(
        "Sampled Plan Diversity"
    )

    print(
        f"{'Model':<28}"
        f"{'Mean distinct':>16}"
        f"{'Median':>12}"
        f"{'All unique':>14}"
        f"{'Any duplicate':>16}"
        f"{'Empty plans':>14}"
    )

    print(
        "-" * 100
    )

    for result in results:
        distinct_counts: list[
            int
        ] = []

        all_unique_count = 0
        duplicate_count = 0
        empty_count = 0

        for record in result.records:
            plans = [
                normalized_plan(
                    candidate.get(
                        "plan",
                        "",
                    )
                )
                for candidate
                in record[
                    "candidates"
                ]
            ]

            non_empty = [
                plan
                for plan in plans
                if plan
            ]

            distinct = len(
                set(
                    non_empty
                )
            )

            distinct_counts.append(
                distinct
            )

            empty_count += sum(
                not plan
                for plan in plans
            )

            if (
                len(non_empty)
                == len(plans)
                and distinct
                == len(plans)
            ):
                all_unique_count += 1

            if distinct < len(
                non_empty
            ):
                duplicate_count += 1

        print(
            f"{result.label:<28}"
            f"{statistics.mean(distinct_counts):>16.2f}"
            f"{statistics.median(distinct_counts):>12.2f}"
            f"{all_unique_count:>14}"
            f"{duplicate_count:>16}"
            f"{empty_count:>14}"
        )


# ---------------------------------------------------------------------------
# Candidate position
# ---------------------------------------------------------------------------


def print_first_success_position(
    results: Sequence[
        ModelResult
    ],
) -> None:
    print_header(
        "First Successful Candidate Position"
    )

    for result in results:
        positions: Counter[
            int
        ] = Counter()

        never = 0

        for record in result.records:
            position: int | None = None

            for index, candidate in enumerate(
                record[
                    "candidates"
                ],
                start=1,
            ):
                if candidate_passed(
                    candidate
                ):
                    position = (
                        index
                    )
                    break

            if position is None:
                never += 1
            else:
                positions[
                    position
                ] += 1

        print_subheader(
            result.label
        )

        for position in sorted(
            positions
        ):
            count = positions[
                position
            ]

            print(
                f"Candidate {position:2d}: "
                f"{count:3d}"
            )

        print(
            f"Never solved : {never}"
        )


# ---------------------------------------------------------------------------
# Generation cost
# ---------------------------------------------------------------------------


def safe_float(
    value: Any,
) -> float:
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


def print_generation_cost(
    results: Sequence[
        ModelResult
    ],
) -> None:
    print_header(
        "Generation Cost"
    )

    print(
        f"{'Model':<28}"
        f"{'Plan Prompt':>14}"
        f"{'Plan Comp':>14}"
        f"{'Code Prompt':>14}"
        f"{'Code Comp':>14}"
        f"{'Total Comp':>14}"
        f"{'Gen Time':>14}"
    )

    print(
        "-" * 112
    )

    for result in results:
        plan_prompt_tokens: list[
            float
        ] = []
        plan_completion_tokens: list[
            float
        ] = []
        code_prompt_tokens: list[
            float
        ] = []
        code_completion_tokens: list[
            float
        ] = []
        total_completion_tokens: list[
            float
        ] = []
        generation_times: list[
            float
        ] = []

        for record in result.records:
            for candidate in record[
                "candidates"
            ]:
                plan_prompt = safe_float(
                    candidate.get(
                        "plan_prompt_tokens",
                        0,
                    )
                )

                plan_completion = safe_float(
                    candidate.get(
                        "plan_completion_tokens",
                        0,
                    )
                )

                code_prompt = safe_float(
                    candidate.get(
                        "code_prompt_tokens",
                        0,
                    )
                )

                code_completion = safe_float(
                    candidate.get(
                        "code_completion_tokens",
                        0,
                    )
                )

                generation_time = (
                    safe_float(
                        candidate.get(
                            "plan_generation_time",
                            0.0,
                        )
                    )
                    + safe_float(
                        candidate.get(
                            "code_generation_time",
                            0.0,
                        )
                    )
                )

                plan_prompt_tokens.append(
                    plan_prompt
                )

                plan_completion_tokens.append(
                    plan_completion
                )

                code_prompt_tokens.append(
                    code_prompt
                )

                code_completion_tokens.append(
                    code_completion
                )

                total_completion_tokens.append(
                    plan_completion
                    + code_completion
                )

                generation_times.append(
                    generation_time
                )

        print(
            f"{result.label:<28}"
            f"{statistics.mean(plan_prompt_tokens):>14.2f}"
            f"{statistics.mean(plan_completion_tokens):>14.2f}"
            f"{statistics.mean(code_prompt_tokens):>14.2f}"
            f"{statistics.mean(code_completion_tokens):>14.2f}"
            f"{statistics.mean(total_completion_tokens):>14.2f}"
            f"{statistics.mean(generation_times):>14.3f}"
        )


# ---------------------------------------------------------------------------
# Status distribution
# ---------------------------------------------------------------------------


def print_status_distribution(
    results: Sequence[
        ModelResult
    ],
) -> None:
    print_header(
        "Candidate Evaluation Status Distribution"
    )

    for result in results:
        statuses = Counter(
            str(
                candidate.get(
                    "status",
                    "UNKNOWN",
                )
            )
            for record
            in result.records
            for candidate
            in record[
                "candidates"
            ]
        )

        total = sum(
            statuses.values()
        )

        print_subheader(
            result.label
        )

        for status, count in (
            statuses.most_common()
        ):
            print(
                f"{status:<28} "
                f"{count:5d}/"
                f"{total:<5d} "
                f"({count / total:.2%})"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    ks = validate_ks(
        args.ks
    )

    results = load_model_results(
        args.result
    )

    for result in results:
        validate_model_result(
            result,
            ks=ks,
            expected_problems=(
                args.expected_problems
            ),
        )

    validate_cross_model_alignment(
        results
    )

    print_dataset_summary(
        results
    )

    print_overall_coverage(
        results,
        ks=ks,
    )

    print_marginal_gain(
        results,
        ks=ks,
    )

    print_difficulty_coverage(
        results,
        ks=ks,
    )

    print_candidate_success_frequency(
        results
    )

    print_best_ratio_analysis(
        results,
        ks=ks,
    )

    print_plan_diversity(
        results
    )

    print_first_success_position(
        results
    )

    print_status_distribution(
        results
    )

    print_generation_cost(
        results
    )


if __name__ == "__main__":
    main()