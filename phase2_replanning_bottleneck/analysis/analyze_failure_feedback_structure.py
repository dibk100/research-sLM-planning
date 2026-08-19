"""
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase2_replanning_bottleneck/analysis/analyze_failure_feedback_structure.py \
  --results /mnt/hdd/project_sLM_planning/phase1/livecodebench_v6_stdin/qwen25Coder3b/direct/results.jsonl \
  --tokenizer Qwen/Qwen2.5-Coder-3B-Instruct

"""

# phase2_replanning_bottleneck/analysis/analyze_failure_feedback_structure.py

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from transformers import AutoTokenizer

from src.datasets.phase1_failure_loader import (
    Phase1FailureRecord,
    load_phase1_failures,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Phase 2 failing-input size distribution "
            "and the relationship between overall failure status "
            "and first failing test status."
        )
    )

    parser.add_argument(
        "--results",
        required=True,
        help="Path to Phase 1 Direct results.jsonl.",
    )

    parser.add_argument(
        "--tokenizer",
        required=True,
        help=(
            "Tokenizer/model name or local path used to measure "
            "input_text token lengths."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of refinable failures to analyze.",
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help=(
            "Number of status-mismatch examples to print."
        ),
    )

    return parser.parse_args()


def percentile(
    values: list[int],
    q: float,
) -> float:
    if not values:
        return 0.0

    values = sorted(values)

    position = (
        (len(values) - 1) * q
    )

    lower = int(position)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    weight = position - lower

    return (
        values[lower] * (1.0 - weight)
        + values[upper] * weight
    )


def print_length_summary(
    *,
    title: str,
    char_lengths: list[int],
    token_lengths: list[int],
) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print()
    print("[Character Length]")

    print(f"Count  : {len(char_lengths)}")
    print(f"Min    : {min(char_lengths)}")
    print(f"Mean   : {mean(char_lengths):.2f}")
    print(f"Median : {median(char_lengths):.2f}")
    print(f"P75    : {percentile(char_lengths, 0.75):.2f}")
    print(f"P90    : {percentile(char_lengths, 0.90):.2f}")
    print(f"P95    : {percentile(char_lengths, 0.95):.2f}")
    print(f"P99    : {percentile(char_lengths, 0.99):.2f}")
    print(f"Max    : {max(char_lengths)}")

    print()
    print("[Token Length]")

    print(f"Count  : {len(token_lengths)}")
    print(f"Min    : {min(token_lengths)}")
    print(f"Mean   : {mean(token_lengths):.2f}")
    print(f"Median : {median(token_lengths):.2f}")
    print(f"P75    : {percentile(token_lengths, 0.75):.2f}")
    print(f"P90    : {percentile(token_lengths, 0.90):.2f}")
    print(f"P95    : {percentile(token_lengths, 0.95):.2f}")
    print(f"P99    : {percentile(token_lengths, 0.99):.2f}")
    print(f"Max    : {max(token_lengths)}")


def print_length_buckets(
    *,
    char_lengths: list[int],
    token_lengths: list[int],
) -> None:
    print()
    print("=" * 80)
    print("Failing Input Length Buckets")
    print("=" * 80)

    char_thresholds = [
        1_000,
        4_000,
        16_000,
        100_000,
    ]

    print()
    print("[Character Buckets]")

    previous = 0

    for threshold in char_thresholds:
        count = sum(
            previous < value <= threshold
            for value in char_lengths
        )

        print(
            f"{previous + 1:>8,d} - "
            f"{threshold:>8,d} : "
            f"{count:3d}"
        )

        previous = threshold

    count = sum(
        value > char_thresholds[-1]
        for value in char_lengths
    )

    print(
        f">{char_thresholds[-1]:>17,d} : "
        f"{count:3d}"
    )

    token_thresholds = [
        512,
        1_024,
        4_096,
        8_192,
        16_384,
        32_768,
    ]

    print()
    print("[Token Buckets]")

    previous = 0

    for threshold in token_thresholds:
        count = sum(
            previous < value <= threshold
            for value in token_lengths
        )

        print(
            f"{previous + 1:>8,d} - "
            f"{threshold:>8,d} : "
            f"{count:3d}"
        )

        previous = threshold

    count = sum(
        value > token_thresholds[-1]
        for value in token_lengths
    )

    print(
        f">{token_thresholds[-1]:>17,d} : "
        f"{count:3d}"
    )


def print_length_by_status(
    failures: list[Phase1FailureRecord],
    *,
    char_lengths_by_id: dict[str, int],
    token_lengths_by_id: dict[str, int],
) -> None:
    grouped: dict[
        str,
        list[Phase1FailureRecord],
    ] = defaultdict(list)

    for failure in failures:
        grouped[failure.status].append(
            failure
        )

    print()
    print("=" * 80)
    print("Failing Input Length by Overall Status")
    print("=" * 80)

    for status, records in grouped.items():
        char_lengths = [
            char_lengths_by_id[
                record.problem_id
            ]
            for record in records
        ]

        token_lengths = [
            token_lengths_by_id[
                record.problem_id
            ]
            for record in records
        ]

        print()
        print(
            f"[{status}] n={len(records)}"
        )

        print(
            "Chars  : "
            f"median={median(char_lengths):.1f}, "
            f"p90={percentile(char_lengths, 0.90):.1f}, "
            f"p95={percentile(char_lengths, 0.95):.1f}, "
            f"max={max(char_lengths)}"
        )

        print(
            "Tokens : "
            f"median={median(token_lengths):.1f}, "
            f"p90={percentile(token_lengths, 0.90):.1f}, "
            f"p95={percentile(token_lengths, 0.95):.1f}, "
            f"max={max(token_lengths)}"
        )


def print_largest_inputs(
    failures: list[Phase1FailureRecord],
    *,
    char_lengths_by_id: dict[str, int],
    token_lengths_by_id: dict[str, int],
    top_k: int = 20,
) -> None:
    ranked = sorted(
        failures,
        key=lambda failure: (
            token_lengths_by_id[
                failure.problem_id
            ]
        ),
        reverse=True,
    )

    print()
    print("=" * 80)
    print(f"Top {top_k} Largest Failing Inputs")
    print("=" * 80)

    for failure in ranked[:top_k]:
        print(
            f"{failure.problem_id:20s} | "
            f"overall={failure.status:20s} | "
            f"test={failure.test_index:3d} | "
            f"chars={char_lengths_by_id[failure.problem_id]:9d} | "
            f"tokens={token_lengths_by_id[failure.problem_id]:9d}"
        )


def print_status_relationship(
    failures: list[Phase1FailureRecord],
) -> None:
    pair_counts = Counter(
        (
            failure.status,
            failure.first_failed_status,
        )
        for failure in failures
    )

    print()
    print("=" * 80)
    print("Overall Status vs First Failed Test Status")
    print("=" * 80)

    print()
    print(
        f"{'Overall Status':25s} "
        f"{'First Failed Status':25s} "
        f"{'Count':>6s}"
    )

    print("-" * 60)

    for (
        overall_status,
        first_failed_status,
    ), count in sorted(
        pair_counts.items()
    ):
        print(
            f"{overall_status:25s} "
            f"{first_failed_status:25s} "
            f"{count:6d}"
        )


def print_status_match_summary(
    failures: list[Phase1FailureRecord],
) -> None:
    matched = [
        failure
        for failure in failures
        if (
            failure.status
            == failure.first_failed_status
        )
    ]

    mismatched = [
        failure
        for failure in failures
        if (
            failure.status
            != failure.first_failed_status
        )
    ]

    print()
    print("=" * 80)
    print("Status Match Summary")
    print("=" * 80)

    print(
        f"Matched    : "
        f"{len(matched)} / {len(failures)} "
        f"({len(matched) / len(failures):.2%})"
    )

    print(
        f"Mismatched : "
        f"{len(mismatched)} / {len(failures)} "
        f"({len(mismatched) / len(failures):.2%})"
    )


def print_mismatch_examples(
    failures: list[Phase1FailureRecord],
    *,
    num_examples: int,
) -> None:
    mismatched = [
        failure
        for failure in failures
        if (
            failure.status
            != failure.first_failed_status
        )
    ]

    print()
    print("=" * 80)
    print("Status Mismatch Examples")
    print("=" * 80)

    if not mismatched:
        print("None")
        return

    for failure in mismatched[
        :num_examples
    ]:
        print()
        print(
            f"problem_id         : "
            f"{failure.problem_id}"
        )

        print(
            f"overall_status     : "
            f"{failure.status}"
        )

        print(
            f"first_failed_status: "
            f"{failure.first_failed_status}"
        )

        print(
            f"test_index         : "
            f"{failure.test_index}"
        )

        print(
            f"stderr             : "
            f"{failure.stderr!r}"
        )


def main() -> None:
    args = parse_args()

    result_path = Path(
        args.results
    )

    if not result_path.exists():
        raise FileNotFoundError(
            f"Phase 1 results not found: "
            f"{result_path}"
        )

    failures = load_phase1_failures(
        result_path=result_path,
        limit=args.limit,
    )

    if not failures:
        raise ValueError(
            "No refinable Phase 1 failures were loaded."
        )

    print("=" * 80)
    print("Phase 2 Failure Feedback Structure Analysis")
    print("=" * 80)

    print(
        f"Source    : {result_path}"
    )

    print(
        f"Failures  : {len(failures)}"
    )

    print(
        f"Tokenizer : {args.tokenizer}"
    )

    # --------------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )

    # --------------------------------------------------------------
    # Input length calculation
    # --------------------------------------------------------------

    char_lengths_by_id: dict[
        str,
        int,
    ] = {}

    token_lengths_by_id: dict[
        str,
        int,
    ] = {}

    for index, failure in enumerate(
        failures,
        start=1,
    ):
        char_length = len(
            failure.input_text
        )

        token_length = len(
            tokenizer.encode(
                failure.input_text,
                add_special_tokens=False,
            )
        )

        char_lengths_by_id[
            failure.problem_id
        ] = char_length

        token_lengths_by_id[
            failure.problem_id
        ] = token_length

        if (
            index % 25 == 0
            or index == len(failures)
        ):
            print(
                f"[Tokenize] "
                f"{index}/{len(failures)}"
            )

    char_lengths = list(
        char_lengths_by_id.values()
    )

    token_lengths = list(
        token_lengths_by_id.values()
    )

    # --------------------------------------------------------------
    # Analysis 1: failing input length
    # --------------------------------------------------------------

    print_length_summary(
        title="Failing Input Length Distribution",
        char_lengths=char_lengths,
        token_lengths=token_lengths,
    )

    print_length_buckets(
        char_lengths=char_lengths,
        token_lengths=token_lengths,
    )

    print_length_by_status(
        failures,
        char_lengths_by_id=(
            char_lengths_by_id
        ),
        token_lengths_by_id=(
            token_lengths_by_id
        ),
    )

    print_largest_inputs(
        failures,
        char_lengths_by_id=(
            char_lengths_by_id
        ),
        token_lengths_by_id=(
            token_lengths_by_id
        ),
    )

    # --------------------------------------------------------------
    # Analysis 2: overall status vs first failure
    # --------------------------------------------------------------

    print_status_relationship(
        failures
    )

    print_status_match_summary(
        failures
    )

    print_mismatch_examples(
        failures,
        num_examples=args.num_examples,
    )


if __name__ == "__main__":
    main()