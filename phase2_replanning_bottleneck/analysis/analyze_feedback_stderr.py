"""
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase2_replanning_bottleneck/analysis/analyze_feedback_stderr.py \
  --results /mnt/hdd/project_sLM_planning/phase1/livecodebench_v6_stdin/qwen25Coder3b/direct/results.jsonl
"""

# phase2_replanning_bottleneck/analysis/analyze_feedback_stderr.py

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from src.datasets.phase1_failure_loader import (
    Phase1FailureRecord,
    load_phase1_failures,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze evaluator stderr feedback from "
            "Phase 1 Direct refinable failures."
        )
    )

    parser.add_argument(
        "--results",
        required=True,
        help=(
            "Path to Phase 1 Direct results.jsonl."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional maximum number of refinable "
            "failures to analyze."
        ),
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help=(
            "Number of stderr examples to print "
            "for each failure status."
        ),
    )

    return parser.parse_args()


def print_status_distribution(
    failures: list[Phase1FailureRecord],
) -> None:
    counts = Counter(
        failure.status
        for failure in failures
    )

    print()
    print("[Failure Status]")

    for status, count in counts.most_common():
        print(
            f"{status:25s}: {count}"
        )


def print_stderr_availability(
    failures: list[Phase1FailureRecord],
) -> None:
    nonempty = sum(
        bool(failure.stderr.strip())
        for failure in failures
    )

    empty = (
        len(failures) - nonempty
    )

    print()
    print("[stderr Availability]")
    print(f"Non-empty : {nonempty}")
    print(f"Empty     : {empty}")

    grouped: dict[
        str,
        list[Phase1FailureRecord],
    ] = defaultdict(list)

    for failure in failures:
        grouped[failure.status].append(
            failure
        )

    print()
    print("[stderr Availability by Status]")

    for status, records in grouped.items():
        status_nonempty = sum(
            bool(record.stderr.strip())
            for record in records
        )

        status_empty = (
            len(records) - status_nonempty
        )

        print(
            f"{status:25s} | "
            f"total={len(records):3d} | "
            f"nonempty={status_nonempty:3d} | "
            f"empty={status_empty:3d}"
        )


def print_stderr_lengths(
    failures: list[Phase1FailureRecord],
) -> None:
    lengths = [
        len(failure.stderr)
        for failure in failures
    ]

    if not lengths:
        return

    sorted_lengths = sorted(
        lengths
    )

    median = (
        sorted_lengths[
            len(sorted_lengths) // 2
        ]
        if len(sorted_lengths) % 2 == 1
        else (
            sorted_lengths[
                len(sorted_lengths) // 2 - 1
            ]
            + sorted_lengths[
                len(sorted_lengths) // 2
            ]
        )
        / 2
    )

    print()
    print("[stderr Length]")
    print(f"Min    : {min(lengths)}")
    print(f"Max    : {max(lengths)}")
    print(
        f"Mean   : "
        f"{sum(lengths) / len(lengths):.2f}"
    )
    print(f"Median : {median}")


def print_stderr_examples(
    failures: list[Phase1FailureRecord],
    *,
    num_examples: int,
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
    print("[stderr Examples]")

    for status, records in grouped.items():
        print()
        print("-" * 80)
        print(status)
        print("-" * 80)

        for failure in records[
            :num_examples
        ]:
            print(
                f"{failure.problem_id:20s} | "
                f"test={failure.test_index:3d} | "
                f"stderr={failure.stderr!r}"
            )


def print_empty_stderr_cases(
    failures: list[Phase1FailureRecord],
) -> None:
    empty_records = [
        failure
        for failure in failures
        if not failure.stderr.strip()
    ]

    print()
    print("[Empty stderr Cases]")

    if not empty_records:
        print("None")
        return

    for failure in empty_records:
        print(
            f"{failure.problem_id:20s} | "
            f"status={failure.status:25s} | "
            f"test={failure.test_index}"
        )


def print_common_messages(
    failures: list[Phase1FailureRecord],
) -> None:
    """
    Show exact stderr message frequencies.

    This is useful for checking whether evaluator feedback
    consists mostly of generic messages or contains
    problem-specific diagnostic information.
    """

    grouped: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    for failure in failures:
        message = (
            failure.stderr.strip()
            if failure.stderr.strip()
            else "<EMPTY>"
        )

        grouped[failure.status][
            message
        ] += 1

    print()
    print("[Most Common Exact stderr Messages]")

    for status, counter in grouped.items():
        print()
        print("-" * 80)
        print(status)
        print("-" * 80)

        for message, count in counter.most_common(
            10
        ):
            display_message = message.replace(
                "\n",
                "\\n",
            )

            if len(display_message) > 300:
                display_message = (
                    display_message[:300]
                    + "..."
                )

            print(
                f"{count:3d}x | "
                f"{display_message}"
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

    if (
        args.limit is not None
        and args.limit <= 0
    ):
        raise ValueError(
            "--limit must be greater than 0."
        )

    if args.num_examples <= 0:
        raise ValueError(
            "--num-examples must be greater than 0."
        )

    # --------------------------------------------------------------
    # Use exactly the same failure selection policy as Phase 2.
    # --------------------------------------------------------------

    failures = load_phase1_failures(
        result_path=result_path,
        limit=args.limit,
    )

    if not failures:
        raise ValueError(
            "No refinable Phase 1 failures were loaded."
        )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print("=" * 80)
    print("Phase 2 Evaluator Feedback Diagnostics")
    print("=" * 80)

    print(
        f"Source   : {result_path}"
    )
    print(
        f"Failures : {len(failures)}"
    )

    print_status_distribution(
        failures
    )

    print_stderr_availability(
        failures
    )

    print_stderr_lengths(
        failures
    )

    print_common_messages(
        failures
    )

    print_stderr_examples(
        failures,
        num_examples=args.num_examples,
    )

    print_empty_stderr_cases(
        failures
    )


if __name__ == "__main__":
    main()