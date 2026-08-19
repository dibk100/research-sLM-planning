"""
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase2_replanning_bottleneck/analysis/analyze_phase2_results.py \
  --feedback /mnt/hdd/project_sLM_planning/phase2/livecodebench_v6_stdin/qwen253b/feedback/results.jsonl \
  --self-replan /mnt/hdd/project_sLM_planning/phase2/livecodebench_v6_stdin/qwen253b/self_replan/results.jsonl \
  --teacher-replan /mnt/hdd/project_sLM_planning/phase2/livecodebench_v6_stdin/qwen253b/teacher_replan/results.jsonl \
  --expected-problems 261

"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from scipy.stats import binomtest


STRATEGIES = (
    "feedback",
    "self_replan",
    "teacher_replan",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Phase 2 feedback / self-replan / "
            "teacher-replan results."
        )
    )

    parser.add_argument(
        "--feedback",
        required=True,
        help="Feedback-regeneration results.jsonl",
    )

    parser.add_argument(
        "--self-replan",
        required=True,
        help="Self-replan results.jsonl",
    )

    parser.add_argument(
        "--teacher-replan",
        required=True,
        help="Teacher-replan results.jsonl",
    )

    parser.add_argument(
        "--expected-problems",
        type=int,
        default=230,
    )

    return parser.parse_args()


def load_jsonl(
    path: str | Path,
) -> dict[str, dict]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    records: dict[str, dict] = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON: {path}:{line_number}"
                ) from error

            problem_id = str(
                record.get("problem_id", "")
            ).strip()

            if not problem_id:
                raise ValueError(
                    f"Missing problem_id: {path}:{line_number}"
                )

            if problem_id in records:
                raise ValueError(
                    f"Duplicate problem_id={problem_id}: {path}"
                )

            records[problem_id] = record

    return records


def get_bool(
    record: dict,
    key: str,
) -> bool:
    return record.get(key) is True


def get_float(
    record: dict,
    key: str,
    default: float = 0.0,
) -> float:
    value = record.get(key)

    if value is None:
        return default

    return float(value)


def recovered(
    record: dict,
) -> bool:
    # Prefer explicit Phase 2 field.
    if "recovered" in record:
        return record["recovered"] is True

    # Fallback: final full-pass.
    if "passed" in record:
        return record["passed"] is True

    return (
        str(record.get("status", "")).strip()
        == "PASS"
    )


def initial_ratio(
    record: dict,
) -> float:
    for key in (
        "initial_test_pass_ratio",
        "previous_test_pass_ratio",
    ):
        if key in record:
            return float(record[key])

    raise KeyError(
        "Could not find initial test pass ratio."
    )


def final_ratio(
    record: dict,
) -> float:
    if "test_pass_ratio" in record:
        return float(
            record["test_pass_ratio"]
        )

    if "final_test_pass_ratio" in record:
        return float(
            record["final_test_pass_ratio"]
        )

    raise KeyError(
        "Could not find final test pass ratio."
    )


def difficulty(
    record: dict,
) -> str:
    value = record.get("difficulty")

    if value is None:
        return "unknown"

    return str(value).strip().lower()


def initial_status(
    record: dict,
) -> str:
    for key in (
        "initial_status",
        "previous_status",
    ):
        if key in record:
            return str(
                record[key]
            ).strip()

    return "UNKNOWN"


def safe_mean(
    values: list[float],
) -> float:
    if not values:
        return float("nan")

    return mean(values)


def print_header(
    title: str,
) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def print_dataset_check(
    data: dict[str, dict[str, dict]],
    expected: int,
) -> list[str]:
    print_header(
        "Dataset Alignment"
    )

    id_sets = {
        strategy: set(records)
        for strategy, records in data.items()
    }

    for strategy in STRATEGIES:
        print(
            f"{strategy:<16}: "
            f"{len(id_sets[strategy])}"
        )

    common_ids = set.intersection(
        *id_sets.values()
    )

    union_ids = set.union(
        *id_sets.values()
    )

    print(
        f"\nCommon problems : {len(common_ids)}"
    )
    print(
        f"Union problems  : {len(union_ids)}"
    )
    print(
        f"Expected        : {expected}"
    )

    if len(common_ids) != expected:
        print(
            "\n[WARNING] Common problem count "
            "does not match expected count."
        )

    if any(
        ids != common_ids
        for ids in id_sets.values()
    ):
        print(
            "[WARNING] Strategy problem sets "
            "are not identical."
        )

        for strategy, ids in id_sets.items():
            missing = sorted(
                union_ids - ids
            )

            if missing:
                print(
                    f"{strategy} missing: "
                    f"{missing[:10]}"
                )
    else:
        print(
            "[PASS] All strategies use the "
            "same problem set."
        )

    return sorted(
        common_ids
    )


def print_overall_results(
    data: dict[str, dict[str, dict]],
    problem_ids: list[str],
) -> None:
    print_header(
        "Overall Results"
    )

    print(
        f"{'Strategy':<16}"
        f"{'Recovered':>12}"
        f"{'Recovery Rate':>16}"
        f"{'Mean Final':>14}"
        f"{'Mean Delta':>14}"
    )

    print("-" * 72)

    for strategy in STRATEGIES:
        records = [
            data[strategy][pid]
            for pid in problem_ids
        ]

        recovered_count = sum(
            recovered(record)
            for record in records
        )

        final_ratios = [
            final_ratio(record)
            for record in records
        ]

        deltas = [
            final_ratio(record)
            - initial_ratio(record)
            for record in records
        ]

        print(
            f"{strategy:<16}"
            f"{recovered_count:>6}/{len(records):<5}"
            f"{recovered_count / len(records):>15.4f}"
            f"{safe_mean(final_ratios):>14.4f}"
            f"{safe_mean(deltas):>14.4f}"
        )


def print_delta_dynamics(
    data: dict[str, dict[str, dict]],
    problem_ids: list[str],
) -> None:
    print_header(
        "Test-Pass-Ratio Dynamics"
    )

    eps = 1e-12

    for strategy in STRATEGIES:
        improved = 0
        unchanged = 0
        degraded = 0

        deltas = []

        for pid in problem_ids:
            record = data[strategy][pid]

            delta = (
                final_ratio(record)
                - initial_ratio(record)
            )

            deltas.append(
                delta
            )

            if delta > eps:
                improved += 1
            elif delta < -eps:
                degraded += 1
            else:
                unchanged += 1

        n = len(problem_ids)

        print()
        print(strategy)
        print("-" * 50)

        print(
            f"Improved  : {improved:3d}/{n} "
            f"({improved / n:.2%})"
        )
        print(
            f"Unchanged : {unchanged:3d}/{n} "
            f"({unchanged / n:.2%})"
        )
        print(
            f"Degraded  : {degraded:3d}/{n} "
            f"({degraded / n:.2%})"
        )
        print(
            f"Mean delta: {safe_mean(deltas):+.6f}"
        )


def print_recovery_by_difficulty(
    data: dict[str, dict[str, dict]],
    problem_ids: list[str],
) -> None:
    print_header(
        "Recovery by Difficulty"
    )

    # Use feedback as canonical metadata source.
    groups: dict[str, list[str]] = defaultdict(
        list
    )

    for pid in problem_ids:
        diff = difficulty(
            data["feedback"][pid]
        )

        groups[diff].append(
            pid
        )

    order = [
        "easy",
        "medium",
        "hard",
        "unknown",
    ]

    for diff in order:
        ids = groups.get(
            diff,
            [],
        )

        if not ids:
            continue

        print()
        print(
            f"[{diff.upper()}] n={len(ids)}"
        )

        for strategy in STRATEGIES:
            count = sum(
                recovered(
                    data[strategy][pid]
                )
                for pid in ids
            )

            print(
                f"{strategy:<16}: "
                f"{count:3d}/{len(ids):3d} "
                f"({count / len(ids):.2%})"
            )


def print_recovery_by_initial_status(
    data: dict[str, dict[str, dict]],
    problem_ids: list[str],
) -> None:
    print_header(
        "Recovery by Initial Failure Status"
    )

    groups: dict[str, list[str]] = defaultdict(
        list
    )

    for pid in problem_ids:
        status = initial_status(
            data["feedback"][pid]
        )

        groups[status].append(
            pid
        )

    for status in sorted(groups):
        ids = groups[status]

        print()
        print(
            f"[{status}] n={len(ids)}"
        )

        for strategy in STRATEGIES:
            count = sum(
                recovered(
                    data[strategy][pid]
                )
                for pid in ids
            )

            print(
                f"{strategy:<16}: "
                f"{count:3d}/{len(ids):3d} "
                f"({count / len(ids):.2%})"
            )


def mcnemar_exact(
    data_a: dict[str, dict],
    data_b: dict[str, dict],
    problem_ids: list[str],
) -> tuple[int, int, int, int, float]:
    both_pass = 0
    a_only = 0
    b_only = 0
    both_fail = 0

    for pid in problem_ids:
        a = recovered(
            data_a[pid]
        )
        b = recovered(
            data_b[pid]
        )

        if a and b:
            both_pass += 1
        elif a and not b:
            a_only += 1
        elif not a and b:
            b_only += 1
        else:
            both_fail += 1

    discordant = (
        a_only
        + b_only
    )

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(
            k=min(
                a_only,
                b_only,
            ),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    return (
        both_pass,
        a_only,
        b_only,
        both_fail,
        p_value,
    )


def print_pairwise_tests(
    data: dict[str, dict[str, dict]],
    problem_ids: list[str],
) -> None:
    print_header(
        "Paired Recovery Comparison / McNemar Exact"
    )

    pairs = [
        (
            "feedback",
            "self_replan",
        ),
        (
            "feedback",
            "teacher_replan",
        ),
        (
            "self_replan",
            "teacher_replan",
        ),
    ]

    for a, b in pairs:
        (
            both_pass,
            a_only,
            b_only,
            both_fail,
            p_value,
        ) = mcnemar_exact(
            data[a],
            data[b],
            problem_ids,
        )

        print()
        print(
            f"{a} vs {b}"
        )
        print("-" * 60)
        print(
            f"Both PASS : {both_pass}"
        )
        print(
            f"{a} only : {a_only}"
        )
        print(
            f"{b} only : {b_only}"
        )
        print(
            f"Both FAIL : {both_fail}"
        )
        print(
            f"McNemar exact p = "
            f"{p_value:.10g}"
        )


def print_three_way_patterns(
    data: dict[str, dict[str, dict]],
    problem_ids: list[str],
) -> None:
    print_header(
        "Three-Way Recovery Patterns"
    )

    patterns = Counter()

    pattern_ids: dict[
        tuple[bool, bool, bool],
        list[str],
    ] = defaultdict(
        list
    )

    for pid in problem_ids:
        pattern = (
            recovered(
                data["feedback"][pid]
            ),
            recovered(
                data["self_replan"][pid]
            ),
            recovered(
                data["teacher_replan"][pid]
            ),
        )

        patterns[pattern] += 1
        pattern_ids[pattern].append(
            pid
        )

    labels = {
        (
            False,
            False,
            False,
        ): "F FAIL / S FAIL / T FAIL",

        (
            False,
            False,
            True,
        ): "F FAIL / S FAIL / T PASS",

        (
            False,
            True,
            False,
        ): "F FAIL / S PASS / T FAIL",

        (
            False,
            True,
            True,
        ): "F FAIL / S PASS / T PASS",

        (
            True,
            False,
            False,
        ): "F PASS / S FAIL / T FAIL",

        (
            True,
            False,
            True,
        ): "F PASS / S FAIL / T PASS",

        (
            True,
            True,
            False,
        ): "F PASS / S PASS / T FAIL",

        (
            True,
            True,
            True,
        ): "F PASS / S PASS / T PASS",
    }

    for pattern, label in labels.items():
        count = patterns[
            pattern
        ]

        print(
            f"{label:<30}: "
            f"{count:3d} "
            f"({count / len(problem_ids):.2%})"
        )

    key_pattern = (
        False,
        False,
        True,
    )

    ids = pattern_ids[
        key_pattern
    ]

    print()
    print(
        "Key pattern: Feedback FAIL + "
        "Self-Replan FAIL + Teacher-Replan PASS"
    )
    print(
        f"Count : {len(ids)}"
    )

    if ids:
        print(
            "IDs   : "
            + ", ".join(ids)
        )


def print_cost_analysis(
    data: dict[str, dict[str, dict]],
    problem_ids: list[str],
) -> None:
    print_header(
        "Generation Cost"
    )

    keys = (
        "prompt_tokens",
        "completion_tokens",
        "generation_time",
    )

    print(
        f"{'Strategy':<16}"
        f"{'Prompt Tok':>14}"
        f"{'Completion':>14}"
        f"{'Total Tok':>14}"
        f"{'Gen Time':>14}"
    )

    print("-" * 72)

    for strategy in STRATEGIES:
        records = [
            data[strategy][pid]
            for pid in problem_ids
        ]

        prompt_tokens = [
            get_float(
                r,
                "prompt_tokens",
            )
            for r in records
        ]

        completion_tokens = [
            get_float(
                r,
                "completion_tokens",
            )
            for r in records
        ]

        generation_times = [
            get_float(
                r,
                "generation_time",
            )
            for r in records
        ]

        mean_prompt = safe_mean(
            prompt_tokens
        )

        mean_completion = safe_mean(
            completion_tokens
        )

        print(
            f"{strategy:<16}"
            f"{mean_prompt:>14.2f}"
            f"{mean_completion:>14.2f}"
            f"{mean_prompt + mean_completion:>14.2f}"
            f"{safe_mean(generation_times):>14.3f}"
        )

    print()
    print(
        "Note: Teacher-Replan cost contains only "
        "student code-generation inference cost; "
        "external teacher labeling cost is excluded."
    )


def main() -> None:
    args = parse_args()

    data = {
        "feedback": load_jsonl(
            args.feedback
        ),
        "self_replan": load_jsonl(
            args.self_replan
        ),
        "teacher_replan": load_jsonl(
            args.teacher_replan
        ),
    }

    print_header(
        "Phase 2 Replanning Bottleneck Analysis"
    )

    problem_ids = print_dataset_check(
        data,
        expected=args.expected_problems,
    )

    if not problem_ids:
        raise ValueError(
            "No common problem IDs."
        )

    print_overall_results(
        data,
        problem_ids,
    )

    print_delta_dynamics(
        data,
        problem_ids,
    )

    print_recovery_by_difficulty(
        data,
        problem_ids,
    )

    print_recovery_by_initial_status(
        data,
        problem_ids,
    )

    print_pairwise_tests(
        data,
        problem_ids,
    )

    print_three_way_patterns(
        data,
        problem_ids,
    )

    print_cost_analysis(
        data,
        problem_ids,
    )


if __name__ == "__main__":
    main()