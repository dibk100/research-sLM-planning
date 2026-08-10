"""
PYTHONPATH=. python archive/analyze_phase2_full.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# Paths
# ============================================================

RESULT_PATHS = {
    "Feedback": Path(
        "/mnt/hdd/project_sLM_planning/output/"
        "phase2_feedback_regeneration_500/results.jsonl"
    ),
    "Self-Replan": Path(
        "/mnt/hdd/project_sLM_planning/output/"
        "phase2_self_replan_500/results.jsonl"
    ),
    "Teacher-Replan": Path(
        "/mnt/hdd/project_sLM_planning/output/"
        "phase2_teacher_replan_500/results.jsonl"
    ),
}


# ============================================================
# Loading
# ============================================================

def load_results(
    path: Path,
) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Result file not found: {path}"
        )

    records = {}

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
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON: "
                    f"{path}:{line_number}"
                ) from e

            problem_id = record.get(
                "problem_id"
            )

            if not problem_id:
                raise ValueError(
                    f"Missing problem_id: "
                    f"{path}:{line_number}"
                )

            if problem_id in records:
                raise ValueError(
                    f"Duplicate problem_id: "
                    f"{problem_id}"
                )

            records[problem_id] = record

    return records


# ============================================================
# Helpers
# ============================================================

def pct(
    count: int,
    total: int,
) -> str:
    if total == 0:
        return "N/A"

    return (
        f"{count}/{total} "
        f"({count / total:.1%})"
    )


def is_exact_same(
    record: dict,
) -> bool:
    initial = (
        record.get("initial_code")
        or ""
    ).strip()

    refined = (
        record.get("refined_code")
        or ""
    ).strip()

    return initial == refined


def is_recovered(
    record: dict,
) -> bool:
    return bool(
        record.get("recovered")
    )


def get_delta(
    record: dict,
) -> int:
    return int(
        record.get(
            "test_pass_delta",
            0,
        )
    )


# ============================================================
# 1. Basic Summary
# ============================================================

def print_basic_summary(
    all_results: dict[
        str,
        dict[str, dict],
    ],
) -> None:
    print()
    print("=" * 100)
    print("1. BASIC SUMMARY")
    print("=" * 100)

    print()
    print(
        "| Strategy | Exact Same | Changed | Recovered |"
    )
    print(
        "|---|---:|---:|---:|"
    )

    for strategy, records in (
        all_results.items()
    ):
        total = len(records)

        same = sum(
            is_exact_same(r)
            for r in records.values()
        )

        changed = total - same

        recovered = sum(
            is_recovered(r)
            for r in records.values()
        )

        print(
            f"| {strategy} "
            f"| {pct(same, total)} "
            f"| {pct(changed, total)} "
            f"| {pct(recovered, total)} |"
        )


# ============================================================
# 2. Test Pass Delta
# ============================================================

def print_delta_analysis(
    all_results: dict[
        str,
        dict[str, dict],
    ],
) -> None:
    print()
    print("=" * 100)
    print("2. TEST-PASS DELTA ANALYSIS")
    print("=" * 100)

    print()
    print(
        "| Strategy | Improved | Unchanged | "
        "Degraded | Mean Delta |"
    )
    print(
        "|---|---:|---:|---:|---:|"
    )

    for strategy, records in (
        all_results.items()
    ):
        deltas = [
            get_delta(r)
            for r in records.values()
        ]

        total = len(deltas)

        improved = sum(
            d > 0
            for d in deltas
        )

        unchanged = sum(
            d == 0
            for d in deltas
        )

        degraded = sum(
            d < 0
            for d in deltas
        )

        mean_delta = (
            sum(deltas) / total
            if total
            else 0.0
        )

        print(
            f"| {strategy} "
            f"| {pct(improved, total)} "
            f"| {pct(unchanged, total)} "
            f"| {pct(degraded, total)} "
            f"| {mean_delta:+.3f} |"
        )


# ============================================================
# 3. Difficulty Analysis
# ============================================================

def print_difficulty_analysis(
    all_results: dict[
        str,
        dict[str, dict],
    ],
) -> None:
    print()
    print("=" * 100)
    print("3. RECOVERY BY DIFFICULTY")
    print("=" * 100)

    difficulties = [
        "easy",
        "medium",
        "hard",
    ]

    print()
    print(
        "| Difficulty | Feedback | "
        "Self-Replan | Teacher-Replan |"
    )
    print(
        "|---|---:|---:|---:|"
    )

    for difficulty in difficulties:
        values = []

        for strategy in [
            "Feedback",
            "Self-Replan",
            "Teacher-Replan",
        ]:
            subset = [
                r
                for r
                in all_results[
                    strategy
                ].values()
                if str(
                    r.get("difficulty", "")
                ).lower()
                == difficulty
            ]

            recovered = sum(
                is_recovered(r)
                for r in subset
            )

            values.append(
                pct(
                    recovered,
                    len(subset),
                )
            )

        print(
            f"| {difficulty.capitalize()} "
            f"| {values[0]} "
            f"| {values[1]} "
            f"| {values[2]} |"
        )


# ============================================================
# 4. Paired Alignment
# ============================================================

def get_common_problem_ids(
    all_results: dict[
        str,
        dict[str, dict],
    ],
) -> list[str]:
    sets = [
        set(records.keys())
        for records
        in all_results.values()
    ]

    common = set.intersection(
        *sets
    )

    return sorted(common)


def print_alignment(
    all_results: dict[
        str,
        dict[str, dict],
    ],
    common_ids: list[str],
) -> None:
    print()
    print("=" * 100)
    print("4. PAIRED DATA ALIGNMENT")
    print("=" * 100)

    for strategy, records in (
        all_results.items()
    ):
        print(
            f"{strategy:15s}: "
            f"{len(records)}"
        )

    print(
        f"{'Common':15s}: "
        f"{len(common_ids)}"
    )

    all_sets = [
        set(r.keys())
        for r in all_results.values()
    ]

    if all(
        s == all_sets[0]
        for s in all_sets[1:]
    ):
        print(
            "[PASS] All strategies contain "
            "the same problem IDs."
        )
    else:
        print(
            "[WARNING] Problem ID sets differ "
            "between strategies."
        )


# ============================================================
# 5. Paired Recovery Patterns
# ============================================================

def print_paired_patterns(
    all_results: dict[
        str,
        dict[str, dict],
    ],
    common_ids: list[str],
) -> None:
    print()
    print("=" * 100)
    print("5. PAIRED RECOVERY PATTERNS")
    print("=" * 100)

    pattern_counts = Counter()

    pattern_ids = defaultdict(list)

    for problem_id in common_ids:
        feedback = is_recovered(
            all_results[
                "Feedback"
            ][problem_id]
        )

        self_replan = is_recovered(
            all_results[
                "Self-Replan"
            ][problem_id]
        )

        teacher = is_recovered(
            all_results[
                "Teacher-Replan"
            ][problem_id]
        )

        pattern = (
            int(feedback),
            int(self_replan),
            int(teacher),
        )

        pattern_counts[pattern] += 1
        pattern_ids[pattern].append(
            problem_id
        )

    labels = {
        (0, 0, 0):
            "All Fail",

        (0, 0, 1):
            "Feedback Fail + Self Fail + Teacher Pass",

        (0, 1, 0):
            "Self Only",

        (0, 1, 1):
            "Self + Teacher Pass",

        (1, 0, 0):
            "Feedback Only",

        (1, 0, 1):
            "Feedback + Teacher Pass",

        (1, 1, 0):
            "Feedback + Self Pass",

        (1, 1, 1):
            "All Pass",
    }

    total = len(common_ids)

    print()

    for pattern in sorted(
        labels.keys()
    ):
        count = pattern_counts[
            pattern
        ]

        print(
            f"{labels[pattern]:45s} "
            f": {pct(count, total)}"
        )

    key_pattern = (
        0,
        0,
        1,
    )

    print()
    print("-" * 100)
    print(
        "KEY PATTERN: "
        "Feedback FAIL + Self FAIL + "
        "Teacher PASS"
    )
    print("-" * 100)

    ids = pattern_ids[
        key_pattern
    ]

    print(
        f"Count: {len(ids)}/{total} "
        f"({len(ids) / total:.1%})"
        if total
        else "Count: 0"
    )

    print()
    print("Problem IDs:")

    for problem_id in ids:
        print(
            f"  {problem_id}"
        )


# ============================================================
# 6. McNemar Exact Test
# ============================================================

def exact_binomial_two_sided(
    b: int,
    c: int,
) -> float:
    """
    McNemar exact test.

    Under H0:
        b ~ Binomial(b+c, 0.5)

    Two-sided exact p-value.
    """

    from math import comb

    n = b + c

    if n == 0:
        return 1.0

    k = min(
        b,
        c,
    )

    tail = sum(
        comb(n, i)
        for i in range(
            k + 1
        )
    ) / (2 ** n)

    return min(
        1.0,
        2.0 * tail,
    )


def mcnemar_pair(
    name_a: str,
    name_b: str,
    all_results: dict[
        str,
        dict[str, dict],
    ],
    common_ids: list[str],
) -> None:
    a_only = 0
    b_only = 0
    both_pass = 0
    both_fail = 0

    for problem_id in common_ids:
        a = is_recovered(
            all_results[
                name_a
            ][problem_id]
        )

        b = is_recovered(
            all_results[
                name_b
            ][problem_id]
        )

        if a and b:
            both_pass += 1

        elif a and not b:
            a_only += 1

        elif not a and b:
            b_only += 1

        else:
            both_fail += 1

    p_value = (
        exact_binomial_two_sided(
            a_only,
            b_only,
        )
    )

    print()
    print(
        f"{name_a} vs {name_b}"
    )
    print("-" * 70)

    print(
        f"Both fail : {both_fail}"
    )

    print(
        f"{name_a} only : "
        f"{a_only}"
    )

    print(
        f"{name_b} only : "
        f"{b_only}"
    )

    print(
        f"Both pass : {both_pass}"
    )

    print(
        f"Discordant: "
        f"{a_only + b_only}"
    )

    print(
        f"Exact McNemar p-value: "
        f"{p_value:.8g}"
    )


def print_mcnemar_analysis(
    all_results: dict[
        str,
        dict[str, dict],
    ],
    common_ids: list[str],
) -> None:
    print()
    print("=" * 100)
    print("6. EXACT McNEMAR TEST")
    print("=" * 100)

    mcnemar_pair(
        "Feedback",
        "Self-Replan",
        all_results,
        common_ids,
    )

    mcnemar_pair(
        "Feedback",
        "Teacher-Replan",
        all_results,
        common_ids,
    )

    mcnemar_pair(
        "Self-Replan",
        "Teacher-Replan",
        all_results,
        common_ids,
    )


# ============================================================
# 7. Recovered Problem IDs
# ============================================================

def print_recovered_ids(
    all_results: dict[
        str,
        dict[str, dict],
    ],
) -> None:
    print()
    print("=" * 100)
    print("7. RECOVERED PROBLEM IDs")
    print("=" * 100)

    for strategy, records in (
        all_results.items()
    ):
        ids = sorted(
            problem_id
            for problem_id, record
            in records.items()
            if is_recovered(record)
        )

        print()
        print(
            f"[{strategy}] "
            f"{len(ids)} recovered"
        )

        for problem_id in ids:
            print(
                f"  {problem_id}"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:
    all_results = {
        strategy: load_results(
            path
        )
        for strategy, path
        in RESULT_PATHS.items()
    }

    common_ids = (
        get_common_problem_ids(
            all_results
        )
    )

    print_alignment(
        all_results,
        common_ids,
    )

    print_basic_summary(
        all_results
    )

    print_delta_analysis(
        all_results
    )

    print_difficulty_analysis(
        all_results
    )

    print_paired_patterns(
        all_results,
        common_ids,
    )

    print_mcnemar_analysis(
        all_results,
        common_ids,
    )

    print_recovered_ids(
        all_results
    )

    print()
    print("=" * 100)
    print(
        "Phase 2 analysis completed."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()