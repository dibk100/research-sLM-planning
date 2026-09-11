"""
PYTHONPATH=. python \
  phase4_method_discovery/trajectory_analysis/analysis/analyze_fff_cases.py
  
================================================================================
FFF Partial-Correctness Analysis
================================================================================
Total FFF cases: 89

tpr_only_improvement             6 ( 6.74%)
vanilla_only_improvement         7 ( 7.87%)
tpr_dominant_improvement         4 ( 4.49%)
vanilla_dominant_improvement     2 ( 2.25%)
both_equal_improvement           4 ( 4.49%)
crossed                          1 ( 1.12%)
both_degraded                    6 ( 6.74%)
tpr_degraded                     4 ( 4.49%)
vanilla_degraded                 4 ( 4.49%)
unchanged_nonzero               25 (28.09%)
all_zero                        26 (29.21%)
other_mixed                      0 ( 0.00%)

================================================================================
Top 15 qualitative candidates
================================================================================
 1. deepcoder_taco_00417 | tpr_only_improvement | B=0.000 V=0.000 T=0.918 | score=1.377
 2. deepcoder_taco_00379 | vanilla_only_improvement | B=0.000 V=0.686 T=0.000 | score=1.029
 3. deepcoder_taco_00022 | tpr_degraded | B=0.550 V=0.550 T=0.000 | score=0.825
 4. deepcoder_taco_00061 | vanilla_degraded | B=0.605 V=0.158 T=0.605 | score=0.671
 5. deepcoder_taco_01069 | both_equal_improvement | B=0.000 V=0.513 T=0.513 | score=0.513
 6. deepcoder_taco_00330 | tpr_dominant_improvement | B=0.110 V=0.256 T=0.500 | score=0.512
 7. deepcoder_taco_00020 | both_equal_improvement | B=0.009 V=0.368 T=0.368 | score=0.358
 8. deepcoder_taco_00881 | vanilla_degraded | B=0.957 V=0.759 T=0.957 | score=0.297
 9. deepcoder_taco_01029 | vanilla_only_improvement | B=0.085 V=0.277 T=0.085 | score=0.287
10. deepcoder_taco_01025 | vanilla_dominant_improvement | B=0.000 V=0.224 T=0.182 | score=0.245
11. deepcoder_taco_00030 | both_degraded | B=0.222 V=0.024 T=0.102 | score=0.237
12. deepcoder_taco_01047 | vanilla_only_improvement | B=0.138 V=0.287 T=0.138 | score=0.224
13. deepcoder_taco_00412 | tpr_dominant_improvement | B=0.000 V=0.046 T=0.148 | score=0.199
14. deepcoder_taco_01113 | both_degraded | B=0.189 V=0.000 T=0.000 | score=0.189
15. deepcoder_taco_00554 | vanilla_dominant_improvement | B=0.000 V=0.125 T=0.010 | score=0.183
"""
# phase4_method_discovery/trajectory_analysis/analysis/analyze_fff_cases.py
#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = (
    "phase4_method_discovery/trajectory_analysis/"
    "outputs/transition_candidates.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    "phase4_method_discovery/trajectory_analysis/outputs"
)

EPS = 1e-12


CATEGORY_DESCRIPTIONS = {
    "tpr_dominant_improvement": (
        "Both RL variants improve over Base, but TPR improves more than Vanilla."
    ),
    "vanilla_dominant_improvement": (
        "Both RL variants improve over Base, but Vanilla improves more than TPR."
    ),
    "both_equal_improvement": (
        "Both RL variants improve over Base by the same amount."
    ),
    "tpr_only_improvement": (
        "TPR improves over Base while Vanilla does not."
    ),
    "vanilla_only_improvement": (
        "Vanilla improves over Base while TPR does not."
    ),
    "both_degraded": (
        "Both RL variants have lower TPR than Base."
    ),
    "tpr_degraded": (
        "TPR is worse than Base while Vanilla is not worse."
    ),
    "vanilla_degraded": (
        "Vanilla is worse than Base while TPR is not worse."
    ),
    "crossed": (
        "One RL variant improves while the other degrades relative to Base."
    ),
    "all_zero": (
        "Base, Vanilla, and TPR all have zero test-pass ratio."
    ),
    "unchanged_nonzero": (
        "All three have the same non-zero test-pass ratio."
    ),
    "other_mixed": (
        "Mixed partial-correctness pattern not covered by the main categories."
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze persistent-failure (FFF) cases from the "
            "Base/Vanilla/TPR trajectory comparison."
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="transition_candidates.jsonl",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            if "problem_id" not in row:
                raise ValueError(
                    f"{path}:{line_no}: missing problem_id"
                )

            rows.append(row)

    return rows


def safe_float(value: Any) -> float:
    if value is None:
        return 0.0

    return float(value)


def is_zero(x: float) -> bool:
    return abs(x) <= EPS


def is_positive(x: float) -> bool:
    return x > EPS


def is_negative(x: float) -> bool:
    return x < -EPS


def classify_fff(
    base_tpr: float,
    vanilla_tpr: float,
    tpr_tpr: float,
) -> str:
    """
    Classify an FFF case according to partial-correctness changes.

    All three runs already failed full execution, so this function
    focuses only on test-pass-ratio differences.
    """

    dv = vanilla_tpr - base_tpr
    dt = tpr_tpr - base_tpr

    # Completely uninformative at the execution level.
    if (
        is_zero(base_tpr)
        and is_zero(vanilla_tpr)
        and is_zero(tpr_tpr)
    ):
        return "all_zero"

    # Exactly unchanged partial correctness.
    if (
        is_zero(dv)
        and is_zero(dt)
    ):
        return "unchanged_nonzero"

    # One improves and the other degrades.
    if (
        (is_positive(dv) and is_negative(dt))
        or
        (is_negative(dv) and is_positive(dt))
    ):
        return "crossed"

    # TPR improves, Vanilla does not improve.
    if is_positive(dt) and not is_positive(dv):
        return "tpr_only_improvement"

    # Vanilla improves, TPR does not improve.
    if is_positive(dv) and not is_positive(dt):
        return "vanilla_only_improvement"

    # Both improve.
    if is_positive(dv) and is_positive(dt):
        diff = dt - dv

        if is_positive(diff):
            return "tpr_dominant_improvement"

        if is_negative(diff):
            return "vanilla_dominant_improvement"

        return "both_equal_improvement"

    # Both degrade.
    if is_negative(dv) and is_negative(dt):
        return "both_degraded"

    # Only TPR degrades.
    if is_negative(dt) and not is_negative(dv):
        return "tpr_degraded"

    # Only Vanilla degrades.
    if is_negative(dv) and not is_negative(dt):
        return "vanilla_degraded"

    return "other_mixed"


def qualitative_score(
    category: str,
    base_tpr: float,
    vanilla_tpr: float,
    tpr_tpr: float,
) -> float:
    """
    Ranking score for manual qualitative inspection.

    This is NOT a performance metric.

    Higher values simply indicate that the case has larger behavioral
    separation among Base / Vanilla / TPR and may therefore be more
    informative for trajectory-level inspection.
    """

    dv = vanilla_tpr - base_tpr
    dt = tpr_tpr - base_tpr
    d_tv = tpr_tpr - vanilla_tpr

    # Maximum pairwise TPR separation.
    separation = max(
        abs(dv),
        abs(dt),
        abs(d_tv),
    )

    # Reward cases where the two RL methods behave differently.
    rl_difference = abs(d_tv)

    score = separation + 0.5 * rl_difference

    # all-zero cases cannot be ranked by execution behavior.
    if category == "all_zero":
        return 0.0

    return score


def build_fff_row(
    row: dict[str, Any],
) -> dict[str, Any]:

    base = row["base"]
    vanilla = row["vanilla"]
    tpr = row["tpr"]

    base_tpr = safe_float(
        base.get("test_pass_ratio")
    )
    vanilla_tpr = safe_float(
        vanilla.get("test_pass_ratio")
    )
    tpr_tpr = safe_float(
        tpr.get("test_pass_ratio")
    )

    dv = vanilla_tpr - base_tpr
    dt = tpr_tpr - base_tpr
    d_tv = tpr_tpr - vanilla_tpr

    category = classify_fff(
        base_tpr,
        vanilla_tpr,
        tpr_tpr,
    )

    score = qualitative_score(
        category,
        base_tpr,
        vanilla_tpr,
        tpr_tpr,
    )

    return {
        "problem_id": row["problem_id"],
        "title": row.get("title"),
        "difficulty": row.get("difficulty"),
        "rating": row.get("rating"),

        "category": category,
        "category_description":
            CATEGORY_DESCRIPTIONS[category],

        "base_tpr": base_tpr,
        "vanilla_tpr": vanilla_tpr,
        "tpr_tpr": tpr_tpr,

        "vanilla_vs_base_tpr": dv,
        "tpr_vs_base_tpr": dt,
        "tpr_vs_vanilla_tpr": d_tv,

        "base_passed_tests":
            base.get("passed_tests"),
        "vanilla_passed_tests":
            vanilla.get("passed_tests"),
        "tpr_passed_tests":
            tpr.get("passed_tests"),

        "total_tests":
            base.get("total_tests"),

        "base_status":
            base.get("status"),
        "vanilla_status":
            vanilla.get("status"),
        "tpr_status":
            tpr.get("status"),

        "qualitative_score": score,

        "base_plan":
            base.get("plan"),
        "vanilla_plan":
            vanilla.get("plan"),
        "tpr_plan":
            tpr.get("plan"),

        "base_code":
            base.get("code"),
        "vanilla_code":
            vanilla.get("code"),
        "tpr_code":
            tpr.get("code"),

        "problem":
            row.get("problem"),
    }


def write_summary_csv(
    path: Path,
    rows: list[dict[str, Any]],
):
    counts = Counter(
        row["category"]
        for row in rows
    )

    total = len(rows)

    category_order = [
        "tpr_only_improvement",
        "vanilla_only_improvement",
        "tpr_dominant_improvement",
        "vanilla_dominant_improvement",
        "both_equal_improvement",
        "crossed",
        "both_degraded",
        "tpr_degraded",
        "vanilla_degraded",
        "unchanged_nonzero",
        "all_zero",
        "other_mixed",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "category",
            "description",
            "count",
            "ratio",
        ])

        for category in category_order:
            count = counts.get(category, 0)

            writer.writerow([
                category,
                CATEGORY_DESCRIPTIONS[category],
                count,
                count / total if total else 0.0,
            ])


def write_cases_csv(
    path: Path,
    rows: list[dict[str, Any]],
):
    fieldnames = [
        "problem_id",
        "category",

        "base_tpr",
        "vanilla_tpr",
        "tpr_tpr",

        "vanilla_vs_base_tpr",
        "tpr_vs_base_tpr",
        "tpr_vs_vanilla_tpr",

        "base_passed_tests",
        "vanilla_passed_tests",
        "tpr_passed_tests",
        "total_tests",

        "base_status",
        "vanilla_status",
        "tpr_status",

        "qualitative_score",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: row.get(key)
                for key in fieldnames
            })


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def print_summary(
    rows: list[dict[str, Any]],
):
    counts = Counter(
        row["category"]
        for row in rows
    )

    print()
    print("=" * 80)
    print("FFF Partial-Correctness Analysis")
    print("=" * 80)

    print(f"Total FFF cases: {len(rows)}")
    print()

    category_order = [
        "tpr_only_improvement",
        "vanilla_only_improvement",
        "tpr_dominant_improvement",
        "vanilla_dominant_improvement",
        "both_equal_improvement",
        "crossed",
        "both_degraded",
        "tpr_degraded",
        "vanilla_degraded",
        "unchanged_nonzero",
        "all_zero",
        "other_mixed",
    ]

    for category in category_order:
        count = counts.get(category, 0)

        if len(rows):
            ratio = count / len(rows)
        else:
            ratio = 0.0

        print(
            f"{category:30s} "
            f"{count:3d} "
            f"({ratio:6.2%})"
        )


def print_top_cases(
    rows: list[dict[str, Any]],
    n: int = 15,
):
    informative = [
        row
        for row in rows
        if row["category"] != "all_zero"
    ]

    print()
    print("=" * 80)
    print(f"Top {min(n, len(informative))} qualitative candidates")
    print("=" * 80)

    for i, row in enumerate(
        informative[:n],
        start=1,
    ):
        print(
            f"{i:2d}. "
            f"{row['problem_id']} | "
            f"{row['category']} | "
            f"B={row['base_tpr']:.3f} "
            f"V={row['vanilla_tpr']:.3f} "
            f"T={row['tpr_tpr']:.3f} | "
            f"score={row['qualitative_score']:.3f}"
        )


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_jsonl(input_path)

    fff_source = [
        row
        for row in rows
        if row.get("transition") == "FFF"
    ]

    if not fff_source:
        raise RuntimeError(
            "No FFF cases found in input."
        )

    fff_rows = [
        build_fff_row(row)
        for row in fff_source
    ]

    # Rank cases by qualitative informativeness.
    fff_rows.sort(
        key=lambda row: (
            -row["qualitative_score"],
            row["problem_id"],
        )
    )

    summary_path = (
        output_dir
        / "fff_case_summary.csv"
    )

    cases_csv_path = (
        output_dir
        / "fff_cases_ranked.csv"
    )

    cases_jsonl_path = (
        output_dir
        / "fff_cases_ranked.jsonl"
    )

    write_summary_csv(
        summary_path,
        fff_rows,
    )

    write_cases_csv(
        cases_csv_path,
        fff_rows,
    )

    write_jsonl(
        cases_jsonl_path,
        fff_rows,
    )

    print_summary(fff_rows)
    print_top_cases(fff_rows)

    print()
    print("[Outputs]")
    print(f"  {summary_path}")
    print(f"  {cases_csv_path}")
    print(f"  {cases_jsonl_path}")


if __name__ == "__main__":
    main()