"""
PYTHONPATH="$HOME/workspace/project_sLM_planning" \
python \
  phase4_method_discovery/diagnostic/analysis/analyze_coverage_conditioned.py \
  --input \
  phase4_method_discovery/diagnostic/outputs/pilot/base_n8_m4_100problems.jsonl \
  --output-dir \
  phase4_method_discovery/diagnostic/outputs/pilot/base_n8_m4_100problems_analysis
"""
# phase4_method_discovery/diagnostic/analysis/analyze_coverage_conditioned.py

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# =============================================================================
# Basic utilities
# =============================================================================


def safe_div(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def mean(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return float(
        statistics.mean(values)
    )


def pvariance(
    values: list[float],
) -> float:
    if len(values) <= 1:
        return 0.0

    return float(
        statistics.pvariance(values)
    )


# =============================================================================
# I/O
# =============================================================================


def load_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input JSONL not found: {path}"
        )

    records: list[
        dict[str, Any]
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSONL record: "
                    f"line={line_number}"
                ) from exc

            if not isinstance(
                record,
                dict,
            ):
                raise TypeError(
                    "Each JSONL line must be a dict."
                )

            records.append(
                record
            )

    if not records:
        raise RuntimeError(
            "No records loaded."
        )

    return records


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

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
        writer.writerows(rows)


# =============================================================================
# Grouping / validation
# =============================================================================


def group_by_problem(
    records: list[dict[str, Any]],
) -> dict[
    str,
    list[dict[str, Any]],
]:
    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        problem_id = str(
            record["problem_id"]
        )

        grouped[
            problem_id
        ].append(
            record
        )

    for problem_id in grouped:
        grouped[
            problem_id
        ] = sorted(
            grouped[
                problem_id
            ],
            key=lambda item: int(
                item["plan_index"]
            ),
        )

    return dict(grouped)


def validate_problem_group(
    problem_id: str,
    plans: list[dict[str, Any]],
) -> tuple[int, int]:
    if not plans:
        raise ValueError(
            f"No plans for {problem_id}"
        )

    num_plans = len(
        plans
    )

    m_values = {
        int(
            plan["num_codes"]
        )
        for plan in plans
    }

    if len(m_values) != 1:
        raise ValueError(
            "Inconsistent M within problem: "
            f"{problem_id}, "
            f"values={sorted(m_values)}"
        )

    num_codes_per_plan = next(
        iter(m_values)
    )

    return (
        num_plans,
        num_codes_per_plan,
    )


# =============================================================================
# Coverage classification
# =============================================================================


def classify_coverage(
    best_q_binary: float,
) -> str:

    if best_q_binary <= 0.0:
        return "NO_COVERAGE"

    if best_q_binary < 0.75:
        return "UNSTABLE_COVERAGE"

    return "STABLE_COVERAGE"


# =============================================================================
# Per-problem analysis
# =============================================================================


def analyze_problem(
    problem_id: str,
    plans: list[dict[str, Any]],
) -> dict[str, Any]:

    (
        num_plans,
        num_codes_per_plan,
    ) = validate_problem_group(
        problem_id,
        plans,
    )

    q_binary = [
        float(
            plan[
                "plan_success_rate"
            ]
        )
        for plan in plans
    ]

    q_tpr = [
        float(
            plan[
                "mean_test_pass_ratio"
            ]
        )
        for plan in plans
    ]

    passed_counts = [
        int(
            plan[
                "num_passed_codes"
            ]
        )
        for plan in plans
    ]

    mean_binary = mean(
        q_binary
    )

    best_binary = max(
        q_binary
    )

    mean_tpr = mean(
        q_tpr
    )

    best_tpr = max(
        q_tpr
    )

    between_binary = pvariance(
        q_binary
    )

    between_tpr = pvariance(
        q_tpr
    )

    # Empirical Bernoulli variance within each fixed plan.
    #
    # q_i(1-q_i)
    within_binary_per_plan = [
        q * (1.0 - q)
        for q in q_binary
    ]

    mean_within_binary = mean(
        within_binary_per_plan
    )

    total_descriptive_variance = (
        between_binary
        + mean_within_binary
    )

    between_share = safe_div(
        between_binary,
        total_descriptive_variance,
    )

    within_share = safe_div(
        mean_within_binary,
        total_descriptive_variance,
    )

    coverage_group = classify_coverage(
        best_binary
    )

    # -------------------------------------------------------------------------
    # Binary reward information loss
    # -------------------------------------------------------------------------

    binary_zero_indices = [
        index
        for index, q
        in enumerate(q_binary)
        if q == 0.0
    ]

    binary_zero_count = len(
        binary_zero_indices
    )

    binary_zero_tpr_positive = sum(
        1
        for index
        in binary_zero_indices
        if q_tpr[index] > 0.0
    )

    binary_zero_tpr_ge_025 = sum(
        1
        for index
        in binary_zero_indices
        if q_tpr[index] >= 0.25
    )

    binary_zero_tpr_ge_050 = sum(
        1
        for index
        in binary_zero_indices
        if q_tpr[index] >= 0.50
    )

    # -------------------------------------------------------------------------
    # Rank disagreement proxy
    #
    # How often does the TPR-best plan differ from the binary-best plan?
    # For ties we use first max only, so this is only descriptive.
    # -------------------------------------------------------------------------

    best_binary_index = int(
        max(
            range(num_plans),
            key=lambda i: q_binary[i],
        )
    )

    best_tpr_index = int(
        max(
            range(num_plans),
            key=lambda i: q_tpr[i],
        )
    )

    return {
        "problem_id": str(
            problem_id
        ),

        "coverage_group": (
            coverage_group
        ),

        "num_plans": int(
            num_plans
        ),

        "num_codes_per_plan": int(
            num_codes_per_plan
        ),

        "mean_q_binary": float(
            mean_binary
        ),

        "best_q_binary": float(
            best_binary
        ),

        "best_mean_gap_binary": float(
            best_binary
            - mean_binary
        ),

        "mean_q_tpr": float(
            mean_tpr
        ),

        "best_q_tpr": float(
            best_tpr
        ),

        "best_mean_gap_tpr": float(
            best_tpr
            - mean_tpr
        ),

        "between_plan_var_binary": float(
            between_binary
        ),

        "within_plan_coder_var_binary": float(
            mean_within_binary
        ),

        "between_share_binary": float(
            between_share
        ),

        "within_share_binary": float(
            within_share
        ),

        "between_plan_var_tpr": float(
            between_tpr
        ),

        "num_usable_plans": int(
            sum(
                1
                for q in q_binary
                if q > 0.0
            )
        ),

        "num_stable_050_plans": int(
            sum(
                1
                for q in q_binary
                if q >= 0.50
            )
        ),

        "num_stable_075_plans": int(
            sum(
                1
                for q in q_binary
                if q >= 0.75
            )
        ),

        "num_perfect_plans": int(
            sum(
                1
                for q in q_binary
                if q >= 1.0
            )
        ),

        # ---------------------------------------------------------------------
        # Binary-zero / TPR-positive analysis
        # ---------------------------------------------------------------------

        "num_binary_zero_plans": int(
            binary_zero_count
        ),

        "num_binary_zero_tpr_positive": int(
            binary_zero_tpr_positive
        ),

        "binary_zero_tpr_positive_rate": float(
            safe_div(
                binary_zero_tpr_positive,
                binary_zero_count,
            )
        ),

        "num_binary_zero_tpr_ge_025": int(
            binary_zero_tpr_ge_025
        ),

        "binary_zero_tpr_ge_025_rate": float(
            safe_div(
                binary_zero_tpr_ge_025,
                binary_zero_count,
            )
        ),

        "num_binary_zero_tpr_ge_050": int(
            binary_zero_tpr_ge_050
        ),

        "binary_zero_tpr_ge_050_rate": float(
            safe_div(
                binary_zero_tpr_ge_050,
                binary_zero_count,
            )
        ),

        "best_binary_plan_index": int(
            best_binary_index
        ),

        "best_tpr_plan_index": int(
            best_tpr_index
        ),

        "best_plan_index_disagreement": int(
            best_binary_index
            != best_tpr_index
        ),

        "binary_profile": " | ".join(
            f"{count}/{num_codes_per_plan}"
            for count in passed_counts
        ),

        "q_binary_profile": " | ".join(
            f"{q:.4f}"
            for q in q_binary
        ),

        "q_tpr_profile": " | ".join(
            f"{q:.4f}"
            for q in q_tpr
        ),
    }


# =============================================================================
# Group-level aggregation
# =============================================================================


def summarize_group(
    group_name: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:

    if not rows:
        return {
            "group": group_name,
            "num_problems": 0,
        }

    num_problems = len(
        rows
    )

    return {
        "group": group_name,

        "num_problems": int(
            num_problems
        ),

        "mean_q_binary": mean(
            [
                float(
                    row[
                        "mean_q_binary"
                    ]
                )
                for row in rows
            ]
        ),

        "mean_best_q_binary": mean(
            [
                float(
                    row[
                        "best_q_binary"
                    ]
                )
                for row in rows
            ]
        ),

        "mean_best_mean_gap_binary": mean(
            [
                float(
                    row[
                        "best_mean_gap_binary"
                    ]
                )
                for row in rows
            ]
        ),

        "mean_q_tpr": mean(
            [
                float(
                    row[
                        "mean_q_tpr"
                    ]
                )
                for row in rows
            ]
        ),

        "mean_best_q_tpr": mean(
            [
                float(
                    row[
                        "best_q_tpr"
                    ]
                )
                for row in rows
            ]
        ),

        "mean_best_mean_gap_tpr": mean(
            [
                float(
                    row[
                        "best_mean_gap_tpr"
                    ]
                )
                for row in rows
            ]
        ),

        "mean_between_plan_var_binary": mean(
            [
                float(
                    row[
                        "between_plan_var_binary"
                    ]
                )
                for row in rows
            ]
        ),

        "mean_within_plan_coder_var_binary": mean(
            [
                float(
                    row[
                        "within_plan_coder_var_binary"
                    ]
                )
                for row in rows
            ]
        ),

        # Recompute the descriptive group-level shares from
        # group-mean variance terms.
        "between_share_binary": safe_div(
            mean(
                [
                    float(
                        row[
                            "between_plan_var_binary"
                        ]
                    )
                    for row in rows
                ]
            ),
            (
                mean(
                    [
                        float(
                            row[
                                "between_plan_var_binary"
                            ]
                        )
                        for row in rows
                    ]
                )
                +
                mean(
                    [
                        float(
                            row[
                                "within_plan_coder_var_binary"
                            ]
                        )
                        for row in rows
                    ]
                )
            ),
        ),

        "within_share_binary": safe_div(
            mean(
                [
                    float(
                        row[
                            "within_plan_coder_var_binary"
                        ]
                    )
                    for row in rows
                ]
            ),
            (
                mean(
                    [
                        float(
                            row[
                                "between_plan_var_binary"
                            ]
                        )
                        for row in rows
                    ]
                )
                +
                mean(
                    [
                        float(
                            row[
                                "within_plan_coder_var_binary"
                            ]
                        )
                        for row in rows
                    ]
                )
            ),
        ),

        "mean_between_plan_var_tpr": mean(
            [
                float(
                    row[
                        "between_plan_var_tpr"
                    ]
                )
                for row in rows
            ]
        ),

        "mean_num_usable_plans": mean(
            [
                float(
                    row[
                        "num_usable_plans"
                    ]
                )
                for row in rows
            ]
        ),

        "mean_num_stable_075_plans": mean(
            [
                float(
                    row[
                        "num_stable_075_plans"
                    ]
                )
                for row in rows
            ]
        ),

        "binary_zero_tpr_positive_rate": safe_div(
            sum(
                int(
                    row[
                        "num_binary_zero_tpr_positive"
                    ]
                )
                for row in rows
            ),
            sum(
                int(
                    row[
                        "num_binary_zero_plans"
                    ]
                )
                for row in rows
            ),
        ),

        "binary_zero_tpr_ge_025_rate": safe_div(
            sum(
                int(
                    row[
                        "num_binary_zero_tpr_ge_025"
                    ]
                )
                for row in rows
            ),
            sum(
                int(
                    row[
                        "num_binary_zero_plans"
                    ]
                )
                for row in rows
            ),
        ),

        "binary_zero_tpr_ge_050_rate": safe_div(
            sum(
                int(
                    row[
                        "num_binary_zero_tpr_ge_050"
                    ]
                )
                for row in rows
            ),
            sum(
                int(
                    row[
                        "num_binary_zero_plans"
                    ]
                )
                for row in rows
            ),
        ),

        "best_plan_index_disagreement_rate": mean(
            [
                float(
                    row[
                        "best_plan_index_disagreement"
                    ]
                )
                for row in rows
            ]
        ),
    }


# =============================================================================
# Global plan-level binary-vs-TPR analysis
# =============================================================================


def analyze_plan_reward_resolution(
    records: list[dict[str, Any]],
) -> dict[str, Any]:

    total_plans = len(
        records
    )

    binary_zero_records = [
        record
        for record in records
        if float(
            record[
                "plan_success_rate"
            ]
        ) == 0.0
    ]

    binary_zero_count = len(
        binary_zero_records
    )

    binary_zero_tpr_positive = [
        record
        for record
        in binary_zero_records
        if float(
            record[
                "mean_test_pass_ratio"
            ]
        ) > 0.0
    ]

    binary_zero_tpr_ge_025 = [
        record
        for record
        in binary_zero_records
        if float(
            record[
                "mean_test_pass_ratio"
            ]
        ) >= 0.25
    ]

    binary_zero_tpr_ge_050 = [
        record
        for record
        in binary_zero_records
        if float(
            record[
                "mean_test_pass_ratio"
            ]
        ) >= 0.50
    ]

    positive_binary_records = [
        record
        for record in records
        if float(
            record[
                "plan_success_rate"
            ]
        ) > 0.0
    ]

    return {
        "total_plans": int(
            total_plans
        ),

        "binary_zero_plans": int(
            binary_zero_count
        ),

        "binary_zero_plan_rate": float(
            safe_div(
                binary_zero_count,
                total_plans,
            )
        ),

        "binary_zero_but_tpr_positive": int(
            len(
                binary_zero_tpr_positive
            )
        ),

        "binary_zero_but_tpr_positive_rate_within_binary_zero": float(
            safe_div(
                len(
                    binary_zero_tpr_positive
                ),
                binary_zero_count,
            )
        ),

        "binary_zero_but_tpr_ge_025": int(
            len(
                binary_zero_tpr_ge_025
            )
        ),

        "binary_zero_but_tpr_ge_025_rate_within_binary_zero": float(
            safe_div(
                len(
                    binary_zero_tpr_ge_025
                ),
                binary_zero_count,
            )
        ),

        "binary_zero_but_tpr_ge_050": int(
            len(
                binary_zero_tpr_ge_050
            )
        ),

        "binary_zero_but_tpr_ge_050_rate_within_binary_zero": float(
            safe_div(
                len(
                    binary_zero_tpr_ge_050
                ),
                binary_zero_count,
            )
        ),

        "binary_positive_plans": int(
            len(
                positive_binary_records
            )
        ),

        "binary_positive_plan_rate": float(
            safe_div(
                len(
                    positive_binary_records
                ),
                total_plans,
            )
        ),

        "mean_tpr_among_binary_zero_plans": mean(
            [
                float(
                    record[
                        "mean_test_pass_ratio"
                    ]
                )
                for record
                in binary_zero_records
            ]
        ),
    }


# =============================================================================
# Code status analysis
# =============================================================================


def analyze_code_statuses(
    records: list[dict[str, Any]],
) -> dict[str, Any]:

    counter: Counter[
        str
    ] = Counter()

    total_codes = 0

    for record in records:
        for code in record[
            "codes"
        ]:
            total_codes += 1

            status = str(
                code.get(
                    "status",
                    "UNKNOWN",
                )
            )

            counter[
                status
            ] += 1

    return {
        "total_codes": int(
            total_codes
        ),

        "status_distribution": {
            status: {
                "count": int(
                    count
                ),
                "ratio": float(
                    safe_div(
                        count,
                        total_codes,
                    )
                ),
            }
            for status, count
            in counter.most_common()
        },
    }


# =============================================================================
# Console report
# =============================================================================


def print_group_summary(
    summary: dict[str, Any],
) -> None:

    print()
    print(
        "-" * 100
    )

    print(
        summary[
            "group"
        ]
    )

    print(
        "-" * 100
    )

    if int(
        summary[
            "num_problems"
        ]
    ) == 0:
        print(
            "No problems in group."
        )
        return

    print(
        f"Problems                     : "
        f"{summary['num_problems']}"
    )

    print(
        f"Mean q_binary                : "
        f"{summary['mean_q_binary']:.4f}"
    )

    print(
        f"Mean best q_binary           : "
        f"{summary['mean_best_q_binary']:.4f}"
    )

    print(
        f"Mean best-mean gap binary    : "
        f"{summary['mean_best_mean_gap_binary']:.4f}"
    )

    print(
        f"Mean q_TPR                   : "
        f"{summary['mean_q_tpr']:.4f}"
    )

    print(
        f"Mean best q_TPR              : "
        f"{summary['mean_best_q_tpr']:.4f}"
    )

    print(
        f"Mean best-mean gap TPR       : "
        f"{summary['mean_best_mean_gap_tpr']:.4f}"
    )

    print(
        f"Between-plan variance        : "
        f"{summary['mean_between_plan_var_binary']:.6f}"
    )

    print(
        f"Within-plan coder variance   : "
        f"{summary['mean_within_plan_coder_var_binary']:.6f}"
    )

    print(
        f"Between share                : "
        f"{summary['between_share_binary']:.4f}"
    )

    print(
        f"Within share                 : "
        f"{summary['within_share_binary']:.4f}"
    )

    print(
        f"Mean usable plans            : "
        f"{summary['mean_num_usable_plans']:.4f}"
    )

    print(
        f"Mean stable>=.75 plans       : "
        f"{summary['mean_num_stable_075_plans']:.4f}"
    )

    print(
        "Binary=0 but TPR>0          : "
        f"{summary['binary_zero_tpr_positive_rate']:.4f}"
    )

    print(
        "Binary=0 but TPR>=.25       : "
        f"{summary['binary_zero_tpr_ge_025_rate']:.4f}"
    )

    print(
        "Binary=0 but TPR>=.50       : "
        f"{summary['binary_zero_tpr_ge_050_rate']:.4f}"
    )

    print(
        "Best binary/TPR disagreement: "
        f"{summary['best_plan_index_disagreement_rate']:.4f}"
    )


# =============================================================================
# Main
# =============================================================================


def run_analysis(
    input_path: str | Path,
    output_dir: str | Path,
) -> None:

    input_path = Path(
        input_path
    )

    output_dir = Path(
        output_dir
    )

    records = load_jsonl(
        input_path
    )

    grouped = group_by_problem(
        records
    )

    problem_rows: list[
        dict[str, Any]
    ] = []

    for problem_id, plans in grouped.items():
        problem_rows.append(
            analyze_problem(
                problem_id,
                plans,
            )
        )

    problem_rows = sorted(
        problem_rows,
        key=lambda row: str(
            row[
                "problem_id"
            ]
        ),
    )

    groups: dict[
        str,
        list[dict[str, Any]],
    ] = {
        "ALL": (
            problem_rows
        ),

        "NO_COVERAGE": [
            row
            for row in problem_rows
            if (
                row[
                    "coverage_group"
                ]
                == "NO_COVERAGE"
            )
        ],

        "ANY_COVERAGE": [
            row
            for row in problem_rows
            if (
                row[
                    "coverage_group"
                ]
                != "NO_COVERAGE"
            )
        ],

        "UNSTABLE_COVERAGE": [
            row
            for row in problem_rows
            if (
                row[
                    "coverage_group"
                ]
                == "UNSTABLE_COVERAGE"
            )
        ],

        "STABLE_COVERAGE": [
            row
            for row in problem_rows
            if (
                row[
                    "coverage_group"
                ]
                == "STABLE_COVERAGE"
            )
        ],
    }

    group_summaries = {
        group_name: summarize_group(
            group_name,
            rows,
        )
        for group_name, rows
        in groups.items()
    }

    reward_resolution = (
        analyze_plan_reward_resolution(
            records
        )
    )

    code_status_summary = (
        analyze_code_statuses(
            records
        )
    )

    coverage_counts = Counter(
        row[
            "coverage_group"
        ]
        for row in problem_rows
    )

    final_summary = {
        "input_path": str(
            input_path
        ),

        "num_problems": int(
            len(
                problem_rows
            )
        ),

        "num_plan_records": int(
            len(
                records
            )
        ),

        "coverage_distribution": {
            key: {
                "count": int(
                    value
                ),
                "rate": float(
                    safe_div(
                        value,
                        len(
                            problem_rows
                        ),
                    )
                ),
            }
            for key, value
            in coverage_counts.items()
        },

        "group_summaries": (
            group_summaries
        ),

        "reward_resolution": (
            reward_resolution
        ),

        "code_status_summary": (
            code_status_summary
        ),
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        output_dir
        / "coverage_conditioned_summary.json",
        final_summary,
    )

    write_csv(
        output_dir
        / "coverage_conditioned_problem_metrics.csv",
        problem_rows,
    )

    group_rows = [
        summary
        for summary
        in group_summaries.values()
    ]

    write_csv(
        output_dir
        / "coverage_conditioned_group_metrics.csv",
        group_rows,
    )

    print()
    print(
        "=" * 100
    )
    print(
        "Coverage-conditioned N x M analysis"
    )
    print(
        "=" * 100
    )

    print(
        f"Problems     : "
        f"{len(problem_rows)}"
    )

    print(
        f"Plan records : "
        f"{len(records)}"
    )

    print()
    print(
        "[Coverage distribution]"
    )

    for group_name in (
        "NO_COVERAGE",
        "UNSTABLE_COVERAGE",
        "STABLE_COVERAGE",
    ):
        count = int(
            coverage_counts.get(
                group_name,
                0,
            )
        )

        print(
            f"{group_name:<22} "
            f"{count:>4}/"
            f"{len(problem_rows)} "
            f"("
            f"{safe_div(count, len(problem_rows)):.4f}"
            f")"
        )

    for group_name in (
        "ALL",
        "ANY_COVERAGE",
        "UNSTABLE_COVERAGE",
        "STABLE_COVERAGE",
        "NO_COVERAGE",
    ):
        print_group_summary(
            group_summaries[
                group_name
            ]
        )

    print()
    print(
        "=" * 100
    )
    print(
        "Binary reward resolution"
    )
    print(
        "=" * 100
    )

    print(
        f"Binary-zero plans          : "
        f"{reward_resolution['binary_zero_plans']}/"
        f"{reward_resolution['total_plans']} "
        f"("
        f"{reward_resolution['binary_zero_plan_rate']:.4f}"
        f")"
    )

    print(
        "Binary=0 but TPR>0        : "
        f"{reward_resolution['binary_zero_but_tpr_positive']} "
        f"("
        f"{reward_resolution['binary_zero_but_tpr_positive_rate_within_binary_zero']:.4f}"
        f" of binary-zero plans)"
    )

    print(
        "Binary=0 but TPR>=.25     : "
        f"{reward_resolution['binary_zero_but_tpr_ge_025']} "
        f"("
        f"{reward_resolution['binary_zero_but_tpr_ge_025_rate_within_binary_zero']:.4f}"
        f")"
    )

    print(
        "Binary=0 but TPR>=.50     : "
        f"{reward_resolution['binary_zero_but_tpr_ge_050']} "
        f"("
        f"{reward_resolution['binary_zero_but_tpr_ge_050_rate_within_binary_zero']:.4f}"
        f")"
    )

    print(
        "Mean TPR among binary-zero: "
        f"{reward_resolution['mean_tpr_among_binary_zero_plans']:.4f}"
    )

    print()
    print(
        "Saved:"
    )

    print(
        "  "
        + str(
            output_dir
            / "coverage_conditioned_summary.json"
        )
    )

    print(
        "  "
        + str(
            output_dir
            / "coverage_conditioned_problem_metrics.csv"
        )
    )

    print(
        "  "
        + str(
            output_dir
            / "coverage_conditioned_group_metrics.csv"
        )
    )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Coverage-conditioned analysis for "
            "Phase 4 N-plans x M-codes diagnostic."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to N x M diagnostic JSONL."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory for analysis outputs."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    run_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
    )