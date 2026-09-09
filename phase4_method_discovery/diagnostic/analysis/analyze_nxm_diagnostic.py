"""
PYTHONPATH="$HOME/workspace/project_sLM_planning" \
python \
  phase4_method_discovery/diagnostic/analysis/analyze_nxm_diagnostic.py \
  --input \
  phase4_method_discovery/diagnostic/outputs/pilot/base_n8_m4_100problems.jsonl \
  --output-dir \
  phase4_method_discovery/diagnostic/outputs/pilot/base_n8_m4_100problems_analysis
"""
# phase4_method_discovery/diagnostic/analysis/analyze_nxm_diagnostic.py

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# =============================================================================
# I/O
# =============================================================================


def load_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Diagnostic JSONL not found: {path}"
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
                    f"path={path}, "
                    f"line={line_number}"
                ) from exc

            if not isinstance(
                record,
                dict,
            ):
                raise TypeError(
                    "Each JSONL line must decode "
                    "to a dict."
                )

            records.append(
                record
            )

    if not records:
        raise RuntimeError(
            "No diagnostic records loaded."
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
        writer.writerows(
            rows
        )


# =============================================================================
# Validation
# =============================================================================


def validate_records(
    records: list[dict[str, Any]],
) -> None:
    required_fields = {
        "problem_id",
        "plan_index",
        "num_codes",
        "num_passed_codes",
        "plan_success_rate",
        "mean_test_pass_ratio",
        "codes",
    }

    for index, record in enumerate(
        records
    ):
        missing = (
            required_fields
            - set(record.keys())
        )

        if missing:
            raise ValueError(
                "Diagnostic record is missing "
                f"required fields at record {index}: "
                + ", ".join(
                    sorted(missing)
                )
            )

        if not isinstance(
            record["codes"],
            list,
        ):
            raise TypeError(
                "record['codes'] must be list: "
                f"record_index={index}"
            )

        num_codes = int(
            record["num_codes"]
        )

        if len(
            record["codes"]
        ) != num_codes:
            raise ValueError(
                "num_codes does not match "
                "codes length: "
                f"problem={record['problem_id']}, "
                f"plan_index={record['plan_index']}, "
                f"num_codes={num_codes}, "
                f"actual={len(record['codes'])}"
            )

        q_binary = float(
            record[
                "plan_success_rate"
            ]
        )

        q_tpr = float(
            record[
                "mean_test_pass_ratio"
            ]
        )

        if not (
            0.0
            <= q_binary
            <= 1.0
        ):
            raise ValueError(
                "plan_success_rate outside "
                "[0, 1]: "
                f"{q_binary}"
            )

        if not (
            0.0
            <= q_tpr
            <= 1.0
        ):
            raise ValueError(
                "mean_test_pass_ratio outside "
                "[0, 1]: "
                f"{q_tpr}"
            )


# =============================================================================
# Numeric helpers
# =============================================================================


def population_variance(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    if len(values) == 1:
        return 0.0

    return float(
        statistics.pvariance(
            values
        )
    )


def mean(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return float(
        statistics.mean(
            values
        )
    )


def safe_div(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


# =============================================================================
# Grouping
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
    ] = defaultdict(
        list
    )

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

    return dict(
        grouped
    )


# =============================================================================
# Plan-level analysis
# =============================================================================


def build_plan_rows(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[
        dict[str, Any]
    ] = []

    for record in records:
        codes = record[
            "codes"
        ]

        status_counter = Counter(
            str(
                code.get(
                    "status",
                    "",
                )
            )
            for code in codes
        )

        num_codes = int(
            record["num_codes"]
        )

        num_passed_codes = int(
            record[
                "num_passed_codes"
            ]
        )

        q_binary = float(
            record[
                "plan_success_rate"
            ]
        )

        q_tpr = float(
            record[
                "mean_test_pass_ratio"
            ]
        )

        row = {
            "problem_id": str(
                record[
                    "problem_id"
                ]
            ),
            "problem_index": int(
                record.get(
                    "problem_index",
                    -1,
                )
            ),
            "plan_index": int(
                record[
                    "plan_index"
                ]
            ),
            "num_codes": int(
                num_codes
            ),
            "num_passed_codes": int(
                num_passed_codes
            ),
            "q_binary": float(
                q_binary
            ),
            "q_tpr": float(
                q_tpr
            ),
            "has_any_pass": int(
                num_passed_codes
                > 0
            ),
            "stable_050": int(
                q_binary
                >= 0.50
            ),
            "stable_075": int(
                q_binary
                >= 0.75
            ),
            "stable_100": int(
                q_binary
                >= 1.00
            ),
            "num_generation_errors": int(
                record.get(
                    "num_generation_errors",
                    0,
                )
            ),
            "num_parsing_errors": int(
                record.get(
                    "num_parsing_errors",
                    0,
                )
            ),
            "num_evaluation_errors": int(
                record.get(
                    "num_evaluation_errors",
                    0,
                )
            ),
            "num_wrong_answer": int(
                status_counter.get(
                    "WRONG_ANSWER",
                    0,
                )
            ),
            "num_runtime_error": int(
                status_counter.get(
                    "RUNTIME_ERROR",
                    0,
                )
            ),
            "num_tle": int(
                status_counter.get(
                    "TIME_LIMIT_EXCEEDED",
                    0,
                )
            ),
            "num_pass_status": int(
                status_counter.get(
                    "PASS",
                    0,
                )
            ),
        }

        rows.append(
            row
        )

    return rows


# =============================================================================
# Problem-level analysis
# =============================================================================


def analyze_problem(
    problem_id: str,
    plans: list[dict[str, Any]],
) -> dict[str, Any]:
    q_binary_values = [
        float(
            plan[
                "plan_success_rate"
            ]
        )
        for plan in plans
    ]

    q_tpr_values = [
        float(
            plan[
                "mean_test_pass_ratio"
            ]
        )
        for plan in plans
    ]

    num_passed_codes = [
        int(
            plan[
                "num_passed_codes"
            ]
        )
        for plan in plans
    ]

    num_codes_values = [
        int(
            plan[
                "num_codes"
            ]
        )
        for plan in plans
    ]

    if len(
        set(
            num_codes_values
        )
    ) != 1:
        raise ValueError(
            "Inconsistent M within problem: "
            f"{problem_id}, "
            f"values={num_codes_values}"
        )

    num_plans = len(
        plans
    )

    num_codes_per_plan = (
        num_codes_values[0]
    )

    mean_q_binary = mean(
        q_binary_values
    )

    best_q_binary = max(
        q_binary_values
    )

    min_q_binary = min(
        q_binary_values
    )

    var_q_binary = (
        population_variance(
            q_binary_values
        )
    )

    mean_q_tpr = mean(
        q_tpr_values
    )

    best_q_tpr = max(
        q_tpr_values
    )

    min_q_tpr = min(
        q_tpr_values
    )

    var_q_tpr = (
        population_variance(
            q_tpr_values
        )
    )

    best_plan_index_binary = int(
        plans[
            q_binary_values.index(
                best_q_binary
            )
        ][
            "plan_index"
        ]
    )

    best_plan_index_tpr = int(
        plans[
            q_tpr_values.index(
                best_q_tpr
            )
        ][
            "plan_index"
        ]
    )

    num_plans_any_pass = sum(
        1
        for value
        in q_binary_values
        if value > 0.0
    )

    num_plans_ge_050 = sum(
        1
        for value
        in q_binary_values
        if value >= 0.50
    )

    num_plans_ge_075 = sum(
        1
        for value
        in q_binary_values
        if value >= 0.75
    )

    num_plans_eq_100 = sum(
        1
        for value
        in q_binary_values
        if math.isclose(
            value,
            1.0,
        )
    )

    total_passed_codes = sum(
        num_passed_codes
    )

    total_codes = (
        num_plans
        * num_codes_per_plan
    )

    any_pass_nxm = (
        total_passed_codes
        > 0
    )

    any_plan_usable = (
        num_plans_any_pass
        > 0
    )

    any_stable_050 = (
        num_plans_ge_050
        > 0
    )

    any_stable_075 = (
        num_plans_ge_075
        > 0
    )

    any_stable_100 = (
        num_plans_eq_100
        > 0
    )

    return {
        "problem_id": str(
            problem_id
        ),
        "num_plans": int(
            num_plans
        ),
        "num_codes_per_plan": int(
            num_codes_per_plan
        ),
        "total_codes": int(
            total_codes
        ),
        "total_passed_codes": int(
            total_passed_codes
        ),
        "overall_code_pass_rate": float(
            safe_div(
                total_passed_codes,
                total_codes,
            )
        ),

        # -------------------------------------------------------------
        # Binary plan utility
        # -------------------------------------------------------------
        "mean_q_binary": float(
            mean_q_binary
        ),
        "best_q_binary": float(
            best_q_binary
        ),
        "min_q_binary": float(
            min_q_binary
        ),
        "best_mean_gap_binary": float(
            best_q_binary
            - mean_q_binary
        ),
        "between_plan_var_binary": float(
            var_q_binary
        ),
        "best_plan_index_binary": int(
            best_plan_index_binary
        ),

        # -------------------------------------------------------------
        # TPR plan utility
        # -------------------------------------------------------------
        "mean_q_tpr": float(
            mean_q_tpr
        ),
        "best_q_tpr": float(
            best_q_tpr
        ),
        "min_q_tpr": float(
            min_q_tpr
        ),
        "best_mean_gap_tpr": float(
            best_q_tpr
            - mean_q_tpr
        ),
        "between_plan_var_tpr": float(
            var_q_tpr
        ),
        "best_plan_index_tpr": int(
            best_plan_index_tpr
        ),

        # -------------------------------------------------------------
        # Coverage / stable plan counts
        # -------------------------------------------------------------
        "num_plans_any_pass": int(
            num_plans_any_pass
        ),
        "num_plans_ge_050": int(
            num_plans_ge_050
        ),
        "num_plans_ge_075": int(
            num_plans_ge_075
        ),
        "num_plans_eq_100": int(
            num_plans_eq_100
        ),

        "any_pass_nxm": int(
            any_pass_nxm
        ),
        "any_plan_usable": int(
            any_plan_usable
        ),
        "any_stable_050": int(
            any_stable_050
        ),
        "any_stable_075": int(
            any_stable_075
        ),
        "any_stable_100": int(
            any_stable_100
        ),

        # -------------------------------------------------------------
        # Plan profile
        #
        # Example:
        #   0/4 | 0/4 | 3/4 | ...
        # -------------------------------------------------------------
        "binary_profile": " | ".join(
            f"{passed}/{num_codes_per_plan}"
            for passed
            in num_passed_codes
        ),

        "q_binary_profile": " | ".join(
            f"{value:.4f}"
            for value
            in q_binary_values
        ),

        "q_tpr_profile": " | ".join(
            f"{value:.4f}"
            for value
            in q_tpr_values
        ),
    }


def build_problem_rows(
    grouped: dict[
        str,
        list[dict[str, Any]],
    ],
) -> list[dict[str, Any]]:
    rows: list[
        dict[str, Any]
    ] = []

    for problem_id, plans in grouped.items():
        rows.append(
            analyze_problem(
                problem_id,
                plans,
            )
        )

    return rows


# =============================================================================
# Within-plan coder stochasticity
# =============================================================================


def compute_within_plan_binary_variance(
    record: dict[str, Any],
) -> float:
    """
    Binary code rollout variance within one fixed plan.

    For Bernoulli outcomes y_j in {0,1}, population variance is:

        q * (1 - q)

    where q is the empirical plan success rate.

    This quantity captures coder realization uncertainty under a fixed plan.
    """

    q = float(
        record[
            "plan_success_rate"
        ]
    )

    return float(
        q
        * (
            1.0
            - q
        )
    )


def analyze_variance_decomposition(
    records: list[
        dict[str, Any]
    ],
    grouped: dict[
        str,
        list[dict[str, Any]],
    ],
) -> dict[str, Any]:
    """
    Descriptive variance decomposition.

    between-plan variance:
        variation of q_i across plans for the same problem.

    within-plan coder variance:
        Bernoulli realization variance q_i(1-q_i), averaged across plans.

    This is not presented as a formal ANOVA estimator.
    It is a diagnostic comparison of two uncertainty sources.
    """

    problem_between_binary: list[
        float
    ] = []

    problem_within_binary: list[
        float
    ] = []

    problem_between_tpr: list[
        float
    ] = []

    for _, plans in grouped.items():
        q_binary_values = [
            float(
                plan[
                    "plan_success_rate"
                ]
            )
            for plan in plans
        ]

        q_tpr_values = [
            float(
                plan[
                    "mean_test_pass_ratio"
                ]
            )
            for plan in plans
        ]

        between_binary = (
            population_variance(
                q_binary_values
            )
        )

        between_tpr = (
            population_variance(
                q_tpr_values
            )
        )

        within_binary = mean(
            [
                compute_within_plan_binary_variance(
                    plan
                )
                for plan
                in plans
            ]
        )

        problem_between_binary.append(
            between_binary
        )

        problem_between_tpr.append(
            between_tpr
        )

        problem_within_binary.append(
            within_binary
        )

    mean_between_binary = mean(
        problem_between_binary
    )

    mean_within_binary = mean(
        problem_within_binary
    )

    total_descriptive_binary = (
        mean_between_binary
        + mean_within_binary
    )

    return {
        "mean_between_plan_variance_binary": float(
            mean_between_binary
        ),
        "mean_within_plan_coder_variance_binary": float(
            mean_within_binary
        ),
        "between_share_binary": float(
            safe_div(
                mean_between_binary,
                total_descriptive_binary,
            )
        ),
        "within_share_binary": float(
            safe_div(
                mean_within_binary,
                total_descriptive_binary,
            )
        ),
        "mean_between_plan_variance_tpr": float(
            mean(
                problem_between_tpr
            )
        ),
    }


# =============================================================================
# Plan success histogram
# =============================================================================


def build_plan_success_histogram(
    records: list[
        dict[str, Any]
    ],
) -> dict[str, int]:
    """
    For M=4, produces bins such as:

        0/4
        1/4
        2/4
        3/4
        4/4
    """

    counter: Counter[
        str
    ] = Counter()

    for record in records:
        num_codes = int(
            record[
                "num_codes"
            ]
        )

        passed = int(
            record[
                "num_passed_codes"
            ]
        )

        key = (
            f"{passed}/"
            f"{num_codes}"
        )

        counter[
            key
        ] += 1

    # Sort numerically by numerator, denominator.
    def sort_key(
        item: tuple[
            str,
            int,
        ],
    ) -> tuple[int, int]:
        label = item[0]

        numerator_text, denominator_text = (
            label.split("/")
        )

        return (
            int(
                denominator_text
            ),
            int(
                numerator_text
            ),
        )

    return {
        key: int(
            value
        )
        for key, value
        in sorted(
            counter.items(),
            key=sort_key,
        )
    }


# =============================================================================
# Failure status analysis
# =============================================================================


def analyze_code_statuses(
    records: list[
        dict[str, Any]
    ],
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

            counter[
                str(
                    code.get(
                        "status",
                        "UNKNOWN",
                    )
                )
            ] += 1

    distribution = {
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
    }

    return {
        "total_codes": int(
            total_codes
        ),
        "status_distribution": (
            distribution
        ),
    }


# =============================================================================
# Global summary
# =============================================================================


def build_global_summary(
    records: list[
        dict[str, Any]
    ],
    problem_rows: list[
        dict[str, Any]
    ],
    variance_summary: dict[
        str,
        Any,
    ],
) -> dict[str, Any]:
    num_plans = len(
        records
    )

    num_problems = len(
        problem_rows
    )

    total_codes = sum(
        int(
            record[
                "num_codes"
            ]
        )
        for record in records
    )

    total_passed_codes = sum(
        int(
            record[
                "num_passed_codes"
            ]
        )
        for record in records
    )

    q_binary_all = [
        float(
            record[
                "plan_success_rate"
            ]
        )
        for record in records
    ]

    q_tpr_all = [
        float(
            record[
                "mean_test_pass_ratio"
            ]
        )
        for record in records
    ]

    problems_any_pass = sum(
        int(
            row[
                "any_pass_nxm"
            ]
        )
        for row in problem_rows
    )

    problems_stable_050 = sum(
        int(
            row[
                "any_stable_050"
            ]
        )
        for row in problem_rows
    )

    problems_stable_075 = sum(
        int(
            row[
                "any_stable_075"
            ]
        )
        for row in problem_rows
    )

    problems_stable_100 = sum(
        int(
            row[
                "any_stable_100"
            ]
        )
        for row in problem_rows
    )

    mean_best_binary = mean(
        [
            float(
                row[
                    "best_q_binary"
                ]
            )
            for row in problem_rows
        ]
    )

    mean_problem_binary = mean(
        [
            float(
                row[
                    "mean_q_binary"
                ]
            )
            for row in problem_rows
        ]
    )

    mean_best_tpr = mean(
        [
            float(
                row[
                    "best_q_tpr"
                ]
            )
            for row in problem_rows
        ]
    )

    mean_problem_tpr = mean(
        [
            float(
                row[
                    "mean_q_tpr"
                ]
            )
            for row in problem_rows
        ]
    )

    return {
        "num_problems": int(
            num_problems
        ),
        "num_plan_records": int(
            num_plans
        ),
        "num_code_records": int(
            total_codes
        ),

        # -------------------------------------------------------------
        # Overall code-level success
        # -------------------------------------------------------------
        "total_passed_codes": int(
            total_passed_codes
        ),
        "overall_code_pass_rate": float(
            safe_div(
                total_passed_codes,
                total_codes,
            )
        ),

        # -------------------------------------------------------------
        # Plan-level averages
        # -------------------------------------------------------------
        "mean_plan_q_binary": float(
            mean(
                q_binary_all
            )
        ),
        "mean_plan_q_tpr": float(
            mean(
                q_tpr_all
            )
        ),

        # -------------------------------------------------------------
        # Problem-level oracle / headroom
        # -------------------------------------------------------------
        "mean_problem_q_binary": float(
            mean_problem_binary
        ),
        "mean_best_plan_q_binary": float(
            mean_best_binary
        ),
        "mean_best_mean_gap_binary": float(
            mean_best_binary
            - mean_problem_binary
        ),

        "mean_problem_q_tpr": float(
            mean_problem_tpr
        ),
        "mean_best_plan_q_tpr": float(
            mean_best_tpr
        ),
        "mean_best_mean_gap_tpr": float(
            mean_best_tpr
            - mean_problem_tpr
        ),

        # -------------------------------------------------------------
        # Coverage
        # -------------------------------------------------------------
        "problems_any_pass": int(
            problems_any_pass
        ),
        "problem_any_pass_rate": float(
            safe_div(
                problems_any_pass,
                num_problems,
            )
        ),

        "problems_with_stable_plan_050": int(
            problems_stable_050
        ),
        "problem_stable_plan_050_rate": float(
            safe_div(
                problems_stable_050,
                num_problems,
            )
        ),

        "problems_with_stable_plan_075": int(
            problems_stable_075
        ),
        "problem_stable_plan_075_rate": float(
            safe_div(
                problems_stable_075,
                num_problems,
            )
        ),

        "problems_with_stable_plan_100": int(
            problems_stable_100
        ),
        "problem_stable_plan_100_rate": float(
            safe_div(
                problems_stable_100,
                num_problems,
            )
        ),

        "variance_decomposition": (
            variance_summary
        ),
    }


# =============================================================================
# Console report
# =============================================================================


def print_problem_profiles(
    problem_rows: list[
        dict[str, Any]
    ],
) -> None:
    print()
    print(
        "=" * 100
    )
    print(
        "Problem-level N x M profiles"
    )
    print(
        "=" * 100
    )

    for row in problem_rows:
        print()
        print(
            f"{row['problem_id']}"
        )

        print(
            "  binary profile : "
            f"{row['binary_profile']}"
        )

        print(
            "  q_binary       : "
            f"{row['q_binary_profile']}"
        )

        print(
            "  q_tpr          : "
            f"{row['q_tpr_profile']}"
        )

        print(
            "  mean/best bin  : "
            f"{row['mean_q_binary']:.4f} / "
            f"{row['best_q_binary']:.4f}"
        )

        print(
            "  best-mean gap  : "
            f"{row['best_mean_gap_binary']:.4f}"
        )

        print(
            "  between var    : "
            f"{row['between_plan_var_binary']:.6f}"
        )

        print(
            "  usable plans   : "
            f"{row['num_plans_any_pass']}/"
            f"{row['num_plans']}"
        )

        print(
            "  stable >=.50   : "
            f"{row['num_plans_ge_050']}"
        )

        print(
            "  stable >=.75   : "
            f"{row['num_plans_ge_075']}"
        )


def print_summary(
    summary: dict[str, Any],
    histogram: dict[str, int],
    status_summary: dict[str, Any],
) -> None:
    print()
    print(
        "=" * 100
    )
    print(
        "N x M Diagnostic Summary"
    )
    print(
        "=" * 100
    )

    print(
        f"Problems                : "
        f"{summary['num_problems']}"
    )

    print(
        f"Plan records            : "
        f"{summary['num_plan_records']}"
    )

    print(
        f"Code records            : "
        f"{summary['num_code_records']}"
    )

    print()
    print(
        "[Code-level]"
    )

    print(
        f"Passed codes             : "
        f"{summary['total_passed_codes']}"
    )

    print(
        f"Overall code pass rate   : "
        f"{summary['overall_code_pass_rate']:.4f}"
    )

    print()
    print(
        "[Plan-level]"
    )

    print(
        f"Mean q_binary            : "
        f"{summary['mean_plan_q_binary']:.4f}"
    )

    print(
        f"Mean q_TPR               : "
        f"{summary['mean_plan_q_tpr']:.4f}"
    )

    print()
    print(
        "[Problem-level / Oracle]"
    )

    print(
        f"Mean-plan q_binary       : "
        f"{summary['mean_problem_q_binary']:.4f}"
    )

    print(
        f"Best-plan q_binary       : "
        f"{summary['mean_best_plan_q_binary']:.4f}"
    )

    print(
        f"Best-Mean gap            : "
        f"{summary['mean_best_mean_gap_binary']:.4f}"
    )

    print(
        f"Mean-plan q_TPR          : "
        f"{summary['mean_problem_q_tpr']:.4f}"
    )

    print(
        f"Best-plan q_TPR          : "
        f"{summary['mean_best_plan_q_tpr']:.4f}"
    )

    print(
        f"Best-Mean TPR gap        : "
        f"{summary['mean_best_mean_gap_tpr']:.4f}"
    )

    print()
    print(
        "[Coverage]"
    )

    print(
        "Any-pass problems        : "
        f"{summary['problems_any_pass']}/"
        f"{summary['num_problems']} "
        f"("
        f"{summary['problem_any_pass_rate']:.4f}"
        f")"
    )

    print(
        "Stable >= 0.50 problems : "
        f"{summary['problems_with_stable_plan_050']}/"
        f"{summary['num_problems']} "
        f"("
        f"{summary['problem_stable_plan_050_rate']:.4f}"
        f")"
    )

    print(
        "Stable >= 0.75 problems : "
        f"{summary['problems_with_stable_plan_075']}/"
        f"{summary['num_problems']} "
        f"("
        f"{summary['problem_stable_plan_075_rate']:.4f}"
        f")"
    )

    print(
        "Stable = 1.00 problems  : "
        f"{summary['problems_with_stable_plan_100']}/"
        f"{summary['num_problems']} "
        f"("
        f"{summary['problem_stable_plan_100_rate']:.4f}"
        f")"
    )

    variance = (
        summary[
            "variance_decomposition"
        ]
    )

    print()
    print(
        "[Descriptive variance decomposition]"
    )

    print(
        "Between-plan var         : "
        f"{variance['mean_between_plan_variance_binary']:.6f}"
    )

    print(
        "Within-plan coder var    : "
        f"{variance['mean_within_plan_coder_variance_binary']:.6f}"
    )

    print(
        "Between share            : "
        f"{variance['between_share_binary']:.4f}"
    )

    print(
        "Within share             : "
        f"{variance['within_share_binary']:.4f}"
    )

    print()
    print(
        "[Plan success histogram]"
    )

    total_plans = sum(
        histogram.values()
    )

    for label, count in histogram.items():
        print(
            f"{label:>8} : "
            f"{count:>4} "
            f"("
            f"{safe_div(count, total_plans):.4f}"
            f")"
        )

    print()
    print(
        "[Code status distribution]"
    )

    for status, payload in (
        status_summary[
            "status_distribution"
        ].items()
    ):
        print(
            f"{status:<24} "
            f"{payload['count']:>5} "
            f"("
            f"{payload['ratio']:.4f}"
            f")"
        )

    print(
        "=" * 100
    )


# =============================================================================
# Main analysis
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

    validate_records(
        records
    )

    grouped = group_by_problem(
        records
    )

    plan_rows = build_plan_rows(
        records
    )

    problem_rows = build_problem_rows(
        grouped
    )

    variance_summary = (
        analyze_variance_decomposition(
            records,
            grouped,
        )
    )

    histogram = (
        build_plan_success_histogram(
            records
        )
    )

    status_summary = (
        analyze_code_statuses(
            records
        )
    )

    global_summary = (
        build_global_summary(
            records,
            problem_rows,
            variance_summary,
        )
    )

    full_summary = {
        "input_path": str(
            input_path
        ),
        "global_summary": (
            global_summary
        ),
        "plan_success_histogram": (
            histogram
        ),
        "code_status_summary": (
            status_summary
        ),
        "problem_results": (
            problem_rows
        ),
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir
        / "summary.json"
    )

    problem_csv_path = (
        output_dir
        / "problem_metrics.csv"
    )

    plan_csv_path = (
        output_dir
        / "plan_metrics.csv"
    )

    write_json(
        summary_path,
        full_summary,
    )

    write_csv(
        problem_csv_path,
        problem_rows,
    )

    write_csv(
        plan_csv_path,
        plan_rows,
    )

    print_problem_profiles(
        problem_rows
    )

    print_summary(
        global_summary,
        histogram,
        status_summary,
    )

    print()
    print(
        "Saved:"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {problem_csv_path}"
    )

    print(
        f"  {plan_csv_path}"
    )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Phase 4 "
            "N-plans x M-codes diagnostic."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to run_nxm_diagnostic "
            "JSONL output."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory for summary JSON "
            "and CSV metrics."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
    )