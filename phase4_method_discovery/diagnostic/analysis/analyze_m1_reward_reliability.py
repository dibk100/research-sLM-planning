"""
PYTHONPATH="$HOME/workspace/project_sLM_planning" \
python \
  phase4_method_discovery/diagnostic/analysis/analyze_m1_reward_reliability.py \
  --input \
  phase4_method_discovery/diagnostic/outputs/pilot/base_n8_m4_100problems.jsonl \
  --output-dir \
  phase4_method_discovery/diagnostic/outputs/pilot/base_n8_m4_100problems_analysis \
  --selection-trials 1000 \
  --simulation-seed 42
"""

# phase4_method_discovery/diagnostic/analysis/analyze_m1_reward_reliability.py

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


# =============================================================================
# Basic utilities
# =============================================================================


def mean(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return float(
        statistics.mean(values)
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


def mae(
    predictions: list[float],
    references: list[float],
) -> float:
    if len(predictions) != len(references):
        raise ValueError(
            "MAE input lengths differ."
        )

    if not predictions:
        return 0.0

    return mean(
        [
            abs(pred - ref)
            for pred, ref
            in zip(
                predictions,
                references,
                strict=True,
            )
        ]
    )


def rmse(
    predictions: list[float],
    references: list[float],
) -> float:
    if len(predictions) != len(references):
        raise ValueError(
            "RMSE input lengths differ."
        )

    if not predictions:
        return 0.0

    squared_error = mean(
        [
            (pred - ref) ** 2
            for pred, ref
            in zip(
                predictions,
                references,
                strict=True,
            )
        ]
    )

    return float(
        math.sqrt(
            squared_error
        )
    )


# =============================================================================
# Correlation helpers
# =============================================================================


def pearson_correlation(
    xs: list[float],
    ys: list[float],
) -> float:
    if len(xs) != len(ys):
        raise ValueError(
            "Pearson input lengths differ."
        )

    if len(xs) < 2:
        return 0.0

    mean_x = mean(xs)
    mean_y = mean(ys)

    numerator = sum(
        (x - mean_x)
        * (y - mean_y)
        for x, y
        in zip(
            xs,
            ys,
            strict=True,
        )
    )

    denom_x = math.sqrt(
        sum(
            (x - mean_x) ** 2
            for x in xs
        )
    )

    denom_y = math.sqrt(
        sum(
            (y - mean_y) ** 2
            for y in ys
        )
    )

    denominator = (
        denom_x
        * denom_y
    )

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def rankdata(
    values: list[float],
) -> list[float]:
    """
    Average-rank implementation with tie handling.

    Example:
        values = [10, 20, 20, 30]
        ranks  = [1, 2.5, 2.5, 4]
    """

    indexed = sorted(
        enumerate(values),
        key=lambda pair: pair[1],
    )

    ranks = [
        0.0
        for _ in values
    ]

    i = 0

    while i < len(indexed):
        j = i + 1

        while (
            j < len(indexed)
            and indexed[j][1]
            == indexed[i][1]
        ):
            j += 1

        # Ranks are 1-based.
        average_rank = (
            (i + 1)
            + j
        ) / 2.0

        for k in range(
            i,
            j,
        ):
            original_index = (
                indexed[k][0]
            )

            ranks[
                original_index
            ] = average_rank

        i = j

    return ranks


def spearman_correlation(
    xs: list[float],
    ys: list[float],
) -> float:
    if len(xs) != len(ys):
        raise ValueError(
            "Spearman input lengths differ."
        )

    if len(xs) < 2:
        return 0.0

    return pearson_correlation(
        rankdata(xs),
        rankdata(ys),
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
                    "Invalid JSONL: "
                    f"line={line_number}"
                ) from exc

            if not isinstance(
                record,
                dict,
            ):
                raise TypeError(
                    "Each JSONL line must "
                    "decode to dict."
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
        writer.writerows(
            rows
        )


# =============================================================================
# Data extraction
# =============================================================================


def get_binary_reward(
    code: dict[str, Any],
) -> float:
    """
    Prefer explicitly stored binary_reward.

    Fall back to passed if necessary.
    """

    if "binary_reward" in code:
        return float(
            code[
                "binary_reward"
            ]
        )

    return (
        1.0
        if bool(
            code.get(
                "passed",
                False,
            )
        )
        else 0.0
    )


def get_tpr_reward(
    code: dict[str, Any],
) -> float:
    return float(
        code.get(
            "test_pass_ratio",
            0.0,
        )
    )


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
        grouped[
            str(
                record[
                    "problem_id"
                ]
            )
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
                item[
                    "plan_index"
                ]
            ),
        )

    return dict(
        grouped
    )


def validate_records(
    records: list[dict[str, Any]],
) -> int:
    """
    Validate constant M and return M.
    """

    m_values: set[int] = set()

    for record in records:
        if "codes" not in record:
            raise KeyError(
                "Record missing codes."
            )

        codes = record[
            "codes"
        ]

        if not isinstance(
            codes,
            list,
        ):
            raise TypeError(
                "codes must be list."
            )

        m_values.add(
            len(codes)
        )

    if len(m_values) != 1:
        raise ValueError(
            "Diagnostic contains inconsistent M: "
            f"{sorted(m_values)}"
        )

    m = next(
        iter(
            m_values
        )
    )

    if m < 2:
        raise ValueError(
            "M must be >= 2 for "
            "M=1 reliability analysis."
        )

    return int(m)


# =============================================================================
# Plan-level M=1 vs M=4 analysis
# =============================================================================


def analyze_plan_reliability(
    records: list[dict[str, Any]],
    *,
    reward_type: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:

    if reward_type not in {
        "binary",
        "tpr",
    }:
        raise ValueError(
            f"Unknown reward type: {reward_type}"
        )

    m1_values: list[
        float
    ] = []

    q_full_values: list[
        float
    ] = []

    q_loo_values: list[
        float
    ] = []

    plan_rows: list[
        dict[str, Any]
    ] = []

    for record in records:
        codes = record[
            "codes"
        ]

        if reward_type == "binary":
            rewards = [
                get_binary_reward(
                    code
                )
                for code in codes
            ]
        else:
            rewards = [
                get_tpr_reward(
                    code
                )
                for code in codes
            ]

        m = len(
            rewards
        )

        q_full = mean(
            rewards
        )

        per_rollout_abs_error: list[
            float
        ] = []

        per_rollout_loo_abs_error: list[
            float
        ] = []

        for j, reward in enumerate(
            rewards
        ):
            other_rewards = [
                value
                for index, value
                in enumerate(rewards)
                if index != j
            ]

            q_loo = mean(
                other_rewards
            )

            m1_values.append(
                reward
            )

            q_full_values.append(
                q_full
            )

            q_loo_values.append(
                q_loo
            )

            per_rollout_abs_error.append(
                abs(
                    reward
                    - q_full
                )
            )

            per_rollout_loo_abs_error.append(
                abs(
                    reward
                    - q_loo
                )
            )

        plan_rows.append(
            {
                "problem_id": str(
                    record[
                        "problem_id"
                    ]
                ),
                "plan_index": int(
                    record[
                        "plan_index"
                    ]
                ),
                "reward_type": (
                    reward_type
                ),
                "M": int(
                    m
                ),
                "q_full": float(
                    q_full
                ),
                "reward_profile": (
                    " | ".join(
                        f"{value:.4f}"
                        for value
                        in rewards
                    )
                ),
                "mean_abs_error_m1_vs_m4": mean(
                    per_rollout_abs_error
                ),
                "mean_abs_error_m1_vs_loo": mean(
                    per_rollout_loo_abs_error
                ),
                "min_single_reward": float(
                    min(
                        rewards
                    )
                ),
                "max_single_reward": float(
                    max(
                        rewards
                    )
                ),
                "single_reward_range": float(
                    max(rewards)
                    - min(rewards)
                ),
            }
        )

    summary = {
        "reward_type": (
            reward_type
        ),

        "num_plans": int(
            len(records)
        ),

        "num_single_rollout_observations": int(
            len(m1_values)
        ),

        # -------------------------------------------------------------
        # Direct M1 -> M4 comparison.
        #
        # Note that the M1 observation is included inside q_full,
        # so these correlations are optimistic.
        # -------------------------------------------------------------

        "m1_vs_m4_mae": float(
            mae(
                m1_values,
                q_full_values,
            )
        ),

        "m1_vs_m4_rmse": float(
            rmse(
                m1_values,
                q_full_values,
            )
        ),

        "m1_vs_m4_pearson": float(
            pearson_correlation(
                m1_values,
                q_full_values,
            )
        ),

        "m1_vs_m4_spearman": float(
            spearman_correlation(
                m1_values,
                q_full_values,
            )
        ),

        # -------------------------------------------------------------
        # Leave-one-out.
        #
        # Single rollout j is compared against the mean of the OTHER
        # M-1 coder realizations.
        #
        # This avoids self-inclusion and is a more conservative
        # reliability diagnostic.
        # -------------------------------------------------------------

        "m1_vs_loo_mae": float(
            mae(
                m1_values,
                q_loo_values,
            )
        ),

        "m1_vs_loo_rmse": float(
            rmse(
                m1_values,
                q_loo_values,
            )
        ),

        "m1_vs_loo_pearson": float(
            pearson_correlation(
                m1_values,
                q_loo_values,
            )
        ),

        "m1_vs_loo_spearman": float(
            spearman_correlation(
                m1_values,
                q_loo_values,
            )
        ),
    }

    return (
        summary,
        plan_rows,
    )


# =============================================================================
# Pairwise reward inversion analysis
# =============================================================================


def analyze_pairwise_inversions(
    grouped: dict[
        str,
        list[dict[str, Any]],
    ],
    *,
    reward_type: str,
) -> dict[str, Any]:
    """
    For every pair of plans with different M=4 empirical utilities:

        Q_A > Q_B

    enumerate all M x M combinations of one sampled rollout from A
    and one sampled rollout from B.

    Measure whether M=1 says:

        A > B     correct
        A = B     tie
        A < B     inversion

    This directly quantifies how often one rollout reverses the
    ordering implied by M=4 empirical utility.
    """

    correct = 0
    ties = 0
    inversions = 0

    comparable_plan_pairs = 0
    single_rollout_pair_trials = 0

    for _, plans in grouped.items():
        reward_lists: list[
            list[float]
        ] = []

        q_values: list[
            float
        ] = []

        for plan in plans:
            if reward_type == "binary":
                rewards = [
                    get_binary_reward(
                        code
                    )
                    for code
                    in plan[
                        "codes"
                    ]
                ]
            else:
                rewards = [
                    get_tpr_reward(
                        code
                    )
                    for code
                    in plan[
                        "codes"
                    ]
                ]

            reward_lists.append(
                rewards
            )

            q_values.append(
                mean(
                    rewards
                )
            )

        n = len(
            plans
        )

        for i in range(n):
            for j in range(
                i + 1,
                n,
            ):
                q_i = (
                    q_values[i]
                )

                q_j = (
                    q_values[j]
                )

                if math.isclose(
                    q_i,
                    q_j,
                    abs_tol=1e-12,
                ):
                    continue

                comparable_plan_pairs += 1

                # Orient pair so A is the M=4-better plan.
                if q_i > q_j:
                    better_rewards = (
                        reward_lists[i]
                    )

                    worse_rewards = (
                        reward_lists[j]
                    )

                else:
                    better_rewards = (
                        reward_lists[j]
                    )

                    worse_rewards = (
                        reward_lists[i]
                    )

                for reward_better in (
                    better_rewards
                ):
                    for reward_worse in (
                        worse_rewards
                    ):
                        single_rollout_pair_trials += 1

                        if (
                            reward_better
                            > reward_worse
                        ):
                            correct += 1

                        elif (
                            reward_better
                            < reward_worse
                        ):
                            inversions += 1

                        else:
                            ties += 1

    return {
        "reward_type": reward_type,

        "comparable_plan_pairs": int(
            comparable_plan_pairs
        ),

        "single_rollout_pair_trials": int(
            single_rollout_pair_trials
        ),

        "correct_order_count": int(
            correct
        ),

        "tie_count": int(
            ties
        ),

        "inversion_count": int(
            inversions
        ),

        "correct_order_rate": float(
            safe_div(
                correct,
                single_rollout_pair_trials,
            )
        ),

        "tie_rate": float(
            safe_div(
                ties,
                single_rollout_pair_trials,
            )
        ),

        "inversion_rate": float(
            safe_div(
                inversions,
                single_rollout_pair_trials,
            )
        ),

        # Among trials where M1 actually expresses an ordering,
        # how often is that ordering wrong?
        "inversion_rate_excluding_ties": float(
            safe_div(
                inversions,
                correct
                + inversions,
            )
        ),
    }


# =============================================================================
# Best-plan selection simulation
# =============================================================================


def simulate_m1_plan_selection(
    grouped: dict[
        str,
        list[dict[str, Any]],
    ],
    *,
    reward_type: str,
    num_trials: int,
    seed: int,
) -> dict[str, Any]:
    """
    Simulate the following training/selection setting:

        - For each candidate plan, observe ONE coder rollout.
        - Rank/select plans using only that M=1 reward.
        - Compare the selected plan against M=4 empirical utility.

    We resample ONLY from the already observed M=4 code trajectories.
    No new model inference is performed.

    Ties among M1 top-scoring plans are broken uniformly at random.
    """

    rng = random.Random(
        seed
    )

    selected_q_values: list[
        float
    ] = []

    oracle_q_values: list[
        float
    ] = []

    regrets: list[
        float
    ] = []

    exact_best_hits = 0

    nonzero_oracle_problem_trials = 0
    false_no_signal_trials = 0

    total_trials = 0

    for _, plans in grouped.items():
        reward_lists: list[
            list[float]
        ] = []

        q_values: list[
            float
        ] = []

        for plan in plans:
            if reward_type == "binary":
                rewards = [
                    get_binary_reward(
                        code
                    )
                    for code
                    in plan[
                        "codes"
                    ]
                ]
            else:
                rewards = [
                    get_tpr_reward(
                        code
                    )
                    for code
                    in plan[
                        "codes"
                    ]
                ]

            reward_lists.append(
                rewards
            )

            q_values.append(
                mean(
                    rewards
                )
            )

        oracle_q = max(
            q_values
        )

        best_plan_indices = {
            index
            for index, q
            in enumerate(
                q_values
            )
            if math.isclose(
                q,
                oracle_q,
                abs_tol=1e-12,
            )
        }

        for _ in range(
            num_trials
        ):
            total_trials += 1

            sampled_m1_rewards = [
                rng.choice(
                    rewards
                )
                for rewards
                in reward_lists
            ]

            max_observed_reward = max(
                sampled_m1_rewards
            )

            top_indices = [
                index
                for index, value
                in enumerate(
                    sampled_m1_rewards
                )
                if math.isclose(
                    value,
                    max_observed_reward,
                    abs_tol=1e-12,
                )
            ]

            selected_index = (
                rng.choice(
                    top_indices
                )
            )

            selected_q = (
                q_values[
                    selected_index
                ]
            )

            regret = (
                oracle_q
                - selected_q
            )

            selected_q_values.append(
                selected_q
            )

            oracle_q_values.append(
                oracle_q
            )

            regrets.append(
                regret
            )

            if (
                selected_index
                in best_plan_indices
            ):
                exact_best_hits += 1

            # For problems where M=4 says some useful signal exists,
            # ask how often M=1 sees no positive reward at all.
            if oracle_q > 0.0:
                nonzero_oracle_problem_trials += 1

                if max_observed_reward <= 0.0:
                    false_no_signal_trials += 1

    return {
        "reward_type": (
            reward_type
        ),

        "num_trials_per_problem": int(
            num_trials
        ),

        "total_selection_trials": int(
            total_trials
        ),

        "mean_oracle_m4_utility": float(
            mean(
                oracle_q_values
            )
        ),

        "mean_m1_selected_plan_m4_utility": float(
            mean(
                selected_q_values
            )
        ),

        "mean_selection_regret": float(
            mean(
                regrets
            )
        ),

        "exact_m4_best_plan_hit_rate": float(
            safe_div(
                exact_best_hits,
                total_trials,
            )
        ),

        "false_no_signal_rate_on_nonzero_oracle_problems": float(
            safe_div(
                false_no_signal_trials,
                nonzero_oracle_problem_trials,
            )
        ),
    }


# =============================================================================
# Threshold reliability
# =============================================================================


def analyze_binary_threshold_reliability(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Binary-specific interpretation.

    For plans that appear reasonably useful under M=4, how often would
    one rollout incorrectly assign zero?

    q4 >= .50:
        at least 2/4 successes

    q4 >= .75:
        at least 3/4 successes

    q4 = 1.00:
        4/4 successes
    """

    thresholds = (
        0.25,
        0.50,
        0.75,
        1.00,
    )

    output: dict[
        str,
        Any,
    ] = {}

    for threshold in thresholds:
        eligible_plans = 0
        total_single_observations = 0
        zero_observations = 0

        for record in records:
            rewards = [
                get_binary_reward(
                    code
                )
                for code
                in record[
                    "codes"
                ]
            ]

            q4 = mean(
                rewards
            )

            if (
                q4 + 1e-12
                < threshold
            ):
                continue

            eligible_plans += 1

            for reward in rewards:
                total_single_observations += 1

                if reward == 0.0:
                    zero_observations += 1

        key = (
            f"q4_ge_{threshold:.2f}"
        )

        output[
            key
        ] = {
            "eligible_plans": int(
                eligible_plans
            ),

            "single_rollout_observations": int(
                total_single_observations
            ),

            "m1_zero_count": int(
                zero_observations
            ),

            "m1_false_zero_rate": float(
                safe_div(
                    zero_observations,
                    total_single_observations,
                )
            ),
        }

    return output


# =============================================================================
# Per-problem selection diagnostics
# =============================================================================


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
        q_binary_values: list[
            float
        ] = []

        q_tpr_values: list[
            float
        ] = []

        for plan in plans:
            binary_rewards = [
                get_binary_reward(
                    code
                )
                for code
                in plan[
                    "codes"
                ]
            ]

            tpr_rewards = [
                get_tpr_reward(
                    code
                )
                for code
                in plan[
                    "codes"
                ]
            ]

            q_binary_values.append(
                mean(
                    binary_rewards
                )
            )

            q_tpr_values.append(
                mean(
                    tpr_rewards
                )
            )

        best_binary = max(
            q_binary_values
        )

        best_tpr = max(
            q_tpr_values
        )

        if best_binary == 0.0:
            coverage_group = (
                "NO_COVERAGE"
            )

        elif best_binary < 0.75:
            coverage_group = (
                "UNSTABLE_COVERAGE"
            )

        else:
            coverage_group = (
                "STABLE_COVERAGE"
            )

        rows.append(
            {
                "problem_id": str(
                    problem_id
                ),

                "coverage_group": (
                    coverage_group
                ),

                "num_plans": int(
                    len(
                        plans
                    )
                ),

                "mean_q4_binary": float(
                    mean(
                        q_binary_values
                    )
                ),

                "best_q4_binary": float(
                    best_binary
                ),

                "best_mean_gap_binary": float(
                    best_binary
                    - mean(
                        q_binary_values
                    )
                ),

                "mean_q4_tpr": float(
                    mean(
                        q_tpr_values
                    )
                ),

                "best_q4_tpr": float(
                    best_tpr
                ),

                "best_mean_gap_tpr": float(
                    best_tpr
                    - mean(
                        q_tpr_values
                    )
                ),

                "q4_binary_profile": (
                    " | ".join(
                        f"{value:.4f}"
                        for value
                        in q_binary_values
                    )
                ),

                "q4_tpr_profile": (
                    " | ".join(
                        f"{value:.4f}"
                        for value
                        in q_tpr_values
                    )
                ),
            }
        )

    return sorted(
        rows,
        key=lambda row: str(
            row[
                "problem_id"
            ]
        ),
    )


# =============================================================================
# Main
# =============================================================================


def run_analysis(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    selection_trials: int,
    simulation_seed: int,
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

    m = validate_records(
        records
    )

    grouped = group_by_problem(
        records
    )

    # -------------------------------------------------------------------------
    # 1. M1 value estimation reliability
    # -------------------------------------------------------------------------

    (
        binary_reliability,
        binary_plan_rows,
    ) = analyze_plan_reliability(
        records,
        reward_type="binary",
    )

    (
        tpr_reliability,
        tpr_plan_rows,
    ) = analyze_plan_reliability(
        records,
        reward_type="tpr",
    )

    # -------------------------------------------------------------------------
    # 2. Pairwise reward ordering / inversion
    # -------------------------------------------------------------------------

    binary_pairwise = (
        analyze_pairwise_inversions(
            grouped,
            reward_type="binary",
        )
    )

    tpr_pairwise = (
        analyze_pairwise_inversions(
            grouped,
            reward_type="tpr",
        )
    )

    # -------------------------------------------------------------------------
    # 3. M1-based best-plan selection simulation
    # -------------------------------------------------------------------------

    binary_selection = (
        simulate_m1_plan_selection(
            grouped,
            reward_type="binary",
            num_trials=(
                selection_trials
            ),
            seed=(
                simulation_seed
            ),
        )
    )

    tpr_selection = (
        simulate_m1_plan_selection(
            grouped,
            reward_type="tpr",
            num_trials=(
                selection_trials
            ),
            seed=(
                simulation_seed
                + 1
            ),
        )
    )

    # -------------------------------------------------------------------------
    # 4. Binary false-zero analysis
    # -------------------------------------------------------------------------

    binary_thresholds = (
        analyze_binary_threshold_reliability(
            records
        )
    )

    # -------------------------------------------------------------------------
    # 5. Problem-level reference summary
    # -------------------------------------------------------------------------

    problem_rows = (
        build_problem_rows(
            grouped
        )
    )

    # -------------------------------------------------------------------------
    # Final payload
    # -------------------------------------------------------------------------

    summary = {
        "input_path": str(
            input_path
        ),

        "num_problems": int(
            len(
                grouped
            )
        ),

        "num_plans": int(
            len(
                records
            )
        ),

        "M": int(
            m
        ),

        "num_code_trajectories": int(
            len(records)
            * m
        ),

        "interpretation_note": (
            "M=4 is an empirical reference estimate, "
            "not ground-truth expected plan utility. "
            "Leave-one-out statistics avoid direct self-inclusion."
        ),

        "binary": {
            "estimation_reliability": (
                binary_reliability
            ),

            "pairwise_ordering": (
                binary_pairwise
            ),

            "m1_plan_selection": (
                binary_selection
            ),

            "threshold_false_zero": (
                binary_thresholds
            ),
        },

        "tpr": {
            "estimation_reliability": (
                tpr_reliability
            ),

            "pairwise_ordering": (
                tpr_pairwise
            ),

            "m1_plan_selection": (
                tpr_selection
            ),
        },
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir
        / "m1_vs_m4_reliability_summary.json"
    )

    binary_plan_csv = (
        output_dir
        / "m1_vs_m4_binary_plan_metrics.csv"
    )

    tpr_plan_csv = (
        output_dir
        / "m1_vs_m4_tpr_plan_metrics.csv"
    )

    problem_csv = (
        output_dir
        / "m1_vs_m4_problem_metrics.csv"
    )

    write_json(
        summary_path,
        summary,
    )

    write_csv(
        binary_plan_csv,
        binary_plan_rows,
    )

    write_csv(
        tpr_plan_csv,
        tpr_plan_rows,
    )

    write_csv(
        problem_csv,
        problem_rows,
    )

    # =========================================================================
    # Console report
    # =========================================================================

    print()
    print(
        "=" * 100
    )

    print(
        "M=1 Reward Reliability vs M=4 Empirical Plan Utility"
    )

    print(
        "=" * 100
    )

    print(
        f"Problems              : "
        f"{len(grouped)}"
    )

    print(
        f"Plans                 : "
        f"{len(records)}"
    )

    print(
        f"M                     : "
        f"{m}"
    )

    print(
        f"Code trajectories     : "
        f"{len(records) * m}"
    )

    # -------------------------------------------------------------------------
    # Binary
    # -------------------------------------------------------------------------

    print()
    print(
        "[Binary reward: M1 -> M4]"
    )

    print(
        f"MAE                   : "
        f"{binary_reliability['m1_vs_m4_mae']:.4f}"
    )

    print(
        f"RMSE                  : "
        f"{binary_reliability['m1_vs_m4_rmse']:.4f}"
    )

    print(
        f"Pearson               : "
        f"{binary_reliability['m1_vs_m4_pearson']:.4f}"
    )

    print(
        f"Spearman              : "
        f"{binary_reliability['m1_vs_m4_spearman']:.4f}"
    )

    print()
    print(
        "[Binary reward: M1 -> leave-one-out M3]"
    )

    print(
        f"MAE                   : "
        f"{binary_reliability['m1_vs_loo_mae']:.4f}"
    )

    print(
        f"RMSE                  : "
        f"{binary_reliability['m1_vs_loo_rmse']:.4f}"
    )

    print(
        f"Pearson               : "
        f"{binary_reliability['m1_vs_loo_pearson']:.4f}"
    )

    print(
        f"Spearman              : "
        f"{binary_reliability['m1_vs_loo_spearman']:.4f}"
    )

    print()
    print(
        "[Binary pairwise ordering]"
    )

    print(
        f"Correct ordering      : "
        f"{binary_pairwise['correct_order_rate']:.4f}"
    )

    print(
        f"Tie                   : "
        f"{binary_pairwise['tie_rate']:.4f}"
    )

    print(
        f"Reward inversion      : "
        f"{binary_pairwise['inversion_rate']:.4f}"
    )

    print(
        f"Inversion excl. ties  : "
        f"{binary_pairwise['inversion_rate_excluding_ties']:.4f}"
    )

    print()
    print(
        "[Binary M1 plan selection]"
    )

    print(
        f"M4 oracle utility     : "
        f"{binary_selection['mean_oracle_m4_utility']:.4f}"
    )

    print(
        f"M1-selected utility   : "
        f"{binary_selection['mean_m1_selected_plan_m4_utility']:.4f}"
    )

    print(
        f"Selection regret      : "
        f"{binary_selection['mean_selection_regret']:.4f}"
    )

    print(
        f"Exact best-plan hit   : "
        f"{binary_selection['exact_m4_best_plan_hit_rate']:.4f}"
    )

    print(
        f"False no-signal rate  : "
        f"{binary_selection['false_no_signal_rate_on_nonzero_oracle_problems']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Binary false zeros
    # -------------------------------------------------------------------------

    print()
    print(
        "[Binary M1 false-zero probability]"
    )

    for key, payload in (
        binary_thresholds.items()
    ):
        print(
            f"{key:<14} "
            f"plans={payload['eligible_plans']:<4} "
            f"M1 zero rate="
            f"{payload['m1_false_zero_rate']:.4f}"
        )

    # -------------------------------------------------------------------------
    # TPR
    # -------------------------------------------------------------------------

    print()
    print(
        "[TPR reward: M1 -> M4]"
    )

    print(
        f"MAE                   : "
        f"{tpr_reliability['m1_vs_m4_mae']:.4f}"
    )

    print(
        f"RMSE                  : "
        f"{tpr_reliability['m1_vs_m4_rmse']:.4f}"
    )

    print(
        f"Pearson               : "
        f"{tpr_reliability['m1_vs_m4_pearson']:.4f}"
    )

    print(
        f"Spearman              : "
        f"{tpr_reliability['m1_vs_m4_spearman']:.4f}"
    )

    print()
    print(
        "[TPR reward: M1 -> leave-one-out M3]"
    )

    print(
        f"MAE                   : "
        f"{tpr_reliability['m1_vs_loo_mae']:.4f}"
    )

    print(
        f"RMSE                  : "
        f"{tpr_reliability['m1_vs_loo_rmse']:.4f}"
    )

    print(
        f"Pearson               : "
        f"{tpr_reliability['m1_vs_loo_pearson']:.4f}"
    )

    print(
        f"Spearman              : "
        f"{tpr_reliability['m1_vs_loo_spearman']:.4f}"
    )

    print()
    print(
        "[TPR pairwise ordering]"
    )

    print(
        f"Correct ordering      : "
        f"{tpr_pairwise['correct_order_rate']:.4f}"
    )

    print(
        f"Tie                   : "
        f"{tpr_pairwise['tie_rate']:.4f}"
    )

    print(
        f"Reward inversion      : "
        f"{tpr_pairwise['inversion_rate']:.4f}"
    )

    print(
        f"Inversion excl. ties  : "
        f"{tpr_pairwise['inversion_rate_excluding_ties']:.4f}"
    )

    print()
    print(
        "[TPR M1 plan selection]"
    )

    print(
        f"M4 oracle utility     : "
        f"{tpr_selection['mean_oracle_m4_utility']:.4f}"
    )

    print(
        f"M1-selected utility   : "
        f"{tpr_selection['mean_m1_selected_plan_m4_utility']:.4f}"
    )

    print(
        f"Selection regret      : "
        f"{tpr_selection['mean_selection_regret']:.4f}"
    )

    print(
        f"Exact best-plan hit   : "
        f"{tpr_selection['exact_m4_best_plan_hit_rate']:.4f}"
    )

    print(
        f"False no-signal rate  : "
        f"{tpr_selection['false_no_signal_rate_on_nonzero_oracle_problems']:.4f}"
    )

    print()
    print(
        "=" * 100
    )

    print(
        "Saved:"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {binary_plan_csv}"
    )

    print(
        f"  {tpr_plan_csv}"
    )

    print(
        f"  {problem_csv}"
    )

    print(
        "=" * 100
    )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify how reliably a single coder rollout "
            "(M=1) estimates M=4 empirical plan utility."
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

    parser.add_argument(
        "--selection-trials",
        type=int,
        default=1000,
        help=(
            "Number of bootstrap/resampling trials per problem "
            "for M=1 plan-selection simulation."
        ),
    )

    parser.add_argument(
        "--simulation-seed",
        type=int,
        default=42,
        help=(
            "Random seed for trajectory-resampling simulation."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_analysis(
        input_path=(
            args.input
        ),
        output_dir=(
            args.output_dir
        ),
        selection_trials=int(
            args.selection_trials
        ),
        simulation_seed=int(
            args.simulation_seed
        ),
    )