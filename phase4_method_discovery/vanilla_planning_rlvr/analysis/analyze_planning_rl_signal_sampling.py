"""
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase4_method_discovery/vanilla_planning_rlvr/analysis/analyze_planning_rl_signal_sampling.py \
  --input phase4_method_discovery/vanilla_planning_rlvr/outputs/planning_rl_signal_sampling/step25_pilot20_g16.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze planning RL signal sampling results: "
            "group reward informativeness, TPR shaping potential, "
            "and token-level entropy structure."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL from sample_planning_rl_signal.py.",
    )

    parser.add_argument(
        "--top-token-count",
        type=int,
        default=20,
        help=(
            "Number of most frequent high-entropy tokens "
            "to print."
        ),
    )

    parser.add_argument(
        "--high-entropy-quantile",
        type=float,
        default=0.90,
        help=(
            "Global entropy quantile used to define "
            "high-entropy tokens. Default: 0.90."
        ),
    )

    return parser.parse_args()


# ============================================================================
# Basic statistics
# ============================================================================


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")

    return sum(values) / len(values)


def variance(values: list[float]) -> float:
    if not values:
        return float("nan")

    m = mean(values)

    return sum(
        (x - m) ** 2
        for x in values
    ) / len(values)


def std(values: list[float]) -> float:
    v = variance(values)

    if math.isnan(v):
        return float("nan")

    return math.sqrt(v)


def median(values: list[float]) -> float:
    if not values:
        return float("nan")

    return statistics.median(values)


def quantile(
    values: list[float],
    q: float,
) -> float:
    """
    Linear-interpolated quantile.
    """

    if not values:
        return float("nan")

    if not 0.0 <= q <= 1.0:
        raise ValueError(
            "q must be in [0, 1]."
        )

    xs = sorted(values)

    if len(xs) == 1:
        return xs[0]

    position = (
        (len(xs) - 1) * q
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return xs[lower]

    weight = (
        position - lower
    )

    return (
        xs[lower] * (1.0 - weight)
        + xs[upper] * weight
    )


def pearson(
    xs: list[float],
    ys: list[float],
) -> float:
    if len(xs) != len(ys):
        raise ValueError(
            "xs and ys must have equal length."
        )

    if len(xs) < 2:
        return float("nan")

    mx = mean(xs)
    my = mean(ys)

    dx = [
        x - mx
        for x in xs
    ]

    dy = [
        y - my
        for y in ys
    ]

    denom_x = math.sqrt(
        sum(x * x for x in dx)
    )

    denom_y = math.sqrt(
        sum(y * y for y in dy)
    )

    if denom_x == 0.0 or denom_y == 0.0:
        return float("nan")

    return (
        sum(
            x * y
            for x, y in zip(dx, dy)
        )
        / (denom_x * denom_y)
    )


def fmt(
    value: float,
    digits: int = 4,
) -> str:
    if math.isnan(value):
        return "N/A"

    return f"{value:.{digits}f}"


def fmt_signed(
    value: float,
    digits: int = 4,
) -> str:
    if math.isnan(value):
        return "N/A"

    return f"{value:+.{digits}f}"


# ============================================================================
# Loading
# ============================================================================


def load_records(
    path: Path,
) -> list[dict[str, Any]]:
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
                    "Invalid JSON at line "
                    f"{line_number}: {exc}"
                ) from exc

            records.append(
                record
            )

    if not records:
        raise ValueError(
            f"No records found: {path}"
        )

    return records


# ============================================================================
# Entropy feature extraction
# ============================================================================


def get_token_entropies(
    sample: dict[str, Any],
) -> list[float]:
    diagnostics = sample.get(
        "token_diagnostics",
        [],
    )

    entropies: list[float] = []

    for token in diagnostics:
        value = token.get(
            "entropy"
        )

        if value is None:
            continue

        entropies.append(
            float(value)
        )

    return entropies


def entropy_features(
    sample: dict[str, Any],
    *,
    global_high_threshold: float,
) -> dict[str, float]:
    entropies = (
        get_token_entropies(
            sample
        )
    )

    if not entropies:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "max": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "top10_mean": float("nan"),
            "high_ratio": float("nan"),
        }

    p90 = quantile(
        entropies,
        0.90,
    )

    p95 = quantile(
        entropies,
        0.95,
    )

    top_count = max(
        1,
        math.ceil(
            len(entropies) * 0.10
        ),
    )

    top_values = sorted(
        entropies,
        reverse=True,
    )[:top_count]

    high_count = sum(
        entropy
        >= global_high_threshold
        for entropy in entropies
    )

    return {
        "mean": mean(entropies),
        "std": std(entropies),
        "max": max(entropies),
        "p90": p90,
        "p95": p95,
        "top10_mean": mean(
            top_values
        ),
        "high_ratio": (
            high_count
            / len(entropies)
        ),
    }


# ============================================================================
# Flatten samples
# ============================================================================


def flatten_samples(
    records: list[dict[str, Any]],
    *,
    global_high_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[
        dict[str, Any]
    ] = []

    for record in records:
        problem_id = str(
            record.get(
                "problem_id",
                "",
            )
        )

        group_type = str(
            record.get(
                "group_type",
                "",
            )
        )

        for sample in record.get(
            "samples",
            []
        ):
            features = entropy_features(
                sample,
                global_high_threshold=(
                    global_high_threshold
                ),
            )

            rows.append(
                {
                    "problem_id": (
                        problem_id
                    ),
                    "group_type": (
                        group_type
                    ),
                    "sample_id": (
                        sample.get(
                            "sample_id"
                        )
                    ),
                    "reward": float(
                        sample.get(
                            "reward",
                            0.0,
                        )
                    ),
                    "tpr": float(
                        sample.get(
                            "test_pass_ratio",
                            0.0,
                        )
                    ),
                    "status": str(
                        sample.get(
                            "status",
                            "UNKNOWN",
                        )
                    ),
                    "token_count": int(
                        sample.get(
                            "plan_token_count",
                            0,
                        )
                    ),
                    **features,
                    "sample": sample,
                }
            )

    return rows


# ============================================================================
# Printing helpers
# ============================================================================


def print_header(
    title: str,
) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def print_subheader(
    title: str,
) -> None:
    print()
    print("-" * 100)
    print(title)
    print("-" * 100)


# ============================================================================
# 1. Group-level reward analysis
# ============================================================================


def analyze_groups(
    records: list[dict[str, Any]],
) -> None:
    print_header(
        "1. GROUP-LEVEL REWARD INFORMATIVENESS"
    )

    counts = Counter(
        str(
            record.get(
                "group_type",
                "unknown",
            )
        )
        for record in records
    )

    total = len(records)

    for group_type in (
        "all_zero",
        "mixed",
        "all_one",
    ):
        count = counts.get(
            group_type,
            0,
        )

        pct = (
            100.0 * count / total
            if total
            else 0.0
        )

        print(
            f"{group_type:10s}: "
            f"{count:3d}/{total:<3d} "
            f"({pct:6.2f}%)"
        )

    informative = counts.get(
        "mixed",
        0,
    )

    print()
    print(
        "Binary-reward informative groups : "
        f"{informative}/{total} "
        f"({100.0 * informative / total:.2f}%)"
    )

    print(
        "Binary-reward flat groups        : "
        f"{total - informative}/{total} "
        f"({100.0 * (total - informative) / total:.2f}%)"
    )

    print_subheader(
        "Per-problem group statistics"
    )

    print(
        f"{'Problem':<20}"
        f"{'Group':<12}"
        f"{'Pos':>7}"
        f"{'R mean':>10}"
        f"{'R var':>10}"
        f"{'TPR mean':>12}"
        f"{'TPR var':>12}"
    )

    for record in records:
        print(
            f"{str(record.get('problem_id')):<20}"
            f"{str(record.get('group_type')):<12}"
            f"{int(record.get('num_positive', 0)):>3}/"
            f"{int(record.get('num_samples', 0)):<3}"
            f"{float(record.get('reward_mean', 0.0)):>10.4f}"
            f"{float(record.get('reward_variance', 0.0)):>10.4f}"
            f"{float(record.get('tpr_mean', 0.0)):>12.4f}"
            f"{float(record.get('tpr_variance', 0.0)):>12.4f}"
        )


# ============================================================================
# 2. Reward shaping analysis
# ============================================================================


def analyze_tpr_signal(
    records: list[dict[str, Any]],
) -> None:
    print_header(
        "2. BINARY REWARD VS PARTIAL TEST-PASS SIGNAL"
    )

    eps = 1e-12

    flat_binary_groups = [
        record
        for record in records
        if float(
            record.get(
                "reward_variance",
                0.0,
            )
        )
        <= eps
    ]

    hidden_tpr_signal = [
        record
        for record in flat_binary_groups
        if float(
            record.get(
                "tpr_variance",
                0.0,
            )
        )
        > eps
    ]

    all_zero = [
        record
        for record in records
        if record.get(
            "group_type"
        )
        == "all_zero"
    ]

    all_zero_with_tpr = [
        record
        for record in all_zero
        if float(
            record.get(
                "tpr_variance",
                0.0,
            )
        )
        > eps
    ]

    all_one = [
        record
        for record in records
        if record.get(
            "group_type"
        )
        == "all_one"
    ]

    print(
        "Binary-flat groups                 : "
        f"{len(flat_binary_groups)}/{len(records)}"
    )

    print(
        "Binary-flat but TPR-variable groups: "
        f"{len(hidden_tpr_signal)}/{len(records)}"
    )

    print(
        "All-zero groups                    : "
        f"{len(all_zero)}"
    )

    print(
        "All-zero with TPR variation        : "
        f"{len(all_zero_with_tpr)}/"
        f"{len(all_zero)}"
        if all_zero
        else
        "All-zero with TPR variation        : N/A"
    )

    print(
        "All-one groups                     : "
        f"{len(all_one)}"
    )

    if hidden_tpr_signal:
        print_subheader(
            "Groups where binary reward loses TPR variation"
        )

        print(
            f"{'Problem':<20}"
            f"{'Group':<12}"
            f"{'R var':>12}"
            f"{'TPR mean':>12}"
            f"{'TPR var':>12}"
        )

        for record in sorted(
            hidden_tpr_signal,
            key=lambda x: float(
                x.get(
                    "tpr_variance",
                    0.0,
                )
            ),
            reverse=True,
        ):
            print(
                f"{str(record.get('problem_id')):<20}"
                f"{str(record.get('group_type')):<12}"
                f"{float(record.get('reward_variance', 0.0)):>12.6f}"
                f"{float(record.get('tpr_mean', 0.0)):>12.4f}"
                f"{float(record.get('tpr_variance', 0.0)):>12.6f}"
            )


# ============================================================================
# 3. Overall entropy analysis
# ============================================================================


ENTROPY_METRICS = (
    "mean",
    "std",
    "max",
    "p90",
    "p95",
    "top10_mean",
    "high_ratio",
)


def analyze_entropy_overall(
    rows: list[dict[str, Any]],
) -> None:
    print_header(
        "3. TOKEN-LEVEL ENTROPY SUMMARY"
    )

    print(
        f"{'Metric':<18}"
        f"{'All':>12}"
        f"{'PASS':>12}"
        f"{'FAIL':>12}"
        f"{'PASS-FAIL':>14}"
    )

    pass_rows = [
        row
        for row in rows
        if row["reward"] > 0.0
    ]

    fail_rows = [
        row
        for row in rows
        if row["reward"] <= 0.0
    ]

    for metric in ENTROPY_METRICS:
        all_values = [
            row[metric]
            for row in rows
            if not math.isnan(
                row[metric]
            )
        ]

        pass_values = [
            row[metric]
            for row in pass_rows
            if not math.isnan(
                row[metric]
            )
        ]

        fail_values = [
            row[metric]
            for row in fail_rows
            if not math.isnan(
                row[metric]
            )
        ]

        all_mean = mean(
            all_values
        )

        pass_mean = mean(
            pass_values
        )

        fail_mean = mean(
            fail_values
        )

        delta = (
            pass_mean - fail_mean
            if (
                not math.isnan(pass_mean)
                and not math.isnan(fail_mean)
            )
            else float("nan")
        )

        print(
            f"{metric:<18}"
            f"{fmt(all_mean):>12}"
            f"{fmt(pass_mean):>12}"
            f"{fmt(fail_mean):>12}"
            f"{fmt_signed(delta):>14}"
        )

    print()
    print(
        f"PASS samples: {len(pass_rows)}"
    )

    print(
        f"FAIL samples: {len(fail_rows)}"
    )


# ============================================================================
# 4. Mixed-group within-problem entropy analysis
# ============================================================================


def analyze_mixed_groups(
    rows: list[dict[str, Any]],
) -> None:
    print_header(
        "4. MIXED-GROUP WITHIN-PROBLEM PASS/FAIL ENTROPY"
    )

    mixed_rows = [
        row
        for row in rows
        if row["group_type"]
        == "mixed"
    ]

    problem_ids = sorted(
        {
            row["problem_id"]
            for row in mixed_rows
        }
    )

    if not problem_ids:
        print(
            "No mixed groups found."
        )
        return

    metric_deltas: dict[
        str,
        list[float],
    ] = {
        metric: []
        for metric in ENTROPY_METRICS
    }

    print(
        f"{'Problem':<20}"
        f"{'P/F':>8}"
        f"{'mean Δ':>12}"
        f"{'p90 Δ':>12}"
        f"{'p95 Δ':>12}"
        f"{'top10 Δ':>12}"
        f"{'max Δ':>12}"
    )

    for problem_id in problem_ids:
        problem_rows = [
            row
            for row in mixed_rows
            if row["problem_id"]
            == problem_id
        ]

        pass_rows = [
            row
            for row in problem_rows
            if row["reward"] > 0.0
        ]

        fail_rows = [
            row
            for row in problem_rows
            if row["reward"] <= 0.0
        ]

        deltas: dict[
            str,
            float,
        ] = {}

        for metric in ENTROPY_METRICS:
            pass_values = [
                row[metric]
                for row in pass_rows
                if not math.isnan(
                    row[metric]
                )
            ]

            fail_values = [
                row[metric]
                for row in fail_rows
                if not math.isnan(
                    row[metric]
                )
            ]

            if (
                not pass_values
                or not fail_values
            ):
                delta = float("nan")
            else:
                # PASS - FAIL within the same problem.
                delta = (
                    mean(pass_values)
                    - mean(fail_values)
                )

                metric_deltas[
                    metric
                ].append(
                    delta
                )

            deltas[
                metric
            ] = delta

        print(
            f"{problem_id:<20}"
            f"{len(pass_rows):>3}/"
            f"{len(fail_rows):<3}"
            f"{fmt_signed(deltas['mean']):>12}"
            f"{fmt_signed(deltas['p90']):>12}"
            f"{fmt_signed(deltas['p95']):>12}"
            f"{fmt_signed(deltas['top10_mean']):>12}"
            f"{fmt_signed(deltas['max']):>12}"
        )

    print_subheader(
        "Mean within-problem delta across mixed groups"
    )

    print(
        "Definition: delta = mean(PASS) - mean(FAIL)"
    )

    print()

    for metric in ENTROPY_METRICS:
        deltas = metric_deltas[
            metric
        ]

        negative = sum(
            delta < 0
            for delta in deltas
        )

        positive = sum(
            delta > 0
            for delta in deltas
        )

        zero = sum(
            delta == 0
            for delta in deltas
        )

        print(
            f"{metric:<18}: "
            f"mean_delta="
            f"{fmt_signed(mean(deltas))} | "
            f"PASS<FAIL={negative}/{len(deltas)} | "
            f"PASS>FAIL={positive}/{len(deltas)} | "
            f"equal={zero}/{len(deltas)}"
        )


# ============================================================================
# 5. Correlation analysis
# ============================================================================


def analyze_correlations(
    rows: list[dict[str, Any]],
) -> None:
    print_header(
        "5. ENTROPY / REWARD / TPR CORRELATIONS"
    )

    rewards = [
        row["reward"]
        for row in rows
    ]

    tprs = [
        row["tpr"]
        for row in rows
    ]

    print(
        f"{'Metric':<18}"
        f"{'corr(R)':>14}"
        f"{'corr(TPR)':>14}"
    )

    for metric in ENTROPY_METRICS:
        metric_values: list[
            float
        ] = []

        metric_rewards: list[
            float
        ] = []

        metric_tprs: list[
            float
        ] = []

        for row in rows:
            value = row[
                metric
            ]

            if math.isnan(value):
                continue

            metric_values.append(
                value
            )

            metric_rewards.append(
                row["reward"]
            )

            metric_tprs.append(
                row["tpr"]
            )

        reward_corr = pearson(
            metric_values,
            metric_rewards,
        )

        tpr_corr = pearson(
            metric_values,
            metric_tprs,
        )

        print(
            f"{metric:<18}"
            f"{fmt(reward_corr):>14}"
            f"{fmt(tpr_corr):>14}"
        )

    # Token count is useful as a possible confound.
    token_counts = [
        float(
            row["token_count"]
        )
        for row in rows
    ]

    print()
    print(
        "Potential sequence-length confound:"
    )

    print(
        "corr(token_count, reward) = "
        f"{fmt(pearson(token_counts, rewards))}"
    )

    print(
        "corr(token_count, TPR)    = "
        f"{fmt(pearson(token_counts, tprs))}"
    )


# ============================================================================
# 6. High-entropy token inspection
# ============================================================================


def normalize_token_for_display(
    token: str,
) -> str:
    return (
        token
        .replace(
            "\n",
            "\\n",
        )
        .replace(
            "\t",
            "\\t",
        )
    )


def analyze_high_entropy_tokens(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    top_token_count: int,
) -> None:
    print_header(
        "6. HIGH-ENTROPY TOKEN CONTENT"
    )

    print(
        "Global high-entropy threshold: "
        f"{threshold:.6f}"
    )

    pass_counter: Counter[
        str
    ] = Counter()

    fail_counter: Counter[
        str
    ] = Counter()

    pass_positions: list[
        float
    ] = []

    fail_positions: list[
        float
    ] = []

    total_high_tokens = 0

    for row in rows:
        sample = row[
            "sample"
        ]

        diagnostics = sample.get(
            "token_diagnostics",
            [],
        )

        token_count = max(
            1,
            len(diagnostics),
        )

        for token in diagnostics:
            entropy = token.get(
                "entropy"
            )

            if entropy is None:
                continue

            entropy = float(
                entropy
            )

            if entropy < threshold:
                continue

            total_high_tokens += 1

            token_text = str(
                token.get(
                    "token_text",
                    "",
                )
            )

            token_text = (
                normalize_token_for_display(
                    token_text
                )
            )

            token_index = int(
                token.get(
                    "token_index",
                    0,
                )
            )

            relative_position = (
                token_index
                / max(
                    1,
                    token_count - 1,
                )
            )

            if row["reward"] > 0.0:
                pass_counter[
                    token_text
                ] += 1

                pass_positions.append(
                    relative_position
                )
            else:
                fail_counter[
                    token_text
                ] += 1

                fail_positions.append(
                    relative_position
                )

    print(
        "High-entropy token count : "
        f"{total_high_tokens}"
    )

    print(
        "Mean relative position "
        "(0=start, 1=end):"
    )

    print(
        "  PASS: "
        f"{fmt(mean(pass_positions))}"
    )

    print(
        "  FAIL: "
        f"{fmt(mean(fail_positions))}"
    )

    print_subheader(
        "Most frequent high-entropy tokens in PASS plans"
    )

    for token, count in (
        pass_counter.most_common(
            top_token_count
        )
    ):
        print(
            f"{count:6d}  {repr(token)}"
        )

    print_subheader(
        "Most frequent high-entropy tokens in FAIL plans"
    )

    for token, count in (
        fail_counter.most_common(
            top_token_count
        )
    ):
        print(
            f"{count:6d}  {repr(token)}"
        )


# ============================================================================
# 7. Extreme examples
# ============================================================================


def analyze_extreme_samples(
    rows: list[dict[str, Any]],
) -> None:
    print_header(
        "7. EXTREME ENTROPY SAMPLES"
    )

    for metric in (
        "mean",
        "p95",
        "top10_mean",
        "max",
    ):
        valid_rows = [
            row
            for row in rows
            if not math.isnan(
                row[metric]
            )
        ]

        highest = sorted(
            valid_rows,
            key=lambda row: row[
                metric
            ],
            reverse=True,
        )[:5]

        lowest = sorted(
            valid_rows,
            key=lambda row: row[
                metric
            ],
        )[:5]

        print_subheader(
            f"Highest {metric}"
        )

        for row in highest:
            print(
                f"{row['problem_id']:<20} "
                f"sample={row['sample_id']:<3} "
                f"R={int(row['reward'])} "
                f"TPR={row['tpr']:.4f} "
                f"{metric}={row[metric]:.4f} "
                f"tokens={row['token_count']}"
            )

        print_subheader(
            f"Lowest {metric}"
        )

        for row in lowest:
            print(
                f"{row['problem_id']:<20} "
                f"sample={row['sample_id']:<3} "
                f"R={int(row['reward'])} "
                f"TPR={row['tpr']:.4f} "
                f"{metric}={row[metric]:.4f} "
                f"tokens={row['token_count']}"
            )


# ============================================================================
# 8. Method-selection summary
# ============================================================================


def print_method_diagnostics(
    records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    print_header(
        "8. METHOD-SELECTION DIAGNOSTICS"
    )

    eps = 1e-12

    mixed = [
        record
        for record in records
        if record.get(
            "group_type"
        )
        == "mixed"
    ]

    all_zero = [
        record
        for record in records
        if record.get(
            "group_type"
        )
        == "all_zero"
    ]

    binary_flat_tpr_variable = [
        record
        for record in records
        if (
            float(
                record.get(
                    "reward_variance",
                    0.0,
                )
            )
            <= eps
            and float(
                record.get(
                    "tpr_variance",
                    0.0,
                )
            )
            > eps
        )
    ]

    mixed_ratio = (
        len(mixed)
        / len(records)
    )

    hidden_tpr_ratio = (
        len(
            binary_flat_tpr_variable
        )
        / len(records)
    )

    all_zero_ratio = (
        len(all_zero)
        / len(records)
    )

    print(
        "These are diagnostics, not automatic "
        "method-selection rules."
    )

    print()

    print(
        "Binary reward informativeness:"
    )

    print(
        f"  mixed-group ratio = "
        f"{mixed_ratio:.4f} "
        f"({len(mixed)}/{len(records)})"
    )

    print()

    print(
        "Potential reward-shaping signal:"
    )

    print(
        "  binary-flat + TPR-variable ratio = "
        f"{hidden_tpr_ratio:.4f} "
        f"({len(binary_flat_tpr_variable)}/"
        f"{len(records)})"
    )

    print(
        f"  all-zero ratio = "
        f"{all_zero_ratio:.4f} "
        f"({len(all_zero)}/{len(records)})"
    )

    # --------------------------------------------------------------
    # Mixed-group upper-tail entropy diagnostic
    # --------------------------------------------------------------

    mixed_rows = [
        row
        for row in rows
        if row["group_type"]
        == "mixed"
    ]

    problem_ids = {
        row["problem_id"]
        for row in mixed_rows
    }

    top10_deltas: list[
        float
    ] = []

    p95_deltas: list[
        float
    ] = []

    for problem_id in problem_ids:
        problem_rows = [
            row
            for row in mixed_rows
            if row["problem_id"]
            == problem_id
        ]

        pass_rows = [
            row
            for row in problem_rows
            if row["reward"] > 0
        ]

        fail_rows = [
            row
            for row in problem_rows
            if row["reward"] <= 0
        ]

        if not pass_rows or not fail_rows:
            continue

        top10_deltas.append(
            mean(
                [
                    row[
                        "top10_mean"
                    ]
                    for row in pass_rows
                ]
            )
            - mean(
                [
                    row[
                        "top10_mean"
                    ]
                    for row in fail_rows
                ]
            )
        )

        p95_deltas.append(
            mean(
                [
                    row["p95"]
                    for row in pass_rows
                ]
            )
            - mean(
                [
                    row["p95"]
                    for row in fail_rows
                ]
            )
        )

    print()

    print(
        "Upper-tail entropy signal "
        "(within mixed groups):"
    )

    print(
        "  mean PASS-FAIL top10 entropy delta = "
        f"{fmt_signed(mean(top10_deltas))}"
    )

    print(
        "  mean PASS-FAIL p95 entropy delta   = "
        f"{fmt_signed(mean(p95_deltas))}"
    )

    print()
    print(
        "Interpretation guide:"
    )

    print(
        "  - Many mixed groups -> binary execution reward "
        "already provides within-group discrimination."
    )

    print(
        "  - Many binary-flat but TPR-variable groups -> "
        "partial-test reward shaping may recover discarded signal."
    )

    print(
        "  - Systematic PASS/FAIL differences in upper-tail entropy -> "
        "investigate entropy-aware or token-selective updates."
    )

    print(
        "  - No systematic entropy difference -> "
        "do not justify an entropy method from sequence-level "
        "statistics alone."
    )

    print(
        "  - High-entropy tokens must still be inspected semantically: "
        "formatting uncertainty is not a planning fork."
    )


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()

    input_path = Path(
        args.input
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found: {input_path}"
        )

    if not (
        0.0
        < args.high_entropy_quantile
        < 1.0
    ):
        raise ValueError(
            "--high-entropy-quantile "
            "must be in (0, 1)."
        )

    records = load_records(
        input_path
    )

    # ------------------------------------------------------------------
    # Determine a GLOBAL high-entropy threshold.
    #
    # Important:
    # This is computed across all generated plan tokens, rather than
    # defining "high entropy" separately for every plan.
    #
    # This preserves absolute entropy magnitude differences between plans.
    # ------------------------------------------------------------------

    all_token_entropies: list[
        float
    ] = []

    for record in records:
        for sample in record.get(
            "samples",
            []
        ):
            all_token_entropies.extend(
                get_token_entropies(
                    sample
                )
            )

    if not all_token_entropies:
        raise ValueError(
            "No token-level entropy values found. "
            "Expected samples[*].token_diagnostics[*].entropy."
        )

    global_high_threshold = quantile(
        all_token_entropies,
        args.high_entropy_quantile,
    )

    rows = flatten_samples(
        records,
        global_high_threshold=(
            global_high_threshold
        ),
    )

    print_header(
        "PLANNING RL SIGNAL SAMPLING ANALYSIS"
    )

    print(
        f"Input                    : "
        f"{input_path}"
    )

    print(
        f"Checkpoint               : "
        f"{records[0].get('checkpoint', 'unknown')}"
    )

    print(
        f"Problems                 : "
        f"{len(records)}"
    )

    print(
        f"Samples                  : "
        f"{len(rows)}"
    )

    print(
        f"Plan tokens              : "
        f"{len(all_token_entropies)}"
    )

    print(
        f"High-entropy quantile    : "
        f"{args.high_entropy_quantile:.2f}"
    )

    print(
        f"Global entropy threshold : "
        f"{global_high_threshold:.6f}"
    )

    analyze_groups(
        records
    )

    analyze_tpr_signal(
        records
    )

    analyze_entropy_overall(
        rows
    )

    analyze_mixed_groups(
        rows
    )

    analyze_correlations(
        rows
    )

    analyze_high_entropy_tokens(
        rows,
        threshold=(
            global_high_threshold
        ),
        top_token_count=(
            args.top_token_count
        ),
    )

    analyze_extreme_samples(
        rows
    )

    print_method_diagnostics(
        records,
        rows,
    )


if __name__ == "__main__":
    main()