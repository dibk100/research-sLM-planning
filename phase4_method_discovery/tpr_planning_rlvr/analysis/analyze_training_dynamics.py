"""
TPR 50-step training dynamics를 Vanilla와 동일 기준으로 비교함.

PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python \
  phase4_method_discovery/tpr_planning_rlvr/analysis/analyze_training_dynamics.py \
  --tpr-log \
  phase4_method_discovery/tpr_planning_rlvr/outputs/training_smoke_20260901_065205.log \
  --output-dir \
  phase4_method_discovery/tpr_planning_rlvr/outputs/training_analysis
  
cat \
  phase4_method_discovery/tpr_planning_rlvr/outputs/training_smoke_20260901_050633.log \
  phase4_method_discovery/tpr_planning_rlvr/outputs/training_smoke_20260901_065205.log \
  > phase4_method_discovery/tpr_planning_rlvr/outputs/training_tpr_combined.log
  
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python \
  phase4_method_discovery/tpr_planning_rlvr/analysis/analyze_training_dynamics.py \
  --tpr-log \
  phase4_method_discovery/tpr_planning_rlvr/outputs/training_tpr_combined.log \
  --output-dir \
  phase4_method_discovery/tpr_planning_rlvr/outputs/training_analysis_full
"""

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


# =============================================================================
# Configuration
# =============================================================================


METRICS = [
    "actor/entropy",
    "actor/pg_loss",
    "actor/loss",
    "actor/grad_norm",
    "actor/lr",

    "critic/score/mean",
    "critic/score/max",
    "critic/score/min",

    "critic/rewards/mean",
    "critic/rewards/max",
    "critic/rewards/min",

    "critic/advantages/mean",
    "critic/advantages/max",
    "critic/advantages/min",

    "response_length/mean",
    "response_length/max",
    "response_length/min",

    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/step",

    "training/global_step",
    "training/epoch",
]


FLOAT_PATTERN = (
    r"(?:np\.float64\()?([+-]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?)\)?"
)


@dataclass
class StepRecord:
    step: int

    reward_mean: float | None
    reward_min: float | None
    reward_max: float | None

    advantage_mean: float | None
    advantage_min: float | None
    advantage_max: float | None

    entropy: float | None
    pg_loss: float | None
    grad_norm: float | None

    response_length_mean: float | None

    generation_time: float | None
    actor_update_time: float | None
    step_time: float | None

    group_type: str
    reward_variable: bool
    positive_reward: bool
    effective_update: bool


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Vanilla vs TPR Planning-RLVR "
            "training dynamics from verl console logs."
        )
    )

    parser.add_argument(
        "--tpr-log",
        type=Path,
        required=True,
        help="TPR training log.",
    )

    parser.add_argument(
        "--vanilla-log",
        type=Path,
        default=None,
        help=(
            "Optional Vanilla training log for direct comparison."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for CSV / JSON reports.",
    )

    parser.add_argument(
        "--expected-steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--grad-eps",
        type=float,
        default=1e-12,
        help=(
            "Gradient norm threshold used to count "
            "effective policy-update steps."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Parsing
# =============================================================================


def extract_metric(
    line: str,
    metric: str,
) -> float | None:
    pattern = (
        re.escape(metric)
        + r":"
        + FLOAT_PATTERN
    )

    match = re.search(
        pattern,
        line,
    )

    if match is None:
        return None

    try:
        return float(
            match.group(1)
        )
    except ValueError:
        return None


def extract_step(
    line: str,
) -> int | None:
    # Prefer explicit training/global_step.
    value = extract_metric(
        line,
        "training/global_step",
    )

    if value is not None:
        return int(value)

    # Fallback to TaskRunner "step:N - ..."
    match = re.search(
        r"\bstep:(\d+)\s+-",
        line,
    )

    if match is None:
        return None

    return int(
        match.group(1)
    )


def classify_group(
    reward_min: float | None,
    reward_max: float | None,
    *,
    eps: float = 1e-12,
) -> str:
    if (
        reward_min is None
        or reward_max is None
    ):
        return "unknown"

    if (
        abs(reward_min) <= eps
        and abs(reward_max) <= eps
    ):
        return "all_zero"

    if (
        abs(reward_min - 1.0) <= eps
        and abs(reward_max - 1.0) <= eps
    ):
        return "all_one"

    return "mixed"


def build_step_record(
    line: str,
    *,
    grad_eps: float,
) -> StepRecord | None:
    step = extract_step(line)

    if step is None:
        return None

    reward_mean = extract_metric(
        line,
        "critic/score/mean",
    )
    reward_min = extract_metric(
        line,
        "critic/score/min",
    )
    reward_max = extract_metric(
        line,
        "critic/score/max",
    )

    advantage_mean = extract_metric(
        line,
        "critic/advantages/mean",
    )
    advantage_min = extract_metric(
        line,
        "critic/advantages/min",
    )
    advantage_max = extract_metric(
        line,
        "critic/advantages/max",
    )

    entropy = extract_metric(
        line,
        "actor/entropy",
    )

    pg_loss = extract_metric(
        line,
        "actor/pg_loss",
    )

    grad_norm = extract_metric(
        line,
        "actor/grad_norm",
    )

    response_length_mean = extract_metric(
        line,
        "response_length/mean",
    )

    generation_time = extract_metric(
        line,
        "timing_s/gen",
    )

    actor_update_time = extract_metric(
        line,
        "timing_s/update_actor",
    )

    step_time = extract_metric(
        line,
        "timing_s/step",
    )

    group_type = classify_group(
        reward_min,
        reward_max,
    )

    reward_variable = (
        reward_min is not None
        and reward_max is not None
        and not math.isclose(
            reward_min,
            reward_max,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
    )

    positive_reward = (
        reward_max is not None
        and reward_max > 0.0
    )

    effective_update = (
        grad_norm is not None
        and abs(grad_norm) > grad_eps
    )

    return StepRecord(
        step=step,

        reward_mean=reward_mean,
        reward_min=reward_min,
        reward_max=reward_max,

        advantage_mean=advantage_mean,
        advantage_min=advantage_min,
        advantage_max=advantage_max,

        entropy=entropy,
        pg_loss=pg_loss,
        grad_norm=grad_norm,

        response_length_mean=(
            response_length_mean
        ),

        generation_time=(
            generation_time
        ),
        actor_update_time=(
            actor_update_time
        ),
        step_time=step_time,

        group_type=group_type,
        reward_variable=reward_variable,
        positive_reward=positive_reward,
        effective_update=effective_update,
    )


def load_training_log(
    path: Path,
    *,
    grad_eps: float,
) -> list[StepRecord]:
    if not path.exists():
        raise FileNotFoundError(
            f"Training log not found: {path}"
        )

    # Dedupe by global step.
    #
    # Some Ray/W&B logs may print a training metric line
    # more than once. Keep the last complete occurrence.
    step_records: dict[
        int,
        StepRecord,
    ] = {}

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            if "critic/score/mean:" not in line:
                continue

            record = build_step_record(
                line,
                grad_eps=grad_eps,
            )

            if record is None:
                continue

            step_records[
                record.step
            ] = record

    return [
        step_records[step]
        for step in sorted(
            step_records
        )
    ]


# =============================================================================
# Statistics
# =============================================================================


def valid_values(
    records: list[StepRecord],
    attr: str,
) -> list[float]:
    values = []

    for record in records:
        value = getattr(
            record,
            attr,
        )

        if value is not None:
            values.append(
                float(value)
            )

    return values


def safe_mean(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return mean(values)


def entropy_change(
    records: list[StepRecord],
) -> dict[str, float | None]:
    entropy_records = [
        record
        for record in records
        if record.entropy is not None
    ]

    if not entropy_records:
        return {
            "initial": None,
            "final": None,
            "delta": None,
        }

    initial = float(
        entropy_records[0].entropy
    )

    final = float(
        entropy_records[-1].entropy
    )

    return {
        "initial": initial,
        "final": final,
        "delta": final - initial,
    }


def summarize(
    records: list[StepRecord],
    *,
    expected_steps: int,
) -> dict[str, Any]:
    num_steps = len(records)

    all_zero = sum(
        record.group_type
        == "all_zero"
        for record in records
    )

    mixed = sum(
        record.group_type
        == "mixed"
        for record in records
    )

    all_one = sum(
        record.group_type
        == "all_one"
        for record in records
    )

    reward_variable_steps = sum(
        record.reward_variable
        for record in records
    )

    positive_reward_steps = sum(
        record.positive_reward
        for record in records
    )

    effective_update_steps = sum(
        record.effective_update
        for record in records
    )

    entropy_stats = entropy_change(
        records
    )

    reward_means = valid_values(
        records,
        "reward_mean",
    )

    pg_losses = valid_values(
        records,
        "pg_loss",
    )

    grad_norms = valid_values(
        records,
        "grad_norm",
    )

    response_lengths = valid_values(
        records,
        "response_length_mean",
    )

    generation_times = valid_values(
        records,
        "generation_time",
    )

    step_times = valid_values(
        records,
        "step_time",
    )

    advantage_ranges = []

    for record in records:
        if (
            record.advantage_min
            is not None
            and record.advantage_max
            is not None
        ):
            advantage_ranges.append(
                record.advantage_max
                - record.advantage_min
            )

    return {
        "parsed_steps": num_steps,
        "expected_steps": (
            expected_steps
        ),
        "complete": (
            num_steps
            >= expected_steps
        ),

        "first_step": (
            records[0].step
            if records
            else None
        ),
        "last_step": (
            records[-1].step
            if records
            else None
        ),

        "group_distribution": {
            "all_zero": all_zero,
            "mixed": mixed,
            "all_one": all_one,

            "all_zero_rate": (
                all_zero / num_steps
                if num_steps
                else 0.0
            ),
            "mixed_rate": (
                mixed / num_steps
                if num_steps
                else 0.0
            ),
            "all_one_rate": (
                all_one / num_steps
                if num_steps
                else 0.0
            ),
        },

        "reward_variable_steps": (
            reward_variable_steps
        ),
        "reward_variable_rate": (
            reward_variable_steps
            / num_steps
            if num_steps
            else 0.0
        ),

        "positive_reward_steps": (
            positive_reward_steps
        ),
        "positive_reward_rate": (
            positive_reward_steps
            / num_steps
            if num_steps
            else 0.0
        ),

        "effective_update_steps": (
            effective_update_steps
        ),
        "effective_update_rate": (
            effective_update_steps
            / num_steps
            if num_steps
            else 0.0
        ),

        "reward_mean": safe_mean(
            reward_means
        ),

        "mean_abs_pg_loss": (
            safe_mean(
                [
                    abs(value)
                    for value in pg_losses
                ]
            )
        ),

        "mean_grad_norm": safe_mean(
            grad_norms
        ),

        "mean_advantage_range": (
            safe_mean(
                advantage_ranges
            )
        ),

        "entropy": entropy_stats,

        "mean_response_length": (
            safe_mean(
                response_lengths
            )
        ),

        "mean_generation_time": (
            safe_mean(
                generation_times
            )
        ),

        "mean_step_time": safe_mean(
            step_times
        ),
    }


# =============================================================================
# Output
# =============================================================================


def fmt(
    value: Any,
    digits: int = 6,
) -> str:
    if value is None:
        return "-"

    if isinstance(
        value,
        float,
    ):
        return f"{value:.{digits}f}"

    return str(value)


def print_summary(
    name: str,
    summary: dict[str, Any],
) -> None:
    groups = summary[
        "group_distribution"
    ]

    entropy = summary["entropy"]

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    print(
        f"Parsed steps             : "
        f"{summary['parsed_steps']}/"
        f"{summary['expected_steps']}"
    )

    print(
        f"Step range               : "
        f"{summary['first_step']} -> "
        f"{summary['last_step']}"
    )

    print()

    print(
        f"All-zero groups          : "
        f"{groups['all_zero']} "
        f"({groups['all_zero_rate']:.2%})"
    )

    print(
        f"Mixed groups             : "
        f"{groups['mixed']} "
        f"({groups['mixed_rate']:.2%})"
    )

    print(
        f"All-one groups           : "
        f"{groups['all_one']} "
        f"({groups['all_one_rate']:.2%})"
    )

    print()

    print(
        f"Reward-variable steps    : "
        f"{summary['reward_variable_steps']} "
        f"({summary['reward_variable_rate']:.2%})"
    )

    print(
        f"Positive-reward steps    : "
        f"{summary['positive_reward_steps']} "
        f"({summary['positive_reward_rate']:.2%})"
    )

    print(
        f"Effective update steps   : "
        f"{summary['effective_update_steps']} "
        f"({summary['effective_update_rate']:.2%})"
    )

    print()

    print(
        f"Mean reward              : "
        f"{fmt(summary['reward_mean'])}"
    )

    print(
        f"Mean |PG loss|           : "
        f"{fmt(summary['mean_abs_pg_loss'])}"
    )

    print(
        f"Mean grad norm           : "
        f"{fmt(summary['mean_grad_norm'])}"
    )

    print(
        f"Mean advantage range     : "
        f"{fmt(summary['mean_advantage_range'])}"
    )

    print()

    print(
        f"Entropy initial          : "
        f"{fmt(entropy['initial'])}"
    )

    print(
        f"Entropy final            : "
        f"{fmt(entropy['final'])}"
    )

    print(
        f"Entropy delta            : "
        f"{fmt(entropy['delta'])}"
    )

    print()

    print(
        f"Mean response length     : "
        f"{fmt(summary['mean_response_length'], 3)}"
    )

    print(
        f"Mean generation time     : "
        f"{fmt(summary['mean_generation_time'], 3)}s"
    )

    print(
        f"Mean step time           : "
        f"{fmt(summary['mean_step_time'], 3)}s"
    )


def print_comparison(
    vanilla: dict[str, Any],
    tpr: dict[str, Any],
) -> None:
    print()
    print("=" * 100)
    print("Vanilla vs TPR")
    print("=" * 100)

    rows = [
        (
            "All-zero group rate",
            vanilla[
                "group_distribution"
            ][
                "all_zero_rate"
            ],
            tpr[
                "group_distribution"
            ][
                "all_zero_rate"
            ],
            "%",
        ),
        (
            "Mixed group rate",
            vanilla[
                "group_distribution"
            ][
                "mixed_rate"
            ],
            tpr[
                "group_distribution"
            ][
                "mixed_rate"
            ],
            "%",
        ),
        (
            "Reward-variable rate",
            vanilla[
                "reward_variable_rate"
            ],
            tpr[
                "reward_variable_rate"
            ],
            "%",
        ),
        (
            "Effective update rate",
            vanilla[
                "effective_update_rate"
            ],
            tpr[
                "effective_update_rate"
            ],
            "%",
        ),
        (
            "Mean reward",
            vanilla["reward_mean"],
            tpr["reward_mean"],
            "float",
        ),
        (
            "Mean |PG loss|",
            vanilla[
                "mean_abs_pg_loss"
            ],
            tpr[
                "mean_abs_pg_loss"
            ],
            "float",
        ),
        (
            "Mean grad norm",
            vanilla[
                "mean_grad_norm"
            ],
            tpr[
                "mean_grad_norm"
            ],
            "float",
        ),
        (
            "Entropy delta",
            vanilla[
                "entropy"
            ][
                "delta"
            ],
            tpr[
                "entropy"
            ][
                "delta"
            ],
            "float",
        ),
    ]

    print(
        f"{'Metric':<30}"
        f"{'Vanilla':>16}"
        f"{'TPR':>16}"
        f"{'TPR - Vanilla':>18}"
    )

    print(
        "-" * 80
    )

    for (
        metric,
        vanilla_value,
        tpr_value,
        kind,
    ) in rows:
        if (
            vanilla_value is None
            or tpr_value is None
        ):
            delta = None
        else:
            delta = (
                tpr_value
                - vanilla_value
            )

        if kind == "%":
            vanilla_text = (
                f"{vanilla_value:.2%}"
                if vanilla_value is not None
                else "-"
            )

            tpr_text = (
                f"{tpr_value:.2%}"
                if tpr_value is not None
                else "-"
            )

            delta_text = (
                f"{delta:+.2%}"
                if delta is not None
                else "-"
            )

        else:
            vanilla_text = fmt(
                vanilla_value
            )
            tpr_text = fmt(
                tpr_value
            )

            delta_text = (
                f"{delta:+.6f}"
                if delta is not None
                else "-"
            )

        print(
            f"{metric:<30}"
            f"{vanilla_text:>16}"
            f"{tpr_text:>16}"
            f"{delta_text:>18}"
        )


def save_step_csv(
    path: Path,
    records: list[StepRecord],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        asdict(
            records[0]
        ).keys()
    ) if records else [
        "step"
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                asdict(record)
            )


def save_json(
    path: Path,
    obj: Any,
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
            obj,
            f,
            indent=2,
            ensure_ascii=False,
        )

        f.write("\n")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    output_dir = (
        args.output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 100
    )
    print(
        "Planning-RLVR Training Dynamics Analysis"
    )
    print(
        "=" * 100
    )

    print(
        f"TPR log     : "
        f"{args.tpr_log}"
    )

    if args.vanilla_log:
        print(
            f"Vanilla log : "
            f"{args.vanilla_log}"
        )

    print(
        f"Output      : "
        f"{output_dir}"
    )

    # -----------------------------------------------------------------
    # TPR
    # -----------------------------------------------------------------

    tpr_records = load_training_log(
        args.tpr_log,
        grad_eps=args.grad_eps,
    )

    tpr_summary = summarize(
        tpr_records,
        expected_steps=(
            args.expected_steps
        ),
    )

    print_summary(
        "TPR Planning-RLVR",
        tpr_summary,
    )

    save_step_csv(
        output_dir
        / "tpr_training_steps.csv",
        tpr_records,
    )

    # -----------------------------------------------------------------
    # Optional Vanilla
    # -----------------------------------------------------------------

    vanilla_records = None
    vanilla_summary = None

    if args.vanilla_log:
        vanilla_records = (
            load_training_log(
                args.vanilla_log,
                grad_eps=args.grad_eps,
            )
        )

        vanilla_summary = summarize(
            vanilla_records,
            expected_steps=(
                args.expected_steps
            ),
        )

        print_summary(
            "Vanilla Planning-RLVR",
            vanilla_summary,
        )

        save_step_csv(
            output_dir
            / "vanilla_training_steps.csv",
            vanilla_records,
        )

        print_comparison(
            vanilla_summary,
            tpr_summary,
        )

    # -----------------------------------------------------------------
    # Save final summary
    # -----------------------------------------------------------------

    summary = {
        "tpr": tpr_summary,
    }

    if vanilla_summary is not None:
        summary[
            "vanilla"
        ] = vanilla_summary

        summary[
            "delta_tpr_minus_vanilla"
        ] = {
            "all_zero_rate": (
                tpr_summary[
                    "group_distribution"
                ][
                    "all_zero_rate"
                ]
                - vanilla_summary[
                    "group_distribution"
                ][
                    "all_zero_rate"
                ]
            ),

            "mixed_rate": (
                tpr_summary[
                    "group_distribution"
                ][
                    "mixed_rate"
                ]
                - vanilla_summary[
                    "group_distribution"
                ][
                    "mixed_rate"
                ]
            ),

            "reward_variable_rate": (
                tpr_summary[
                    "reward_variable_rate"
                ]
                - vanilla_summary[
                    "reward_variable_rate"
                ]
            ),

            "effective_update_rate": (
                tpr_summary[
                    "effective_update_rate"
                ]
                - vanilla_summary[
                    "effective_update_rate"
                ]
            ),
        }

    summary_path = (
        output_dir
        / "training_dynamics_summary.json"
    )

    save_json(
        summary_path,
        summary,
    )

    print()
    print("=" * 100)
    print("Saved Outputs")
    print("=" * 100)

    print(
        "[Saved] "
        f"{output_dir / 'tpr_training_steps.csv'}"
    )

    if vanilla_records is not None:
        print(
            "[Saved] "
            f"{output_dir / 'vanilla_training_steps.csv'}"
        )

    print(
        "[Saved] "
        f"{summary_path}"
    )

    print()
    print(
        "Analysis complete."
    )


if __name__ == "__main__":
    main()