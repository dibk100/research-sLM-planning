"""
1. 50-step 학습 dynamics 분석

Analyze a vanilla Planning-RLVR pilot training log.

This script parses verl training stdout/stderr logs containing lines such as:

    (TaskRunner pid=...) step:4 - actor/entropy:0.81 - actor/pg_loss:0.0 ...

and summarizes:

1. Reward / score dynamics
2. Effective RL update frequency
3. Actor optimization dynamics
4. Entropy
5. Response lengths
6. Timing / throughput
7. CPU / GPU memory metrics
8. Reward-update consistency

Example
-------
python \
  phase4_method_discovery/vanilla_planning_rlvr/analysis/analyze_pilot_training.py \
  --log phase4_method_discovery/vanilla_planning_rlvr/outputs/<TRAINING_LOG>.log \
  --csv phase4_method_discovery/vanilla_planning_rlvr/outputs/pilot50_metrics.csv
  

python \
  phase4_method_discovery/vanilla_planning_rlvr/analysis/analyze_pilot_training.py \
  --log /tmp/ray/session_latest/logs/worker-2c2eb4507ecdc358f84a4f9291e43a492c122aa638296bdb577e0a00-01000000-1936857.out \
  --csv phase4_method_discovery/vanilla_planning_rlvr/outputs/pilot50_training_metrics.csv

Optional CSV output
-------------------
python \
    phase4_method_discovery/vanilla_planning_rlvr/analysis/analyze_pilot_training.py \
    --log path/to/training.log \
    --csv outputs/pilot_training_metrics.csv
"""
# phase4_method_discovery/vanilla_planning_rlvr/analysis/analyze_pilot_training.py
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path
from typing import Any


# =============================================================================
# Parsing
# =============================================================================

STEP_PATTERN = re.compile(r"(?:^|\s)step:(\d+)\s+-\s+")

VALUE_PATTERNS = (
    re.compile(r"^np\.float64\(([-+0-9.eE]+)\)$"),
    re.compile(r"^np\.float32\(([-+0-9.eE]+)\)$"),
    re.compile(r"^np\.int64\(([-+0-9]+)\)$"),
    re.compile(r"^np\.int32\(([-+0-9]+)\)$"),
)


def parse_numeric(value: str) -> float | None:
    """Convert a verl metric value into float when possible."""
    value = value.strip()

    for pattern in VALUE_PATTERNS:
        match = pattern.match(value)
        if match:
            value = match.group(1)
            break

    try:
        result = float(value)
    except ValueError:
        return None

    if not math.isfinite(result):
        return None

    return result


def parse_training_line(line: str) -> dict[str, float] | None:
    """
    Parse one verl training metric line.

    Returns
    -------
    dict or None
        Dictionary containing `step` and parsed numerical metrics.
    """
    match = STEP_PATTERN.search(line)

    if match is None:
        return None

    step = int(match.group(1))

    # Everything after "step:N - "
    payload = line[match.end():]

    record: dict[str, float] = {
        "step": float(step),
    }

    for item in payload.split(" - "):
        item = item.strip()

        if ":" not in item:
            continue

        key, value = item.split(":", 1)

        key = key.strip()
        numeric_value = parse_numeric(value)

        if numeric_value is not None:
            record[key] = numeric_value

    return record


def load_training_records(log_path: Path) -> list[dict[str, float]]:
    """Load all training-step metric records from a log."""
    records_by_step: dict[int, dict[str, float]] = {}

    with log_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            record = parse_training_line(line)

            if record is None:
                continue

            step = int(record["step"])

            # If a step somehow appears more than once, merge metrics.
            if step not in records_by_step:
                records_by_step[step] = record
            else:
                records_by_step[step].update(record)

    return [
        records_by_step[step]
        for step in sorted(records_by_step)
    ]


# =============================================================================
# Statistics helpers
# =============================================================================

def metric_values(
    records: list[dict[str, float]],
    key: str,
) -> list[float]:
    return [
        record[key]
        for record in records
        if key in record
    ]


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def safe_min(values: list[float]) -> float | None:
    if not values:
        return None
    return min(values)


def safe_max(values: list[float]) -> float | None:
    if not values:
        return None
    return max(values)


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "N/A"

    return f"{value:.{digits}f}"


def fmt_pct(
    numerator: int,
    denominator: int,
) -> str:
    if denominator == 0:
        return "N/A"

    return f"{100.0 * numerator / denominator:.2f}%"


def count_nonzero(
    records: list[dict[str, float]],
    key: str,
    eps: float = 1e-12,
) -> int:
    return sum(
        abs(record[key]) > eps
        for record in records
        if key in record
    )


def count_positive(
    records: list[dict[str, float]],
    key: str,
    eps: float = 1e-12,
) -> int:
    return sum(
        record[key] > eps
        for record in records
        if key in record
    )


def steps_matching(
    records: list[dict[str, float]],
    key: str,
    predicate,
) -> list[int]:
    result = []

    for record in records:
        if key not in record:
            continue

        if predicate(record[key]):
            result.append(int(record["step"]))

    return result


# =============================================================================
# Analysis
# =============================================================================

def analyze(records: list[dict[str, float]]) -> dict[str, Any]:
    if not records:
        raise RuntimeError(
            "No training-step records were found in the log."
        )

    total_steps = len(records)

    # -------------------------------------------------------------------------
    # Reward / score
    # -------------------------------------------------------------------------

    score_mean = metric_values(records, "critic/score/mean")
    score_max = metric_values(records, "critic/score/max")
    score_min = metric_values(records, "critic/score/min")

    reward_mean = metric_values(records, "critic/rewards/mean")
    reward_max = metric_values(records, "critic/rewards/max")
    reward_min = metric_values(records, "critic/rewards/min")

    positive_score_steps = steps_matching(
        records,
        "critic/score/max",
        lambda x: x > 1e-12,
    )

    positive_mean_score_steps = steps_matching(
        records,
        "critic/score/mean",
        lambda x: x > 1e-12,
    )

    # -------------------------------------------------------------------------
    # Actor updates
    # -------------------------------------------------------------------------

    grad_norm = metric_values(records, "actor/grad_norm")
    pg_loss = metric_values(records, "actor/pg_loss")
    actor_loss = metric_values(records, "actor/loss")

    effective_update_steps = steps_matching(
        records,
        "actor/grad_norm",
        lambda x: abs(x) > 1e-12,
    )

    nonzero_pg_steps = steps_matching(
        records,
        "actor/pg_loss",
        lambda x: abs(x) > 1e-12,
    )

    # -------------------------------------------------------------------------
    # Advantage
    # -------------------------------------------------------------------------

    advantage_mean = metric_values(
        records,
        "critic/advantages/mean",
    )
    advantage_max = metric_values(
        records,
        "critic/advantages/max",
    )
    advantage_min = metric_values(
        records,
        "critic/advantages/min",
    )

    advantage_signal_steps = []

    for record in records:
        values = [
            record.get("critic/advantages/min", 0.0),
            record.get("critic/advantages/max", 0.0),
        ]

        if any(abs(v) > 1e-12 for v in values):
            advantage_signal_steps.append(int(record["step"]))

    # -------------------------------------------------------------------------
    # Entropy
    # -------------------------------------------------------------------------

    entropy = metric_values(records, "actor/entropy")

    # -------------------------------------------------------------------------
    # Response length
    # -------------------------------------------------------------------------

    response_mean = metric_values(
        records,
        "response_length/mean",
    )
    response_max = metric_values(
        records,
        "response_length/max",
    )
    response_min = metric_values(
        records,
        "response_length/min",
    )
    response_clip = metric_values(
        records,
        "response_length/clip_ratio",
    )

    # -------------------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------------------

    step_time = metric_values(records, "timing_s/step")
    gen_time = metric_values(records, "timing_s/gen")
    actor_update_time = metric_values(
        records,
        "timing_s/update_actor",
    )
    weight_update_time = metric_values(
        records,
        "timing_s/update_weights",
    )

    throughput = metric_values(
        records,
        "perf/throughput",
    )

    # -------------------------------------------------------------------------
    # Memory
    # -------------------------------------------------------------------------

    gpu_allocated = metric_values(
        records,
        "actor/perf/max_memory_allocated_gb",
    )

    gpu_reserved = metric_values(
        records,
        "actor/perf/max_memory_reserved_gb",
    )

    cpu_memory = metric_values(
        records,
        "actor/perf/cpu_memory_used_gb",
    )

    # -------------------------------------------------------------------------
    # PPO / rollout consistency
    # -------------------------------------------------------------------------

    ppo_kl = metric_values(records, "actor/ppo_kl")

    rollout_prob_diff = metric_values(
        records,
        "training/rollout_probs_diff_mean",
    )

    rollout_corr = metric_values(
        records,
        "training/rollout_actor_probs_pearson_corr",
    )

    # -------------------------------------------------------------------------
    # Reward <-> update consistency
    # -------------------------------------------------------------------------

    positive_set = set(positive_score_steps)
    update_set = set(effective_update_steps)

    reward_and_update = sorted(positive_set & update_set)
    reward_without_update = sorted(positive_set - update_set)
    update_without_reward = sorted(update_set - positive_set)

    return {
        "total_steps": total_steps,
        "first_step": int(records[0]["step"]),
        "last_step": int(records[-1]["step"]),

        "score_mean": score_mean,
        "score_max": score_max,
        "score_min": score_min,

        "reward_mean": reward_mean,
        "reward_max": reward_max,
        "reward_min": reward_min,

        "positive_score_steps": positive_score_steps,
        "positive_mean_score_steps": positive_mean_score_steps,

        "grad_norm": grad_norm,
        "pg_loss": pg_loss,
        "actor_loss": actor_loss,

        "effective_update_steps": effective_update_steps,
        "nonzero_pg_steps": nonzero_pg_steps,

        "advantage_mean": advantage_mean,
        "advantage_max": advantage_max,
        "advantage_min": advantage_min,
        "advantage_signal_steps": advantage_signal_steps,

        "entropy": entropy,

        "response_mean": response_mean,
        "response_max": response_max,
        "response_min": response_min,
        "response_clip": response_clip,

        "step_time": step_time,
        "gen_time": gen_time,
        "actor_update_time": actor_update_time,
        "weight_update_time": weight_update_time,
        "throughput": throughput,

        "gpu_allocated": gpu_allocated,
        "gpu_reserved": gpu_reserved,
        "cpu_memory": cpu_memory,

        "ppo_kl": ppo_kl,
        "rollout_prob_diff": rollout_prob_diff,
        "rollout_corr": rollout_corr,

        "reward_and_update": reward_and_update,
        "reward_without_update": reward_without_update,
        "update_without_reward": update_without_reward,
    }


# =============================================================================
# Reporting
# =============================================================================

def print_step_list(
    label: str,
    steps: list[int],
) -> None:
    if steps:
        value = ", ".join(map(str, steps))
    else:
        value = "none"

    print(f"{label:<30}: {value}")


def print_summary(
    log_path: Path,
    result: dict[str, Any],
) -> None:
    total_steps = result["total_steps"]

    positive_steps = result["positive_score_steps"]
    effective_steps = result["effective_update_steps"]
    advantage_steps = result["advantage_signal_steps"]

    print()
    print("=" * 80)
    print("Vanilla Planning RLVR Pilot Training Analysis")
    print("=" * 80)

    print(f"Log file                      : {log_path}")
    print(f"Parsed steps                  : {total_steps}")
    print(
        f"Step range                    : "
        f"{result['first_step']} -> {result['last_step']}"
    )

    # -------------------------------------------------------------------------
    print()
    print("-" * 80)
    print("1. Reward / Score Signal")
    print("-" * 80)

    print(
        f"Positive-score steps          : "
        f"{len(positive_steps)} / {total_steps} "
        f"({fmt_pct(len(positive_steps), total_steps)})"
    )

    print(
        f"Positive mean-score steps     : "
        f"{len(result['positive_mean_score_steps'])} / "
        f"{total_steps} "
        f"({fmt_pct(len(result['positive_mean_score_steps']), total_steps)})"
    )

    print(
        f"Mean critic score             : "
        f"{fmt(safe_mean(result['score_mean']))}"
    )

    print(
        f"Maximum score/mean            : "
        f"{fmt(safe_max(result['score_mean']))}"
    )

    print(
        f"Maximum score/max             : "
        f"{fmt(safe_max(result['score_max']))}"
    )

    print_step_list(
        "Positive-score step IDs",
        positive_steps,
    )

    # -------------------------------------------------------------------------
    print()
    print("-" * 80)
    print("2. Effective RL Updates")
    print("-" * 80)

    print(
        f"Effective update steps        : "
        f"{len(effective_steps)} / {total_steps} "
        f"({fmt_pct(len(effective_steps), total_steps)})"
    )

    print(
        f"Advantage-signal steps        : "
        f"{len(advantage_steps)} / {total_steps} "
        f"({fmt_pct(len(advantage_steps), total_steps)})"
    )

    print(
        f"Non-zero PG-loss steps        : "
        f"{len(result['nonzero_pg_steps'])} / {total_steps} "
        f"({fmt_pct(len(result['nonzero_pg_steps']), total_steps)})"
    )

    print(
        f"Mean grad norm                : "
        f"{fmt(safe_mean(result['grad_norm']))}"
    )

    print(
        f"Max grad norm                 : "
        f"{fmt(safe_max(result['grad_norm']))}"
    )

    print(
        f"Mean |PG loss|                : "
        f"{fmt(safe_mean([abs(x) for x in result['pg_loss']]))}"
    )

    print_step_list(
        "Effective update step IDs",
        effective_steps,
    )

    # -------------------------------------------------------------------------
    print()
    print("-" * 80)
    print("3. Reward <-> Update Consistency")
    print("-" * 80)

    print(
        f"Reward + update               : "
        f"{len(result['reward_and_update'])}"
    )

    print(
        f"Reward but no update          : "
        f"{len(result['reward_without_update'])}"
    )

    print(
        f"Update but no positive score  : "
        f"{len(result['update_without_reward'])}"
    )

    print_step_list(
        "Reward + update step IDs",
        result["reward_and_update"],
    )

    print_step_list(
        "Reward/no-update step IDs",
        result["reward_without_update"],
    )

    print_step_list(
        "Update/no-reward step IDs",
        result["update_without_reward"],
    )

    # -------------------------------------------------------------------------
    print()
    print("-" * 80)
    print("4. Policy Entropy")
    print("-" * 80)

    entropy = result["entropy"]

    print(f"Initial entropy               : {fmt(entropy[0] if entropy else None)}")
    print(f"Final entropy                 : {fmt(entropy[-1] if entropy else None)}")
    print(f"Mean entropy                  : {fmt(safe_mean(entropy))}")
    print(f"Min entropy                   : {fmt(safe_min(entropy))}")
    print(f"Max entropy                   : {fmt(safe_max(entropy))}")

    if len(entropy) >= 2:
        print(
            f"Entropy change                : "
            f"{entropy[-1] - entropy[0]:.6f}"
        )

    # -------------------------------------------------------------------------
    print()
    print("-" * 80)
    print("5. Response Length")
    print("-" * 80)

    print(
        f"Mean response length          : "
        f"{fmt(safe_mean(result['response_mean']), 2)}"
    )

    print(
        f"Observed maximum length       : "
        f"{fmt(safe_max(result['response_max']), 2)}"
    )

    print(
        f"Mean clipping ratio           : "
        f"{fmt(safe_mean(result['response_clip']), 4)}"
    )

    # -------------------------------------------------------------------------
    print()
    print("-" * 80)
    print("6. Runtime")
    print("-" * 80)

    print(
        f"Mean step time                : "
        f"{fmt(safe_mean(result['step_time']), 3)} s"
    )

    print(
        f"Mean generation time          : "
        f"{fmt(safe_mean(result['gen_time']), 3)} s"
    )

    print(
        f"Mean actor update time        : "
        f"{fmt(safe_mean(result['actor_update_time']), 3)} s"
    )

    print(
        f"Mean weight sync time         : "
        f"{fmt(safe_mean(result['weight_update_time']), 3)} s"
    )

    print(
        f"Mean throughput               : "
        f"{fmt(safe_mean(result['throughput']), 2)} tokens/s"
    )

    # -------------------------------------------------------------------------
    print()
    print("-" * 80)
    print("7. Memory")
    print("-" * 80)

    print(
        f"Peak actor GPU allocated      : "
        f"{fmt(safe_max(result['gpu_allocated']), 3)} GiB"
    )

    print(
        f"Peak actor GPU reserved       : "
        f"{fmt(safe_max(result['gpu_reserved']), 3)} GiB"
    )

    print(
        f"Peak actor CPU memory         : "
        f"{fmt(safe_max(result['cpu_memory']), 3)} GiB"
    )

    print(
        f"Mean actor CPU memory         : "
        f"{fmt(safe_mean(result['cpu_memory']), 3)} GiB"
    )

    # -------------------------------------------------------------------------
    print()
    print("-" * 80)
    print("8. Rollout / Training Consistency")
    print("-" * 80)

    print(
        f"Mean PPO KL                   : "
        f"{fmt(safe_mean(result['ppo_kl']), 8)}"
    )

    print(
        f"Mean rollout prob diff        : "
        f"{fmt(safe_mean(result['rollout_prob_diff']), 8)}"
    )

    print(
        f"Mean rollout/actor corr       : "
        f"{fmt(safe_mean(result['rollout_corr']), 8)}"
    )

    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("Pilot Diagnosis")
    print("=" * 80)

    update_rate = (
        len(effective_steps) / total_steps
        if total_steps
        else 0.0
    )

    positive_rate = (
        len(positive_steps) / total_steps
        if total_steps
        else 0.0
    )

    if effective_steps:
        print("[PASS] The actor received non-zero policy-gradient updates.")
    else:
        print("[WARN] No effective actor update was detected.")

    if positive_steps:
        print(
            "[PASS] Verifiable positive reward/score signals were observed."
        )
    else:
        print(
            "[WARN] No positive score signal was observed."
        )

    if update_rate < 0.20:
        print(
            "[WARN] Effective update density is low "
            f"({100.0 * update_rate:.1f}%)."
        )
    else:
        print(
            "[INFO] Effective update density is "
            f"{100.0 * update_rate:.1f}%."
        )

    if positive_rate < 0.20:
        print(
            "[INFO] Reward signal is sparse "
            f"({100.0 * positive_rate:.1f}% positive-score steps)."
        )

    if result["reward_without_update"]:
        print(
            "[CHECK] Some positive-score steps did not produce a "
            "non-zero actor gradient."
        )

    if result["update_without_reward"]:
        print(
            "[INFO] Some actor updates occurred on steps whose "
            "critic/score/max was zero; inspect group-relative "
            "advantages if unexpected."
        )

    print()
    print(
        "NOTE: This analysis diagnoses training dynamics only. "
        "It does NOT establish that the RL-trained planner "
        "improves downstream coding performance."
    )

    print(
        "Next required experiment: evaluate the final RL planner "
        "checkpoint against the frozen base Self-Plan baseline "
        "on a held-out evaluation set."
    )

    print("=" * 80)


# =============================================================================
# CSV
# =============================================================================

def write_csv(
    records: list[dict[str, float]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_keys = set()

    for record in records:
        all_keys.update(record.keys())

    keys = ["step"] + sorted(
        key
        for key in all_keys
        if key != "step"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(record)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a vanilla Planning-RLVR verl pilot training log."
        )
    )

    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to the training stdout/stderr log.",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "Optional path for exporting per-step parsed metrics "
            "as CSV."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.log.exists():
        raise FileNotFoundError(
            f"Training log not found: {args.log}"
        )

    records = load_training_records(args.log)

    if not records:
        raise RuntimeError(
            "Could not find any '(TaskRunner ...) step:N - ...' "
            f"training metric lines in {args.log}"
        )

    result = analyze(records)

    print_summary(
        log_path=args.log,
        result=result,
    )

    if args.csv is not None:
        write_csv(
            records=records,
            output_path=args.csv,
        )

        print()
        print(f"[OK] Per-step CSV saved to: {args.csv}")


if __name__ == "__main__":
    main()