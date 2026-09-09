#!/usr/bin/env python3

"""
Analyze a full vanilla Planning-RLVR training run.

Main questions
--------------
1. How sparse was the execution reward?
2. How many GRPO groups were actually informative?
3. How often did informative groups produce advantages / actor updates?
4. Did reward and update density change over training?
5. Did entropy or response length drift during training?
6. Was optimization numerically stable?

Assumption
----------
The current experiment uses:
    rollout.n = 16
    binary execution reward in {0, 1}

Therefore, for one training step:

    critic/score/mean = (# successful rollouts) / 16

and the rollout group can be classified as:

    all-fail     : 0 / 16 successful
    mixed        : 1..15 / 16 successful
    all-pass     : 16 / 16 successful

Only mixed groups provide non-zero group-relative reward variation for GRPO
under the binary reward setting.

Example
-------
python \
  phase4_method_discovery/vanilla_planning_rlvr/analysis/analyze_full_training.py \
  --log phase4_method_discovery/vanilla_planning_rlvr/outputs/full900_training.log \
  --group-size 16 \
  --window-size 100 \
  --csv phase4_method_discovery/vanilla_planning_rlvr/outputs/full900_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import Counter
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
    match = STEP_PATTERN.search(line)

    if match is None:
        return None

    step = int(match.group(1))
    payload = line[match.end():]

    record: dict[str, float] = {
        "step": float(step),
    }

    for item in payload.split(" - "):
        item = item.strip()

        if ":" not in item:
            continue

        key, value = item.split(":", 1)

        numeric_value = parse_numeric(value)

        if numeric_value is not None:
            record[key.strip()] = numeric_value

    return record


def load_training_records(log_path: Path) -> list[dict[str, float]]:
    """
    Load training records and merge duplicate metric lines for the same step.

    This is useful for resumed runs or logs where one step appears more than once.
    """
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

            if step not in records_by_step:
                records_by_step[step] = record
            else:
                records_by_step[step].update(record)

    return [
        records_by_step[step]
        for step in sorted(records_by_step)
    ]


# =============================================================================
# Helpers
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


def safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def fmt(
    value: float | None,
    digits: int = 6,
) -> str:
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


def is_nonzero(
    value: float | None,
    eps: float = 1e-12,
) -> bool:
    return value is not None and abs(value) > eps


def get_advantage_signal(
    record: dict[str, float],
    eps: float = 1e-12,
) -> bool:
    adv_min = record.get("critic/advantages/min")
    adv_max = record.get("critic/advantages/max")

    return (
        is_nonzero(adv_min, eps)
        or is_nonzero(adv_max, eps)
    )


def get_update_signal(
    record: dict[str, float],
    eps: float = 1e-12,
) -> bool:
    return is_nonzero(
        record.get("actor/grad_norm"),
        eps,
    )


def get_pg_signal(
    record: dict[str, float],
    eps: float = 1e-12,
) -> bool:
    return is_nonzero(
        record.get("actor/pg_loss"),
        eps,
    )


# =============================================================================
# GRPO group classification
# =============================================================================

def infer_success_count(
    record: dict[str, float],
    group_size: int,
) -> int | None:
    """
    Infer number of successful rollouts from critic/score/mean.

    Assumes binary reward and one GRPO group of size `group_size`.
    """
    score_mean = record.get("critic/score/mean")

    if score_mean is None:
        return None

    raw_count = score_mean * group_size
    count = int(round(raw_count))

    # Guard against small floating-point deviations.
    count = max(0, min(group_size, count))

    return count


def classify_group(
    success_count: int | None,
    group_size: int,
) -> str:
    if success_count is None:
        return "unknown"

    if success_count == 0:
        return "all_fail"

    if success_count == group_size:
        return "all_pass"

    return "mixed"


# =============================================================================
# Per-step augmentation
# =============================================================================

def augment_records(
    records: list[dict[str, float]],
    group_size: int,
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []

    for original in records:
        record: dict[str, Any] = dict(original)

        success_count = infer_success_count(
            original,
            group_size,
        )

        group_type = classify_group(
            success_count,
            group_size,
        )

        record["success_count"] = success_count
        record["group_type"] = group_type

        record["is_positive"] = (
            success_count is not None
            and success_count > 0
        )

        record["is_informative"] = (
            group_type == "mixed"
        )

        record["has_advantage_signal"] = (
            get_advantage_signal(original)
        )

        record["has_actor_update"] = (
            get_update_signal(original)
        )

        record["has_pg_signal"] = (
            get_pg_signal(original)
        )

        augmented.append(record)

    return augmented


# =============================================================================
# Global analysis
# =============================================================================

def analyze_global(
    records: list[dict[str, Any]],
    group_size: int,
) -> dict[str, Any]:
    total_steps = len(records)

    group_counter = Counter(
        record["group_type"]
        for record in records
    )

    success_counter = Counter(
        record["success_count"]
        for record in records
        if record["success_count"] is not None
    )

    positive_steps = [
        int(record["step"])
        for record in records
        if record["is_positive"]
    ]

    informative_steps = [
        int(record["step"])
        for record in records
        if record["is_informative"]
    ]

    advantage_steps = [
        int(record["step"])
        for record in records
        if record["has_advantage_signal"]
    ]

    update_steps = [
        int(record["step"])
        for record in records
        if record["has_actor_update"]
    ]

    pg_steps = [
        int(record["step"])
        for record in records
        if record["has_pg_signal"]
    ]

    informative_set = set(informative_steps)
    advantage_set = set(advantage_steps)
    update_set = set(update_steps)

    informative_and_advantage = sorted(
        informative_set & advantage_set
    )

    informative_without_advantage = sorted(
        informative_set - advantage_set
    )

    advantage_without_informative = sorted(
        advantage_set - informative_set
    )

    informative_and_update = sorted(
        informative_set & update_set
    )

    informative_without_update = sorted(
        informative_set - update_set
    )

    update_without_informative = sorted(
        update_set - informative_set
    )

    return {
        "total_steps": total_steps,
        "first_step": int(records[0]["step"]),
        "last_step": int(records[-1]["step"]),

        "group_counter": group_counter,
        "success_counter": success_counter,

        "positive_steps": positive_steps,
        "informative_steps": informative_steps,
        "advantage_steps": advantage_steps,
        "update_steps": update_steps,
        "pg_steps": pg_steps,

        "informative_and_advantage":
            informative_and_advantage,
        "informative_without_advantage":
            informative_without_advantage,
        "advantage_without_informative":
            advantage_without_informative,

        "informative_and_update":
            informative_and_update,
        "informative_without_update":
            informative_without_update,
        "update_without_informative":
            update_without_informative,

        "score_mean": metric_values(
            records,
            "critic/score/mean",
        ),

        "entropy": metric_values(
            records,
            "actor/entropy",
        ),

        "grad_norm": metric_values(
            records,
            "actor/grad_norm",
        ),

        "pg_loss": metric_values(
            records,
            "actor/pg_loss",
        ),

        "response_mean": metric_values(
            records,
            "response_length/mean",
        ),

        "response_clip": metric_values(
            records,
            "response_length/clip_ratio",
        ),

        "step_time": metric_values(
            records,
            "timing_s/step",
        ),

        "gen_time": metric_values(
            records,
            "timing_s/gen",
        ),

        "update_actor_time": metric_values(
            records,
            "timing_s/update_actor",
        ),

        "update_weights_time": metric_values(
            records,
            "timing_s/update_weights",
        ),

        "gpu_allocated": metric_values(
            records,
            "actor/perf/max_memory_allocated_gb",
        ),

        "gpu_reserved": metric_values(
            records,
            "actor/perf/max_memory_reserved_gb",
        ),

        "cpu_memory": metric_values(
            records,
            "actor/perf/cpu_memory_used_gb",
        ),

        "ppo_kl": metric_values(
            records,
            "actor/ppo_kl",
        ),

        "rollout_prob_diff": metric_values(
            records,
            "training/rollout_probs_diff_mean",
        ),

        "rollout_corr": metric_values(
            records,
            "training/rollout_actor_probs_pearson_corr",
        ),

        "group_size": group_size,
        
        "observed_steps": sorted(
            int(record["step"])
            for record in records
        ),

        "missing_steps": sorted(
            set(
                range(
                    int(records[0]["step"]),
                    int(records[-1]["step"]) + 1,
                )
            )
            - {
                int(record["step"])
                for record in records
            }
        ),
    }


# =============================================================================
# Window analysis
# =============================================================================

def analyze_window(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(records)

    all_fail = sum(
        record["group_type"] == "all_fail"
        for record in records
    )

    mixed = sum(
        record["group_type"] == "mixed"
        for record in records
    )

    all_pass = sum(
        record["group_type"] == "all_pass"
        for record in records
    )

    advantage = sum(
        record["has_advantage_signal"]
        for record in records
    )

    updates = sum(
        record["has_actor_update"]
        for record in records
    )

    positive = sum(
        record["is_positive"]
        for record in records
    )

    success_counts = [
        record["success_count"]
        for record in records
        if record["success_count"] is not None
    ]

    entropy = metric_values(
        records,
        "actor/entropy",
    )

    response_mean = metric_values(
        records,
        "response_length/mean",
    )

    score_mean = metric_values(
        records,
        "critic/score/mean",
    )

    grad_norm = metric_values(
        records,
        "actor/grad_norm",
    )

    return {
        "start": int(records[0]["step"]),
        "end": int(records[-1]["step"]),
        "n": total,

        "positive": positive,
        "all_fail": all_fail,
        "mixed": mixed,
        "all_pass": all_pass,
        "advantage": advantage,
        "updates": updates,

        "mean_success_count":
            safe_mean(success_counts),

        "mean_score":
            safe_mean(score_mean),

        "mean_entropy":
            safe_mean(entropy),

        "mean_response_length":
            safe_mean(response_mean),

        "mean_grad_norm":
            safe_mean(grad_norm),
    }

def build_windows(
    records: list[dict[str, Any]],
    window_size: int,
) -> list[dict[str, Any]]:
    """
    Build training windows using the actual global step number.

    Example for window_size=100:
        1-100
        101-200
        ...
        801-900

    Missing metric steps (e.g. checkpoint-boundary crashes) remain missing
    observations and do not shift subsequent window boundaries.
    """
    if window_size <= 0:
        raise ValueError(
            "window_size must be > 0"
        )

    if not records:
        return []

    first_step = int(records[0]["step"])
    last_step = int(records[-1]["step"])

    # Align windows to 1-based training-step boundaries.
    first_window_start = (
        ((first_step - 1) // window_size)
        * window_size
        + 1
    )

    windows = []

    for window_start in range(
        first_window_start,
        last_step + 1,
        window_size,
    ):
        window_end = (
            window_start
            + window_size
            - 1
        )

        chunk = [
            record
            for record in records
            if (
                window_start
                <= int(record["step"])
                <= window_end
            )
        ]

        if not chunk:
            continue

        result = analyze_window(chunk)

        # Use intended step boundaries rather than
        # first/last observed metric steps.
        result["start"] = window_start
        result["end"] = min(
            window_end,
            last_step,
        )

        expected_n = (
            result["end"]
            - result["start"]
            + 1
        )

        result["expected_n"] = expected_n
        result["missing_n"] = (
            expected_n
            - result["n"]
        )

        windows.append(result)

    return windows


# =============================================================================
# Reporting
# =============================================================================

def print_step_list(
    label: str,
    steps: list[int],
    max_items: int = 50,
) -> None:
    if not steps:
        print(f"{label:<34}: none")
        return

    if len(steps) <= max_items:
        text = ", ".join(map(str, steps))
    else:
        head = ", ".join(
            map(str, steps[:max_items])
        )
        text = (
            f"{head}, ... "
            f"(+{len(steps) - max_items} more)"
        )

    print(f"{label:<34}: {text}")


def print_global_summary(
    result: dict[str, Any],
) -> None:
    total = result["total_steps"]
    counter = result["group_counter"]

    all_fail = counter.get("all_fail", 0)
    mixed = counter.get("mixed", 0)
    all_pass = counter.get("all_pass", 0)
    unknown = counter.get("unknown", 0)

    print()
    print("=" * 88)
    print("Vanilla Planning-RLVR Full Training Analysis")
    print("=" * 88)
    
    print(
        f"Observed metric records            : "
        f"{total}"
    )

    print(
        f"Missing metric steps               : "
        f"{result['missing_steps']}"
    )

    print(
        f"Parsed training steps             : "
        f"{total}"
    )
    print(
        f"Step range                        : "
        f"{result['first_step']} -> "
        f"{result['last_step']}"
    )
    print(
        f"GRPO group size                   : "
        f"{result['group_size']}"
    )

    # -------------------------------------------------------------------------
    print()
    print("-" * 88)
    print("1. GRPO Reward-Group Structure")
    print("-" * 88)

    print(
        f"All-fail groups (0/N)             : "
        f"{all_fail:4d} / {total} "
        f"({fmt_pct(all_fail, total)})"
    )

    print(
        f"Mixed informative groups          : "
        f"{mixed:4d} / {total} "
        f"({fmt_pct(mixed, total)})"
    )

    print(
        f"All-pass groups (N/N)             : "
        f"{all_pass:4d} / {total} "
        f"({fmt_pct(all_pass, total)})"
    )

    if unknown:
        print(
            f"Unknown groups                     : "
            f"{unknown:4d} / {total} "
            f"({fmt_pct(unknown, total)})"
        )

    print(
        f"Any-positive groups                : "
        f"{len(result['positive_steps']):4d} / {total} "
        f"({fmt_pct(len(result['positive_steps']), total)})"
    )

    # -------------------------------------------------------------------------
    print()
    print("-" * 88)
    print("2. Success Count Distribution")
    print("-" * 88)

    success_counter = result["success_counter"]
    group_size = result["group_size"]

    for count in range(group_size + 1):
        frequency = success_counter.get(
            count,
            0,
        )

        print(
            f"{count:2d}/{group_size:<2d} successful rollouts"
            f"             : "
            f"{frequency:4d} "
            f"({fmt_pct(frequency, total)})"
        )

    # -------------------------------------------------------------------------
    print()
    print("-" * 88)
    print("3. Effective GRPO Learning Signal")
    print("-" * 88)

    informative = result["informative_steps"]
    advantage = result["advantage_steps"]
    updates = result["update_steps"]
    pg_steps = result["pg_steps"]

    print(
        f"Informative reward groups          : "
        f"{len(informative):4d} / {total} "
        f"({fmt_pct(len(informative), total)})"
    )

    print(
        f"Advantage-signal steps             : "
        f"{len(advantage):4d} / {total} "
        f"({fmt_pct(len(advantage), total)})"
    )

    print(
        f"Non-zero gradient steps            : "
        f"{len(updates):4d} / {total} "
        f"({fmt_pct(len(updates), total)})"
    )

    print(
        f"Non-zero PG-loss steps             : "
        f"{len(pg_steps):4d} / {total} "
        f"({fmt_pct(len(pg_steps), total)})"
    )

    print()
    print(
        f"Informative -> advantage           : "
        f"{len(result['informative_and_advantage'])}"
    )

    print(
        f"Informative but no advantage       : "
        f"{len(result['informative_without_advantage'])}"
    )

    print(
        f"Advantage without informative      : "
        f"{len(result['advantage_without_informative'])}"
    )

    print(
        f"Informative -> actor update        : "
        f"{len(result['informative_and_update'])}"
    )

    print(
        f"Informative but no actor update    : "
        f"{len(result['informative_without_update'])}"
    )

    print(
        f"Actor update without informative   : "
        f"{len(result['update_without_informative'])}"
    )

    print()

    print_step_list(
        "Informative step IDs",
        informative,
    )

    print_step_list(
        "Informative/no-update IDs",
        result["informative_without_update"],
    )

    print_step_list(
        "Update/non-informative IDs",
        result["update_without_informative"],
    )

    # -------------------------------------------------------------------------
    print()
    print("-" * 88)
    print("4. Optimization Dynamics")
    print("-" * 88)

    print(
        f"Mean grad norm                     : "
        f"{fmt(safe_mean(result['grad_norm']))}"
    )

    print(
        f"Median grad norm                   : "
        f"{fmt(safe_median(result['grad_norm']))}"
    )

    print(
        f"Max grad norm                      : "
        f"{fmt(safe_max(result['grad_norm']))}"
    )

    print(
        f"Mean |PG loss|                     : "
        f"{fmt(safe_mean([
            abs(x)
            for x in result['pg_loss']
        ]))}"
    )

    print(
        f"Mean PPO KL                        : "
        f"{fmt(safe_mean(result['ppo_kl']), 8)}"
    )

    # -------------------------------------------------------------------------
    print()
    print("-" * 88)
    print("5. Policy Entropy")
    print("-" * 88)

    entropy = result["entropy"]

    print(
        f"Initial entropy                    : "
        f"{fmt(entropy[0] if entropy else None)}"
    )

    print(
        f"Final entropy                      : "
        f"{fmt(entropy[-1] if entropy else None)}"
    )

    print(
        f"Mean entropy                       : "
        f"{fmt(safe_mean(entropy))}"
    )

    print(
        f"Min / Max entropy                  : "
        f"{fmt(safe_min(entropy))} / "
        f"{fmt(safe_max(entropy))}"
    )

    if len(entropy) >= 2:
        print(
            f"Endpoint entropy change            : "
            f"{entropy[-1] - entropy[0]:.6f}"
        )

    # -------------------------------------------------------------------------
    print()
    print("-" * 88)
    print("6. Response Length")
    print("-" * 88)

    print(
        f"Mean response length               : "
        f"{fmt(safe_mean(result['response_mean']), 2)}"
    )

    print(
        f"Mean response clipping ratio       : "
        f"{fmt(safe_mean(result['response_clip']), 4)}"
    )

    # -------------------------------------------------------------------------
    print()
    print("-" * 88)
    print("7. Runtime / Memory")
    print("-" * 88)

    print(
        f"Mean step time                     : "
        f"{fmt(safe_mean(result['step_time']), 3)} s"
    )

    print(
        f"Mean generation time               : "
        f"{fmt(safe_mean(result['gen_time']), 3)} s"
    )

    print(
        f"Mean actor-update time             : "
        f"{fmt(safe_mean(result['update_actor_time']), 3)} s"
    )

    print(
        f"Mean weight-sync time              : "
        f"{fmt(safe_mean(result['update_weights_time']), 3)} s"
    )

    print(
        f"Peak actor GPU allocated           : "
        f"{fmt(safe_max(result['gpu_allocated']), 3)} GiB"
    )

    print(
        f"Peak actor GPU reserved            : "
        f"{fmt(safe_max(result['gpu_reserved']), 3)} GiB"
    )

    print(
        f"Peak actor CPU memory              : "
        f"{fmt(safe_max(result['cpu_memory']), 3)} GiB"
    )

    # -------------------------------------------------------------------------
    print()
    print("-" * 88)
    print("8. Rollout / Actor Consistency")
    print("-" * 88)

    print(
        f"Mean rollout probability diff      : "
        f"{fmt(safe_mean(result['rollout_prob_diff']), 8)}"
    )

    print(
        f"Mean rollout/actor Pearson corr    : "
        f"{fmt(safe_mean(result['rollout_corr']), 8)}"
    )


def print_window_summary(
    windows: list[dict[str, Any]],
) -> None:
    print()
    print("=" * 118)
    print("Training Dynamics by Window")
    print("=" * 118)

    header = (
        f"{'Steps':<13}"
        f"{'N':>6}"
        f"{'Miss':>6}"
        f"{'Positive':>11}"
        f"{'AllFail':>10}"
        f"{'Mixed':>10}"
        f"{'AllPass':>10}"
        f"{'Adv':>8}"
        f"{'Update':>9}"
        f"{'Succ/N':>10}"
        f"{'Score':>10}"
        f"{'Entropy':>10}"
        f"{'RespLen':>10}"
    )

    print(header)
    print("-" * 118)

    for window in windows:
        step_range = (
            f"{window['start']}-"
            f"{window['end']}"
        )

        print(
            f"{step_range:<13}"
            f"{window['n']:>6d}"
            f"{window['missing_n']:>6d}"
            f"{window['positive']:>11d}"
            f"{window['all_fail']:>10d}"
            f"{window['mixed']:>10d}"
            f"{window['all_pass']:>10d}"
            f"{window['advantage']:>8d}"
            f"{window['updates']:>9d}"
            f"{fmt(window['mean_success_count'], 2):>10}"
            f"{fmt(window['mean_score'], 4):>10}"
            f"{fmt(window['mean_entropy'], 4):>10}"
            f"{fmt(window['mean_response_length'], 1):>10}"
        )

    print("=" * 118)


def print_diagnosis(
    result: dict[str, Any],
) -> None:
    total = result["total_steps"]

    informative = len(
        result["informative_steps"]
    )

    updates = len(
        result["update_steps"]
    )

    informative_rate = (
        informative / total
        if total
        else 0.0
    )

    print()
    print("=" * 88)
    print("Training Diagnosis")
    print("=" * 88)

    if informative == 0:
        print(
            "[CRITICAL] No mixed-reward GRPO group was observed."
        )
        print(
            "           Under binary group-relative reward, "
            "the run had effectively no useful preference signal."
        )
    else:
        print(
            "[PASS] Mixed-reward GRPO groups were observed."
        )

    print(
        f"[INFO] Informative-group density: "
        f"{100.0 * informative_rate:.2f}% "
        f"({informative}/{total})."
    )

    if informative_rate < 0.10:
        print(
            "[WARN] The effective GRPO signal is extremely sparse."
        )
    elif informative_rate < 0.20:
        print(
            "[WARN] The effective GRPO signal is sparse."
        )
    else:
        print(
            "[INFO] The run received a non-trivial density "
            "of informative reward groups."
        )

    if result["informative_without_advantage"]:
        print(
            "[CHECK] Some mixed-reward groups did not produce "
            "a detected advantage signal."
        )

    if result["advantage_without_informative"]:
        print(
            "[CHECK] Advantage signal occurred on groups classified "
            "as non-informative. Verify reward semantics or metrics."
        )

    if result["informative_without_update"]:
        print(
            "[CHECK] Some informative groups did not produce "
            "a non-zero actor gradient."
        )

    if result["update_without_informative"]:
        print(
            "[CHECK] Some actor updates occurred without a mixed "
            "binary-reward group. Inspect GRPO/reward semantics."
        )

    if updates == 0:
        print(
            "[CRITICAL] No non-zero actor gradient was detected."
        )
    else:
        print(
            f"[PASS] Non-zero actor updates were observed on "
            f"{updates}/{total} steps "
            f"({100.0 * updates / total:.2f}%)."
        )

    print()
    print(
        "IMPORTANT: Training dynamics alone do not establish "
        "planner improvement."
    )
    print(
        "Next: evaluate base and RL planner checkpoints on the "
        "same held-out problems with the same frozen coder."
    )

    print("=" * 88)


# =============================================================================
# CSV export
# =============================================================================

def write_step_csv(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_keys = set()

    for record in records:
        all_keys.update(record.keys())

    preferred = [
        "step",
        "success_count",
        "group_type",
        "is_positive",
        "is_informative",
        "has_advantage_signal",
        "has_actor_update",
        "has_pg_signal",
    ]

    remaining = sorted(
        key
        for key in all_keys
        if key not in preferred
    )

    fieldnames = preferred + remaining

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(record)


def write_window_csv(
    windows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not windows:
        return

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(windows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(windows)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a full vanilla Planning-RLVR "
            "training log."
        )
    )

    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Training stdout/stderr log.",
    )

    parser.add_argument(
        "--group-size",
        type=int,
        default=16,
        help=(
            "Number of planner rollouts per GRPO group. "
            "Default: 16."
        ),
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help=(
            "Number of training steps per dynamics window. "
            "Default: 100."
        ),
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional per-step CSV output.",
    )

    parser.add_argument(
        "--window-csv",
        type=Path,
        default=None,
        help="Optional window-level CSV output.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.log.exists():
        raise FileNotFoundError(
            f"Training log not found: {args.log}"
        )

    if args.group_size <= 1:
        raise ValueError(
            "--group-size must be > 1"
        )

    raw_records = load_training_records(
        args.log
    )

    if not raw_records:
        raise RuntimeError(
            "No training metric lines were found."
        )

    records = augment_records(
        raw_records,
        group_size=args.group_size,
    )

    result = analyze_global(
        records,
        group_size=args.group_size,
    )

    windows = build_windows(
        records,
        window_size=args.window_size,
    )

    print_global_summary(result)
    print_window_summary(windows)
    print_diagnosis(result)

    if args.csv is not None:
        write_step_csv(
            records,
            args.csv,
        )

        print(
            f"\n[OK] Per-step CSV saved: "
            f"{args.csv}"
        )

    if args.window_csv is not None:
        write_window_csv(
            windows,
            args.window_csv,
        )

        print(
            f"[OK] Window CSV saved: "
            f"{args.window_csv}"
        )


if __name__ == "__main__":
    main()