"""
python \
  phase4_method_discovery/tpr_planning_rlvr/analysis/analyze_full_training.py \
  --log \
  phase4_method_discovery/tpr_planning_rlvr/outputs/full900_training.log \
  --group-size 16 \
  --window-size 100 \
  --expected-steps 900 \
  --csv \
  phase4_method_discovery/tpr_planning_rlvr/outputs/full900_metrics.csv
"""
# phase4_method_discovery/tpr_planning_rlvr/analysis/analyze_full_training.py

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


EPS = 1.0e-8

STEP_PATTERN = re.compile(
    r"(?:^|\s)step:(\d+)\s+-\s+"
)

METRIC_PATTERN = re.compile(
    r"([A-Za-z0-9_./-]+):"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


# ======================================================================
# Parsing
# ======================================================================


def parse_metric_line(
    line: str,
) -> dict[str, float | int] | None:
    step_match = STEP_PATTERN.search(line)

    if step_match is None:
        return None

    step = int(
        step_match.group(1)
    )

    record: dict[str, float | int] = {
        "step": step,
    }

    for key, value in METRIC_PATTERN.findall(line):
        try:
            record[key] = float(value)
        except ValueError:
            continue

    return record


def parse_training_log(
    path: Path,
) -> pd.DataFrame:
    records: dict[int, dict[str, float | int]] = {}

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            record = parse_metric_line(line)

            if record is None:
                continue

            step = int(record["step"])

            # If the same metric step somehow appears more than once,
            # keep the last complete record.
            records[step] = record

    if not records:
        raise RuntimeError(
            f"No training metric records found in: {path}"
        )

    df = pd.DataFrame(
        [
            records[step]
            for step in sorted(records)
        ]
    )

    return df


# ======================================================================
# Utilities
# ======================================================================


def metric(
    row: pd.Series,
    key: str,
    default: float = float("nan"),
) -> float:
    value = row.get(key, default)

    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    return value


def finite(
    x: float,
) -> bool:
    return math.isfinite(x)


def safe_mean(
    series: pd.Series,
) -> float:
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    return float(
        values.mean()
    )


def safe_median(
    series: pd.Series,
) -> float:
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    return float(
        values.median()
    )


def pct(
    n: int,
    d: int,
) -> float:
    if d <= 0:
        return 0.0

    return 100.0 * n / d


# ======================================================================
# Derived TPR / GRPO diagnostics
# ======================================================================


def add_derived_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    required = [
        "critic/score/mean",
        "critic/score/max",
        "critic/score/min",
    ]

    missing = [
        key
        for key in required
        if key not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required TPR score metrics: "
            + ", ".join(missing)
        )

    score_mean = pd.to_numeric(
        df["critic/score/mean"],
        errors="coerce",
    )

    score_max = pd.to_numeric(
        df["critic/score/max"],
        errors="coerce",
    )

    score_min = pd.to_numeric(
        df["critic/score/min"],
        errors="coerce",
    )

    df["tpr_group_mean"] = score_mean
    df["tpr_group_min"] = score_min
    df["tpr_group_max"] = score_max

    df["tpr_group_range"] = (
        score_max - score_min
    )

    # ------------------------------------------------------------------
    # Reward-group taxonomy
    # ------------------------------------------------------------------

    df["group_informative"] = (
        df["tpr_group_range"].abs() > EPS
    )

    df["group_all_zero"] = (
        (score_min.abs() <= EPS)
        & (score_max.abs() <= EPS)
    )

    df["group_all_one"] = (
        ((score_min - 1.0).abs() <= EPS)
        & ((score_max - 1.0).abs() <= EPS)
    )

    df["group_constant_partial"] = (
        (~df["group_informative"])
        & (~df["group_all_zero"])
        & (~df["group_all_one"])
        & (score_mean > EPS)
        & (score_mean < 1.0 - EPS)
    )

    df["group_any_positive"] = (
        score_max > EPS
    )

    df["group_any_perfect"] = (
        score_max >= 1.0 - EPS
    )

    df["group_has_partial_reward"] = (
        (
            (score_min > EPS)
            & (score_min < 1.0 - EPS)
        )
        |
        (
            (score_mean > EPS)
            & (score_mean < 1.0 - EPS)
        )
        |
        (
            (score_max > EPS)
            & (score_max < 1.0 - EPS)
        )
    )

    # ------------------------------------------------------------------
    # Advantage signal
    # ------------------------------------------------------------------

    adv_min = pd.to_numeric(
        df.get(
            "critic/advantages/min",
            pd.Series(
                float("nan"),
                index=df.index,
            ),
        ),
        errors="coerce",
    )

    adv_max = pd.to_numeric(
        df.get(
            "critic/advantages/max",
            pd.Series(
                float("nan"),
                index=df.index,
            ),
        ),
        errors="coerce",
    )

    df["advantage_signal"] = (
        (adv_min.abs() > EPS)
        | (adv_max.abs() > EPS)
    )

    # ------------------------------------------------------------------
    # Actor optimization signal
    # ------------------------------------------------------------------

    grad_norm = pd.to_numeric(
        df.get(
            "actor/grad_norm",
            pd.Series(
                float("nan"),
                index=df.index,
            ),
        ),
        errors="coerce",
    )

    pg_loss = pd.to_numeric(
        df.get(
            "actor/pg_loss",
            pd.Series(
                float("nan"),
                index=df.index,
            ),
        ),
        errors="coerce",
    )

    df["nonzero_grad"] = (
        grad_norm.abs() > EPS
    )

    df["nonzero_pg_loss"] = (
        pg_loss.abs() > EPS
    )

    return df


# ======================================================================
# Full-run summary
# ======================================================================


def print_full_summary(
    df: pd.DataFrame,
    *,
    expected_steps: int,
    group_size: int,
) -> None:
    observed_steps = set(
        df["step"].astype(int)
    )

    expected = set(
        range(1, expected_steps + 1)
    )

    missing = sorted(
        expected - observed_steps
    )

    n = len(df)

    all_zero = int(
        df["group_all_zero"].sum()
    )

    informative = int(
        df["group_informative"].sum()
    )

    constant_partial = int(
        df["group_constant_partial"].sum()
    )

    all_one = int(
        df["group_all_one"].sum()
    )

    any_positive = int(
        df["group_any_positive"].sum()
    )

    any_perfect = int(
        df["group_any_perfect"].sum()
    )

    partial = int(
        df["group_has_partial_reward"].sum()
    )

    advantage = int(
        df["advantage_signal"].sum()
    )

    nonzero_grad = int(
        df["nonzero_grad"].sum()
    )

    nonzero_pg = int(
        df["nonzero_pg_loss"].sum()
    )

    print()
    print("=" * 88)
    print("TPR Planning-RLVR Full Training Analysis")
    print("=" * 88)

    print(
        f"Observed metric records            : {n}"
    )
    print(
        f"Missing metric steps               : {missing}"
    )
    print(
        f"Step range                         : "
        f"{int(df['step'].min())} -> "
        f"{int(df['step'].max())}"
    )
    print(
        f"GRPO group size                    : {group_size}"
    )

    # ------------------------------------------------------------------
    print()
    print("-" * 88)
    print("1. TPR Reward-Group Structure")
    print("-" * 88)

    print(
        f"All-zero groups                    : "
        f"{all_zero:4d} / {n} "
        f"({pct(all_zero, n):.2f}%)"
    )

    print(
        f"Informative groups (max > min)     : "
        f"{informative:4d} / {n} "
        f"({pct(informative, n):.2f}%)"
    )

    print(
        f"Constant partial-reward groups     : "
        f"{constant_partial:4d} / {n} "
        f"({pct(constant_partial, n):.2f}%)"
    )

    print(
        f"All-one groups                     : "
        f"{all_one:4d} / {n} "
        f"({pct(all_one, n):.2f}%)"
    )

    print(
        f"Any-positive groups                : "
        f"{any_positive:4d} / {n} "
        f"({pct(any_positive, n):.2f}%)"
    )

    print(
        f"Any-perfect-rollout groups         : "
        f"{any_perfect:4d} / {n} "
        f"({pct(any_perfect, n):.2f}%)"
    )

    print(
        f"Groups exposing partial reward     : "
        f"{partial:4d} / {n} "
        f"({pct(partial, n):.2f}%)"
    )

    # ------------------------------------------------------------------
    print()
    print("-" * 88)
    print("2. Dense Reward Statistics")
    print("-" * 88)

    print(
        "Mean rollout/group TPR             : "
        f"{safe_mean(df['tpr_group_mean']):.6f}"
    )

    print(
        "Median group TPR                   : "
        f"{safe_median(df['tpr_group_mean']):.6f}"
    )

    print(
        "Mean group min TPR                 : "
        f"{safe_mean(df['tpr_group_min']):.6f}"
    )

    print(
        "Mean group max TPR                 : "
        f"{safe_mean(df['tpr_group_max']):.6f}"
    )

    print(
        "Mean within-group reward range     : "
        f"{safe_mean(df['tpr_group_range']):.6f}"
    )

    print(
        "Median within-group reward range   : "
        f"{safe_median(df['tpr_group_range']):.6f}"
    )

    print(
        "Max within-group reward range      : "
        f"{float(df['tpr_group_range'].max()):.6f}"
    )

    # ------------------------------------------------------------------
    print()
    print("-" * 88)
    print("3. Effective GRPO Learning Signal")
    print("-" * 88)

    print(
        f"Reward-variance steps              : "
        f"{informative:4d} / {n} "
        f"({pct(informative, n):.2f}%)"
    )

    print(
        f"Advantage-signal steps             : "
        f"{advantage:4d} / {n} "
        f"({pct(advantage, n):.2f}%)"
    )

    print(
        f"Non-zero gradient steps            : "
        f"{nonzero_grad:4d} / {n} "
        f"({pct(nonzero_grad, n):.2f}%)"
    )

    print(
        f"Non-zero PG-loss steps             : "
        f"{nonzero_pg:4d} / {n} "
        f"({pct(nonzero_pg, n):.2f}%)"
    )

    informative_adv = int(
        (
            df["group_informative"]
            & df["advantage_signal"]
        ).sum()
    )

    informative_no_adv = int(
        (
            df["group_informative"]
            & ~df["advantage_signal"]
        ).sum()
    )

    adv_without_info = int(
        (
            ~df["group_informative"]
            & df["advantage_signal"]
        ).sum()
    )

    informative_update = int(
        (
            df["group_informative"]
            & df["nonzero_grad"]
        ).sum()
    )

    informative_no_update = int(
        (
            df["group_informative"]
            & ~df["nonzero_grad"]
        ).sum()
    )

    update_without_info = int(
        (
            ~df["group_informative"]
            & df["nonzero_grad"]
        ).sum()
    )

    print()
    print(
        f"Informative -> advantage           : "
        f"{informative_adv}"
    )
    print(
        f"Informative but no advantage       : "
        f"{informative_no_adv}"
    )
    print(
        f"Advantage without informative      : "
        f"{adv_without_info}"
    )
    print(
        f"Informative -> actor update        : "
        f"{informative_update}"
    )
    print(
        f"Informative but no actor update    : "
        f"{informative_no_update}"
    )
    print(
        f"Actor update without informative   : "
        f"{update_without_info}"
    )

    # ------------------------------------------------------------------
    print()
    print("-" * 88)
    print("4. Optimization Dynamics")
    print("-" * 88)

    if "actor/grad_norm" in df.columns:
        grad = pd.to_numeric(
            df["actor/grad_norm"],
            errors="coerce",
        )

        print(
            f"Mean grad norm                     : "
            f"{grad.mean():.6f}"
        )
        print(
            f"Median grad norm                   : "
            f"{grad.median():.6f}"
        )
        print(
            f"Max grad norm                      : "
            f"{grad.max():.6f}"
        )

    if "actor/pg_loss" in df.columns:
        pg = pd.to_numeric(
            df["actor/pg_loss"],
            errors="coerce",
        )

        print(
            f"Mean |PG loss|                     : "
            f"{pg.abs().mean():.6f}"
        )

    if "actor/ppo_kl" in df.columns:
        print(
            f"Mean PPO KL                        : "
            f"{safe_mean(df['actor/ppo_kl']):.8f}"
        )

    # ------------------------------------------------------------------
    print()
    print("-" * 88)
    print("5. Policy Entropy")
    print("-" * 88)

    if "actor/entropy" in df.columns:
        entropy = pd.to_numeric(
            df["actor/entropy"],
            errors="coerce",
        )

        print(
            f"Initial entropy                    : "
            f"{entropy.iloc[0]:.6f}"
        )

        print(
            f"Final entropy                      : "
            f"{entropy.iloc[-1]:.6f}"
        )

        print(
            f"Mean entropy                       : "
            f"{entropy.mean():.6f}"
        )

        print(
            f"Min / Max entropy                  : "
            f"{entropy.min():.6f} / "
            f"{entropy.max():.6f}"
        )

        print(
            f"Endpoint entropy change            : "
            f"{entropy.iloc[-1] - entropy.iloc[0]:+.6f}"
        )

    # ------------------------------------------------------------------
    print()
    print("-" * 88)
    print("6. Response Length")
    print("-" * 88)

    if "response_length/mean" in df.columns:
        print(
            f"Mean response length               : "
            f"{safe_mean(df['response_length/mean']):.2f}"
        )

    if "response_length/clip_ratio" in df.columns:
        print(
            f"Mean response clipping ratio       : "
            f"{safe_mean(df['response_length/clip_ratio']):.4f}"
        )

    # ------------------------------------------------------------------
    print()
    print("-" * 88)
    print("7. Runtime / Memory")
    print("-" * 88)

    timing_keys = [
        (
            "Mean step time",
            "timing_s/step",
        ),
        (
            "Mean generation time",
            "timing_s/gen",
        ),
        (
            "Mean actor-update time",
            "timing_s/update_actor",
        ),
        (
            "Mean weight-sync time",
            "timing_s/update_weights",
        ),
    ]

    for label, key in timing_keys:
        if key in df.columns:
            print(
                f"{label:<35}: "
                f"{safe_mean(df[key]):.3f} s"
            )

    memory_keys = [
        (
            "Peak actor GPU allocated",
            "actor/perf/max_memory_allocated_gb",
        ),
        (
            "Peak actor GPU reserved",
            "actor/perf/max_memory_reserved_gb",
        ),
        (
            "Peak actor CPU memory",
            "actor/perf/cpu_memory_used_gb",
        ),
    ]

    for label, key in memory_keys:
        if key in df.columns:
            value = pd.to_numeric(
                df[key],
                errors="coerce",
            ).max()

            print(
                f"{label:<35}: "
                f"{value:.3f} GiB"
            )

    # ------------------------------------------------------------------
    print()
    print("-" * 88)
    print("8. Rollout / Actor Consistency")
    print("-" * 88)

    if (
        "training/rollout_probs_diff_mean"
        in df.columns
    ):
        print(
            "Mean rollout probability diff      : "
            f"{safe_mean(df['training/rollout_probs_diff_mean']):.8f}"
        )

    if (
        "training/rollout_actor_probs_pearson_corr"
        in df.columns
    ):
        print(
            "Mean rollout/actor Pearson corr    : "
            f"{safe_mean(df['training/rollout_actor_probs_pearson_corr']):.8f}"
        )


# ======================================================================
# Window analysis
# ======================================================================


def print_window_analysis(
    df: pd.DataFrame,
    *,
    expected_steps: int,
    window_size: int,
) -> None:
    print()
    print("=" * 132)
    print("TPR Training Dynamics by Window")
    print("=" * 132)

    header = (
        f"{'Steps':<13}"
        f"{'N':>5}"
        f"{'All0':>7}"
        f"{'Inform':>8}"
        f"{'ConstP':>8}"
        f"{'All1':>7}"
        f"{'Any+':>7}"
        f"{'Adv':>7}"
        f"{'Update':>8}"
        f"{'MeanTPR':>10}"
        f"{'Range':>9}"
        f"{'Entropy':>10}"
        f"{'RespLen':>10}"
        f"{'CPU':>9}"
    )

    print(header)
    print("-" * 132)

    for start in range(
        1,
        expected_steps + 1,
        window_size,
    ):
        end = min(
            start + window_size - 1,
            expected_steps,
        )

        sub = df[
            (df["step"] >= start)
            & (df["step"] <= end)
        ]

        n = len(sub)

        if n == 0:
            continue

        all_zero = int(
            sub["group_all_zero"].sum()
        )

        informative = int(
            sub["group_informative"].sum()
        )

        constant_partial = int(
            sub["group_constant_partial"].sum()
        )

        all_one = int(
            sub["group_all_one"].sum()
        )

        any_positive = int(
            sub["group_any_positive"].sum()
        )

        advantage = int(
            sub["advantage_signal"].sum()
        )

        update = int(
            sub["nonzero_grad"].sum()
        )

        mean_tpr = safe_mean(
            sub["tpr_group_mean"]
        )

        mean_range = safe_mean(
            sub["tpr_group_range"]
        )

        entropy = (
            safe_mean(
                sub["actor/entropy"]
            )
            if "actor/entropy" in sub.columns
            else float("nan")
        )

        response_len = (
            safe_mean(
                sub["response_length/mean"]
            )
            if "response_length/mean" in sub.columns
            else float("nan")
        )

        cpu_memory = (
            safe_mean(
                sub["actor/perf/cpu_memory_used_gb"]
            )
            if (
                "actor/perf/cpu_memory_used_gb"
                in sub.columns
            )
            else float("nan")
        )

        print(
            f"{start}-{end:<8}"
            f"{n:>5}"
            f"{all_zero:>7}"
            f"{informative:>8}"
            f"{constant_partial:>8}"
            f"{all_one:>7}"
            f"{any_positive:>7}"
            f"{advantage:>7}"
            f"{update:>8}"
            f"{mean_tpr:>10.4f}"
            f"{mean_range:>9.4f}"
            f"{entropy:>10.4f}"
            f"{response_len:>10.1f}"
            f"{cpu_memory:>9.2f}"
        )

    print("=" * 132)


# ======================================================================
# Diagnosis
# ======================================================================


def print_diagnosis(
    df: pd.DataFrame,
) -> None:
    n = len(df)

    informative = int(
        df["group_informative"].sum()
    )

    all_zero = int(
        df["group_all_zero"].sum()
    )

    partial = int(
        df["group_has_partial_reward"].sum()
    )

    advantage = int(
        df["advantage_signal"].sum()
    )

    update = int(
        df["nonzero_grad"].sum()
    )

    print()
    print("=" * 88)
    print("TPR Training Diagnosis")
    print("=" * 88)

    if informative > 0:
        print(
            "[PASS] Reward-varying TPR groups were observed."
        )
    else:
        print(
            "[FAIL] No within-group TPR variation was observed."
        )

    print(
        "[INFO] Informative-group density: "
        f"{pct(informative, n):.2f}% "
        f"({informative}/{n})."
    )

    print(
        "[INFO] All-zero-group density: "
        f"{pct(all_zero, n):.2f}% "
        f"({all_zero}/{n})."
    )

    print(
        "[INFO] Partial-reward exposure: "
        f"{pct(partial, n):.2f}% "
        f"({partial}/{n})."
    )

    print(
        "[INFO] Advantage-signal density: "
        f"{pct(advantage, n):.2f}% "
        f"({advantage}/{n})."
    )

    print(
        "[INFO] Non-zero actor-gradient density: "
        f"{pct(update, n):.2f}% "
        f"({update}/{n})."
    )

    print()

    print(
        "Vanilla comparison reference:"
    )

    print(
        "  Vanilla informative groups = "
        "256 / 897 = 28.54%"
    )

    print(
        "Compare the TPR informative-group density above "
        "against 28.54% to determine whether dense reward "
        "actually increased usable GRPO credit signal."
    )

    print()

    print(
        "IMPORTANT: Denser training signal alone does not "
        "establish better planning."
    )

    print(
        "Next: evaluate Base, Vanilla-step900, and "
        "TPR-step900 on the same held-out val100."
    )

    print("=" * 88)


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze full TPR Planning-RLVR training."
        )
    )

    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Canonical full training log.",
    )

    parser.add_argument(
        "--group-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--expected-steps",
        type=int,
        default=900,
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.log.exists():
        raise FileNotFoundError(
            f"Training log not found: {args.log}"
        )

    if args.group_size <= 0:
        raise ValueError(
            "--group-size must be > 0."
        )

    if args.window_size <= 0:
        raise ValueError(
            "--window-size must be > 0."
        )

    df = parse_training_log(
        args.log
    )

    df = add_derived_columns(
        df
    )

    print_full_summary(
        df,
        expected_steps=args.expected_steps,
        group_size=args.group_size,
    )

    print_window_analysis(
        df,
        expected_steps=args.expected_steps,
        window_size=args.window_size,
    )

    print_diagnosis(
        df
    )

    if args.csv is not None:
        args.csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            args.csv,
            index=False,
        )

        print()
        print(
            f"[OK] Per-step CSV saved: {args.csv}"
        )


if __name__ == "__main__":
    main()