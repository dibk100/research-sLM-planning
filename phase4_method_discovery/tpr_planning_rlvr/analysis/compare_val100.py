"""
PYTHONPATH="$PWD:$HOME/workspace/LiveCodeBench" \
python \
phase4_method_discovery/tpr_planning_rlvr/analysis/compare_val100.py \
  --base phase4_method_discovery/vanilla_planning_rlvr/outputs/checkpoint_eval/base_val100.jsonl \
  --vanilla phase4_method_discovery/vanilla_planning_rlvr/outputs/checkpoint_eval/step900_val100.jsonl \
  --tpr \
phase4_method_discovery/tpr_planning_rlvr/outputs/checkpoint_eval/step900_val100.jsonl \
  --csv \
phase4_method_discovery/tpr_planning_rlvr/outputs/checkpoint_eval/base_vanilla_tpr_val100.csv

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12


def load_jsonl(path: Path, label: str) -> pd.DataFrame:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            required = {
                "problem_id",
                "passed",
                "passed_tests",
                "total_tests",
                "test_pass_ratio",
            }

            missing = required - obj.keys()
            if missing:
                raise RuntimeError(
                    f"{path}:{line_no}: missing fields: "
                    f"{sorted(missing)}"
                )

            rows.append(
                {
                    "problem_id": str(obj["problem_id"]),
                    f"{label}_passed": bool(obj["passed"]),
                    f"{label}_passed_tests": int(
                        obj["passed_tests"]
                    ),
                    f"{label}_total_tests": int(
                        obj["total_tests"]
                    ),
                    f"{label}_tpr": float(
                        obj["test_pass_ratio"]
                    ),
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(f"No records found: {path}")

    if df["problem_id"].duplicated().any():
        dup = df.loc[
            df["problem_id"].duplicated(),
            "problem_id",
        ].tolist()

        raise RuntimeError(
            f"Duplicate problem_id in {path}: {dup[:10]}"
        )

    return df


def validate_method(
    df: pd.DataFrame,
    label: str,
) -> None:
    passed = df[f"{label}_passed"]
    passed_tests = df[f"{label}_passed_tests"]
    total_tests = df[f"{label}_total_tests"]
    tpr = df[f"{label}_tpr"]

    if (total_tests <= 0).any():
        raise RuntimeError(
            f"{label}: total_tests <= 0 detected."
        )

    expected_tpr = passed_tests / total_tests

    if not np.allclose(
        tpr.to_numpy(),
        expected_tpr.to_numpy(),
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"{label}: test_pass_ratio is inconsistent "
            "with passed_tests / total_tests."
        )

    expected_passed = passed_tests == total_tests

    if not np.array_equal(
        passed.to_numpy(),
        expected_passed.to_numpy(),
    ):
        raise RuntimeError(
            f"{label}: passed flag is inconsistent "
            "with passed_tests == total_tests."
        )


def summarize(
    df: pd.DataFrame,
    label: str,
) -> dict:
    passed = df[f"{label}_passed"]
    tpr = df[f"{label}_tpr"]

    return {
        "method": label,
        "n": len(df),
        "passed": int(passed.sum()),
        "pass_rate": float(passed.mean()),
        "mean_tpr": float(tpr.mean()),
        "median_tpr": float(tpr.median()),
    }


def paired_stats(
    df: pd.DataFrame,
    a: str,
    b: str,
) -> dict:
    a_pass = df[f"{a}_passed"]
    b_pass = df[f"{b}_passed"]

    a_tpr = df[f"{a}_tpr"]
    b_tpr = df[f"{b}_tpr"]

    delta = b_tpr - a_tpr

    pp = int((a_pass & b_pass).sum())
    fp = int((~a_pass & b_pass).sum())
    pf = int((a_pass & ~b_pass).sum())
    ff = int((~a_pass & ~b_pass).sum())

    improved = int((delta > EPS).sum())
    unchanged = int((delta.abs() <= EPS).sum())
    degraded = int((delta < -EPS).sum())

    return {
        "comparison": f"{a} -> {b}",
        "PP": pp,
        "FP": fp,
        "PF": pf,
        "FF": ff,
        "tpr_improved": improved,
        "tpr_unchanged": unchanged,
        "tpr_degraded": degraded,
        "mean_delta_tpr": float(delta.mean()),
        "median_delta_tpr": float(delta.median()),
        "max_delta_tpr": float(delta.max()),
        "min_delta_tpr": float(delta.min()),
    }


def print_summary_table(
    summaries: list[dict],
) -> None:
    print()
    print("=" * 88)
    print("Aggregate Evaluation")
    print("=" * 88)

    print(
        f"{'Method':<12}"
        f"{'N':>6}"
        f"{'Pass':>8}"
        f"{'Pass@1':>12}"
        f"{'Mean TPR':>14}"
        f"{'Median TPR':>14}"
    )
    print("-" * 88)

    for s in summaries:
        print(
            f"{s['method']:<12}"
            f"{s['n']:>6}"
            f"{s['passed']:>8}"
            f"{100.0 * s['pass_rate']:>11.2f}%"
            f"{s['mean_tpr']:>14.6f}"
            f"{s['median_tpr']:>14.6f}"
        )


def print_pairwise(
    result: dict,
) -> None:
    print()
    print("-" * 88)
    print(result["comparison"])
    print("-" * 88)

    print(
        "Exact-solve transitions:"
    )
    print(
        f"  PASS -> PASS : {result['PP']}"
    )
    print(
        f"  FAIL -> PASS : {result['FP']}"
    )
    print(
        f"  PASS -> FAIL : {result['PF']}"
    )
    print(
        f"  FAIL -> FAIL : {result['FF']}"
    )

    print()
    print("Full-test TPR transitions:")
    print(
        f"  Improved     : {result['tpr_improved']}"
    )
    print(
        f"  Unchanged    : {result['tpr_unchanged']}"
    )
    print(
        f"  Degraded     : {result['tpr_degraded']}"
    )
    print(
        f"  Mean delta   : "
        f"{result['mean_delta_tpr']:+.6f}"
    )
    print(
        f"  Median delta : "
        f"{result['median_delta_tpr']:+.6f}"
    )
    print(
        f"  Min / Max    : "
        f"{result['min_delta_tpr']:+.6f} / "
        f"{result['max_delta_tpr']:+.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--vanilla",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--tpr",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base = load_jsonl(args.base, "Base")
    vanilla = load_jsonl(args.vanilla, "Vanilla")
    tpr = load_jsonl(args.tpr, "TPR")

    validate_method(base, "Base")
    validate_method(vanilla, "Vanilla")
    validate_method(tpr, "TPR")

    # Inner join by problem_id rather than relying on file order.
    merged = (
        base
        .merge(
            vanilla,
            on="problem_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            tpr,
            on="problem_id",
            how="inner",
            validate="one_to_one",
        )
    )

    expected_n = {
        len(base),
        len(vanilla),
        len(tpr),
    }

    if len(expected_n) != 1:
        raise RuntimeError(
            "The three files have different numbers "
            "of problems."
        )

    expected = len(base)

    if len(merged) != expected:
        raise RuntimeError(
            "Problem sets do not match exactly: "
            f"expected={expected}, merged={len(merged)}"
        )

    # Evaluation protocol should use exactly the same
    # number of tests for a given problem across methods.
    if not (
        (
            merged["Base_total_tests"]
            == merged["Vanilla_total_tests"]
        ).all()
        and (
            merged["Base_total_tests"]
            == merged["TPR_total_tests"]
        ).all()
    ):
        raise RuntimeError(
            "total_tests mismatch across methods."
        )

    summaries = [
        summarize(merged, "Base"),
        summarize(merged, "Vanilla"),
        summarize(merged, "TPR"),
    ]

    print()
    print("=" * 88)
    print("Base vs Vanilla vs TPR — val100")
    print("=" * 88)

    print(
        f"Matched problems : {len(merged)}"
    )

    print_summary_table(
        summaries
    )

    comparisons = [
        paired_stats(
            merged,
            "Base",
            "Vanilla",
        ),
        paired_stats(
            merged,
            "Base",
            "TPR",
        ),
        paired_stats(
            merged,
            "Vanilla",
            "TPR",
        ),
    ]

    print()
    print("=" * 88)
    print("Paired Comparisons")
    print("=" * 88)

    for result in comparisons:
        print_pairwise(result)

    # Useful per-problem table for later qualitative analysis.
    merged["delta_tpr_base_to_vanilla"] = (
        merged["Vanilla_tpr"]
        - merged["Base_tpr"]
    )

    merged["delta_tpr_base_to_tpr"] = (
        merged["TPR_tpr"]
        - merged["Base_tpr"]
    )

    merged["delta_tpr_vanilla_to_tpr"] = (
        merged["TPR_tpr"]
        - merged["Vanilla_tpr"]
    )

    if args.csv is not None:
        args.csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        merged.to_csv(
            args.csv,
            index=False,
        )

        print()
        print(
            f"[OK] Paired CSV saved: {args.csv}"
        )


if __name__ == "__main__":
    main()