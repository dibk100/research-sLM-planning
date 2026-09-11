"""
Base
phase4_method_discovery/vanilla_planning_rlvr/outputs/checkpoint_eval/base_val100.jsonl

Vanilla Step900
phase4_method_discovery/vanilla_planning_rlvr/outputs/checkpoint_eval/step900_val100.jsonl

TPR Step900
phase4_method_discovery/tpr_planning_rlvr/outputs/checkpoint_eval/step900_val100.jsonl

PYTHONPATH=. python \
  phase4_method_discovery/trajectory_analysis/scripts/select_transition_cases.py
  
  
[Record counts]
  Base    : 100
  Vanilla : 100
  TPR     : 100

[Common problems] 100

========================================================================
Transition Summary
========================================================================
PPP stable_success          10 (10.00%)
PPF tpr_regression           0 ( 0.00%)
PFP vanilla_regression       0 ( 0.00%)
PFF both_rl_regression       0 ( 0.00%)
FPP common_rl_recovery       1 ( 1.00%)
FPF vanilla_only_recovery    0 ( 0.00%)
FFP tpr_only_recovery        0 ( 0.00%)
FFF persistent_failure      89 (89.00%)
"""
#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_BASE = (
    "phase4_method_discovery/vanilla_planning_rlvr/"
    "outputs/checkpoint_eval/base_val100.jsonl"
)

DEFAULT_VANILLA = (
    "phase4_method_discovery/vanilla_planning_rlvr/"
    "outputs/checkpoint_eval/step900_val100.jsonl"
)

DEFAULT_TPR = (
    "phase4_method_discovery/tpr_planning_rlvr/"
    "outputs/checkpoint_eval/step900_val100.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    "phase4_method_discovery/trajectory_analysis/outputs"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Join Base / Vanilla / TPR evaluation JSONLs by problem_id "
            "and classify pass/fail transitions."
        )
    )

    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--vanilla", default=DEFAULT_VANILLA)
    parser.add_argument("--tpr", default=DEFAULT_TPR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    return parser.parse_args()


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            problem_id = row.get("problem_id")
            if not problem_id:
                raise ValueError(
                    f"{path}:{line_no}: missing problem_id"
                )

            if problem_id in records:
                raise ValueError(
                    f"{path}:{line_no}: duplicate problem_id={problem_id}"
                )

            records[problem_id] = row

    return records


def validate_common_problem(
    problem_id: str,
    base: dict[str, Any],
    vanilla: dict[str, Any],
    tpr: dict[str, Any],
):
    """
    Basic integrity checks.

    The three evaluation records should refer to the same problem and
    evaluation test set.
    """

    for name, row in [
        ("base", base),
        ("vanilla", vanilla),
        ("tpr", tpr),
    ]:
        if row.get("problem_id") != problem_id:
            raise ValueError(
                f"{problem_id}: {name} problem_id mismatch"
            )

    problems = {
        base.get("problem"),
        vanilla.get("problem"),
        tpr.get("problem"),
    }

    if len(problems) != 1:
        raise ValueError(
            f"{problem_id}: problem statement mismatch"
        )

    totals = {
        base.get("total_tests"),
        vanilla.get("total_tests"),
        tpr.get("total_tests"),
    }

    if len(totals) != 1:
        raise ValueError(
            f"{problem_id}: total_tests mismatch: {totals}"
        )


def pass_symbol(passed: bool) -> str:
    return "P" if bool(passed) else "F"


def transition_label(
    base_passed: bool,
    vanilla_passed: bool,
    tpr_passed: bool,
) -> str:
    return (
        f"{pass_symbol(base_passed)}"
        f"{pass_symbol(vanilla_passed)}"
        f"{pass_symbol(tpr_passed)}"
    )


TRANSITION_DESCRIPTIONS = {
    "PPP": "stable_success",
    "PPF": "tpr_regression",
    "PFP": "vanilla_regression",
    "PFF": "both_rl_regression",
    "FPP": "common_rl_recovery",
    "FPF": "vanilla_only_recovery",
    "FFP": "tpr_only_recovery",
    "FFF": "persistent_failure",
}


def extract_plan(row: dict[str, Any]) -> str | None:
    """
    Extract the generated planner output from strategy_trace.

    We avoid assuming its exact position in the trace and instead search
    for the plan_generation stage.
    """

    trace = row.get("strategy_trace") or []

    for item in trace:
        if not isinstance(item, dict):
            continue

        if item.get("name") != "plan_generation":
            continue

        # Try likely output field names.
        for key in (
            "output",
            "raw_output",
            "generated_text",
            "response",
            "plan",
            "text",
        ):
            value = item.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def compact_run(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": row.get("model_name"),
        "passed": bool(row.get("passed")),
        "status": row.get("status"),
        "passed_tests": row.get("passed_tests"),
        "total_tests": row.get("total_tests"),
        "test_pass_ratio": row.get("test_pass_ratio"),
        "plan": extract_plan(row),
        "code": row.get("extracted_code"),
        "error_message": row.get("error_message"),
    }


def build_candidate(
    problem_id: str,
    base: dict[str, Any],
    vanilla: dict[str, Any],
    tpr: dict[str, Any],
) -> dict[str, Any]:

    base_passed = bool(base.get("passed"))
    vanilla_passed = bool(vanilla.get("passed"))
    tpr_passed = bool(tpr.get("passed"))

    transition = transition_label(
        base_passed,
        vanilla_passed,
        tpr_passed,
    )

    base_tpr = float(base.get("test_pass_ratio") or 0.0)
    vanilla_tpr = float(vanilla.get("test_pass_ratio") or 0.0)
    tpr_tpr = float(tpr.get("test_pass_ratio") or 0.0)

    return {
        "problem_id": problem_id,
        "title": base.get("title"),
        "dataset": base.get("dataset"),
        "platform": base.get("platform"),
        "difficulty": base.get("difficulty"),
        "rating": base.get("rating"),

        "transition": transition,
        "transition_description": TRANSITION_DESCRIPTIONS[transition],

        "problem": base.get("problem"),

        "base": compact_run(base),
        "vanilla": compact_run(vanilla),
        "tpr": compact_run(tpr),

        "delta": {
            "vanilla_vs_base_tpr": vanilla_tpr - base_tpr,
            "tpr_vs_base_tpr": tpr_tpr - base_tpr,
            "tpr_vs_vanilla_tpr": tpr_tpr - vanilla_tpr,
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def write_summary_csv(
    path: Path,
    candidates: list[dict[str, Any]],
):
    counts = Counter(
        row["transition"]
        for row in candidates
    )

    total = len(candidates)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "transition",
            "description",
            "count",
            "ratio",
        ])

        for transition in [
            "PPP",
            "PPF",
            "PFP",
            "PFF",
            "FPP",
            "FPF",
            "FFP",
            "FFF",
        ]:
            count = counts.get(transition, 0)

            writer.writerow([
                transition,
                TRANSITION_DESCRIPTIONS[transition],
                count,
                count / total if total else 0.0,
            ])


def write_ranking_csv(
    path: Path,
    candidates: list[dict[str, Any]],
):
    """
    Human-readable index for choosing qualitative cases.
    """

    rows = []

    for row in candidates:
        rows.append({
            "problem_id": row["problem_id"],
            "transition": row["transition"],
            "description": row["transition_description"],

            "base_passed": row["base"]["passed"],
            "vanilla_passed": row["vanilla"]["passed"],
            "tpr_passed": row["tpr"]["passed"],

            "base_tpr": row["base"]["test_pass_ratio"],
            "vanilla_tpr": row["vanilla"]["test_pass_ratio"],
            "tpr_tpr": row["tpr"]["test_pass_ratio"],

            "vanilla_vs_base_tpr":
                row["delta"]["vanilla_vs_base_tpr"],

            "tpr_vs_base_tpr":
                row["delta"]["tpr_vs_base_tpr"],

            "tpr_vs_vanilla_tpr":
                row["delta"]["tpr_vs_vanilla_tpr"],

            "base_status": row["base"]["status"],
            "vanilla_status": row["vanilla"]["status"],
            "tpr_status": row["tpr"]["status"],

            "base_plan_found":
                row["base"]["plan"] is not None,

            "vanilla_plan_found":
                row["vanilla"]["plan"] is not None,

            "tpr_plan_found":
                row["tpr"]["plan"] is not None,
        })

    rows.sort(
        key=lambda x: (
            x["transition"],
            -abs(x["tpr_vs_base_tpr"]),
            x["problem_id"],
        )
    )

    if not rows:
        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    base_path = Path(args.base)
    vanilla_path = Path(args.vanilla)
    tpr_path = Path(args.tpr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Load]")
    print(f"  Base    : {base_path}")
    print(f"  Vanilla : {vanilla_path}")
    print(f"  TPR     : {tpr_path}")

    base_records = load_jsonl(base_path)
    vanilla_records = load_jsonl(vanilla_path)
    tpr_records = load_jsonl(tpr_path)

    print()
    print("[Record counts]")
    print(f"  Base    : {len(base_records)}")
    print(f"  Vanilla : {len(vanilla_records)}")
    print(f"  TPR     : {len(tpr_records)}")

    base_ids = set(base_records)
    vanilla_ids = set(vanilla_records)
    tpr_ids = set(tpr_records)

    common_ids = (
        base_ids
        & vanilla_ids
        & tpr_ids
    )

    print()
    print(f"[Common problems] {len(common_ids)}")

    if not common_ids:
        raise RuntimeError(
            "No common problem_id across the three files."
        )

    if not (
        base_ids == vanilla_ids == tpr_ids
    ):
        print(
            "[WARNING] The three files do not contain "
            "identical problem_id sets."
        )

        print(
            f"  Base-only-ish    : "
            f"{len(base_ids - common_ids)}"
        )
        print(
            f"  Vanilla-only-ish : "
            f"{len(vanilla_ids - common_ids)}"
        )
        print(
            f"  TPR-only-ish     : "
            f"{len(tpr_ids - common_ids)}"
        )

    candidates = []

    for problem_id in sorted(common_ids):
        base = base_records[problem_id]
        vanilla = vanilla_records[problem_id]
        tpr = tpr_records[problem_id]

        validate_common_problem(
            problem_id,
            base,
            vanilla,
            tpr,
        )

        candidate = build_candidate(
            problem_id,
            base,
            vanilla,
            tpr,
        )

        candidates.append(candidate)

    # Keep JSONL deterministic.
    candidates.sort(
        key=lambda x: x["problem_id"]
    )

    transition_jsonl = (
        output_dir
        / "transition_candidates.jsonl"
    )

    summary_csv = (
        output_dir
        / "transition_summary.csv"
    )

    ranking_csv = (
        output_dir
        / "transition_candidates.csv"
    )

    write_jsonl(
        transition_jsonl,
        candidates,
    )

    write_summary_csv(
        summary_csv,
        candidates,
    )

    write_ranking_csv(
        ranking_csv,
        candidates,
    )

    counts = Counter(
        row["transition"]
        for row in candidates
    )

    print()
    print("=" * 72)
    print("Transition Summary")
    print("=" * 72)

    for transition in [
        "PPP",
        "PPF",
        "PFP",
        "PFF",
        "FPP",
        "FPF",
        "FFP",
        "FFF",
    ]:
        count = counts.get(transition, 0)
        ratio = count / len(candidates)

        print(
            f"{transition} "
            f"{TRANSITION_DESCRIPTIONS[transition]:22s} "
            f"{count:3d} "
            f"({ratio:6.2%})"
        )

    print()
    print("[Outputs]")
    print(f"  {transition_jsonl}")
    print(f"  {summary_csv}")
    print(f"  {ranking_csv}")


if __name__ == "__main__":
    main()