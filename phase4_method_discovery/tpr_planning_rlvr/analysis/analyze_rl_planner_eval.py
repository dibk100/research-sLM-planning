"""
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase4_method_discovery/tpr_planning_rlvr/analysis/analyze_rl_planner_eval.py \
  --base /mnt/hdd/project_sLM_planning/phase1/livecodebench_v6_stdin/qwen25Coder3b/self_plan/results.jsonl \
  --step25 /mnt/hdd/project_sLM_planning/output/phase4_tpr_rl_planner_eval/step25/results.jsonl \
  --step50 /mnt/hdd/project_sLM_planning/output/phase4_tpr_rl_planner_eval/step50/results.jsonl \
  --output-dir /home/dibaeck/workspace/project_sLM_planning/phase4_method_discovery/tpr_planning_rlvr/outputs/analysis
"""

# phase4_method_discovery/tpr_planning_rlvr/analysis/analyze_rl_planner_eval.py

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ============================================================================
# Data structures
# ============================================================================


@dataclass(frozen=True)
class EvalRecord:
    problem_id: str
    difficulty: str
    passed: bool
    test_pass_ratio: float
    status: str


@dataclass(frozen=True)
class Summary:
    total: int
    passed: int
    pass_rate: float
    mean_test_pass_ratio: float


@dataclass(frozen=True)
class TransitionSummary:
    total: int

    both_pass: int
    both_fail: int

    recovered: int
    regressed: int

    net_gain: int

    mcnemar_exact_p: float


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Phase 4 RL planner evaluation results "
            "against the Phase 1 Self-Plan baseline."
        )
    )

    parser.add_argument(
        "--base",
        required=True,
        help="Phase 1 Self-Plan results.jsonl",
    )

    parser.add_argument(
        "--step25",
        required=True,
        help="RL planner step25 results.jsonl",
    )

    parser.add_argument(
        "--step50",
        required=True,
        help="RL planner step50 results.jsonl",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional directory for analysis CSV/JSON outputs. "
            "If omitted, only console output is produced."
        ),
    )

    return parser.parse_args()


# ============================================================================
# Loading
# ============================================================================


def load_jsonl(
    path: str | Path,
) -> dict[str, EvalRecord]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    records: dict[str, EvalRecord] = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                raw = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at "
                    f"{path}:{line_number}"
                ) from error

            problem_id = raw.get(
                "problem_id"
            )

            if not problem_id:
                raise ValueError(
                    f"Missing problem_id at "
                    f"{path}:{line_number}"
                )

            if problem_id in records:
                raise ValueError(
                    f"Duplicate problem_id "
                    f"in {path}: {problem_id}"
                )

            difficulty = raw.get(
                "difficulty"
            )

            if difficulty is None:
                difficulty = "unknown"

            difficulty = str(
                difficulty
            ).strip().lower()

            passed = bool(
                raw.get(
                    "passed",
                    False,
                )
            )

            test_pass_ratio = float(
                raw.get(
                    "test_pass_ratio",
                    0.0,
                )
                or 0.0
            )

            status = str(
                raw.get(
                    "status",
                    "UNKNOWN",
                )
            )

            records[problem_id] = EvalRecord(
                problem_id=problem_id,
                difficulty=difficulty,
                passed=passed,
                test_pass_ratio=test_pass_ratio,
                status=status,
            )

    if not records:
        raise ValueError(
            f"No records found: {path}"
        )

    return records


# ============================================================================
# Validation
# ============================================================================


def validate_problem_sets(
    datasets: dict[
        str,
        dict[str, EvalRecord],
    ],
) -> list[str]:
    names = list(
        datasets.keys()
    )

    reference_name = names[0]

    reference_ids = set(
        datasets[
            reference_name
        ].keys()
    )

    for name in names[1:]:
        current_ids = set(
            datasets[name].keys()
        )

        if current_ids != reference_ids:
            missing = sorted(
                reference_ids
                - current_ids
            )

            extra = sorted(
                current_ids
                - reference_ids
            )

            raise ValueError(
                "\nProblem sets do not match.\n"
                f"Reference : {reference_name}\n"
                f"Current   : {name}\n"
                f"Missing   : {len(missing)}\n"
                f"Extra     : {len(extra)}\n"
                f"Missing examples: "
                f"{missing[:10]}\n"
                f"Extra examples: "
                f"{extra[:10]}"
            )

    problem_ids = sorted(
        reference_ids
    )

    # Also validate difficulty metadata.
    for problem_id in problem_ids:
        difficulties = {
            name: records[
                problem_id
            ].difficulty
            for name, records
            in datasets.items()
        }

        unique = set(
            difficulties.values()
        )

        if len(unique) > 1:
            raise ValueError(
                "Difficulty mismatch for "
                f"{problem_id}: "
                f"{difficulties}"
            )

    return problem_ids


# ============================================================================
# Aggregate metrics
# ============================================================================


def summarize(
    records: dict[str, EvalRecord],
    problem_ids: list[str],
) -> Summary:
    selected = [
        records[problem_id]
        for problem_id in problem_ids
    ]

    total = len(
        selected
    )

    passed = sum(
        record.passed
        for record in selected
    )

    pass_rate = (
        passed / total
        if total > 0
        else 0.0
    )

    mean_test_pass_ratio = (
        sum(
            record.test_pass_ratio
            for record in selected
        )
        / total
        if total > 0
        else 0.0
    )

    return Summary(
        total=total,
        passed=passed,
        pass_rate=pass_rate,
        mean_test_pass_ratio=(
            mean_test_pass_ratio
        ),
    )


def summarize_by_difficulty(
    records: dict[str, EvalRecord],
    problem_ids: list[str],
) -> dict[str, Summary]:
    grouped: dict[
        str,
        list[str],
    ] = {}

    for problem_id in problem_ids:
        difficulty = (
            records[
                problem_id
            ].difficulty
        )

        grouped.setdefault(
            difficulty,
            [],
        ).append(
            problem_id
        )

    return {
        difficulty: summarize(
            records,
            ids,
        )
        for difficulty, ids
        in grouped.items()
    }


# ============================================================================
# Exact McNemar test
# ============================================================================


def binomial_probability(
    n: int,
    k: int,
) -> float:
    return (
        math.comb(n, k)
        * (0.5 ** n)
    )


def exact_mcnemar_p_value(
    recovered: int,
    regressed: int,
) -> float:
    """
    Exact two-sided McNemar test.

    Under H0, among discordant pairs:
        X ~ Binomial(n, 0.5)

    where:
        recovered = A fail -> B pass
        regressed = A pass -> B fail

    Equivalent to scipy.stats.binomtest(
        min(recovered, regressed),
        recovered + regressed,
        p=0.5,
        alternative="two-sided",
    )

    Implemented directly to avoid a scipy dependency.
    """

    n = (
        recovered
        + regressed
    )

    if n == 0:
        return 1.0

    k = min(
        recovered,
        regressed,
    )

    lower_tail = sum(
        binomial_probability(
            n,
            i,
        )
        for i in range(
            k + 1
        )
    )

    p_value = min(
        1.0,
        2.0 * lower_tail,
    )

    return p_value


# ============================================================================
# Paired transition analysis
# ============================================================================


def analyze_transition(
    before: dict[str, EvalRecord],
    after: dict[str, EvalRecord],
    problem_ids: list[str],
) -> TransitionSummary:
    both_pass = 0
    both_fail = 0
    recovered = 0
    regressed = 0

    for problem_id in problem_ids:
        before_pass = (
            before[
                problem_id
            ].passed
        )

        after_pass = (
            after[
                problem_id
            ].passed
        )

        if before_pass and after_pass:
            both_pass += 1

        elif (
            not before_pass
            and not after_pass
        ):
            both_fail += 1

        elif (
            not before_pass
            and after_pass
        ):
            recovered += 1

        elif (
            before_pass
            and not after_pass
        ):
            regressed += 1

    return TransitionSummary(
        total=len(problem_ids),
        both_pass=both_pass,
        both_fail=both_fail,
        recovered=recovered,
        regressed=regressed,
        net_gain=(
            recovered
            - regressed
        ),
        mcnemar_exact_p=(
            exact_mcnemar_p_value(
                recovered,
                regressed,
            )
        ),
    )


def get_transition_ids(
    before: dict[str, EvalRecord],
    after: dict[str, EvalRecord],
    problem_ids: list[str],
) -> dict[str, list[str]]:
    result = {
        "both_pass": [],
        "both_fail": [],
        "recovered": [],
        "regressed": [],
    }

    for problem_id in problem_ids:
        before_pass = (
            before[
                problem_id
            ].passed
        )

        after_pass = (
            after[
                problem_id
            ].passed
        )

        if before_pass and after_pass:
            key = "both_pass"

        elif (
            not before_pass
            and not after_pass
        ):
            key = "both_fail"

        elif (
            not before_pass
            and after_pass
        ):
            key = "recovered"

        else:
            key = "regressed"

        result[key].append(
            problem_id
        )

    return result


# ============================================================================
# Status analysis
# ============================================================================


def count_statuses(
    records: dict[str, EvalRecord],
    problem_ids: list[str],
) -> Counter[str]:
    return Counter(
        records[
            problem_id
        ].status
        for problem_id
        in problem_ids
    )


# ============================================================================
# Printing
# ============================================================================


def print_overall_summary(
    summaries: dict[str, Summary],
) -> None:
    print()
    print("=" * 96)
    print("Overall Performance")
    print("=" * 96)

    print(
        f"{'Planner':<16}"
        f"{'Solved':>12}"
        f"{'Pass Rate':>14}"
        f"{'Mean Test Pass Ratio':>24}"
        f"{'Delta vs Base':>18}"
    )

    print("-" * 96)

    base = summaries["base"]

    for name in (
        "base",
        "step25",
        "step50",
    ):
        summary = summaries[name]

        delta = (
            summary.pass_rate
            - base.pass_rate
        )

        solved_text = (
            f"{summary.passed}/"
            f"{summary.total}"
        )

        delta_text = (
            "-"
            if name == "base"
            else f"{delta * 100:+.2f}%p"
        )

        print(
            f"{name:<16}"
            f"{solved_text:>12}"
            f"{summary.pass_rate * 100:>13.2f}%"
            f"{summary.mean_test_pass_ratio:>24.6f}"
            f"{delta_text:>18}"
        )


def print_difficulty_summary(
    datasets: dict[
        str,
        dict[str, EvalRecord],
    ],
    problem_ids: list[str],
) -> None:
    difficulties = sorted(
        {
            datasets[
                "base"
            ][problem_id].difficulty
            for problem_id
            in problem_ids
        }
    )

    preferred_order = [
        "easy",
        "medium",
        "hard",
    ]

    difficulties = (
        [
            difficulty
            for difficulty
            in preferred_order
            if difficulty
            in difficulties
        ]
        + [
            difficulty
            for difficulty
            in difficulties
            if difficulty
            not in preferred_order
        ]
    )

    print()
    print("=" * 96)
    print("Performance by Difficulty")
    print("=" * 96)

    for difficulty in difficulties:
        ids = [
            problem_id
            for problem_id
            in problem_ids
            if datasets[
                "base"
            ][problem_id].difficulty
            == difficulty
        ]

        print()
        print(
            f"[{difficulty.upper()}] "
            f"N={len(ids)}"
        )

        print(
            f"{'Planner':<16}"
            f"{'Solved':>12}"
            f"{'Pass Rate':>14}"
            f"{'Mean Test Pass Ratio':>24}"
            f"{'Delta vs Base':>18}"
        )

        print("-" * 84)

        base_summary = summarize(
            datasets["base"],
            ids,
        )

        for name in (
            "base",
            "step25",
            "step50",
        ):
            summary = summarize(
                datasets[name],
                ids,
            )

            delta = (
                summary.pass_rate
                - base_summary.pass_rate
            )

            solved_text = (
                f"{summary.passed}/"
                f"{summary.total}"
            )

            delta_text = (
                "-"
                if name == "base"
                else (
                    f"{delta * 100:+.2f}%p"
                )
            )

            print(
                f"{name:<16}"
                f"{solved_text:>12}"
                f"{summary.pass_rate * 100:>13.2f}%"
                f"{summary.mean_test_pass_ratio:>24.6f}"
                f"{delta_text:>18}"
            )


def print_transition(
    name: str,
    transition: TransitionSummary,
) -> None:
    print()
    print(
        f"[{name}]"
    )

    print(
        f"Both pass  : "
        f"{transition.both_pass}"
    )

    print(
        f"Both fail  : "
        f"{transition.both_fail}"
    )

    print(
        f"Recovered  : "
        f"{transition.recovered}"
        "  (before fail -> after pass)"
    )

    print(
        f"Regressed  : "
        f"{transition.regressed}"
        "  (before pass -> after fail)"
    )

    print(
        f"Net gain   : "
        f"{transition.net_gain:+d}"
    )

    discordant = (
        transition.recovered
        + transition.regressed
    )

    print(
        f"Discordant : "
        f"{discordant}"
    )

    print(
        f"McNemar exact p-value : "
        f"{transition.mcnemar_exact_p:.6f}"
    )


def print_status_summary(
    datasets: dict[
        str,
        dict[str, EvalRecord],
    ],
    problem_ids: list[str],
) -> None:
    print()
    print("=" * 96)
    print("Evaluation Status Distribution")
    print("=" * 96)

    all_statuses = sorted(
        {
            record.status
            for records
            in datasets.values()
            for record
            in records.values()
        }
    )

    header = (
        f"{'Status':<28}"
        f"{'base':>12}"
        f"{'step25':>12}"
        f"{'step50':>12}"
    )

    print(header)
    print("-" * len(header))

    counters = {
        name: count_statuses(
            records,
            problem_ids,
        )
        for name, records
        in datasets.items()
    }

    for status in all_statuses:
        print(
            f"{status:<28}"
            f"{counters['base'][status]:>12}"
            f"{counters['step25'][status]:>12}"
            f"{counters['step50'][status]:>12}"
        )


# ============================================================================
# Output files
# ============================================================================


def write_summary_json(
    output_dir: Path,
    summaries: dict[str, Summary],
    transitions: dict[
        str,
        TransitionSummary,
    ],
) -> None:
    output = {
        "overall": {
            name: {
                "total": summary.total,
                "passed": summary.passed,
                "pass_rate": (
                    summary.pass_rate
                ),
                "mean_test_pass_ratio": (
                    summary.mean_test_pass_ratio
                ),
            }
            for name, summary
            in summaries.items()
        },
        "transitions": {
            name: {
                "total": transition.total,
                "both_pass": (
                    transition.both_pass
                ),
                "both_fail": (
                    transition.both_fail
                ),
                "recovered": (
                    transition.recovered
                ),
                "regressed": (
                    transition.regressed
                ),
                "net_gain": (
                    transition.net_gain
                ),
                "mcnemar_exact_p": (
                    transition.mcnemar_exact_p
                ),
            }
            for name, transition
            in transitions.items()
        },
    }

    path = (
        output_dir
        / "summary.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[Saved] {path}"
    )


def write_transition_csv(
    output_dir: Path,
    name: str,
    before: dict[str, EvalRecord],
    after: dict[str, EvalRecord],
    problem_ids: list[str],
) -> None:
    path = (
        output_dir
        / f"{name}.csv"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "problem_id,"
            "difficulty,"
            "before_pass,"
            "after_pass,"
            "transition,"
            "before_test_pass_ratio,"
            "after_test_pass_ratio,"
            "test_pass_ratio_delta,"
            "before_status,"
            "after_status\n"
        )

        for problem_id in problem_ids:
            before_record = (
                before[
                    problem_id
                ]
            )

            after_record = (
                after[
                    problem_id
                ]
            )

            if (
                before_record.passed
                and after_record.passed
            ):
                transition = (
                    "both_pass"
                )

            elif (
                not before_record.passed
                and not after_record.passed
            ):
                transition = (
                    "both_fail"
                )

            elif (
                not before_record.passed
                and after_record.passed
            ):
                transition = (
                    "recovered"
                )

            else:
                transition = (
                    "regressed"
                )

            delta = (
                after_record.test_pass_ratio
                - before_record.test_pass_ratio
            )

            values = [
                problem_id,
                before_record.difficulty,
                str(
                    before_record.passed
                ),
                str(
                    after_record.passed
                ),
                transition,
                (
                    f"{before_record.test_pass_ratio:.8f}"
                ),
                (
                    f"{after_record.test_pass_ratio:.8f}"
                ),
                f"{delta:.8f}",
                before_record.status,
                after_record.status,
            ]

            # Current fields should not contain commas in normal usage,
            # but quote defensively for CSV correctness.
            encoded = [
                '"'
                + str(value).replace(
                    '"',
                    '""',
                )
                + '"'
                for value in values
            ]

            file.write(
                ",".join(encoded)
                + "\n"
            )

    print(
        f"[Saved] {path}"
    )


def write_transition_ids_json(
    output_dir: Path,
    transitions: dict[
        str,
        tuple[
            dict[str, EvalRecord],
            dict[str, EvalRecord],
        ],
    ],
    problem_ids: list[str],
) -> None:
    output: dict[
        str,
        dict[str, list[str]],
    ] = {}

    for name, (
        before,
        after,
    ) in transitions.items():
        output[name] = (
            get_transition_ids(
                before,
                after,
                problem_ids,
            )
        )

    path = (
        output_dir
        / "transition_problem_ids.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[Saved] {path}"
    )


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()

    print("=" * 96)
    print("RL Planner Evaluation Analysis")
    print("=" * 96)

    print(
        f"Base   : {args.base}"
    )
    print(
        f"Step25 : {args.step25}"
    )
    print(
        f"Step50 : {args.step50}"
    )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    datasets = {
        "base": load_jsonl(
            args.base
        ),
        "step25": load_jsonl(
            args.step25
        ),
        "step50": load_jsonl(
            args.step50
        ),
    }

    problem_ids = (
        validate_problem_sets(
            datasets
        )
    )

    print()
    print(
        f"[OK] Matched "
        f"{len(problem_ids)} problems "
        "across all evaluations."
    )

    # ------------------------------------------------------------------
    # Overall
    # ------------------------------------------------------------------

    summaries = {
        name: summarize(
            records,
            problem_ids,
        )
        for name, records
        in datasets.items()
    }

    print_overall_summary(
        summaries
    )

    # ------------------------------------------------------------------
    # Difficulty
    # ------------------------------------------------------------------

    print_difficulty_summary(
        datasets,
        problem_ids,
    )

    # ------------------------------------------------------------------
    # Paired transitions
    # ------------------------------------------------------------------

    transition_pairs = {
        "base_to_step25": (
            datasets["base"],
            datasets["step25"],
        ),
        "base_to_step50": (
            datasets["base"],
            datasets["step50"],
        ),
        "step25_to_step50": (
            datasets["step25"],
            datasets["step50"],
        ),
    }

    transitions = {
        name: analyze_transition(
            before,
            after,
            problem_ids,
        )
        for name, (
            before,
            after,
        )
        in transition_pairs.items()
    }

    print()
    print("=" * 96)
    print("Paired Transition Analysis")
    print("=" * 96)

    for name in (
        "base_to_step25",
        "base_to_step50",
        "step25_to_step50",
    ):
        print_transition(
            name,
            transitions[name],
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    print_status_summary(
        datasets,
        problem_ids,
    )

    # ------------------------------------------------------------------
    # Output files
    # ------------------------------------------------------------------

    if args.output_dir is not None:
        output_dir = Path(
            args.output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print("=" * 96)
        print("Saving Analysis")
        print("=" * 96)

        write_summary_json(
            output_dir,
            summaries,
            transitions,
        )

        for name, (
            before,
            after,
        ) in transition_pairs.items():
            write_transition_csv(
                output_dir=output_dir,
                name=name,
                before=before,
                after=after,
                problem_ids=problem_ids,
            )

        write_transition_ids_json(
            output_dir=output_dir,
            transitions=transition_pairs,
            problem_ids=problem_ids,
        )

    print()
    print("=" * 96)
    print("Analysis Complete")
    print("=" * 96)


if __name__ == "__main__":
    main()