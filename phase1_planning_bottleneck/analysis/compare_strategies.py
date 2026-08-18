"""
# Phase1 결과 분석 스크립트

PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase1_planning_bottleneck/analysis/compare_strategies.py \
  --model-dir qwen25Coder3b \
  --expected-problems 300 \
  --output-dir phase1_planning_bottleneck/archive/comparison_qwen25Coer3b_300

output :
archive폴더에 기록
  
"""
# phase1_planning_bottleneck.analysis.compare_three_strategies.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


STRATEGIES = {
    "Direct": "direct",
    "Self-Plan": "self_plan",
    "Teacher-Plan": "teacher_plan",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Phase 1 Direct, Self-Plan, "
            "and Teacher-Plan results."
        )
    )

    parser.add_argument(
        "--phase1-root",
        default="/mnt/hdd/project_sLM_planning/phase1",
    )

    parser.add_argument(
        "--model-dir",
        required=True,
        help="Model result directory name, e.g. qwen25Coder3b",
    )

    parser.add_argument(
        "--dataset",
        default="livecodebench_v6_stdin",
    )

    parser.add_argument(
        "--expected-problems",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    return parser.parse_args()


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    records = []

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
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON: "
                    f"{path}:{line_number}"
                ) from error

            records.append(record)

    return records


def validate_records(
    *,
    name: str,
    records: list[dict[str, Any]],
    expected_problems: int,
) -> None:
    if len(records) != expected_problems:
        raise ValueError(
            f"{name}: expected "
            f"{expected_problems} records, "
            f"found {len(records)}."
        )

    ids = [
        record["problem_id"]
        for record in records
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            f"{name}: duplicate problem_id found."
        )

    required_fields = {
        "problem_id",
        "difficulty",
        "passed",
        "status",
        "passed_tests",
        "total_tests",
        "test_pass_ratio",
        "prompt_tokens",
        "completion_tokens",
        "generation_time",
        "execution_time",
    }

    for index, record in enumerate(
        records,
        start=1,
    ):
        missing = (
            required_fields
            - set(record)
        )

        if missing:
            raise ValueError(
                f"{name} record #{index} "
                f"missing fields: "
                f"{sorted(missing)}"
            )


def validate_same_problem_set(
    data: dict[
        str,
        list[dict[str, Any]],
    ],
) -> list[str]:

    names = list(data)

    reference_name = names[0]

    reference_ids = [
        record["problem_id"]
        for record in data[reference_name]
    ]

    for name in names[1:]:
        current_ids = [
            record["problem_id"]
            for record in data[name]
        ]

        if current_ids != reference_ids:
            raise ValueError(
                "Problem order/set mismatch: "
                f"{reference_name} vs {name}"
            )

    print(
        f"[PASS] Same problem set: "
        f"{len(reference_ids)} problems"
    )

    return reference_ids


def mean(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def summarize(
    records: list[dict[str, Any]],
) -> dict[str, Any]:

    num_problems = len(records)

    num_passed = sum(
        1
        for record in records
        if record["passed"]
    )

    prompt_tokens = [
        float(record["prompt_tokens"])
        for record in records
    ]

    completion_tokens = [
        float(record["completion_tokens"])
        for record in records
    ]

    generation_times = [
        float(record["generation_time"])
        for record in records
    ]

    execution_times = [
        float(record["execution_time"])
        for record in records
    ]

    return {
        "num_problems": num_problems,
        "num_passed": num_passed,
        "pass_rate": (
            num_passed / num_problems
        ),
        "mean_test_pass_ratio": mean(
            [
                float(
                    record[
                        "test_pass_ratio"
                    ]
                )
                for record in records
            ]
        ),
        "mean_prompt_tokens": mean(
            prompt_tokens
        ),
        "mean_completion_tokens": mean(
            completion_tokens
        ),
        "mean_total_tokens": mean(
            [
                p + c
                for p, c in zip(
                    prompt_tokens,
                    completion_tokens,
                )
            ]
        ),
        "mean_generation_time": mean(
            generation_times
        ),
        "mean_execution_time": mean(
            execution_times
        ),
        "mean_total_runtime": mean(
            [
                g + e
                for g, e in zip(
                    generation_times,
                    execution_times,
                )
            ]
        ),
    }


def build_overall_summary(
    data: dict[
        str,
        list[dict[str, Any]],
    ],
) -> pd.DataFrame:

    rows = []

    for strategy, records in data.items():
        rows.append(
            {
                "strategy": strategy,
                **summarize(records),
            }
        )

    return pd.DataFrame(rows)


def build_difficulty_summary(
    data: dict[
        str,
        list[dict[str, Any]],
    ],
) -> pd.DataFrame:

    rows = []

    difficulty_order = [
        "easy",
        "medium",
        "hard",
    ]

    for difficulty in difficulty_order:
        for strategy, records in data.items():

            subset = [
                record
                for record in records
                if record.get(
                    "difficulty"
                ) == difficulty
            ]

            if not subset:
                continue

            summary = summarize(
                subset
            )

            rows.append(
                {
                    "strategy": strategy,
                    "difficulty": difficulty,
                    **summary,
                }
            )

    return pd.DataFrame(rows)


def build_transition_summary(
    data: dict[
        str,
        list[dict[str, Any]],
    ],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    comparisons = [
        ("Direct", "Self-Plan"),
        ("Direct", "Teacher-Plan"),
        ("Self-Plan", "Teacher-Plan"),
    ]

    summary_rows = []
    detail_rows = []

    for source, target in comparisons:

        source_map = {
            record["problem_id"]: record
            for record in data[source]
        }

        target_map = {
            record["problem_id"]: record
            for record in data[target]
        }

        counts = {
            "FAIL_TO_PASS": 0,
            "PASS_TO_PASS": 0,
            "PASS_TO_FAIL": 0,
            "FAIL_TO_FAIL": 0,
        }

        for problem_id in source_map:

            source_record = (
                source_map[problem_id]
            )

            target_record = (
                target_map[problem_id]
            )

            source_pass = bool(
                source_record["passed"]
            )

            target_pass = bool(
                target_record["passed"]
            )

            if (
                not source_pass
                and target_pass
            ):
                transition = (
                    "FAIL_TO_PASS"
                )

            elif (
                source_pass
                and target_pass
            ):
                transition = (
                    "PASS_TO_PASS"
                )

            elif (
                source_pass
                and not target_pass
            ):
                transition = (
                    "PASS_TO_FAIL"
                )

            else:
                transition = (
                    "FAIL_TO_FAIL"
                )

            counts[transition] += 1

            detail_rows.append(
                {
                    "comparison": (
                        f"{source}_TO_{target}"
                    ),
                    "problem_id": problem_id,
                    "difficulty": (
                        source_record.get(
                            "difficulty"
                        )
                    ),
                    "source_passed": (
                        source_pass
                    ),
                    "target_passed": (
                        target_pass
                    ),
                    "transition": transition,
                    "source_test_pass_ratio": (
                        source_record[
                            "test_pass_ratio"
                        ]
                    ),
                    "target_test_pass_ratio": (
                        target_record[
                            "test_pass_ratio"
                        ]
                    ),
                    "test_pass_ratio_delta": (
                        target_record[
                            "test_pass_ratio"
                        ]
                        - source_record[
                            "test_pass_ratio"
                        ]
                    ),
                }
            )

        for transition, count in (
            counts.items()
        ):
            summary_rows.append(
                {
                    "comparison": (
                        f"{source}_TO_{target}"
                    ),
                    "transition": transition,
                    "count": count,
                }
            )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(detail_rows),
    )


def build_cost_multipliers(
    overall_df: pd.DataFrame,
) -> pd.DataFrame:

    indexed = (
        overall_df
        .set_index("strategy")
    )

    comparisons = [
        ("Self-Plan", "Direct"),
        ("Teacher-Plan", "Direct"),
        ("Teacher-Plan", "Self-Plan"),
    ]

    metrics = [
        "mean_prompt_tokens",
        "mean_completion_tokens",
        "mean_total_tokens",
        "mean_generation_time",
        "mean_total_runtime",
    ]

    rows = []

    for numerator, denominator in (
        comparisons
    ):
        row = {
            "comparison": (
                f"{numerator} / "
                f"{denominator}"
            )
        }

        for metric in metrics:

            denominator_value = float(
                indexed.loc[
                    denominator,
                    metric,
                ]
            )

            numerator_value = float(
                indexed.loc[
                    numerator,
                    metric,
                ]
            )

            if denominator_value == 0:
                multiplier = float("nan")
            else:
                multiplier = (
                    numerator_value
                    / denominator_value
                )

            row[
                f"{metric}_multiplier"
            ] = multiplier

        rows.append(row)

    return pd.DataFrame(rows)


def build_pattern_summary(
    data: dict[
        str,
        list[dict[str, Any]],
    ],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    direct = {
        r["problem_id"]: r
        for r in data["Direct"]
    }

    self_plan = {
        r["problem_id"]: r
        for r in data["Self-Plan"]
    }

    teacher = {
        r["problem_id"]: r
        for r in data["Teacher-Plan"]
    }

    counts: dict[str, int] = {}
    detail_rows = []

    for problem_id in direct:

        d = direct[problem_id]
        s = self_plan[problem_id]
        t = teacher[problem_id]

        pattern = "-".join(
            [
                (
                    "P"
                    if d["passed"]
                    else "F"
                ),
                (
                    "P"
                    if s["passed"]
                    else "F"
                ),
                (
                    "P"
                    if t["passed"]
                    else "F"
                ),
            ]
        )

        counts[pattern] = (
            counts.get(pattern, 0)
            + 1
        )

        detail_rows.append(
            {
                "problem_id": problem_id,
                "difficulty": (
                    d.get("difficulty")
                ),
                "pattern": pattern,
                "direct_passed": (
                    d["passed"]
                ),
                "self_plan_passed": (
                    s["passed"]
                ),
                "teacher_plan_passed": (
                    t["passed"]
                ),
                "direct_test_pass_ratio": (
                    d["test_pass_ratio"]
                ),
                "self_plan_test_pass_ratio": (
                    s["test_pass_ratio"]
                ),
                "teacher_plan_test_pass_ratio": (
                    t["test_pass_ratio"]
                ),
            }
        )

    summary_df = pd.DataFrame(
        [
            {
                "pattern": pattern,
                "count": count,
            }
            for pattern, count
            in counts.items()
        ]
    ).sort_values(
        "count",
        ascending=False,
    )

    return (
        summary_df,
        pd.DataFrame(detail_rows),
    )


def print_table(
    title: str,
    df: pd.DataFrame,
) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    print(
        df.to_string(
            index=False,
            float_format=(
                lambda x: f"{x:.6f}"
            ),
        )
    )


def main() -> None:
    args = parse_args()

    root = (
        Path(args.phase1_root)
        / args.dataset
        / args.model_dir
    )

    paths = {
        strategy_name: (
            root
            / directory
            / "results.jsonl"
        )
        for strategy_name, directory
        in STRATEGIES.items()
    }

    data = {}

    for strategy_name, path in (
        paths.items()
    ):
        records = load_jsonl(path)

        validate_records(
            name=strategy_name,
            records=records,
            expected_problems=(
                args.expected_problems
            ),
        )

        data[strategy_name] = records

    validate_same_problem_set(
        data
    )

    overall_df = (
        build_overall_summary(data)
    )

    difficulty_df = (
        build_difficulty_summary(data)
    )

    transition_df, transition_detail_df = (
        build_transition_summary(data)
    )

    cost_df = (
        build_cost_multipliers(
            overall_df
        )
    )

    pattern_df, problem_df = (
        build_pattern_summary(data)
    )

    print_table(
        "Overall Comparison",
        overall_df,
    )

    print_table(
        "Difficulty Comparison",
        difficulty_df,
    )

    print_table(
        "Pairwise Transition Summary",
        transition_df,
    )

    print_table(
        "Cost Multipliers",
        cost_df,
    )

    print_table(
        "Direct / Self / Teacher Pattern",
        pattern_df,
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "overall_summary.csv":
            overall_df,

        "difficulty_summary.csv":
            difficulty_df,

        "transition_summary.csv":
            transition_df,

        "transition_detail.csv":
            transition_detail_df,

        "cost_multipliers.csv":
            cost_df,

        "problem_comparison.csv":
            problem_df,

        "pattern_summary.csv":
            pattern_df,
    }

    for filename, dataframe in (
        outputs.items()
    ):
        path = (
            output_dir
            / filename
        )

        dataframe.to_csv(
            path,
            index=False,
        )

        print(
            f"[SAVED] {path}"
        )

    print()
    print(
        "[DONE] Three-strategy "
        "comparison completed."
    )


if __name__ == "__main__":
    main()