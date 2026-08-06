"""
Direct / Self-Plan / Teacher-Plan 정량 비교.

Usage:

python -m archive.sample50_compare_three_strategies

python -m archive.sample50_compare_three_strategies \
  --direct-path /mnt/hdd/project_sLM_planning/output/direct_50_stdin/results.jsonl \
  --self-plan-path /mnt/hdd/project_sLM_planning/output/self_plan_50_stdin/results.jsonl \
  --teacher-plan-path /mnt/hdd/project_sLM_planning/output/teacher_plan_50_stdin/results.jsonl \
  --output-dir ./archive/comparison_50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DIRECT_PATH = Path(
    "/mnt/hdd/project_sLM_planning/output/"
    "direct_50_stdin/results.jsonl"
)

DEFAULT_SELF_PLAN_PATH = Path(
    "/mnt/hdd/project_sLM_planning/output/"
    "self_plan_50_stdin/results.jsonl"
)

DEFAULT_TEACHER_PLAN_PATH = Path(
    "/mnt/hdd/project_sLM_planning/output/"
    "teacher_plan_50_stdin/results.jsonl"
)

DEFAULT_OUTPUT_DIR = Path(
    "/mnt/hdd/project_sLM_planning/output/"
    "comparison_50"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Direct, Self-Plan, and "
            "Teacher-Plan experiment results."
        )
    )

    parser.add_argument(
        "--direct-path",
        type=Path,
        default=DEFAULT_DIRECT_PATH,
    )

    parser.add_argument(
        "--self-plan-path",
        type=Path,
        default=DEFAULT_SELF_PLAN_PATH,
    )

    parser.add_argument(
        "--teacher-plan-path",
        type=Path,
        default=DEFAULT_TEACHER_PLAN_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Result file not found: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}: {path}"
                ) from error

            records.append(record)

    if not records:
        raise ValueError(
            f"No records found: {path}"
        )

    df = pd.DataFrame(records)

    required_columns = {
        "problem_id",
        "difficulty",
        "passed",
        "status",
        "passed_tests",
        "total_tests",
        "prompt_tokens",
        "completion_tokens",
        "generation_time",
        "execution_time",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: "
            f"{sorted(missing)}"
        )

    if df["problem_id"].duplicated().any():
        duplicated = (
            df.loc[
                df["problem_id"].duplicated(),
                "problem_id",
            ]
            .tolist()
        )

        raise ValueError(
            f"Duplicated problem IDs in {path}: "
            f"{duplicated}"
        )

    return df


def prepare_strategy_df(
    df: pd.DataFrame,
    strategy_name: str,
) -> pd.DataFrame:
    result = df.copy()

    result["strategy_name"] = strategy_name

    result["passed"] = (
        result["passed"]
        .fillna(False)
        .astype(bool)
    )

    result["passed_tests"] = pd.to_numeric(
        result["passed_tests"],
        errors="coerce",
    ).fillna(0)

    result["total_tests"] = pd.to_numeric(
        result["total_tests"],
        errors="coerce",
    ).fillna(0)

    result["prompt_tokens"] = pd.to_numeric(
        result["prompt_tokens"],
        errors="coerce",
    ).fillna(0)

    result["completion_tokens"] = pd.to_numeric(
        result["completion_tokens"],
        errors="coerce",
    ).fillna(0)

    result["generation_time"] = pd.to_numeric(
        result["generation_time"],
        errors="coerce",
    ).fillna(0.0)

    result["execution_time"] = pd.to_numeric(
        result["execution_time"],
        errors="coerce",
    ).fillna(0.0)

    result["test_pass_ratio"] = 0.0

    valid_test_mask = (
        result["total_tests"] > 0
    )

    result.loc[
        valid_test_mask,
        "test_pass_ratio",
    ] = (
        result.loc[
            valid_test_mask,
            "passed_tests",
        ]
        / result.loc[
            valid_test_mask,
            "total_tests",
        ]
    )

    result["total_tokens"] = (
        result["prompt_tokens"]
        + result["completion_tokens"]
    )

    result["total_runtime"] = (
        result["generation_time"]
        + result["execution_time"]
    )

    return result


def validate_same_problem_set(
    strategy_dfs: dict[str, pd.DataFrame],
) -> None:
    id_sets = {
        name: set(df["problem_id"])
        for name, df in strategy_dfs.items()
    }

    names = list(id_sets)
    reference_name = names[0]
    reference_ids = id_sets[reference_name]

    for name in names[1:]:
        if id_sets[name] != reference_ids:
            only_reference = sorted(
                reference_ids - id_sets[name]
            )
            only_current = sorted(
                id_sets[name] - reference_ids
            )

            raise ValueError(
                "Problem ID mismatch between "
                f"{reference_name} and {name}. "
                f"{reference_name} only={only_reference}, "
                f"{name} only={only_current}"
            )

    print(
        f"[PASS] Same problem set: "
        f"{len(reference_ids)} problems"
    )


def build_overall_summary(
    strategy_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for name, df in strategy_dfs.items():
        rows.append(
            {
                "strategy": name,
                "num_problems": len(df),
                "num_passed": int(
                    df["passed"].sum()
                ),
                "pass_rate": float(
                    df["passed"].mean()
                ),
                "mean_test_pass_ratio": float(
                    df["test_pass_ratio"].mean()
                ),
                "mean_prompt_tokens": float(
                    df["prompt_tokens"].mean()
                ),
                "mean_completion_tokens": float(
                    df["completion_tokens"].mean()
                ),
                "mean_total_tokens": float(
                    df["total_tokens"].mean()
                ),
                "mean_generation_time": float(
                    df["generation_time"].mean()
                ),
                "mean_execution_time": float(
                    df["execution_time"].mean()
                ),
                "mean_total_runtime": float(
                    df["total_runtime"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_difficulty_summary(
    strategy_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for strategy_name, df in strategy_dfs.items():
        grouped = df.groupby(
            "difficulty",
            dropna=False,
        )

        for difficulty, group in grouped:
            rows.append(
                {
                    "strategy": strategy_name,
                    "difficulty": difficulty,
                    "num_problems": len(group),
                    "num_passed": int(
                        group["passed"].sum()
                    ),
                    "pass_rate": float(
                        group["passed"].mean()
                    ),
                    "mean_test_pass_ratio": float(
                        group[
                            "test_pass_ratio"
                        ].mean()
                    ),
                    "mean_total_tokens": float(
                        group[
                            "total_tokens"
                        ].mean()
                    ),
                    "mean_generation_time": float(
                        group[
                            "generation_time"
                        ].mean()
                    ),
                }
            )

    result = pd.DataFrame(rows)

    difficulty_order = {
        "easy": 0,
        "medium": 1,
        "hard": 2,
    }

    result["_difficulty_order"] = (
        result["difficulty"]
        .map(difficulty_order)
        .fillna(99)
    )

    result = (
        result.sort_values(
            [
                "_difficulty_order",
                "strategy",
            ]
        )
        .drop(
            columns="_difficulty_order"
        )
        .reset_index(drop=True)
    )

    return result


def transition_label(
    source_passed: bool,
    target_passed: bool,
) -> str:
    if source_passed and target_passed:
        return "PASS_TO_PASS"

    if source_passed and not target_passed:
        return "PASS_TO_FAIL"

    if not source_passed and target_passed:
        return "FAIL_TO_PASS"

    return "FAIL_TO_FAIL"


def build_pairwise_transition(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_name: str,
    target_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = source_df[
        [
            "problem_id",
            "difficulty",
            "passed",
            "test_pass_ratio",
        ]
    ].rename(
        columns={
            "passed": "source_passed",
            "test_pass_ratio": (
                "source_test_pass_ratio"
            ),
        }
    )

    target = target_df[
        [
            "problem_id",
            "passed",
            "test_pass_ratio",
        ]
    ].rename(
        columns={
            "passed": "target_passed",
            "test_pass_ratio": (
                "target_test_pass_ratio"
            ),
        }
    )

    merged = source.merge(
        target,
        on="problem_id",
        how="inner",
        validate="one_to_one",
    )

    merged["transition"] = [
        transition_label(
            source_passed=bool(source_passed),
            target_passed=bool(target_passed),
        )
        for source_passed, target_passed
        in zip(
            merged["source_passed"],
            merged["target_passed"],
        )
    ]

    merged["test_pass_ratio_delta"] = (
        merged["target_test_pass_ratio"]
        - merged["source_test_pass_ratio"]
    )

    merged.insert(
        0,
        "comparison",
        f"{source_name}_TO_{target_name}",
    )

    summary = (
        merged["transition"]
        .value_counts()
        .reindex(
            [
                "FAIL_TO_PASS",
                "PASS_TO_PASS",
                "PASS_TO_FAIL",
                "FAIL_TO_FAIL",
            ],
            fill_value=0,
        )
        .rename_axis("transition")
        .reset_index(name="count")
    )

    summary.insert(
        0,
        "comparison",
        f"{source_name}_TO_{target_name}",
    )

    return merged, summary


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float | None:
    if denominator == 0:
        return None

    return numerator / denominator


def build_cost_multiplier_table(
    overall_summary: pd.DataFrame,
) -> pd.DataFrame:
    indexed = overall_summary.set_index(
        "strategy"
    )

    metrics = [
        "mean_prompt_tokens",
        "mean_completion_tokens",
        "mean_total_tokens",
        "mean_generation_time",
        "mean_total_runtime",
    ]

    comparisons = [
        ("Self-Plan", "Direct"),
        ("Teacher-Plan", "Direct"),
        ("Teacher-Plan", "Self-Plan"),
    ]

    rows = []

    for numerator_name, denominator_name in comparisons:
        row: dict[str, Any] = {
            "comparison": (
                f"{numerator_name} / "
                f"{denominator_name}"
            )
        }

        for metric in metrics:
            numerator = float(
                indexed.loc[
                    numerator_name,
                    metric,
                ]
            )

            denominator = float(
                indexed.loc[
                    denominator_name,
                    metric,
                ]
            )

            row[f"{metric}_multiplier"] = (
                safe_ratio(
                    numerator,
                    denominator,
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_problem_comparison(
    strategy_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    columns = [
        "problem_id",
        "title",
        "difficulty",
        "passed",
        "status",
        "test_pass_ratio",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "generation_time",
        "execution_time",
        "total_runtime",
    ]

    merged: pd.DataFrame | None = None

    for strategy_name, df in strategy_dfs.items():
        suffix = (
            strategy_name
            .lower()
            .replace("-", "_")
        )

        current = df[columns].copy()

        rename_map = {
            column: f"{column}_{suffix}"
            for column in columns
            if column not in {
                "problem_id",
                "title",
                "difficulty",
            }
        }

        current = current.rename(
            columns=rename_map
        )

        if merged is None:
            merged = current
        else:
            current = current.drop(
                columns=[
                    "title",
                    "difficulty",
                ],
                errors="ignore",
            )

            merged = merged.merge(
                current,
                on="problem_id",
                how="inner",
                validate="one_to_one",
            )

    if merged is None:
        raise ValueError(
            "No strategy dataframes provided."
        )

    merged["direct_self_teacher_pattern"] = (
        merged[
            "passed_direct"
        ].map(
            {
                True: "P",
                False: "F",
            }
        )
        + "-"
        + merged[
            "passed_self_plan"
        ].map(
            {
                True: "P",
                False: "F",
            }
        )
        + "-"
        + merged[
            "passed_teacher_plan"
        ].map(
            {
                True: "P",
                False: "F",
            }
        )
    )

    return merged


def print_table(
    title: str,
    table: pd.DataFrame,
) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
    ):
        print(table.to_string(index=False))


def main() -> None:
    args = parse_args()

    direct_df = prepare_strategy_df(
        load_jsonl(args.direct_path),
        "Direct",
    )

    self_plan_df = prepare_strategy_df(
        load_jsonl(args.self_plan_path),
        "Self-Plan",
    )

    teacher_plan_df = prepare_strategy_df(
        load_jsonl(args.teacher_plan_path),
        "Teacher-Plan",
    )

    strategy_dfs = {
        "Direct": direct_df,
        "Self-Plan": self_plan_df,
        "Teacher-Plan": teacher_plan_df,
    }

    validate_same_problem_set(
        strategy_dfs
    )

    overall_summary = build_overall_summary(
        strategy_dfs
    )

    difficulty_summary = (
        build_difficulty_summary(
            strategy_dfs
        )
    )

    direct_to_self_detail, direct_to_self_summary = (
        build_pairwise_transition(
            direct_df,
            self_plan_df,
            "Direct",
            "Self-Plan",
        )
    )

    direct_to_teacher_detail, direct_to_teacher_summary = (
        build_pairwise_transition(
            direct_df,
            teacher_plan_df,
            "Direct",
            "Teacher-Plan",
        )
    )

    self_to_teacher_detail, self_to_teacher_summary = (
        build_pairwise_transition(
            self_plan_df,
            teacher_plan_df,
            "Self-Plan",
            "Teacher-Plan",
        )
    )

    transition_summary = pd.concat(
        [
            direct_to_self_summary,
            direct_to_teacher_summary,
            self_to_teacher_summary,
        ],
        ignore_index=True,
    )

    transition_detail = pd.concat(
        [
            direct_to_self_detail,
            direct_to_teacher_detail,
            self_to_teacher_detail,
        ],
        ignore_index=True,
    )

    cost_multipliers = (
        build_cost_multiplier_table(
            overall_summary
        )
    )

    problem_comparison = (
        build_problem_comparison(
            strategy_dfs
        )
    )

    pattern_summary = (
        problem_comparison[
            "direct_self_teacher_pattern"
        ]
        .value_counts()
        .rename_axis("pattern")
        .reset_index(name="count")
    )

    print_table(
        "Overall Comparison",
        overall_summary,
    )

    print_table(
        "Difficulty Comparison",
        difficulty_summary,
    )

    print_table(
        "Pairwise Transition Summary",
        transition_summary,
    )

    print_table(
        "Cost Multipliers",
        cost_multipliers,
    )

    print_table(
        "Direct / Self / Teacher Pattern",
        pattern_summary,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "overall_summary.csv":
            overall_summary,
        "difficulty_summary.csv":
            difficulty_summary,
        "transition_summary.csv":
            transition_summary,
        "transition_detail.csv":
            transition_detail,
        "cost_multipliers.csv":
            cost_multipliers,
        "problem_comparison.csv":
            problem_comparison,
        "pattern_summary.csv":
            pattern_summary,
    }

    for filename, table in outputs.items():
        output_path = (
            args.output_dir / filename
        )

        table.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"[SAVED] {output_path}")

    print()
    print("[DONE] Three-strategy comparison completed.")


if __name__ == "__main__":
    main()