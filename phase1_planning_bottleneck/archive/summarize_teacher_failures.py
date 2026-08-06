"""
수동 라벨링이 완료된 Teacher-Plan 실패 사례를 집계한다.

Input:
    teacher_failure_cases.csv

Outputs:
    label_completeness.csv
    teacher_plan_correctness_summary.csv
    implementation_fidelity_summary.csv
    primary_bottleneck_summary.csv
    failure_type_summary.csv
    algorithm_category_summary.csv
    difficulty_failure_summary.csv
    plan_correctness_x_fidelity.csv
    plan_correctness_x_bottleneck.csv
    bottleneck_x_difficulty.csv
    failure_type_x_status.csv
    bottleneck_x_status.csv
    labeled_failure_table.csv
    teacher_failure_summary.txt

Usage:

python -m archive.summarize_teacher_failures \
  --input-path ./archive/teacher_failures_50/teacher_failure_cases.csv \
  --output-dir ./archive/teacher_failures_50/summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path(
    "./archive/teacher_failures_50/"
    "teacher_failure_cases.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    "./archive/teacher_failures_50/summary"
)

LABEL_COLUMNS = [
    "teacher_plan_correctness",
    "implementation_fidelity",
    "primary_bottleneck",
    "failure_type",
    "algorithm_category",
    "missing_or_wrong_information",
    "analysis_note",
]

CATEGORICAL_LABEL_COLUMNS = [
    "teacher_plan_correctness",
    "implementation_fidelity",
    "primary_bottleneck",
    "failure_type",
    "algorithm_category",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize manually labeled "
            "Teacher-Plan failure cases."
        )
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Labeled teacher_failure_cases.csv path.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary outputs.",
    )

    parser.add_argument(
        "--expected-count",
        type=int,
        default=26,
        help=(
            "Expected number of Teacher-Plan failures. "
            "Use -1 to disable this validation."
        ),
    )

    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Continue even if some manual labels are empty. "
            "Without this flag, incomplete labels raise an error."
        ),
    )

    return parser.parse_args()


def normalize_boolean_series(
    series: pd.Series,
) -> pd.Series:
    """
    bool, 0/1, 문자열 True/False를 bool로 통일한다.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "pass": True,
                "fail": False,
                "passed": True,
                "failed": False,
            }
        )
    )

    if normalized.isna().any():
        invalid_values = (
            series[normalized.isna()]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Cannot normalize boolean values: "
            f"{invalid_values}"
        )

    return normalized.astype(bool)


def load_labeled_failures(
    input_path: Path,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path.resolve()}"
        )

    dataframe = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
    )

    required_columns = {
        "problem_id",
        "title",
        "difficulty",
        "three_strategy_pattern",
        "direct_passed",
        "self_plan_passed",
        "teacher_plan_passed",
        "teacher_status",
        "teacher_passed_tests",
        "teacher_total_tests",
        "teacher_test_pass_ratio",
        *LABEL_COLUMNS,
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe["problem_id"].duplicated().any():
        duplicated_ids = (
            dataframe.loc[
                dataframe["problem_id"].duplicated(
                    keep=False
                ),
                "problem_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicated problem IDs: "
            f"{sorted(duplicated_ids)}"
        )

    for column in [
        "direct_passed",
        "self_plan_passed",
        "teacher_plan_passed",
    ]:
        dataframe[column] = (
            normalize_boolean_series(
                dataframe[column]
            )
        )

    for column in LABEL_COLUMNS:
        dataframe[column] = (
            dataframe[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    numeric_columns = [
        "teacher_passed_tests",
        "teacher_total_tests",
        "teacher_test_pass_ratio",
    ]

    optional_numeric_columns = [
        "teacher_total_tokens",
        "teacher_generation_time",
        "teacher_execution_time",
        "teacher_minus_self_ratio",
        "self_test_pass_ratio",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        if dataframe[column].isna().any():
            invalid_ids = dataframe.loc[
                dataframe[column].isna(),
                "problem_id",
            ].tolist()

            raise ValueError(
                f"Invalid numeric values in {column}: "
                f"{invalid_ids}"
            )

    for column in optional_numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    return dataframe


def build_label_completeness(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for column in LABEL_COLUMNS:
        labeled_count = int(
            dataframe[column].ne("").sum()
        )

        total_count = len(dataframe)

        rows.append(
            {
                "label_column": column,
                "labeled_count": labeled_count,
                "missing_count": (
                    total_count - labeled_count
                ),
                "completion_rate": (
                    labeled_count / total_count
                    if total_count
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def validate_failure_cases(
    dataframe: pd.DataFrame,
    *,
    expected_count: int,
    allow_incomplete: bool,
) -> pd.DataFrame:
    if (
        expected_count >= 0
        and len(dataframe) != expected_count
    ):
        raise ValueError(
            "Unexpected failure case count: "
            f"expected={expected_count}, "
            f"actual={len(dataframe)}"
        )

    invalid_mask = dataframe[
        "teacher_plan_passed"
    ]

    if invalid_mask.any():
        invalid_ids = dataframe.loc[
            invalid_mask,
            "problem_id",
        ].tolist()

        raise ValueError(
            "Rows containing Teacher-Plan PASS: "
            f"{invalid_ids}"
        )

    completeness = build_label_completeness(
        dataframe
    )

    incomplete = completeness[
        completeness["missing_count"] > 0
    ]

    if (
        not incomplete.empty
        and not allow_incomplete
    ):
        details = ", ".join(
            (
                f"{row.label_column}="
                f"{int(row.missing_count)} missing"
            )
            for row in incomplete.itertuples()
        )

        raise ValueError(
            "Manual labeling is incomplete: "
            f"{details}. "
            "Use --allow-incomplete to summarize "
            "partially labeled data."
        )

    return completeness


def count_summary(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    labeled = dataframe[
        dataframe[column].ne("")
    ]

    result = (
        labeled[column]
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="count")
    )

    total_labeled = len(labeled)

    result["proportion"] = (
        result["count"] / total_labeled
        if total_labeled
        else 0.0
    )

    return result


def build_difficulty_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    aggregation: dict[str, tuple[str, str]] = {
        "num_failures": (
            "problem_id",
            "count",
        ),
        "mean_teacher_test_pass_ratio": (
            "teacher_test_pass_ratio",
            "mean",
        ),
        "median_teacher_test_pass_ratio": (
            "teacher_test_pass_ratio",
            "median",
        ),
    }

    if "teacher_total_tokens" in dataframe.columns:
        aggregation[
            "mean_teacher_total_tokens"
        ] = (
            "teacher_total_tokens",
            "mean",
        )

    if "teacher_generation_time" in dataframe.columns:
        aggregation[
            "mean_teacher_generation_time"
        ] = (
            "teacher_generation_time",
            "mean",
        )

    result = (
        dataframe.groupby(
            "difficulty",
            dropna=False,
        )
        .agg(**aggregation)
        .reset_index()
    )

    difficulty_order = {
        "easy": 0,
        "medium": 1,
        "hard": 2,
    }

    result["_order"] = (
        result["difficulty"]
        .map(difficulty_order)
        .fillna(99)
    )

    return (
        result.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def build_crosstab(
    dataframe: pd.DataFrame,
    row_column: str,
    column_column: str,
) -> pd.DataFrame:
    filtered = dataframe[
        dataframe[row_column].ne("")
        & dataframe[column_column].ne("")
    ]

    table = pd.crosstab(
        filtered[row_column],
        filtered[column_column],
        margins=True,
    )

    return table.reset_index()


def build_bottleneck_test_ratio_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    filtered = dataframe[
        dataframe["primary_bottleneck"].ne("")
    ]

    return (
        filtered.groupby(
            "primary_bottleneck",
            dropna=False,
        )
        .agg(
            count=(
                "problem_id",
                "count",
            ),
            mean_test_pass_ratio=(
                "teacher_test_pass_ratio",
                "mean",
            ),
            median_test_pass_ratio=(
                "teacher_test_pass_ratio",
                "median",
            ),
            zero_test_pass_count=(
                "teacher_test_pass_ratio",
                lambda values: int(
                    (values == 0).sum()
                ),
            ),
        )
        .reset_index()
        .sort_values(
            [
                "count",
                "primary_bottleneck",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def build_plan_correctness_test_ratio_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    filtered = dataframe[
        dataframe[
            "teacher_plan_correctness"
        ].ne("")
    ]

    return (
        filtered.groupby(
            "teacher_plan_correctness",
            dropna=False,
        )
        .agg(
            count=(
                "problem_id",
                "count",
            ),
            mean_test_pass_ratio=(
                "teacher_test_pass_ratio",
                "mean",
            ),
            median_test_pass_ratio=(
                "teacher_test_pass_ratio",
                "median",
            ),
        )
        .reset_index()
    )


def build_case_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "problem_id",
        "title",
        "difficulty",
        "three_strategy_pattern",
        "teacher_status",
        "teacher_passed_tests",
        "teacher_total_tests",
        "teacher_test_pass_ratio",
        "teacher_plan_correctness",
        "implementation_fidelity",
        "primary_bottleneck",
        "failure_type",
        "algorithm_category",
        "missing_or_wrong_information",
        "analysis_note",
    ]

    return dataframe[columns].copy()


def format_dataframe(
    dataframe: pd.DataFrame,
) -> str:
    if dataframe.empty:
        return "(no rows)"

    return dataframe.to_string(index=False)


def print_section(
    title: str,
    dataframe: pd.DataFrame,
) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(format_dataframe(dataframe))


def write_text_report(
    *,
    output_path: Path,
    dataframe: pd.DataFrame,
    completeness: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
) -> None:
    lines = [
        "=" * 100,
        "Teacher-Plan Failure Case Summary",
        "=" * 100,
        "",
        f"Number of failure cases: {len(dataframe)}",
        (
            "Mean Teacher-Plan test pass ratio: "
            f"{dataframe['teacher_test_pass_ratio'].mean():.4f}"
        ),
        (
            "Median Teacher-Plan test pass ratio: "
            f"{dataframe['teacher_test_pass_ratio'].median():.4f}"
        ),
        (
            "Zero-test-pass failures: "
            f"{int((dataframe['teacher_test_pass_ratio'] == 0).sum())}"
        ),
        "",
        "=" * 100,
        "Label Completeness",
        "=" * 100,
        format_dataframe(completeness),
    ]

    section_titles = {
        "teacher_plan_correctness":
            "Teacher-Plan Correctness",
        "implementation_fidelity":
            "Implementation Fidelity",
        "primary_bottleneck":
            "Primary Bottleneck",
        "failure_type":
            "Failure Type",
        "algorithm_category":
            "Algorithm Category",
        "difficulty_summary":
            "Difficulty Summary",
        "plan_correctness_x_fidelity":
            "Plan Correctness × Implementation Fidelity",
        "plan_correctness_x_bottleneck":
            "Plan Correctness × Primary Bottleneck",
        "bottleneck_x_difficulty":
            "Primary Bottleneck × Difficulty",
        "failure_type_x_status":
            "Failure Type × Evaluation Status",
        "bottleneck_x_status":
            "Primary Bottleneck × Evaluation Status",
        "bottleneck_test_ratio":
            "Test Pass Ratio by Primary Bottleneck",
        "plan_correctness_test_ratio":
            "Test Pass Ratio by Plan Correctness",
    }

    for key, title in section_titles.items():
        lines.extend(
            [
                "",
                "=" * 100,
                title,
                "=" * 100,
                format_dataframe(
                    summaries[key]
                ),
            ]
        )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    dataframe = load_labeled_failures(
        args.input_path
    )

    completeness = validate_failure_cases(
        dataframe,
        expected_count=args.expected_count,
        allow_incomplete=args.allow_incomplete,
    )

    summaries = {
        column: count_summary(
            dataframe,
            column,
        )
        for column in CATEGORICAL_LABEL_COLUMNS
    }

    summaries["difficulty_summary"] = (
        build_difficulty_summary(
            dataframe
        )
    )

    summaries["plan_correctness_x_fidelity"] = (
        build_crosstab(
            dataframe,
            "teacher_plan_correctness",
            "implementation_fidelity",
        )
    )

    summaries["plan_correctness_x_bottleneck"] = (
        build_crosstab(
            dataframe,
            "teacher_plan_correctness",
            "primary_bottleneck",
        )
    )

    summaries["bottleneck_x_difficulty"] = (
        build_crosstab(
            dataframe,
            "primary_bottleneck",
            "difficulty",
        )
    )

    summaries["failure_type_x_status"] = (
        build_crosstab(
            dataframe,
            "failure_type",
            "teacher_status",
        )
    )

    summaries["bottleneck_x_status"] = (
        build_crosstab(
            dataframe,
            "primary_bottleneck",
            "teacher_status",
        )
    )

    summaries["bottleneck_test_ratio"] = (
        build_bottleneck_test_ratio_summary(
            dataframe
        )
    )

    summaries["plan_correctness_test_ratio"] = (
        build_plan_correctness_test_ratio_summary(
            dataframe
        )
    )

    case_table = build_case_table(
        dataframe
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_tables = {
        "label_completeness.csv":
            completeness,
        "teacher_plan_correctness_summary.csv":
            summaries[
                "teacher_plan_correctness"
            ],
        "implementation_fidelity_summary.csv":
            summaries[
                "implementation_fidelity"
            ],
        "primary_bottleneck_summary.csv":
            summaries[
                "primary_bottleneck"
            ],
        "failure_type_summary.csv":
            summaries[
                "failure_type"
            ],
        "algorithm_category_summary.csv":
            summaries[
                "algorithm_category"
            ],
        "difficulty_failure_summary.csv":
            summaries[
                "difficulty_summary"
            ],
        "plan_correctness_x_fidelity.csv":
            summaries[
                "plan_correctness_x_fidelity"
            ],
        "plan_correctness_x_bottleneck.csv":
            summaries[
                "plan_correctness_x_bottleneck"
            ],
        "bottleneck_x_difficulty.csv":
            summaries[
                "bottleneck_x_difficulty"
            ],
        "failure_type_x_status.csv":
            summaries[
                "failure_type_x_status"
            ],
        "bottleneck_x_status.csv":
            summaries[
                "bottleneck_x_status"
            ],
        "bottleneck_test_ratio.csv":
            summaries[
                "bottleneck_test_ratio"
            ],
        "plan_correctness_test_ratio.csv":
            summaries[
                "plan_correctness_test_ratio"
            ],
        "labeled_failure_table.csv":
            case_table,
    }

    for filename, table in output_tables.items():
        output_path = (
            args.output_dir / filename
        )

        table.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"[SAVED] {output_path}")

    report_path = (
        args.output_dir
        / "teacher_failure_summary.txt"
    )

    write_text_report(
        output_path=report_path,
        dataframe=dataframe,
        completeness=completeness,
        summaries=summaries,
    )

    print(f"[SAVED] {report_path}")

    print_section(
        "Label Completeness",
        completeness,
    )

    print_section(
        "Teacher-Plan Correctness",
        summaries["teacher_plan_correctness"],
    )

    print_section(
        "Implementation Fidelity",
        summaries["implementation_fidelity"],
    )

    print_section(
        "Primary Bottleneck",
        summaries["primary_bottleneck"],
    )

    print_section(
        "Failure Type",
        summaries["failure_type"],
    )

    print_section(
        "Algorithm Category",
        summaries["algorithm_category"],
    )

    print_section(
        "Difficulty Summary",
        summaries["difficulty_summary"],
    )

    print_section(
        "Plan Correctness × Implementation Fidelity",
        summaries["plan_correctness_x_fidelity"],
    )

    print_section(
        "Plan Correctness × Primary Bottleneck",
        summaries["plan_correctness_x_bottleneck"],
    )

    print_section(
        "Primary Bottleneck × Difficulty",
        summaries["bottleneck_x_difficulty"],
    )

    print_section(
        "Failure Type × Evaluation Status",
        summaries["failure_type_x_status"],
    )

    print_section(
        "Primary Bottleneck × Evaluation Status",
        summaries["bottleneck_x_status"],
    )

    print_section(
        "Test Pass Ratio by Primary Bottleneck",
        summaries["bottleneck_test_ratio"],
    )

    print()
    print(
        "[DONE] Teacher-Plan failure labels "
        "summarized successfully."
    )


if __name__ == "__main__":
    main()