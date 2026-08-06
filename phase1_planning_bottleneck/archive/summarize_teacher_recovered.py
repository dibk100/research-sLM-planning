"""
라벨링이 완료된 Teacher-Plan recovered cases를 집계한다.

Input:
    teacher_recovered_cases.csv

Outputs:
    label_completeness.csv
    plan_correctness_summary.csv
    plan_difference_summary.csv
    recovery_mechanism_summary.csv
    algorithm_category_summary.csv
    difficulty_recovery_summary.csv
    plan_correctness_crosstab.csv
    difference_by_difficulty.csv
    recovery_by_difficulty.csv
    labeled_case_table.csv
    teacher_recovered_summary.txt

Usage:

python -m archive.summarize_teacher_recovered \
  --input-path ./archive/teacher_recovered_50/teacher_recovered_cases.csv \
  --output-dir ./archive/teacher_recovered_50/summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path(
    "./archive/teacher_recovered_50/"
    "teacher_recovered_cases.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    "./archive/teacher_recovered_50/summary"
)

LABEL_COLUMNS = [
    "self_plan_correctness",
    "teacher_plan_correctness",
    "plan_difference_category",
    "recovery_mechanism",
    "algorithm_category",
    "key_teacher_information",
    "analysis_note",
]

CATEGORICAL_LABEL_COLUMNS = [
    "self_plan_correctness",
    "teacher_plan_correctness",
    "plan_difference_category",
    "recovery_mechanism",
    "algorithm_category",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize manually labeled "
            "Teacher-Plan recovered cases."
        )
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Labeled teacher_recovered_cases.csv path.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary CSV and TXT files.",
    )

    parser.add_argument(
        "--expected-count",
        type=int,
        default=16,
        help="Expected number of recovered cases.",
    )

    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Continue even when some manual labels "
            "are empty. By default, incomplete labels "
            "raise an error."
        ),
    )

    return parser.parse_args()


def load_labeled_cases(
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
        "direct_passed",
        "self_plan_passed",
        "teacher_plan_passed",
        "three_strategy_pattern",
        "self_status",
        "teacher_status",
        "self_test_pass_ratio",
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

    for column in LABEL_COLUMNS:
        dataframe[column] = (
            dataframe[column]
            .fillna("")
            .astype(str)
            .str.strip()
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


def validate_labels(
    dataframe: pd.DataFrame,
    *,
    expected_count: int,
    allow_incomplete: bool,
) -> pd.DataFrame:
    if len(dataframe) != expected_count:
        raise ValueError(
            "Unexpected recovered case count: "
            f"expected={expected_count}, "
            f"actual={len(dataframe)}"
        )

    invalid_recovery_mask = ~(
        (~dataframe["self_plan_passed"].astype(bool))
        & dataframe["teacher_plan_passed"].astype(bool)
    )

    if invalid_recovery_mask.any():
        invalid_ids = dataframe.loc[
            invalid_recovery_mask,
            "problem_id",
        ].tolist()

        raise ValueError(
            "Rows that are not "
            "Self-Plan FAIL -> Teacher-Plan PASS: "
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
            "partial labels."
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
    return (
        dataframe.groupby(
            "difficulty",
            dropna=False,
        )
        .agg(
            num_recovered=(
                "problem_id",
                "count",
            ),
            mean_self_test_pass_ratio=(
                "self_test_pass_ratio",
                "mean",
            ),
            mean_teacher_test_pass_ratio=(
                "teacher_test_pass_ratio",
                "mean",
            ),
            mean_test_pass_ratio_delta=(
                "test_pass_ratio_delta",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "difficulty",
            key=lambda series: series.map(
                {
                    "easy": 0,
                    "medium": 1,
                    "hard": 2,
                }
            ).fillna(99),
        )
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


def build_category_by_difficulty(
    dataframe: pd.DataFrame,
    category_column: str,
) -> pd.DataFrame:
    filtered = dataframe[
        dataframe[category_column].ne("")
    ]

    table = pd.crosstab(
        filtered[category_column],
        filtered["difficulty"],
        margins=True,
    )

    return table.reset_index()


def build_case_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "problem_id",
        "title",
        "difficulty",
        "three_strategy_pattern",
        "self_status",
        "self_test_pass_ratio",
        "self_plan_correctness",
        "teacher_plan_correctness",
        "plan_difference_category",
        "recovery_mechanism",
        "algorithm_category",
        "key_teacher_information",
        "analysis_note",
    ]

    return dataframe[columns].copy()


def format_dataframe(
    dataframe: pd.DataFrame,
) -> str:
    if dataframe.empty:
        return "(no rows)"

    return dataframe.to_string(index=False)


def write_text_report(
    *,
    output_path: Path,
    dataframe: pd.DataFrame,
    completeness: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
) -> None:
    lines = [
        "=" * 100,
        "Teacher-Plan Successfully Recovered Cases Summary",
        "=" * 100,
        "",
        f"Number of recovered cases: {len(dataframe)}",
        (
            "Mean Self-Plan test pass ratio: "
            f"{dataframe['self_test_pass_ratio'].mean():.4f}"
        ),
        (
            "Mean Teacher-Plan test pass ratio: "
            f"{dataframe['teacher_test_pass_ratio'].mean():.4f}"
        ),
        (
            "Mean test pass ratio improvement: "
            f"{dataframe['test_pass_ratio_delta'].mean():.4f}"
        ),
        "",
        "=" * 100,
        "Label Completeness",
        "=" * 100,
        format_dataframe(completeness),
    ]

    section_titles = {
        "self_plan_correctness":
            "Self-Plan Correctness",
        "teacher_plan_correctness":
            "Teacher-Plan Correctness",
        "plan_difference_category":
            "Plan Difference Category",
        "recovery_mechanism":
            "Recovery Mechanism",
        "algorithm_category":
            "Algorithm Category",
        "difficulty_summary":
            "Difficulty Summary",
        "plan_correctness_crosstab":
            "Self-Plan Correctness × "
            "Teacher-Plan Correctness",
        "difference_by_difficulty":
            "Plan Difference Category × Difficulty",
        "recovery_by_difficulty":
            "Recovery Mechanism × Difficulty",
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


def print_section(
    title: str,
    dataframe: pd.DataFrame,
) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(format_dataframe(dataframe))


def main() -> None:
    args = parse_args()

    dataframe = load_labeled_cases(
        args.input_path
    )

    completeness = validate_labels(
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
        build_difficulty_summary(dataframe)
    )

    summaries["plan_correctness_crosstab"] = (
        build_crosstab(
            dataframe,
            "self_plan_correctness",
            "teacher_plan_correctness",
        )
    )

    summaries["difference_by_difficulty"] = (
        build_category_by_difficulty(
            dataframe,
            "plan_difference_category",
        )
    )

    summaries["recovery_by_difficulty"] = (
        build_category_by_difficulty(
            dataframe,
            "recovery_mechanism",
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
        "self_plan_correctness_summary.csv":
            summaries[
                "self_plan_correctness"
            ],
        "teacher_plan_correctness_summary.csv":
            summaries[
                "teacher_plan_correctness"
            ],
        "plan_difference_summary.csv":
            summaries[
                "plan_difference_category"
            ],
        "recovery_mechanism_summary.csv":
            summaries[
                "recovery_mechanism"
            ],
        "algorithm_category_summary.csv":
            summaries[
                "algorithm_category"
            ],
        "difficulty_recovery_summary.csv":
            summaries[
                "difficulty_summary"
            ],
        "plan_correctness_crosstab.csv":
            summaries[
                "plan_correctness_crosstab"
            ],
        "difference_by_difficulty.csv":
            summaries[
                "difference_by_difficulty"
            ],
        "recovery_by_difficulty.csv":
            summaries[
                "recovery_by_difficulty"
            ],
        "labeled_case_table.csv":
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
        / "teacher_recovered_summary.txt"
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
        "Self-Plan Correctness",
        summaries["self_plan_correctness"],
    )

    print_section(
        "Teacher-Plan Correctness",
        summaries["teacher_plan_correctness"],
    )

    print_section(
        "Plan Difference Category",
        summaries["plan_difference_category"],
    )

    print_section(
        "Recovery Mechanism",
        summaries["recovery_mechanism"],
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
        "Self-Plan Correctness × "
        "Teacher-Plan Correctness",
        summaries["plan_correctness_crosstab"],
    )

    print()
    print(
        "[DONE] Teacher recovered labels "
        "summarized successfully."
    )


if __name__ == "__main__":
    main()