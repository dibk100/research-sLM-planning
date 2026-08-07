"""
usage :
python -m archive.compare_mcnemar

"""

from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


PROBLEM_COMPARISON_PATH = Path(
    "./archive/comparison_500/problem_comparison.csv"
)

if not PROBLEM_COMPARISON_PATH.exists():
    raise FileNotFoundError(
        f"File not found: {PROBLEM_COMPARISON_PATH.resolve()}"
    )

df = pd.read_csv(PROBLEM_COMPARISON_PATH)

required_columns = {
    "problem_id",
    "passed_direct",
    "passed_self_plan",
    "passed_teacher_plan",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns: {sorted(missing_columns)}\n"
        f"Available columns: {df.columns.tolist()}"
    )


def normalize_boolean(series: pd.Series) -> pd.Series:
    """CSV의 bool/string/0-1 값을 일관된 bool로 변환한다."""
    if pd.api.types.is_bool_dtype(series):
        return series

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
            }
        )
    )

    if normalized.isna().any():
        invalid_values = series[normalized.isna()].unique().tolist()
        raise ValueError(
            f"Cannot convert values to bool: {invalid_values}"
        )

    return normalized.astype(bool)


for column in [
    "passed_direct",
    "passed_self_plan",
    "passed_teacher_plan",
]:
    df[column] = normalize_boolean(df[column])


def run_mcnemar(
    dataframe: pd.DataFrame,
    source_column: str,
    target_column: str,
    comparison_name: str,
) -> dict:
    source = dataframe[source_column]
    target = dataframe[target_column]

    pass_to_pass = int((source & target).sum())
    pass_to_fail = int((source & ~target).sum())
    fail_to_pass = int((~source & target).sum())
    fail_to_fail = int((~source & ~target).sum())

    contingency_table = [
        [pass_to_pass, pass_to_fail],
        [fail_to_pass, fail_to_fail],
    ]

    # 불일치 쌍의 수가 작으므로 exact McNemar test 사용
    result = mcnemar(
        contingency_table,
        exact=True,
    )

    output = {
        "comparison": comparison_name,
        "pass_to_pass": pass_to_pass,
        "pass_to_fail": pass_to_fail,
        "fail_to_pass": fail_to_pass,
        "fail_to_fail": fail_to_fail,
        "discordant_pairs": pass_to_fail + fail_to_pass,
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "significant_0.05": result.pvalue < 0.05,
    }

    print("=" * 70)
    print(comparison_name)
    print("=" * 70)
    print("Contingency table")
    print("                    Target PASS   Target FAIL")
    print(
        f"Source PASS        {pass_to_pass:>11}   "
        f"{pass_to_fail:>11}"
    )
    print(
        f"Source FAIL        {fail_to_pass:>11}   "
        f"{fail_to_fail:>11}"
    )
    print()
    print(f"PASS -> FAIL    : {pass_to_fail}")
    print(f"FAIL -> PASS    : {fail_to_pass}")
    print(f"Discordant pairs: {pass_to_fail + fail_to_pass}")
    print(f"Statistic       : {result.statistic}")
    print(f"Exact p-value   : {result.pvalue:.6f}")
    print(
        "Conclusion      : "
        + (
            "Statistically significant"
            if result.pvalue < 0.05
            else "Not statistically significant"
        )
    )
    print()

    return output


results = [
    run_mcnemar(
        df,
        "passed_direct",
        "passed_self_plan",
        "Direct vs Self-Plan",
    ),
    run_mcnemar(
        df,
        "passed_direct",
        "passed_teacher_plan",
        "Direct vs Teacher-Plan",
    ),
    run_mcnemar(
        df,
        "passed_self_plan",
        "passed_teacher_plan",
        "Self-Plan vs Teacher-Plan",
    ),
]

result_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("McNemar Summary")
print("=" * 70)
print(result_df.to_string(index=False))