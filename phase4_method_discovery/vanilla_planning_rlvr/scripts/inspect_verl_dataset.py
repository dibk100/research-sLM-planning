"""
PYTHONPATH=. python \
  phase4_method_discovery/vanilla_planning_rlvr/scripts/inspect_verl_dataset.py \
  --input phase4_method_discovery/vanilla_planning_rlvr/data/processed/train.parquet \
  --num-samples 3

PYTHONPATH=. python \
  phase4_method_discovery/vanilla_planning_rlvr/scripts/inspect_verl_dataset.py \
  --input phase4_method_discovery/vanilla_planning_rlvr/data/processed/train.parquet \
  --num-samples 1 \
  --show-tests
  
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# Project root
# ============================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.schemas import ProblemExample


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a verl-compatible Planning-RLVR parquet dataset."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to train.parquet or val.parquet.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of rows to print in detail.",
    )

    parser.add_argument(
        "--show-tests",
        action="store_true",
        help="Print public/private tests from extra_info.problem.",
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def resolve_path(
    path: str | Path,
) -> Path:
    resolved = Path(path)

    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved

    if not resolved.exists():
        raise FileNotFoundError(
            f"Dataset not found: {resolved}"
        )

    return resolved


def normalize_mapping(
    value: Any,
) -> dict[str, Any]:
    """
    Convert parquet-loaded mapping-like objects into a plain dict.

    Depending on pyarrow/pandas versions, nested structs may already
    be dicts, but this keeps inspection robust.
    """

    if isinstance(value, dict):
        return value

    if hasattr(value, "as_py"):
        converted = value.as_py()

        if isinstance(converted, dict):
            return converted

    raise TypeError(
        "Expected mapping-like value, "
        f"got {type(value).__name__}"
    )


def normalize_prompt(
    value: Any,
) -> list[dict[str, str]]:
    """
    Normalize the prompt field to:

        [
            {
                "role": "...",
                "content": "..."
            }
        ]
    """

    if hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, list):
        raise TypeError(
            "prompt must be list, "
            f"got {type(value).__name__}"
        )

    messages: list[dict[str, str]] = []

    for index, item in enumerate(value):
        if hasattr(item, "as_py"):
            item = item.as_py()

        if not isinstance(item, dict):
            raise TypeError(
                f"prompt[{index}] must be dict, "
                f"got {type(item).__name__}"
            )

        role = item.get("role")
        content = item.get("content")

        if not isinstance(role, str):
            raise TypeError(
                f"prompt[{index}]['role'] must be str."
            )

        if not isinstance(content, str):
            raise TypeError(
                f"prompt[{index}]['content'] must be str."
            )

        messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    return messages


def restore_problem(
    extra_info: dict[str, Any],
) -> ProblemExample:
    if "problem" not in extra_info:
        raise KeyError(
            "extra_info['problem'] is missing."
        )

    problem_payload = extra_info["problem"]

    if hasattr(problem_payload, "as_py"):
        problem_payload = problem_payload.as_py()

    if isinstance(problem_payload, ProblemExample):
        return problem_payload

    if not isinstance(problem_payload, dict):
        raise TypeError(
            "extra_info['problem'] must be dict or ProblemExample, "
            f"got {type(problem_payload).__name__}"
        )

    return ProblemExample(
        **problem_payload
    )


def validate_row(
    row: pd.Series,
    *,
    row_index: int,
) -> tuple[
    list[dict[str, str]],
    dict[str, Any],
    dict[str, Any],
    ProblemExample,
]:
    required_columns = (
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info",
    )

    for column in required_columns:
        if column not in row:
            raise KeyError(
                f"Row {row_index}: missing column {column!r}"
            )

    data_source = row["data_source"]

    if data_source != "livecodebench_v6":
        raise ValueError(
            f"Row {row_index}: unexpected data_source="
            f"{data_source!r}"
        )

    if row["ability"] != "code_planning":
        raise ValueError(
            f"Row {row_index}: unexpected ability="
            f"{row['ability']!r}"
        )

    prompt = normalize_prompt(
        row["prompt"]
    )

    if not prompt:
        raise ValueError(
            f"Row {row_index}: prompt is empty."
        )

    reward_model = normalize_mapping(
        row["reward_model"]
    )

    extra_info = normalize_mapping(
        row["extra_info"]
    )

    if "ground_truth" not in reward_model:
        raise KeyError(
            f"Row {row_index}: "
            "reward_model['ground_truth'] missing."
        )

    if "problem_id" not in extra_info:
        raise KeyError(
            f"Row {row_index}: "
            "extra_info['problem_id'] missing."
        )

    if "problem_text" not in extra_info:
        raise KeyError(
            f"Row {row_index}: "
            "extra_info['problem_text'] missing."
        )

    problem = restore_problem(
        extra_info
    )

    if (
        extra_info["problem_id"]
        != problem.problem_id
    ):
        raise ValueError(
            f"Row {row_index}: problem_id mismatch: "
            f"extra_info={extra_info['problem_id']!r}, "
            f"problem={problem.problem_id!r}"
        )

    if (
        extra_info["problem_text"].strip()
        != problem.problem.strip()
    ):
        raise ValueError(
            f"Row {row_index}: problem_text mismatch."
        )

    if problem.dataset != "livecodebench_v6":
        raise ValueError(
            f"Row {row_index}: restored problem has "
            f"unexpected dataset={problem.dataset!r}"
        )

    total_tests = (
        len(problem.public_tests)
        + len(problem.private_tests)
    )

    if total_tests == 0:
        raise ValueError(
            f"Row {row_index}: restored problem has no tests."
        )

    return (
        prompt,
        reward_model,
        extra_info,
        problem,
    )


# ============================================================
# Printing
# ============================================================

def print_dataset_summary(
    df: pd.DataFrame,
    input_path: Path,
) -> None:
    print("=" * 80)
    print("Planning-RLVR Dataset Inspection")
    print("=" * 80)

    print(
        f"path           : {input_path}"
    )

    print(
        f"rows           : {len(df)}"
    )

    print(
        f"columns        : {list(df.columns)}"
    )

    print()

    for column in df.columns:
        print(
            f"dtype[{column}] : {df[column].dtype}"
        )

    print("=" * 80)


def print_sample(
    *,
    index: int,
    prompt: list[dict[str, str]],
    reward_model: dict[str, Any],
    extra_info: dict[str, Any],
    problem: ProblemExample,
    show_tests: bool,
) -> None:
    print()
    print("=" * 80)
    print(f"Sample {index}")
    print("=" * 80)

    print(
        f"problem_id      : {problem.problem_id}"
    )

    print(
        f"title           : {problem.title}"
    )

    print(
        f"difficulty      : {problem.difficulty}"
    )

    print(
        f"rating          : {problem.rating}"
    )

    print(
        f"evaluation_type : {problem.evaluation_type}"
    )

    print(
        f"public_tests    : {len(problem.public_tests)}"
    )

    print(
        f"private_tests   : {len(problem.private_tests)}"
    )

    print(
        f"split           : {extra_info.get('split')}"
    )

    print(
        f"dataset index   : {extra_info.get('index')}"
    )

    print(
        f"reward style    : {reward_model.get('style')}"
    )

    print(
        f"ground_truth    : "
        f"{reward_model.get('ground_truth')!r}"
    )

    print()
    print("-" * 80)
    print("Prompt")
    print("-" * 80)

    for message_index, message in enumerate(
        prompt
    ):
        print(
            f"[{message_index}] role="
            f"{message['role']}"
        )
        print(
            message["content"]
        )

    print()
    print("-" * 80)
    print("Problem")
    print("-" * 80)

    print(
        problem.problem
    )

    if problem.starter_code:
        print()
        print("-" * 80)
        print("Starter Code")
        print("-" * 80)
        print(
            problem.starter_code
        )

    if show_tests:
        print()
        print("-" * 80)
        print("Public Tests")
        print("-" * 80)

        print(
            json.dumps(
                problem.public_tests,
                indent=2,
                ensure_ascii=False,
            )
        )

        print()
        print("-" * 80)
        print("Private Tests")
        print("-" * 80)

        print(
            json.dumps(
                problem.private_tests,
                indent=2,
                ensure_ascii=False,
            )
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    if args.num_samples <= 0:
        raise ValueError(
            "--num-samples must be > 0."
        )

    input_path = resolve_path(
        args.input
    )

    df = pd.read_parquet(
        input_path,
        engine="pyarrow",
    )

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    print_dataset_summary(
        df,
        input_path,
    )

    num_to_inspect = min(
        args.num_samples,
        len(df),
    )

    validated = 0

    for index in range(len(df)):
        row = df.iloc[index]

        try:
            (
                prompt,
                reward_model,
                extra_info,
                problem,
            ) = validate_row(
                row,
                row_index=index,
            )

        except Exception as exc:
            print()
            print("=" * 80)
            print("[FAIL] Dataset validation failed")
            print("=" * 80)
            print(
                f"row   : {index}"
            )
            print(
                f"error : {type(exc).__name__}: {exc}"
            )
            raise

        validated += 1

        if index < num_to_inspect:
            print_sample(
                index=index,
                prompt=prompt,
                reward_model=reward_model,
                extra_info=extra_info,
                problem=problem,
                show_tests=args.show_tests,
            )

    print()
    print("=" * 80)
    print("Validation Summary")
    print("=" * 80)

    print(
        f"validated rows : {validated}/{len(df)}"
    )

    print(
        "[PASS] All rows are structurally valid."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()