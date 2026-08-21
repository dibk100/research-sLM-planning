"""

PYTHONPATH="$HOME/workspace/project_sLM_planning" \
python \
  phase4_method_discovery/vanilla_planning_rlvr/scripts/inspect_verl_dataset.py \
  --input /mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/vanilla_planning_rlvr/train.parquet \
  --num-samples 2 \
  --show-tests \
  --max-tests-to-show 2

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ======================================================================
# Project root
# ======================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.schemas import ProblemExample


# ======================================================================
# Constants
# ======================================================================

EXPECTED_DATA_SOURCE = "deepcoder_taco"
EXPECTED_ABILITY = "code_planning"
EXPECTED_EVALUATION_TYPE = "stdin"


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect and validate a verl-compatible "
            "DeepCoder TACO parquet dataset."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Path to train.parquet or val.parquet."
        ),
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help=(
            "Number of validated rows to print in detail."
        ),
    )

    parser.add_argument(
        "--show-tests",
        action="store_true",
        help=(
            "Print private evaluator tests for inspected rows."
        ),
    )

    parser.add_argument(
        "--max-tests-to-show",
        type=int,
        default=3,
        help=(
            "Maximum number of private tests to print per sample."
        ),
    )

    return parser.parse_args()


# ======================================================================
# Helpers
# ======================================================================

def resolve_path(
    path: str | Path,
) -> Path:
    resolved = Path(
        path
    )

    if not resolved.is_absolute():
        resolved = (
            PROJECT_ROOT
            / resolved
        )

    if not resolved.exists():
        raise FileNotFoundError(
            f"Dataset not found: {resolved}"
        )

    return resolved


def normalize_mapping(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any]:
    """
    Normalize parquet-loaded struct-like values to plain dict.
    """

    if isinstance(
        value,
        dict,
    ):
        return value

    if hasattr(
        value,
        "as_py",
    ):
        converted = value.as_py()

        if isinstance(
            converted,
            dict,
        ):
            return converted

    raise TypeError(
        f"{field_name} must be mapping-like, "
        f"got {type(value).__name__}"
    )


def normalize_prompt(
    value: Any,
) -> list[dict[str, str]]:
    """
    Normalize verl prompt field.

    Expected:

        [
            {
                "role": "user",
                "content": "..."
            }
        ]
    """

    if hasattr(
        value,
        "tolist",
    ):
        value = value.tolist()

    if not isinstance(
        value,
        list,
    ):
        raise TypeError(
            "prompt must be list, "
            f"got {type(value).__name__}"
        )

    messages: list[
        dict[str, str]
    ] = []

    for index, item in enumerate(
        value
    ):
        if hasattr(
            item,
            "as_py",
        ):
            item = item.as_py()

        if not isinstance(
            item,
            dict,
        ):
            raise TypeError(
                f"prompt[{index}] must be dict, "
                f"got {type(item).__name__}"
            )

        role = item.get(
            "role"
        )

        content = item.get(
            "content"
        )

        if not isinstance(
            role,
            str,
        ):
            raise TypeError(
                f"prompt[{index}]['role'] must be str."
            )

        if not isinstance(
            content,
            str,
        ):
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
    """
    Restore ProblemExample from extra_info.problem_json.
    """

    if "problem_json" not in extra_info:
        raise KeyError(
            "extra_info['problem_json'] is missing."
        )

    problem_json = extra_info[
        "problem_json"
    ]

    if not isinstance(
        problem_json,
        str,
    ):
        raise TypeError(
            "extra_info['problem_json'] must be str, "
            f"got {type(problem_json).__name__}"
        )

    try:
        payload = json.loads(
            problem_json
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Failed to parse extra_info['problem_json']."
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            "Decoded problem_json must be dict."
        )

    try:
        problem = ProblemExample(
            **payload
        )

    except TypeError as exc:
        raise TypeError(
            "Failed to restore ProblemExample "
            "from problem_json."
        ) from exc

    return problem


# ======================================================================
# Row validation
# ======================================================================

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
                f"row={row_index}: "
                f"missing column {column!r}"
            )

    # ------------------------------------------------------------------
    # data_source
    # ------------------------------------------------------------------

    data_source = row[
        "data_source"
    ]

    if (
        data_source
        != EXPECTED_DATA_SOURCE
    ):
        raise ValueError(
            f"row={row_index}: "
            f"unexpected data_source="
            f"{data_source!r}"
        )

    # ------------------------------------------------------------------
    # ability
    # ------------------------------------------------------------------

    ability = row[
        "ability"
    ]

    if (
        ability
        != EXPECTED_ABILITY
    ):
        raise ValueError(
            f"row={row_index}: "
            f"unexpected ability="
            f"{ability!r}"
        )

    # ------------------------------------------------------------------
    # prompt
    # ------------------------------------------------------------------

    prompt = normalize_prompt(
        row["prompt"]
    )

    if not prompt:
        raise ValueError(
            f"row={row_index}: "
            "prompt is empty."
        )

    if (
        prompt[0]["role"]
        != "user"
    ):
        raise ValueError(
            f"row={row_index}: "
            f"first prompt role must be 'user'."
        )

    if not prompt[0][
        "content"
    ].strip():
        raise ValueError(
            f"row={row_index}: "
            "prompt content is empty."
        )

    # ------------------------------------------------------------------
    # reward_model
    # ------------------------------------------------------------------

    reward_model = normalize_mapping(
        row["reward_model"],
        field_name="reward_model",
    )

    if (
        reward_model.get(
            "style"
        )
        != "rule"
    ):
        raise ValueError(
            f"row={row_index}: "
            "reward_model['style'] must be 'rule'."
        )

    if "ground_truth" not in reward_model:
        raise KeyError(
            f"row={row_index}: "
            "reward_model['ground_truth'] missing."
        )

    # ------------------------------------------------------------------
    # extra_info
    # ------------------------------------------------------------------

    extra_info = normalize_mapping(
        row["extra_info"],
        field_name="extra_info",
    )

    required_extra_fields = (
        "schema_version",
        "split",
        "index",
        "problem_id",
        "problem_text",
        "problem_json",
    )

    for field_name in (
        required_extra_fields
    ):
        if (
            field_name
            not in extra_info
        ):
            raise KeyError(
                f"row={row_index}: "
                f"extra_info[{field_name!r}] missing."
            )

    # ------------------------------------------------------------------
    # ProblemExample restoration
    # ------------------------------------------------------------------

    problem = restore_problem(
        extra_info
    )

    # ------------------------------------------------------------------
    # Cross-field consistency
    # ------------------------------------------------------------------

    if (
        extra_info[
            "problem_id"
        ]
        != problem.problem_id
    ):
        raise ValueError(
            f"row={row_index}: "
            "problem_id mismatch: "
            f"extra_info="
            f"{extra_info['problem_id']!r}, "
            f"problem="
            f"{problem.problem_id!r}"
        )

    if (
        str(
            extra_info[
                "problem_text"
            ]
        ).strip()
        != problem.problem.strip()
    ):
        raise ValueError(
            f"row={row_index}: "
            "problem_text mismatch."
        )

    if (
        problem.dataset
        != EXPECTED_DATA_SOURCE
    ):
        raise ValueError(
            f"row={row_index}: "
            f"restored dataset="
            f"{problem.dataset!r}"
        )

    if (
        problem.evaluation_type
        != EXPECTED_EVALUATION_TYPE
    ):
        raise ValueError(
            f"row={row_index}: "
            f"unexpected evaluation_type="
            f"{problem.evaluation_type!r}"
        )

    # ------------------------------------------------------------------
    # Test integrity
    # ------------------------------------------------------------------

    if problem.public_tests:
        raise ValueError(
            f"row={row_index}: "
            "public_tests should be empty "
            "for TACO RLVR training."
        )

    if not problem.private_tests:
        raise ValueError(
            f"row={row_index}: "
            "private_tests is empty."
        )

    for test_index, test_case in enumerate(
        problem.private_tests
    ):
        if not isinstance(
            test_case,
            dict,
        ):
            raise TypeError(
                f"row={row_index}, "
                f"test={test_index}: "
                "test case must be dict."
            )

        if "input" not in test_case:
            raise KeyError(
                f"row={row_index}, "
                f"test={test_index}: "
                "missing input."
            )

        if "output" not in test_case:
            raise KeyError(
                f"row={row_index}, "
                f"test={test_index}: "
                "missing output."
            )

        if (
            test_case["input"]
            is None
        ):
            raise ValueError(
                f"row={row_index}, "
                f"test={test_index}: "
                "input is None."
            )

        if (
            test_case["output"]
            is None
        ):
            raise ValueError(
                f"row={row_index}, "
                f"test={test_index}: "
                "output is None."
            )

    # ------------------------------------------------------------------
    # Leakage structural check
    # ------------------------------------------------------------------

    prompt_serialized = json.dumps(
        prompt,
        ensure_ascii=False,
    )

    if (
        '"private_tests"'
        in prompt_serialized
        or '"problem_json"'
        in prompt_serialized
    ):
        raise ValueError(
            f"row={row_index}: "
            "reward payload leaked into prompt."
        )

    return (
        prompt,
        reward_model,
        extra_info,
        problem,
    )


# ======================================================================
# Summary
# ======================================================================

def print_dataset_summary(
    df: pd.DataFrame,
    input_path: Path,
) -> None:
    print("=" * 90)
    print(
        "DeepCoder TACO verl Dataset Inspection"
    )
    print("=" * 90)

    print(
        f"path       : {input_path}"
    )

    print(
        f"rows       : {len(df)}"
    )

    print(
        f"columns    : {list(df.columns)}"
    )

    print()

    for column in df.columns:
        print(
            f"dtype[{column}] : "
            f"{df[column].dtype}"
        )

    print("=" * 90)


# ======================================================================
# Sample printing
# ======================================================================

def preview_value(
    value: Any,
    *,
    max_chars: int = 1000,
) -> str:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )

    except TypeError:
        text = repr(
            value
        )

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + "\n... <truncated>"
    )


def print_sample(
    *,
    index: int,
    prompt: list[
        dict[str, str]
    ],
    reward_model: dict[
        str,
        Any
    ],
    extra_info: dict[
        str,
        Any
    ],
    problem: ProblemExample,
    show_tests: bool,
    max_tests_to_show: int,
) -> None:
    print()
    print("=" * 90)
    print(
        f"Sample {index}"
    )
    print("=" * 90)

    print(
        f"problem_id      : "
        f"{problem.problem_id}"
    )

    print(
        f"dataset         : "
        f"{problem.dataset}"
    )

    print(
        f"platform        : "
        f"{problem.platform}"
    )

    print(
        f"evaluation_type : "
        f"{problem.evaluation_type}"
    )

    print(
        f"split           : "
        f"{extra_info.get('split')}"
    )

    print(
        f"dataset index   : "
        f"{extra_info.get('index')}"
    )

    print(
        f"private_tests   : "
        f"{len(problem.private_tests)}"
    )

    print(
        f"public_tests    : "
        f"{len(problem.public_tests)}"
    )

    print(
        f"reward style    : "
        f"{reward_model.get('style')}"
    )

    print(
        f"ground_truth    : "
        f"{reward_model.get('ground_truth')!r}"
    )

    problem_json_length = len(
        extra_info["problem_json"]
    )

    print(
        f"problem_json    : "
        f"{problem_json_length} chars"
    )

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    print()
    print("-" * 90)
    print("Prompt")
    print("-" * 90)

    for message_index, message in enumerate(
        prompt
    ):
        print(
            f"[{message_index}] "
            f"role={message['role']}"
        )

        print(
            message["content"]
        )

    # ------------------------------------------------------------------
    # Problem
    # ------------------------------------------------------------------

    print()
    print("-" * 90)
    print("Restored Problem")
    print("-" * 90)

    print(
        problem.problem
    )

    if problem.starter_code:
        print()
        print(
            "[Starter Code]"
        )

        print(
            problem.starter_code
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    if show_tests:
        print()
        print("-" * 90)
        print("Private Tests")
        print("-" * 90)

        num_to_show = min(
            max_tests_to_show,
            len(
                problem.private_tests
            ),
        )

        for test_index in range(
            num_to_show
        ):
            test_case = (
                problem.private_tests[
                    test_index
                ]
            )

            print()
            print(
                f"[test {test_index}]"
            )

            print(
                preview_value(
                    test_case,
                    max_chars=2000,
                )
            )

        if (
            len(problem.private_tests)
            > num_to_show
        ):
            print()
            print(
                "... "
                f"{len(problem.private_tests) - num_to_show} "
                "more tests omitted"
            )


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()

    if args.num_samples < 0:
        raise ValueError(
            "--num-samples must be >= 0."
        )

    if args.max_tests_to_show <= 0:
        raise ValueError(
            "--max-tests-to-show must be > 0."
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

    num_to_print = min(
        args.num_samples,
        len(df),
    )

    validated = 0

    total_private_tests = 0

    min_private_tests: (
        int | None
    ) = None

    max_private_tests: (
        int | None
    ) = None

    split_counts: dict[
        str,
        int
    ] = {}

    input_type_counts: dict[
        str,
        int
    ] = {}

    output_type_counts: dict[
        str,
        int
    ] = {}

    for row_index in range(
        len(df)
    ):
        row = df.iloc[
            row_index
        ]

        try:
            (
                prompt,
                reward_model,
                extra_info,
                problem,
            ) = validate_row(
                row,
                row_index=row_index,
            )

        except Exception as exc:
            print()
            print("=" * 90)
            print(
                "[FAIL] Dataset validation failed"
            )
            print("=" * 90)

            print(
                f"row   : {row_index}"
            )

            print(
                f"error : "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            raise

        validated += 1

        # --------------------------------------------------------------
        # Aggregate statistics
        # --------------------------------------------------------------

        num_tests = len(
            problem.private_tests
        )

        total_private_tests += (
            num_tests
        )

        if (
            min_private_tests
            is None
            or num_tests
            < min_private_tests
        ):
            min_private_tests = (
                num_tests
            )

        if (
            max_private_tests
            is None
            or num_tests
            > max_private_tests
        ):
            max_private_tests = (
                num_tests
            )

        split = str(
            extra_info.get(
                "split",
                "",
            )
        )

        split_counts[
            split
        ] = (
            split_counts.get(
                split,
                0,
            )
            + 1
        )

        for test_case in (
            problem.private_tests
        ):
            input_type = type(
                test_case[
                    "input"
                ]
            ).__name__

            output_type = type(
                test_case[
                    "output"
                ]
            ).__name__

            input_type_counts[
                input_type
            ] = (
                input_type_counts.get(
                    input_type,
                    0,
                )
                + 1
            )

            output_type_counts[
                output_type
            ] = (
                output_type_counts.get(
                    output_type,
                    0,
                )
                + 1
            )

        # --------------------------------------------------------------
        # Sample print
        # --------------------------------------------------------------

        if (
            row_index
            < num_to_print
        ):
            print_sample(
                index=row_index,
                prompt=prompt,
                reward_model=reward_model,
                extra_info=extra_info,
                problem=problem,
                show_tests=(
                    args.show_tests
                ),
                max_tests_to_show=(
                    args.max_tests_to_show
                ),
            )

    # ==================================================================
    # Final summary
    # ==================================================================

    print()
    print("=" * 90)
    print(
        "Validation Summary"
    )
    print("=" * 90)

    print(
        f"validated rows      : "
        f"{validated}/{len(df)}"
    )

    print(
        f"split counts        : "
        f"{split_counts}"
    )

    print(
        f"total private tests : "
        f"{total_private_tests}"
    )

    if validated:
        print(
            f"mean tests/problem  : "
            f"{total_private_tests / validated:.2f}"
        )

    print(
        f"min tests/problem   : "
        f"{min_private_tests}"
    )

    print(
        f"max tests/problem   : "
        f"{max_private_tests}"
    )

    print(
        f"input types         : "
        f"{input_type_counts}"
    )

    print(
        f"output types        : "
        f"{output_type_counts}"
    )

    print()

    print(
        "[PASS] All rows are structurally valid."
    )

    print("=" * 90)


if __name__ == "__main__":
    main()