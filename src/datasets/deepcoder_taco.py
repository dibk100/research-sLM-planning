# src/datasets/deepcoder_taco.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from src.schemas import ProblemExample


DATASET_NAME = "deepcoder_taco"


class DeepCoderTACOFormatError(ValueError):
    """Raised when a DeepCoder TACO row cannot be converted."""


def parse_tests(
    raw_tests: Any,
) -> dict[str, Any]:
    """
    Parse the DeepCoder TACO `tests` field.

    Expected schema:

        {
            "inputs": [...],
            "outputs": [...]
        }

    Functional examples additionally contain `fn_name`, but
    Phase 4 Vanilla Planning-RLVR initially uses stdin only.
    """

    if isinstance(raw_tests, dict):
        tests = raw_tests

    elif isinstance(raw_tests, str):
        try:
            tests = json.loads(raw_tests)
        except json.JSONDecodeError as exc:
            raise DeepCoderTACOFormatError(
                "Failed to parse tests as JSON."
            ) from exc

    else:
        raise DeepCoderTACOFormatError(
            "tests must be dict or JSON string, "
            f"got {type(raw_tests).__name__}"
        )

    if not isinstance(tests, dict):
        raise DeepCoderTACOFormatError(
            "Parsed tests must be dict."
        )

    inputs = tests.get("inputs")
    outputs = tests.get("outputs")

    if not isinstance(inputs, list):
        raise DeepCoderTACOFormatError(
            "tests['inputs'] must be list."
        )

    if not isinstance(outputs, list):
        raise DeepCoderTACOFormatError(
            "tests['outputs'] must be list."
        )

    if len(inputs) != len(outputs):
        raise DeepCoderTACOFormatError(
            "Number of inputs and outputs does not match: "
            f"{len(inputs)} != {len(outputs)}"
        )

    if len(inputs) == 0:
        raise DeepCoderTACOFormatError(
            "No test cases."
        )

    return tests


def is_stdin_problem(
    tests: dict[str, Any],
) -> bool:
    """
    DeepCoder TACO functional problems contain fn_name.
    """

    return not bool(
        tests.get("fn_name")
    )

def build_test_cases(
    *,
    tests: dict[str, Any],
    row_index: int,
) -> list[dict[str, Any]]:
    """
    Convert DeepCoder TACO inputs/outputs arrays to the local
    ProblemExample test schema.

    Important:
    TACO stdin inputs are not always strings.
    Some examples represent stdin as list[str].

    Preserve the original representation because the
    DeepCoder/rLLM evaluator handles these formats directly.
    """

    inputs = tests["inputs"]
    outputs = tests["outputs"]

    test_cases: list[dict[str, Any]] = []

    for test_index, (
        test_input,
        test_output,
    ) in enumerate(
        zip(
            inputs,
            outputs,
        )
    ):
        if test_input is None:
            raise DeepCoderTACOFormatError(
                f"row={row_index}, "
                f"test={test_index}: "
                "input is None"
            )

        if test_output is None:
            raise DeepCoderTACOFormatError(
                f"row={row_index}, "
                f"test={test_index}: "
                "output is None"
            )

        test_cases.append(
            {
                "input": test_input,
                "output": test_output,
            }
        )

    return test_cases


def convert_row_to_problem(
    row: dict[str, Any],
    *,
    row_index: int,
) -> ProblemExample | None:
    """
    Convert one DeepCoder TACO row into ProblemExample.

    Returns None for functional problems because Phase 4 initially
    uses stdin-only training data.
    """

    problem_text = row.get(
        "problem"
    )

    if (
        not isinstance(problem_text, str)
        or not problem_text.strip()
    ):
        raise DeepCoderTACOFormatError(
            f"row={row_index}: invalid problem text."
        )

    tests = parse_tests(
        row.get("tests")
    )

    if not is_stdin_problem(
        tests
    ):
        return None

    test_cases = build_test_cases(
        tests=tests,
        row_index=row_index,
    )

    problem_id = (
        f"deepcoder_taco_{row_index:05d}"
    )

    return ProblemExample(
        problem_id=problem_id,
        title=problem_id,

        problem=problem_text.strip(),
        starter_code="",

        dataset=DATASET_NAME,
        platform="taco",

        difficulty=None,
        rating=None,
        contest_date="",

        evaluation_type="stdin",

        # Do not expose evaluator tests as model-visible data.
        public_tests=[],
        private_tests=test_cases,

        time_limit=None,
        memory_limit=None,

        function_name=None,
    )


def iter_raw_rows(
    path: str | Path,
) -> Iterator[
    tuple[int, dict[str, Any]]
]:
    """
    Stream raw DeepCoder TACO JSONL without loading all rows
    into memory.
    """

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"DeepCoder TACO JSONL not found: {input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for row_index, line in enumerate(f):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise DeepCoderTACOFormatError(
                    f"Invalid JSON at row {row_index}."
                ) from exc

            if not isinstance(row, dict):
                raise DeepCoderTACOFormatError(
                    f"row={row_index}: expected JSON object."
                )

            yield row_index, row


def load_deepcoder_taco_stdin(
    path: str | Path,
    *,
    strict: bool = True,
) -> list[ProblemExample]:
    """
    Load the stdin subset of DeepCoder TACO as ProblemExample.

    Expected count for the current released dataset:
        6387 stdin problems
    """

    problems: list[
        ProblemExample
    ] = []

    skipped_invalid = 0
    skipped_functional = 0

    for row_index, row in iter_raw_rows(
        path
    ):
        try:
            problem = (
                convert_row_to_problem(
                    row,
                    row_index=row_index,
                )
            )

        except Exception:
            if strict:
                raise

            skipped_invalid += 1
            continue

        if problem is None:
            skipped_functional += 1
            continue

        problems.append(
            problem
        )

    print(
        "[DeepCoderTACO] "
        f"stdin={len(problems)}, "
        f"functional_skipped={skipped_functional}, "
        f"invalid_skipped={skipped_invalid}"
    )

    return problems