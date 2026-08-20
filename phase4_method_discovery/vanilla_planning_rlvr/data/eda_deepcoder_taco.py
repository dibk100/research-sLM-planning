"""
PYTHONPATH=. python \
  phase4_method_discovery/vanilla_planning_rlvr/data/eda_deepcoder_taco.py \
  --num-samples 5 \
  --output-dir /home/dibaeck/workspace/project_sLM_planning/phase4_method_discovery/archive
  
| 항목               |            결과 | 판단              |
| ---------------- | ------------: | --------------- |
| 전체 문제            |         7,436 | 충분              |
| problem 누락       |             0 | 좋음              |
| tests 누락         |             0 | 좋음              |
| solutions 누락     |             0 | 좋음              |
| tests parsing 실패 |             0 | 좋음              |
| stdin            | 6,387 (85.9%) | 주 구성            |
| functional       | 1,049 (14.1%) | 별도 evaluator 필요 |
| 최소 tests         |             6 | 충분              |
| median tests     |           102 | 매우 충분           |
| mean tests       |         104.1 | 매우 충분           |
| solution 없는 문제   |             0 | 좋음              |
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_INPUT = Path(
    "/mnt/hdd/project_sLM_planning/data/"
    "deepcoder_taco/raw/deepcoder_taco_train.jsonl"
)


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "EDA for DeepCoder TACO training dataset."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help=(
            "Path to deepcoder_taco_train.jsonl"
        ),
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help=(
            "Number of rows to print in detail."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "/mnt/hdd/project_sLM_planning/data/"
            "deepcoder_taco/eda"
        ),
    )

    return parser.parse_args()


# ======================================================================
# Loading
# ======================================================================

def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    rows: list[dict[str, Any]] = []

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
                payload = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line "
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(payload, dict):
                raise TypeError(
                    f"Line {line_number} must be JSON object."
                )

            rows.append(payload)

    if not rows:
        raise ValueError(
            "Dataset is empty."
        )

    return rows


# ======================================================================
# Parsing helpers
# ======================================================================

def try_parse_nested_value(
    value: Any,
) -> tuple[Any, str]:
    """
    Attempt to parse values that may be stored as:
    - native dict/list
    - JSON string
    - Python literal string

    Returns:
        parsed_value
        parsing_method
    """

    if isinstance(
        value,
        (dict, list),
    ):
        return value, "native"

    if not isinstance(value, str):
        return value, "other"

    text = value.strip()

    if not text:
        return value, "empty_string"

    try:
        return json.loads(text), "json"

    except Exception:
        pass

    try:
        return ast.literal_eval(text), "literal_eval"

    except Exception:
        return value, "raw_string"


def infer_tests_structure(
    parsed_tests: Any,
) -> str:
    """
    Roughly classify the test schema.
    """

    if isinstance(parsed_tests, dict):

        keys = set(parsed_tests.keys())

        if {
            "inputs",
            "outputs",
        }.issubset(keys):

            if "fn_name" in keys:
                return "functional_like"

            return "stdin_like"

        return (
            "dict:"
            + ",".join(
                sorted(
                    str(key)
                    for key in keys
                )
            )
        )

    if isinstance(parsed_tests, list):
        return "list"

    if isinstance(parsed_tests, str):
        return "string"

    return type(parsed_tests).__name__


def count_tests(
    parsed_tests: Any,
) -> int | None:
    """
    Estimate number of test cases.
    """

    if isinstance(parsed_tests, dict):

        inputs = parsed_tests.get(
            "inputs"
        )

        outputs = parsed_tests.get(
            "outputs"
        )

        if isinstance(inputs, list):
            return len(inputs)

        if isinstance(outputs, list):
            return len(outputs)

        return None

    if isinstance(parsed_tests, list):
        return len(parsed_tests)

    return None


def count_solutions(
    parsed_solutions: Any,
) -> int | None:
    if isinstance(
        parsed_solutions,
        list,
    ):
        return len(parsed_solutions)

    if isinstance(
        parsed_solutions,
        str,
    ):
        return (
            0
            if not parsed_solutions.strip()
            else 1
        )

    return None


# ======================================================================
# Statistics helpers
# ======================================================================

def percentile(
    values: list[int],
    p: float,
) -> float:
    if not values:
        return float("nan")

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    position = (
        p
        * (len(ordered) - 1)
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(
            ordered[lower]
        )

    weight = (
        position - lower
    )

    return (
        ordered[lower]
        * (1 - weight)
        + ordered[upper]
        * weight
    )


def summarize_numeric(
    values: list[int],
) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
        }

    return {
        "count": len(values),
        "min": min(values),
        "p25": percentile(
            values,
            0.25,
        ),
        "median": median(values),
        "mean": mean(values),
        "p75": percentile(
            values,
            0.75,
        ),
        "p90": percentile(
            values,
            0.90,
        ),
        "p95": percentile(
            values,
            0.95,
        ),
        "max": max(values),
    }


# ======================================================================
# Sample print
# ======================================================================

def print_sample(
    *,
    index: int,
    row: dict[str, Any],
) -> None:
    print()
    print("=" * 100)
    print(f"Sample {index}")
    print("=" * 100)

    problem = row.get(
        "problem"
    )

    tests = row.get(
        "tests"
    )

    solutions = row.get(
        "solutions"
    )

    parsed_tests, tests_method = (
        try_parse_nested_value(
            tests
        )
    )

    parsed_solutions, solutions_method = (
        try_parse_nested_value(
            solutions
        )
    )

    print()
    print("[problem]")

    if isinstance(
        problem,
        str,
    ):
        print(
            problem[:3000]
        )

        if len(problem) > 3000:
            print(
                f"... <truncated, chars={len(problem)}>"
            )
    else:
        print(
            repr(problem)
        )

    print()
    print("[tests]")
    print(
        f"raw type       : {type(tests).__name__}"
    )
    print(
        f"parse method   : {tests_method}"
    )
    print(
        f"parsed type    : "
        f"{type(parsed_tests).__name__}"
    )
    print(
        f"structure      : "
        f"{infer_tests_structure(parsed_tests)}"
    )
    print(
        f"test count     : "
        f"{count_tests(parsed_tests)}"
    )

    print()

    if isinstance(
        parsed_tests,
        (dict, list),
    ):
        print(
            json.dumps(
                parsed_tests,
                indent=2,
                ensure_ascii=False,
            )[:5000]
        )
    else:
        print(
            str(parsed_tests)[:5000]
        )

    print()
    print("[solutions]")
    print(
        f"raw type       : "
        f"{type(solutions).__name__}"
    )
    print(
        f"parse method   : "
        f"{solutions_method}"
    )
    print(
        f"parsed type    : "
        f"{type(parsed_solutions).__name__}"
    )
    print(
        f"solution count : "
        f"{count_solutions(parsed_solutions)}"
    )

    print()

    if isinstance(
        parsed_solutions,
        list,
    ):
        for solution_index, solution in enumerate(
            parsed_solutions[:2]
        ):
            print(
                f"--- solution {solution_index} ---"
            )
            print(
                str(solution)[:3000]
            )

            if len(str(solution)) > 3000:
                print(
                    "... <truncated>"
                )
    else:
        print(
            str(parsed_solutions)[:3000]
        )


# ======================================================================
# Main analysis
# ======================================================================

def analyze(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:

    problem_lengths: list[int] = []
    test_counts: list[int] = []
    solution_counts: list[int] = []

    tests_parse_methods: Counter[str] = (
        Counter()
    )

    solution_parse_methods: Counter[str] = (
        Counter()
    )

    test_structures: Counter[str] = (
        Counter()
    )

    missing_problem = 0
    missing_tests = 0
    missing_solutions = 0

    tests_parse_fail = 0
    solutions_parse_fail = 0

    no_tests = 0
    no_solutions = 0

    solution_lengths: list[int] = []

    for row in rows:

        # ------------------------------------------------------
        # Problem
        # ------------------------------------------------------

        problem = row.get(
            "problem"
        )

        if (
            not isinstance(
                problem,
                str,
            )
            or not problem.strip()
        ):
            missing_problem += 1
        else:
            problem_lengths.append(
                len(problem)
            )

        # ------------------------------------------------------
        # Tests
        # ------------------------------------------------------

        tests = row.get(
            "tests"
        )

        if tests is None:
            missing_tests += 1

        parsed_tests, tests_method = (
            try_parse_nested_value(
                tests
            )
        )

        tests_parse_methods[
            tests_method
        ] += 1

        if tests_method == "raw_string":
            tests_parse_fail += 1

        structure = infer_tests_structure(
            parsed_tests
        )

        test_structures[
            structure
        ] += 1

        test_count = count_tests(
            parsed_tests
        )

        if test_count is not None:
            test_counts.append(
                test_count
            )

            if test_count == 0:
                no_tests += 1

        # ------------------------------------------------------
        # Solutions
        # ------------------------------------------------------

        solutions = row.get(
            "solutions"
        )

        if solutions is None:
            missing_solutions += 1

        (
            parsed_solutions,
            solution_method,
        ) = try_parse_nested_value(
            solutions
        )

        solution_parse_methods[
            solution_method
        ] += 1

        if solution_method == "raw_string":
            solutions_parse_fail += 1

        solution_count = count_solutions(
            parsed_solutions
        )

        if solution_count is not None:
            solution_counts.append(
                solution_count
            )

            if solution_count == 0:
                no_solutions += 1

        if isinstance(
            parsed_solutions,
            list,
        ):
            for solution in parsed_solutions:
                if isinstance(
                    solution,
                    str,
                ):
                    solution_lengths.append(
                        len(solution)
                    )

        elif isinstance(
            parsed_solutions,
            str,
        ):
            if parsed_solutions.strip():
                solution_lengths.append(
                    len(parsed_solutions)
                )

    return {
        "num_examples": len(rows),

        "missing": {
            "problem": missing_problem,
            "tests": missing_tests,
            "solutions": missing_solutions,
        },

        "tests": {
            "parse_methods": dict(
                tests_parse_methods
            ),
            "parse_failures": tests_parse_fail,
            "structures": dict(
                test_structures
            ),
            "count_distribution": (
                summarize_numeric(
                    test_counts
                )
            ),
            "zero_test_problems": no_tests,
        },

        "solutions": {
            "parse_methods": dict(
                solution_parse_methods
            ),
            "parse_failures": solutions_parse_fail,
            "count_distribution": (
                summarize_numeric(
                    solution_counts
                )
            ),
            "zero_solution_problems": (
                no_solutions
            ),
            "length_distribution": (
                summarize_numeric(
                    solution_lengths
                )
            ),
        },

        "problem_length_chars": (
            summarize_numeric(
                problem_lengths
            )
        ),
    }


# ======================================================================
# Print summary
# ======================================================================

def print_summary(
    summary: dict[str, Any],
) -> None:

    print()
    print("=" * 100)
    print("DeepCoder TACO EDA Summary")
    print("=" * 100)

    print(
        f"examples : "
        f"{summary['num_examples']}"
    )

    print()

    print("[Missing]")
    for key, value in (
        summary["missing"].items()
    ):
        print(
            f"{key:12s}: {value}"
        )

    print()

    print("[Tests - parsing]")
    for key, value in (
        summary["tests"][
            "parse_methods"
        ].items()
    ):
        print(
            f"{key:20s}: {value}"
        )

    print(
        f"parse failures       : "
        f"{summary['tests']['parse_failures']}"
    )

    print()

    print("[Tests - structures]")
    for key, value in (
        summary["tests"][
            "structures"
        ].items()
    ):
        print(
            f"{key:30s}: {value}"
        )

    print()

    print("[Tests - count distribution]")
    for key, value in (
        summary["tests"][
            "count_distribution"
        ].items()
    ):
        print(
            f"{key:10s}: {value}"
        )

    print(
        f"zero-test problems : "
        f"{summary['tests']['zero_test_problems']}"
    )

    print()

    print("[Solutions - parsing]")
    for key, value in (
        summary["solutions"][
            "parse_methods"
        ].items()
    ):
        print(
            f"{key:20s}: {value}"
        )

    print(
        f"parse failures       : "
        f"{summary['solutions']['parse_failures']}"
    )

    print()

    print("[Solutions - count distribution]")
    for key, value in (
        summary["solutions"][
            "count_distribution"
        ].items()
    ):
        print(
            f"{key:10s}: {value}"
        )

    print(
        f"zero-solution problems : "
        f"{summary['solutions']['zero_solution_problems']}"
    )

    print()

    print("[Problem length - chars]")
    for key, value in (
        summary[
            "problem_length_chars"
        ].items()
    ):
        print(
            f"{key:10s}: {value}"
        )

    print()

    print("[Solution length - chars]")
    for key, value in (
        summary["solutions"][
            "length_distribution"
        ].items()
    ):
        print(
            f"{key:10s}: {value}"
        )

    print("=" * 100)


# ======================================================================
# Save summary
# ======================================================================

def save_summary(
    *,
    summary: dict[str, Any],
    output_dir: Path,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "deepcoder_taco_eda_summary.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[Save] summary -> "
        f"{output_path}"
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

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output_dir
    )

    print("=" * 100)
    print("DeepCoder TACO EDA")
    print("=" * 100)

    print(
        f"input      : {input_path}"
    )

    print(
        f"output dir : {output_dir}"
    )

    print()

    rows = load_jsonl(
        input_path
    )

    print(
        f"[Load] examples={len(rows)}"
    )

    summary = analyze(
        rows
    )

    print_summary(
        summary
    )

    save_summary(
        summary=summary,
        output_dir=output_dir,
    )

    # ----------------------------------------------------------
    # Detailed sample inspection
    # ----------------------------------------------------------

    num_samples = min(
        args.num_samples,
        len(rows),
    )

    for index in range(
        num_samples
    ):
        print_sample(
            index=index,
            row=rows[index],
        )


if __name__ == "__main__":
    main()