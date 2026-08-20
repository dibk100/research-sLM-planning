"""
taco 데이터셋 스키마 확인하기

PYTHONPATH=. python \
  phase4_method_discovery/vanilla_planning_rlvr/scripts/sanity_check_deepcoder_taco.py \
  --evaluation-type stdin \
  --num-problems 10 \
  --num-tests 5 \
  --output-jsonl /home/dibaeck/workspace/project_sLM_planning/phase4_method_discovery/archive/deepcoder_taco_eda_sanity_check_results.jsonl \
  --show-failures

==========================================================================================
DeepCoder TACO Sanity Check
==========================================================================================
input            : /mnt/hdd/project_sLM_planning/data/deepcoder_taco/raw/deepcoder_taco_train.jsonl
evaluation type  : stdin
target problems  : 10
tests/problem    : 5
solution index   : 0
timeout/test     : 6.0s

[001] row=0 type=stdin status=PASS passed=True

[002] row=1 type=stdin status=PASS passed=True

[003] row=3 type=stdin status=PASS passed=True

[004] row=4 type=stdin status=PASS passed=True

[005] row=5 type=stdin status=PASS passed=True

[006] row=7 type=stdin status=PASS passed=True

[007] row=10 type=stdin status=PASS passed=True

[008] row=11 type=stdin status=PASS passed=True

[009] row=12 type=stdin status=PASS passed=True

[010] row=13 type=stdin status=PASS passed=True

==========================================================================================
Sanity Check Summary
==========================================================================================
evaluated       : 10
passed problems : 10
failed problems : 0
pass rate       : 100.000%
stdin           : 10
functional      : 0
results         : /home/dibaeck/workspace/project_sLM_planning/phase4_method_discovery/archive/deepcoder_taco_eda_sanity_check_results.jsonl
==========================================================================================
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
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
            "Sanity-check DeepCoder TACO reference solutions "
            "against released tests."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
    )

    parser.add_argument(
        "--evaluation-type",
        choices=[
            "stdin",
            "functional",
            "both",
        ],
        default="stdin",
    )

    parser.add_argument(
        "--num-problems",
        type=int,
        default=10,
        help="Number of compatible problems to evaluate.",
    )

    parser.add_argument(
        "--num-tests",
        type=int,
        default=5,
        help=(
            "Maximum number of tests per problem. "
            "Use 0 to run all tests."
        ),
    )

    parser.add_argument(
        "--solution-index",
        type=int,
        default=0,
        help=(
            "Reference solution index to test. "
            "Default: solutions[0]."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=6.0,
        help="Timeout per test case in seconds.",
    )

    parser.add_argument(
        "--show-failures",
        action="store_true",
    )

    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=(
            "/mnt/hdd/project_sLM_planning/data/"
            "deepcoder_taco/eda/"
            "sanity_check_results.jsonl"
        ),
    )

    return parser.parse_args()


# ======================================================================
# Loading / parsing
# ======================================================================

def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Input not found: {path}"
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
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}: {exc}"
                ) from exc

            rows.append(row)

    return rows


def parse_tests(
    raw_tests: Any,
) -> dict[str, Any]:
    if isinstance(raw_tests, dict):
        parsed = raw_tests

    elif isinstance(raw_tests, str):
        text = raw_tests.strip()

        try:
            parsed = json.loads(text)

        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)

    else:
        raise TypeError(
            "tests must be dict or str, "
            f"got {type(raw_tests).__name__}"
        )

    if not isinstance(parsed, dict):
        raise TypeError(
            "Parsed tests must be dict."
        )

    inputs = parsed.get("inputs")
    outputs = parsed.get("outputs")

    if not isinstance(inputs, list):
        raise ValueError(
            "tests['inputs'] must be list."
        )

    if not isinstance(outputs, list):
        raise ValueError(
            "tests['outputs'] must be list."
        )

    if len(inputs) != len(outputs):
        raise ValueError(
            "inputs/outputs length mismatch: "
            f"{len(inputs)} vs {len(outputs)}"
        )

    return parsed


def parse_solutions(
    raw_solutions: Any,
) -> list[str]:
    if isinstance(raw_solutions, list):
        solutions = raw_solutions

    elif isinstance(raw_solutions, str):
        text = raw_solutions.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = text

        if isinstance(parsed, list):
            solutions = parsed
        else:
            solutions = [parsed]

    else:
        raise TypeError(
            "solutions must be list or str."
        )

    normalized: list[str] = []

    for solution in solutions:
        if isinstance(solution, str) and solution.strip():
            normalized.append(
                solution.strip()
            )

    return normalized


def infer_evaluation_type(
    tests: dict[str, Any],
) -> str:
    if tests.get("fn_name"):
        return "functional"

    return "stdin"


# ======================================================================
# Output normalization
# ======================================================================

def normalize_stdout(
    value: str,
) -> str:
    """
    Conservative stdout normalization.

    Normalize line endings and trailing whitespace, but do not
    perform aggressive semantic normalization.
    """

    lines = (
        value
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .split("\n")
    )

    normalized = [
        line.rstrip()
        for line in lines
    ]

    while (
        normalized
        and normalized[-1] == ""
    ):
        normalized.pop()

    return "\n".join(normalized)


# ======================================================================
# stdin evaluator
# ======================================================================

def run_stdin_test(
    *,
    code: str,
    test_input: Any,
    expected_output: Any,
    timeout: float,
) -> dict[str, Any]:
    input_text = str(test_input)

    expected_text = str(
        expected_output
    )

    start = time.perf_counter()

    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
            ],
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "status": "TIMEOUT",
            "actual_output": "",
            "expected_output": expected_text,
            "stderr": "",
            "execution_time": (
                time.perf_counter()
                - start
            ),
        }

    execution_time = (
        time.perf_counter()
        - start
    )

    if completed.returncode != 0:
        return {
            "passed": False,
            "status": "RUNTIME_ERROR",
            "actual_output": completed.stdout,
            "expected_output": expected_text,
            "stderr": completed.stderr,
            "execution_time": execution_time,
        }

    actual_normalized = normalize_stdout(
        completed.stdout
    )

    expected_normalized = normalize_stdout(
        expected_text
    )

    passed = (
        actual_normalized
        == expected_normalized
    )

    return {
        "passed": passed,
        "status": (
            "PASS"
            if passed
            else "WRONG_ANSWER"
        ),
        "actual_output": (
            completed.stdout
        ),
        "expected_output": (
            expected_text
        ),
        "stderr": completed.stderr,
        "execution_time": execution_time,
    }


# ======================================================================
# functional evaluator
# ======================================================================

def python_literal(
    value: Any,
) -> str:
    """
    Return Python source representation of a test value.
    """

    return repr(value)


def build_functional_runner(
    *,
    code: str,
    function_name: str,
    test_input: Any,
) -> str:
    """
    Build a standalone script that executes the submitted solution.

    TACO functional inputs may represent:
    - one positional argument
    - multiple positional arguments
    """

    serialized_input = python_literal(
        test_input
    )

    runner = f"""
import json

{code}

_test_input = {serialized_input}

if isinstance(_test_input, list):
    _result = {function_name}(*_test_input)
else:
    _result = {function_name}(_test_input)

print(json.dumps(_result, ensure_ascii=False))
"""

    return textwrap.dedent(
        runner
    )


def parse_expected_functional(
    value: Any,
) -> Any:
    """
    Normalize released functional expected output.
    """

    if not isinstance(value, str):
        return value

    text = value.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def run_functional_test(
    *,
    code: str,
    function_name: str,
    test_input: Any,
    expected_output: Any,
    timeout: float,
) -> dict[str, Any]:

    runner = build_functional_runner(
        code=code,
        function_name=function_name,
        test_input=test_input,
    )

    start = time.perf_counter()

    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                runner,
            ],
            text=True,
            capture_output=True,
            timeout=timeout,
        )

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "status": "TIMEOUT",
            "actual_output": "",
            "expected_output": expected_output,
            "stderr": "",
            "execution_time": (
                time.perf_counter()
                - start
            ),
        }

    execution_time = (
        time.perf_counter()
        - start
    )

    if completed.returncode != 0:
        return {
            "passed": False,
            "status": "RUNTIME_ERROR",
            "actual_output": completed.stdout,
            "expected_output": expected_output,
            "stderr": completed.stderr,
            "execution_time": execution_time,
        }

    stdout = completed.stdout.strip()

    try:
        actual_value = json.loads(
            stdout
        )

    except Exception:
        actual_value = stdout

    expected_value = (
        parse_expected_functional(
            expected_output
        )
    )

    passed = (
        actual_value
        == expected_value
    )

    return {
        "passed": passed,
        "status": (
            "PASS"
            if passed
            else "WRONG_ANSWER"
        ),
        "actual_output": actual_value,
        "expected_output": expected_value,
        "stderr": completed.stderr,
        "execution_time": execution_time,
    }


# ======================================================================
# Problem evaluation
# ======================================================================

def evaluate_problem(
    *,
    row_index: int,
    row: dict[str, Any],
    solution_index: int,
    max_tests: int,
    timeout: float,
) -> dict[str, Any]:

    tests = parse_tests(
        row["tests"]
    )

    solutions = parse_solutions(
        row["solutions"]
    )

    evaluation_type = (
        infer_evaluation_type(
            tests
        )
    )

    if solution_index >= len(
        solutions
    ):
        raise IndexError(
            f"solution_index={solution_index} "
            f"but only {len(solutions)} solutions."
        )

    solution = solutions[
        solution_index
    ]

    inputs = tests["inputs"]
    outputs = tests["outputs"]

    if max_tests > 0:
        inputs = inputs[:max_tests]
        outputs = outputs[:max_tests]

    test_results: list[
        dict[str, Any]
    ] = []

    function_name = tests.get(
        "fn_name"
    )

    for test_index, (
        test_input,
        expected_output,
    ) in enumerate(
        zip(
            inputs,
            outputs,
        )
    ):

        if evaluation_type == "stdin":
            result = run_stdin_test(
                code=solution,
                test_input=test_input,
                expected_output=expected_output,
                timeout=timeout,
            )

        else:
            if not function_name:
                raise ValueError(
                    "functional test missing fn_name."
                )

            result = (
                run_functional_test(
                    code=solution,
                    function_name=function_name,
                    test_input=test_input,
                    expected_output=expected_output,
                    timeout=timeout,
                )
            )

        result["test_index"] = (
            test_index
        )

        test_results.append(
            result
        )

        # Fail-fast for sanity checking.
        if not result["passed"]:
            break

    passed_tests = sum(
        1
        for result in test_results
        if result["passed"]
    )

    total_selected = len(inputs)

    passed = (
        len(test_results)
        == total_selected
        and passed_tests
        == total_selected
    )

    status = (
        "PASS"
        if passed
        else test_results[-1]["status"]
        if test_results
        else "NO_TESTS"
    )

    return {
        "row_index": row_index,
        "evaluation_type": (
            evaluation_type
        ),
        "problem_chars": len(
            str(row["problem"])
        ),
        "num_solutions": len(
            solutions
        ),
        "available_tests": len(
            tests["inputs"]
        ),
        "selected_tests": (
            total_selected
        ),
        "solution_index": (
            solution_index
        ),
        "passed": passed,
        "status": status,
        "passed_tests": passed_tests,
        "executed_tests": len(
            test_results
        ),
        "test_results": (
            test_results
        ),
    }


# ======================================================================
# Logging
# ======================================================================

def append_jsonl(
    path: Path,
    payload: dict[str, Any],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                payload,
                ensure_ascii=False,
            )
        )
        f.write("\n")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()

    if args.num_problems <= 0:
        raise ValueError(
            "--num-problems must be > 0."
        )

    if args.num_tests < 0:
        raise ValueError(
            "--num-tests must be >= 0."
        )

    if args.timeout <= 0:
        raise ValueError(
            "--timeout must be > 0."
        )

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output_jsonl
    )

    rows = load_jsonl(
        input_path
    )

    print("=" * 90)
    print("DeepCoder TACO Sanity Check")
    print("=" * 90)

    print(
        f"input            : {input_path}"
    )
    print(
        f"evaluation type  : "
        f"{args.evaluation_type}"
    )
    print(
        f"target problems  : "
        f"{args.num_problems}"
    )
    print(
        f"tests/problem    : "
        f"{args.num_tests or 'ALL'}"
    )
    print(
        f"solution index   : "
        f"{args.solution_index}"
    )
    print(
        f"timeout/test     : "
        f"{args.timeout}s"
    )

    # Clear previous result for a clean sanity run.
    if output_path.exists():
        output_path.unlink()

    evaluated = 0
    passed_problems = 0
    failed_problems = 0

    type_counter = {
        "stdin": 0,
        "functional": 0,
    }

    for row_index, row in enumerate(
        rows
    ):
        tests = parse_tests(
            row["tests"]
        )

        evaluation_type = (
            infer_evaluation_type(
                tests
            )
        )

        if (
            args.evaluation_type
            != "both"
            and evaluation_type
            != args.evaluation_type
        ):
            continue

        try:
            result = (
                evaluate_problem(
                    row_index=row_index,
                    row=row,
                    solution_index=(
                        args.solution_index
                    ),
                    max_tests=(
                        args.num_tests
                    ),
                    timeout=args.timeout,
                )
            )

        except Exception as exc:
            result = {
                "row_index": row_index,
                "evaluation_type": (
                    evaluation_type
                ),
                "passed": False,
                "status": (
                    "SANITY_CHECK_ERROR"
                ),
                "error": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            }

        append_jsonl(
            output_path,
            result,
        )

        evaluated += 1

        type_counter[
            evaluation_type
        ] += 1

        if result["passed"]:
            passed_problems += 1
        else:
            failed_problems += 1

        print()
        print(
            f"[{evaluated:03d}] "
            f"row={row_index} "
            f"type={evaluation_type} "
            f"status={result['status']} "
            f"passed={result['passed']}"
        )

        if (
            not result["passed"]
            and args.show_failures
        ):
            print(
                json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False,
                )[:6000]
            )

        if evaluated >= (
            args.num_problems
        ):
            break

    print()
    print("=" * 90)
    print("Sanity Check Summary")
    print("=" * 90)

    print(
        f"evaluated       : {evaluated}"
    )
    print(
        f"passed problems : "
        f"{passed_problems}"
    )
    print(
        f"failed problems : "
        f"{failed_problems}"
    )

    if evaluated:
        print(
            f"pass rate       : "
            f"{passed_problems / evaluated:.3%}"
        )

    print(
        f"stdin           : "
        f"{type_counter['stdin']}"
    )
    print(
        f"functional      : "
        f"{type_counter['functional']}"
    )
    print(
        f"results         : "
        f"{output_path}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()