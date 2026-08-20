"""
데이터 연결 후 수정ver

PYTHONPATH="$HOME/workspace/project_sLM_planning" \
python \
  phase4_method_discovery/vanilla_planning_rlvr/scripts/sanity_check_deepcoder_taco.py \
  --num-problems 10 \
  --solution-index 0 \
  --try-solutions 20 \
  --output-jsonl /home/dibaeck/workspace/project_sLM_planning/phase4_method_discovery/archive/deepcoder_taco_multi_solution_sanity.jsonl \
  --show-failures \
  --show-attempts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.datasets.deepcoder_taco import (
    convert_row_to_problem,
    iter_raw_rows,
    parse_tests,
)
from src.execution.taco_evaluator import (
    TACOEvaluator,
)


DEFAULT_INPUT = Path(
    "/mnt/hdd/project_sLM_planning/data/"
    "deepcoder_taco/raw/deepcoder_taco_train.jsonl"
)

DEFAULT_OUTPUT = Path(
    "/home/dibaeck/workspace/project_sLM_planning/"
    "phase4_method_discovery/archive/"
    "deepcoder_taco_multi_solution_sanity.jsonl"
)


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity-check DeepCoder TACO reference solutions "
            "using the production TACOEvaluator."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="DeepCoder TACO raw JSONL path.",
    )

    parser.add_argument(
        "--num-problems",
        type=int,
        default=10,
        help=(
            "Number of stdin problems to inspect. "
            "Functional problems are skipped."
        ),
    )

    parser.add_argument(
        "--solution-index",
        type=int,
        default=0,
        help=(
            "Starting solution index. "
            "Default: 0."
        ),
    )

    parser.add_argument(
        "--try-solutions",
        type=int,
        default=20,
        help=(
            "Maximum number of reference solutions to try "
            "per problem. Evaluation stops at the first PASS."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=6,
        help="Timeout per test used by TACOEvaluator.",
    )

    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=str(DEFAULT_OUTPUT),
    )

    parser.add_argument(
        "--show-failures",
        action="store_true",
        help=(
            "Print detailed information when no tested "
            "reference solution passes."
        ),
    )

    parser.add_argument(
        "--show-attempts",
        action="store_true",
        help=(
            "Print every attempted solution result."
        ),
    )

    return parser.parse_args()


# ======================================================================
# Solution parsing
# ======================================================================

def parse_solutions(
    raw_solutions: Any,
) -> list[str]:
    """
    Normalize the DeepCoder TACO solutions column.

    Current released dataset normally stores this as list[str].
    """

    if isinstance(
        raw_solutions,
        list,
    ):
        raw_items = raw_solutions

    elif isinstance(
        raw_solutions,
        str,
    ):
        text = raw_solutions.strip()

        if not text:
            return []

        try:
            parsed = json.loads(
                text
            )
        except json.JSONDecodeError:
            parsed = text

        if isinstance(
            parsed,
            list,
        ):
            raw_items = parsed
        else:
            raw_items = [
                parsed
            ]

    else:
        raise TypeError(
            "solutions must be list or str, "
            f"got {type(raw_solutions).__name__}"
        )

    solutions: list[str] = []

    for item in raw_items:
        if (
            isinstance(item, str)
            and item.strip()
        ):
            solutions.append(
                item.strip()
            )

    return solutions


# ======================================================================
# Logging
# ======================================================================

def write_jsonl(
    path: Path,
    records: list[
        dict[str, Any]
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            f.write("\n")


# ======================================================================
# Problem evaluation
# ======================================================================

def evaluate_reference_solutions(
    *,
    evaluator: TACOEvaluator,
    problem: Any,
    solutions: list[str],
    start_index: int,
    max_solutions: int,
    show_attempts: bool,
) -> dict[str, Any]:
    """
    Try reference solutions sequentially until one passes.

    This tests the hypothesis:

        solutions[0] is not necessarily the same verified/official
        solution used during DeepCoder dataset curation.
    """

    if start_index >= len(
        solutions
    ):
        raise IndexError(
            f"start solution index "
            f"{start_index} >= "
            f"num solutions {len(solutions)}"
        )

    end_index = min(
        start_index
        + max_solutions,
        len(solutions),
    )

    attempts: list[
        dict[str, Any]
    ] = []

    first_passing_index: (
        int | None
    ) = None

    for solution_index in range(
        start_index,
        end_index,
    ):
        code = solutions[
            solution_index
        ]

        result = evaluator.evaluate(
            problem=problem,
            code=code,
        )

        attempt = {
            "solution_index": (
                solution_index
            ),
            "passed": result.passed,
            "status": result.status,
            "passed_tests": (
                result.passed_tests
            ),
            "total_tests": (
                result.total_tests
            ),
            "error_message": (
                result.error_message
            ),
        }

        attempts.append(
            attempt
        )

        if show_attempts:
            print(
                "    "
                f"solution[{solution_index}] "
                f"status={result.status} "
                f"passed={result.passed} "
                f"tests="
                f"{result.passed_tests}/"
                f"{result.total_tests}"
            )

        if result.passed:
            first_passing_index = (
                solution_index
            )
            break

    has_passing_solution = (
        first_passing_index
        is not None
    )

    return {
        "has_passing_solution": (
            has_passing_solution
        ),
        "first_passing_solution_index": (
            first_passing_index
        ),
        "num_solutions_tried": len(
            attempts
        ),
        "solution_attempts": attempts,
    }


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if args.num_problems <= 0:
        raise ValueError(
            "--num-problems must be > 0."
        )

    if args.solution_index < 0:
        raise ValueError(
            "--solution-index must be >= 0."
        )

    if args.try_solutions <= 0:
        raise ValueError(
            "--try-solutions must be > 0."
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

    # ------------------------------------------------------------------
    # Evaluator
    # ------------------------------------------------------------------

    evaluator = TACOEvaluator(
        timeout_seconds=args.timeout,
        debug=False,
    )

    # ------------------------------------------------------------------
    # Summary counters
    # ------------------------------------------------------------------

    evaluated = 0

    problems_with_pass = 0
    problems_without_pass = 0

    solution0_pass = 0
    later_solution_pass = 0

    functional_skipped = 0
    invalid_rows = 0

    total_solutions_tried = 0

    records: list[
        dict[str, Any]
    ] = []

    print("=" * 90)
    print(
        "DeepCoder TACO "
        "Multi-Solution Production Sanity Check"
    )
    print("=" * 90)

    print(
        f"input            : "
        f"{input_path}"
    )

    print(
        f"target problems  : "
        f"{args.num_problems}"
    )

    print(
        f"start solution   : "
        f"{args.solution_index}"
    )

    print(
        f"max solutions    : "
        f"{args.try_solutions}"
    )

    print(
        f"timeout          : "
        f"{args.timeout}s"
    )

    # ------------------------------------------------------------------
    # Dataset iteration
    # ------------------------------------------------------------------

    for row_index, row in (
        iter_raw_rows(
            input_path
        )
    ):
        # --------------------------------------------------------------
        # Skip functional problems.
        # --------------------------------------------------------------

        try:
            tests = parse_tests(
                row["tests"]
            )

        except Exception as exc:
            invalid_rows += 1

            if args.show_failures:
                print(
                    f"[SKIP] row={row_index} "
                    f"test parsing failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            continue

        if tests.get(
            "fn_name"
        ):
            functional_skipped += 1
            continue

        # --------------------------------------------------------------
        # Convert to ProblemExample.
        # --------------------------------------------------------------

        try:
            problem = (
                convert_row_to_problem(
                    row,
                    row_index=row_index,
                )
            )

            if problem is None:
                functional_skipped += 1
                continue

            solutions = (
                parse_solutions(
                    row["solutions"]
                )
            )

            if not solutions:
                raise ValueError(
                    "No valid reference solutions."
                )

            # ----------------------------------------------------------
            # Evaluate multiple reference solutions.
            # ----------------------------------------------------------

            multi_result = (
                evaluate_reference_solutions(
                    evaluator=evaluator,
                    problem=problem,
                    solutions=solutions,
                    start_index=(
                        args.solution_index
                    ),
                    max_solutions=(
                        args.try_solutions
                    ),
                    show_attempts=(
                        args.show_attempts
                    ),
                )
            )

            attempts = (
                multi_result[
                    "solution_attempts"
                ]
            )

            total_solutions_tried += len(
                attempts
            )

            first_attempt_passed = (
                bool(attempts)
                and bool(
                    attempts[0][
                        "passed"
                    ]
                )
            )

            first_passing_index = (
                multi_result[
                    "first_passing_solution_index"
                ]
            )

            has_passing_solution = (
                multi_result[
                    "has_passing_solution"
                ]
            )

            # ----------------------------------------------------------
            # Classify outcome.
            # ----------------------------------------------------------

            if has_passing_solution:
                problems_with_pass += 1

                if first_attempt_passed:
                    solution0_pass += 1
                else:
                    later_solution_pass += 1

            else:
                problems_without_pass += 1

            record = {
                "row_index": row_index,

                "problem_id": (
                    problem.problem_id
                ),

                "dataset": (
                    problem.dataset
                ),

                "evaluation_type": (
                    problem.evaluation_type
                ),

                "num_tests": len(
                    problem.private_tests
                ),

                "num_solutions": len(
                    solutions
                ),

                "start_solution_index": (
                    args.solution_index
                ),

                "max_solutions_requested": (
                    args.try_solutions
                ),

                "num_solutions_tried": (
                    multi_result[
                        "num_solutions_tried"
                    ]
                ),

                "has_passing_solution": (
                    has_passing_solution
                ),

                "first_passing_solution_index": (
                    first_passing_index
                ),

                "first_attempt_passed": (
                    first_attempt_passed
                ),

                "solution_attempts": (
                    attempts
                ),
            }

        except Exception as exc:
            invalid_rows += 1
            problems_without_pass += 1

            record = {
                "row_index": row_index,

                "problem_id": (
                    f"deepcoder_taco_"
                    f"{row_index:05d}"
                ),

                "has_passing_solution": (
                    False
                ),

                "status": (
                    "SANITY_CHECK_ERROR"
                ),

                "error_message": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            }

        records.append(
            record
        )

        evaluated += 1

        # --------------------------------------------------------------
        # Print compact result.
        # --------------------------------------------------------------
        
        has_pass = record.get(
            "has_passing_solution",
            False,
        )

        first_pass_index = record.get(
            "first_passing_solution_index"
        )

        num_tried = record.get(
            "num_solutions_tried",
            0,
        )

        num_solutions = record.get(
            "num_solutions",
            "?",
        )

        if has_pass:
            print(
                f"[{evaluated:03d}] "
                f"row={row_index} "
                f"PASS "
                f"first_pass={first_pass_index} "
                f"tried={num_tried}/{num_solutions}"
            )

        else:
            print(
                f"[{evaluated:03d}] "
                f"row={row_index} "
                f"NO_PASS "
                f"tried={num_tried}/{num_solutions}"
            )

            if args.show_failures:
                print(
                    json.dumps(
                        record,
                        indent=2,
                        ensure_ascii=False,
                    )
                )

        if evaluated >= args.num_problems:
            break

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    write_jsonl(
        output_path,
        records,
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print()
    print("=" * 90)
    print("Summary")
    print("=" * 90)

    print(
        f"evaluated problems     : "
        f"{evaluated}"
    )

    print(
        f"problems with PASS     : "
        f"{problems_with_pass}"
    )

    print(
        f"problems without PASS  : "
        f"{problems_without_pass}"
    )

    if evaluated:
        print(
            f"problem pass coverage  : "
            f"{problems_with_pass / evaluated:.3%}"
        )

    print()

    print(
        f"first solution PASS    : "
        f"{solution0_pass}"
    )

    print(
        f"later solution PASS    : "
        f"{later_solution_pass}"
    )

    print(
        f"total solutions tried  : "
        f"{total_solutions_tried}"
    )

    print(
        f"functional skipped     : "
        f"{functional_skipped}"
    )

    print(
        f"invalid/error rows     : "
        f"{invalid_rows}"
    )

    print(
        f"results                : "
        f"{output_path}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()