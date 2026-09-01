"""
PYTHONPATH="$(pwd)" python \
phase4_method_discovery/tpr_planning_rlvr/reward/test_non_fail_fast_equivalence.py
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any

from src.execution.taco_evaluator import TACOEvaluator
from src.schemas import ProblemExample


TIMEOUT_SECONDS = 6


@dataclass
class ReferenceResult:
    passed_tests: int
    total_tests: int
    test_pass_ratio: float
    per_test_passed: list[bool]
    elapsed_seconds: float


@dataclass
class OptimizedResult:
    passed_tests: int
    total_tests: int
    test_pass_ratio: float
    per_test_passed: list[bool]
    elapsed_seconds: float


# =====================================================================
# Test problem
# =====================================================================


def build_test_problem() -> ProblemExample:
    """
    Build a deterministic stdin problem with multiple independent tests.

    Task:
        Read one integer n and print n * 2.

    The test set is deliberately chosen so that intentionally incorrect
    programs can produce mixtures of PASS and FAIL.
    """

    private_tests = [
        {
            "input": "1\n",
            "output": "2\n",
        },
        {
            "input": "2\n",
            "output": "4\n",
        },
        {
            "input": "3\n",
            "output": "6\n",
        },
        {
            "input": "4\n",
            "output": "8\n",
        },
        {
            "input": "5\n",
            "output": "10\n",
        },
        {
            "input": "10\n",
            "output": "20\n",
        },
    ]

    return ProblemExample(
        problem_id="tpr_non_fail_fast_equivalence",
        title="TPR Non-Fail-Fast Equivalence Test",
        problem=(
            "Read an integer n and print n * 2."
        ),
        dataset="deepcoder_taco",
        evaluation_type="stdin",
        private_tests=private_tests,
    )


# =====================================================================
# Reference evaluator
# =====================================================================


def evaluate_reference(
    *,
    problem: ProblemExample,
    code: str,
) -> ReferenceResult:
    """
    Reference TPR semantics.

    Evaluate every selected test independently using the original
    fail-fast TACOEvaluator.

    This reproduces the current slow planning_tpr_reward.py behavior:
        N tests -> N independent evaluator subprocesses.
    """

    evaluator = TACOEvaluator(
        timeout_seconds=TIMEOUT_SECONDS,
        debug=False,
    )

    per_test_passed: list[bool] = []

    start = time.perf_counter()

    for test_case in problem.private_tests:
        one_test_problem = copy.deepcopy(
            problem
        )

        one_test_problem.private_tests = [
            copy.deepcopy(test_case)
        ]

        result = evaluator.evaluate(
            problem=one_test_problem,
            code=code,
        )

        passed = bool(
            result.passed
            and result.passed_tests == 1
        )

        per_test_passed.append(
            passed
        )

    elapsed = (
        time.perf_counter() - start
    )

    passed_tests = sum(
        1
        for passed in per_test_passed
        if passed
    )

    total_tests = len(
        per_test_passed
    )

    test_pass_ratio = (
        passed_tests / total_tests
        if total_tests > 0
        else 0.0
    )

    return ReferenceResult(
        passed_tests=passed_tests,
        total_tests=total_tests,
        test_pass_ratio=test_pass_ratio,
        per_test_passed=per_test_passed,
        elapsed_seconds=elapsed,
    )


# =====================================================================
# Optimized evaluator
# =====================================================================


def evaluate_optimized(
    *,
    problem: ProblemExample,
    code: str,
) -> OptimizedResult:
    """
    Optimized TPR semantics.

    Evaluate all selected tests using one spawned non-fail-fast
    evaluator process.
    """

    evaluator = TACOEvaluator(
        timeout_seconds=TIMEOUT_SECONDS,
        debug=False,
    )

    start = time.perf_counter()

    result = evaluator.evaluate_non_fail_fast(
        problem=problem,
        code=code,
    )

    elapsed = (
        time.perf_counter() - start
    )

    per_test_passed = [
        bool(test_result.passed)
        for test_result in result.test_results
    ]

    passed_tests = int(
        result.passed_tests
    )

    total_tests = int(
        result.total_tests
    )

    test_pass_ratio = (
        passed_tests / total_tests
        if total_tests > 0
        else 0.0
    )

    return OptimizedResult(
        passed_tests=passed_tests,
        total_tests=total_tests,
        test_pass_ratio=test_pass_ratio,
        per_test_passed=per_test_passed,
        elapsed_seconds=elapsed,
    )


# =====================================================================
# Comparison
# =====================================================================


def assert_equivalent(
    *,
    case_name: str,
    reference: ReferenceResult,
    optimized: OptimizedResult,
) -> None:
    """
    Assert exact reward-semantic equivalence.
    """

    if (
        reference.total_tests
        != optimized.total_tests
    ):
        raise AssertionError(
            f"[{case_name}] total_tests mismatch: "
            f"reference={reference.total_tests}, "
            f"optimized={optimized.total_tests}"
        )

    if (
        reference.passed_tests
        != optimized.passed_tests
    ):
        raise AssertionError(
            f"[{case_name}] passed_tests mismatch: "
            f"reference={reference.passed_tests}, "
            f"optimized={optimized.passed_tests}"
        )

    if not math.isclose(
        reference.test_pass_ratio,
        optimized.test_pass_ratio,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError(
            f"[{case_name}] TPR mismatch: "
            f"reference={reference.test_pass_ratio}, "
            f"optimized={optimized.test_pass_ratio}"
        )

    if (
        reference.per_test_passed
        != optimized.per_test_passed
    ):
        raise AssertionError(
            f"[{case_name}] per-test mismatch:\n"
            f"reference={reference.per_test_passed}\n"
            f"optimized={optimized.per_test_passed}"
        )


# =====================================================================
# Test cases
# =====================================================================


TEST_CASES: list[
    tuple[str, str]
] = [
    (
        "all_pass",
        """
n = int(input())
print(n * 2)
""".strip(),
    ),

    (
        "all_fail",
        """
n = int(input())
print(n * 3)
""".strip(),
    ),

    (
        "partial_pass",
        """
n = int(input())

if n <= 3:
    print(n * 2)
else:
    print(n * 3)
""".strip(),
    ),

    (
        "runtime_error",
        """
n = int(input())

if n == 3:
    raise RuntimeError("intentional")

print(n * 2)
""".strip(),
    ),

    (
        "syntax_error",
        """
n = int(input(
print(n * 2)
""".strip(),
    ),
]


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    problem = build_test_problem()

    print(
        "=" * 80
    )
    print(
        "TPR non-fail-fast backend equivalence test"
    )
    print(
        "=" * 80
    )

    all_passed = True

    total_reference_time = 0.0
    total_optimized_time = 0.0

    for case_name, code in TEST_CASES:
        print()
        print(
            f"[CASE] {case_name}"
        )

        reference = evaluate_reference(
            problem=problem,
            code=code,
        )

        optimized = evaluate_optimized(
            problem=problem,
            code=code,
        )

        total_reference_time += (
            reference.elapsed_seconds
        )

        total_optimized_time += (
            optimized.elapsed_seconds
        )

        print(
            "  reference : "
            f"{reference.passed_tests}/"
            f"{reference.total_tests} "
            f"TPR={reference.test_pass_ratio:.6f} "
            f"per_test={reference.per_test_passed} "
            f"time={reference.elapsed_seconds:.3f}s"
        )

        print(
            "  optimized : "
            f"{optimized.passed_tests}/"
            f"{optimized.total_tests} "
            f"TPR={optimized.test_pass_ratio:.6f} "
            f"per_test={optimized.per_test_passed} "
            f"time={optimized.elapsed_seconds:.3f}s"
        )

        try:
            assert_equivalent(
                case_name=case_name,
                reference=reference,
                optimized=optimized,
            )

        except AssertionError as exc:
            all_passed = False

            print(
                "  RESULT    : FAIL"
            )

            print(
                f"  ERROR     : {exc}"
            )

        else:
            print(
                "  RESULT    : PASS"
            )

    print()
    print(
        "=" * 80
    )

    print(
        "Timing summary"
    )

    print(
        "=" * 80
    )

    print(
        f"reference total : "
        f"{total_reference_time:.3f}s"
    )

    print(
        f"optimized total : "
        f"{total_optimized_time:.3f}s"
    )

    if total_optimized_time > 0:
        speedup = (
            total_reference_time
            / total_optimized_time
        )

        print(
            f"speedup         : "
            f"{speedup:.2f}x"
        )

    print()

    if not all_passed:
        raise SystemExit(
            "Equivalence test FAILED."
        )

    print(
        "All equivalence tests PASSED."
    )


if __name__ == "__main__":
    main()