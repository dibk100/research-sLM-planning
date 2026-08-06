"""Evaluator unit tests.

Usage:
    python -m src.execution.test_evaluator
"""

from __future__ import annotations

from src.execution.evaluator import Evaluator
from src.schemas import ProblemExample


def make_example() -> ProblemExample:
    return ProblemExample(
        problem_id="test_double",
        title="Double Number",
        prompt="Read an integer and print twice its value.",
        platform="unit_test",
        contest_id="unit_test",
        contest_date="2026-08-06",
        difficulty="easy",
        starter_code="",
        public_tests=[
            {
                "input": "2\n",
                "output": "4\n",
                "testtype": "stdin",
            }
        ],
        private_tests=[
            {
                "input": "10\n",
                "output": "20\n",
                "testtype": "stdin",
            },
            {
                "input": "-3\n",
                "output": "-6\n",
                "testtype": "stdin",
            },
        ],
        metadata={},
    )


def test_correct_code(
    evaluator: Evaluator,
) -> None:
    code = """
value = int(input())
print(value * 2)
""".strip()

    result = evaluator.evaluate(
        make_example(),
        code,
    )

    assert result.passed
    assert result.status == "PASS"
    assert result.passed_tests == 3
    assert result.total_tests == 3


def test_wrong_answer(
    evaluator: Evaluator,
) -> None:
    code = """
value = int(input())
print(value + 2)
""".strip()

    result = evaluator.evaluate(
        make_example(),
        code,
    )

    assert not result.passed
    assert result.status == "WRONG_ANSWER"
    assert result.passed_tests < result.total_tests


def test_runtime_error(
    evaluator: Evaluator,
) -> None:
    code = """
value = int(input())
raise RuntimeError("intentional error")
""".strip()

    result = evaluator.evaluate(
        make_example(),
        code,
    )

    assert not result.passed
    assert result.status == "RUNTIME_ERROR"
    assert "RuntimeError" in (
        result.error_message or ""
    )


def test_syntax_error(
    evaluator: Evaluator,
) -> None:
    code = """
def solve(
    print("broken")
""".strip()

    result = evaluator.evaluate(
        make_example(),
        code,
    )

    assert not result.passed
    assert result.status == "SYNTAX_ERROR"
    assert result.total_tests == 0


def test_timeout() -> None:
    evaluator = Evaluator(
        timeout_seconds=0.2,
    )

    code = """
while True:
    pass
""".strip()

    result = evaluator.evaluate(
        make_example(),
        code,
    )

    assert not result.passed
    assert result.status == "TIMEOUT"


def test_whitespace_normalization(
    evaluator: Evaluator,
) -> None:
    example = make_example()

    example.public_tests = [
        {
            "input": "2\n",
            "output": "4\n\n",
            "testtype": "stdin",
        }
    ]
    example.private_tests = []

    code = """
value = int(input())
print("  ", value * 2, "  ")
""".strip()

    result = evaluator.evaluate(
        example,
        code,
    )

    assert result.passed


def test_empty_code(
    evaluator: Evaluator,
) -> None:
    result = evaluator.evaluate(
        make_example(),
        "   ",
    )

    assert not result.passed
    assert result.status == "EMPTY_CODE"


def main() -> None:
    evaluator = Evaluator(
        timeout_seconds=1.0,
    )

    tests = [
        (
            "correct code",
            lambda: test_correct_code(evaluator),
        ),
        (
            "wrong answer",
            lambda: test_wrong_answer(evaluator),
        ),
        (
            "runtime error",
            lambda: test_runtime_error(evaluator),
        ),
        (
            "syntax error",
            lambda: test_syntax_error(evaluator),
        ),
        (
            "timeout",
            test_timeout,
        ),
        (
            "whitespace normalization",
            lambda: test_whitespace_normalization(
                evaluator
            ),
        ),
        (
            "empty code",
            lambda: test_empty_code(evaluator),
        ),
    ]

    print("=" * 80)
    print("Evaluator Test")
    print("=" * 80)

    for name, test_function in tests:
        test_function()
        print(f"[PASS] {name}")

    print()
    print("[PASS] All Evaluator tests passed.")


if __name__ == "__main__":
    main()