"""Evaluator test with a real LiveCodeBench problem.
고정된 정답 코드와 오답 코드를 이용해 실제 LCB 문제 1873_A에 테스트함

Usage:
    python -m src.execution.test_evaluator_livecodebench
"""

from __future__ import annotations

from src.datasets.dataset_loader import DatasetLoader
from src.execution.evaluator import Evaluator


CORRECT_CODE = """
import sys


def solve():
    input = sys.stdin.readline

    t = int(input())

    for _ in range(t):
        cards = input().strip()

        mismatch_count = sum(
            cards[index] != "abc"[index]
            for index in range(3)
        )

        if mismatch_count in {0, 2}:
            print("YES")
        else:
            print("NO")


if __name__ == "__main__":
    solve()
""".strip()


WRONG_CODE = """
t = int(input())

for _ in range(t):
    cards = input().strip()

    if cards == "abc":
        print("YES")
    else:
        print("NO")
""".strip()


def main() -> None:
    loader = DatasetLoader(
        dataset_name="livecodebench_v6",
        split="test",
        limit=1,
    )

    example = loader.load()[0]

    assert example.problem_id == "1873_A"

    evaluator = Evaluator(
        timeout_seconds=3.0,
        include_public_tests=True,
        include_private_tests=True,
    )

    print("=" * 80)
    print("Correct Code Evaluation")
    print("=" * 80)

    correct_result = evaluator.evaluate(
        example,
        CORRECT_CODE,
    )

    print(f"Status       : {correct_result.status}")
    print(
        f"Passed tests : "
        f"{correct_result.passed_tests}/"
        f"{correct_result.total_tests}"
    )
    print(
        f"Execution    : "
        f"{correct_result.execution_time:.4f}s"
    )

    assert correct_result.passed
    assert correct_result.status == "PASS"

    print()
    print("=" * 80)
    print("Wrong Code Evaluation")
    print("=" * 80)

    wrong_result = evaluator.evaluate(
        example,
        WRONG_CODE,
    )

    print(f"Status       : {wrong_result.status}")
    print(
        f"Passed tests : "
        f"{wrong_result.passed_tests}/"
        f"{wrong_result.total_tests}"
    )
    print(
        f"First error  : "
        f"{wrong_result.error_message}"
    )

    assert not wrong_result.passed
    assert wrong_result.status == "WRONG_ANSWER"

    print()
    print(
        "[PASS] LiveCodeBench evaluator test passed."
    )


if __name__ == "__main__":
    main()