"""Evaluator 결과를 JSONL에 저장하는 통합 테스트.

Usage:
    python -m src.utils.test_logger_with_evaluator
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.datasets.dataset_loader import DatasetLoader
from src.execution.evaluator import Evaluator
from src.schemas import StrategyOutput
from src.utils.jsonl_logger import JSONLLogger
from src.utils.record_builder import (
    build_experiment_record,
)


CORRECT_CODE = """
import sys


def solve():
    input = sys.stdin.readline

    t = int(input())

    for _ in range(t):
        cards = input().strip()
        mismatches = sum(
            cards[index] != "abc"[index]
            for index in range(3)
        )

        print("YES" if mismatches in {0, 2} else "NO")


if __name__ == "__main__":
    solve()
""".strip()


def main() -> None:
    loader = DatasetLoader(
        dataset_name="livecodebench_v6",
        split="test",
        limit=1,
    )
    example = loader.load()[0]

    evaluator = Evaluator(
        timeout_seconds=3.0,
        include_public_tests=True,
        include_private_tests=True,
    )

    evaluation = evaluator.evaluate(
        example,
        CORRECT_CODE,
    )

    strategy_output = StrategyOutput(
        problem_id=example.problem_id,
        strategy="direct",
        formatted_prompt="test prompt",
        raw_output=f"```python\n{CORRECT_CODE}\n```",
        prompt_tokens=100,
        completion_tokens=80,
        generation_time=1.5,
    )

    record = build_experiment_record(
        example=example,
        strategy_output=strategy_output,
        extracted_code=CORRECT_CODE,
        evaluation=evaluation,
        dataset_name="livecodebench_v6",
        model_name="test-model",
        seed=42,
    )

    with tempfile.TemporaryDirectory(
        prefix="phase1_logger_integration_"
    ) as temp_dir:
        output_path = (
            Path(temp_dir)
            / "direct_results.jsonl"
        )

        logger = JSONLLogger(output_path)
        logger.append(record.to_dict())

        loaded_records = logger.load_records()

        assert len(loaded_records) == 1

        loaded = loaded_records[0]

        assert loaded["problem_id"] == "1873_A"
        assert loaded["strategy"] == "direct"
        assert loaded["passed"] is True
        assert loaded["status"] == "PASS"
        assert loaded["passed_tests"] == 5
        assert loaded["total_tests"] == 5
        assert len(loaded["test_results"]) == 5

        assert logger.completed_ids() == {
            "1873_A",
        }

        print("=" * 80)
        print("Saved Experiment Record")
        print("=" * 80)
        print(f"Output path   : {output_path}")
        print(f"Problem ID    : {loaded['problem_id']}")
        print(f"Status        : {loaded['status']}")
        print(
            f"Passed tests  : "
            f"{loaded['passed_tests']}/"
            f"{loaded['total_tests']}"
        )
        print(
            f"Completed IDs : "
            f"{logger.completed_ids()}"
        )

    print()
    print(
        "[PASS] Evaluator + JSONLLogger "
        "integration test passed."
    )


if __name__ == "__main__":
    main()