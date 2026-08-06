"""TeacherPlanStrategy smoke test.

Usage:
    python -m src.strategies.test_teacher_plan \
        --model Qwen/Qwen2.5-Coder-3B-Instruct
"""

from __future__ import annotations

import argparse

from src.datasets.dataset_loader import DatasetLoader
from src.models.generator import ModelGenerator
from src.plans.teacher_plan_store import (
    TeacherPlanStore,
)
from src.strategies.teacher_plan import (
    TeacherPlanStrategy,
)

from src.execution.code_extractor import (
    CodeExtractionError,
    CodeExtractor,
)
from src.execution.evaluator import Evaluator


SYSTEM_PROMPT = (
    "You are an expert competitive programming "
    "assistant. Implement the provided expert plan "
    "correctly and produce an efficient Python solution."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TeacherPlanStrategy smoke test."
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--problem-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--code-max-new-tokens",
        type=int,
        default=1024,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader = DatasetLoader(
        dataset_name="livecodebench_v6",
        split="test",
        limit=args.problem_index + 1,
        test_type="stdin",
        release_version="release_v6",
    )

    example = loader.load()[args.problem_index]

    plan_store = TeacherPlanStore(
        plan_path=(
            "data/teacher_plans/"
            "livecodebench_v6_teacher_plans.jsonl"
        ),
        require_verified=True,
    )

    generator = ModelGenerator(
        model_name_or_path=args.model,
        dtype="bfloat16",
    )

    strategy = TeacherPlanStrategy(
        generator=generator,
        plan_store=plan_store,
        code_prompt_path=(
            "prompts/teacher_plan_code.txt"
        ),
        system_prompt=SYSTEM_PROMPT,
        code_max_new_tokens=(
            args.code_max_new_tokens
        ),
        temperature=0.0,
        top_p=1.0,
    )

    result = strategy.run(example)

    extractor = CodeExtractor()

    evaluator = Evaluator(
        timeout_seconds=5.0,
        include_public_tests=True,
        include_private_tests=True,
    )

    try:
        extracted_code = extractor.extract(
            result.raw_output
        )
    except CodeExtractionError as error:
        raise AssertionError(
            f"Code extraction failed: {error}"
        ) from error

    evaluation = evaluator.evaluate(
        example=example,
        code=extracted_code,
    )

    print("=" * 80)
    print("TeacherPlanStrategy Smoke Test")
    print("=" * 80)
    print(f"Problem ID : {example.problem_id}")
    print(f"Title      : {example.title}")
    print(f"Difficulty : {example.difficulty}")
    print(f"Strategy   : {result.strategy}")

    print()
    print("=" * 80)
    print("Teacher Plan")
    print("=" * 80)
    print(result.teacher_plan)

    print()
    print("=" * 80)
    print("Formatted Prompt")
    print("=" * 80)
    print(result.formatted_prompt)

    print()
    print("=" * 80)
    print("Raw Code Output")
    print("=" * 80)
    print(result.raw_output)

    print()
    print("=" * 80)
    print("Extracted Code")
    print("=" * 80)
    print(extracted_code)

    print()
    print("=" * 80)
    print("Evaluation Result")
    print("=" * 80)
    print(f"Passed       : {evaluation.passed}")
    print(f"Status       : {evaluation.status}")
    print(
        f"Passed tests : "
        f"{evaluation.passed_tests}/"
        f"{evaluation.total_tests}"
    )
    print(
        f"Execution    : "
        f"{evaluation.execution_time:.4f}s"
    )

    if evaluation.error_message:
        print(
            f"Error        : "
            f"{evaluation.error_message}"
        )

    print()
    print("=" * 80)
    print("Generation Statistics")
    print("=" * 80)
    print(
        f"Prompt tokens     : "
        f"{result.prompt_tokens}"
    )
    print(
        f"Completion tokens : "
        f"{result.completion_tokens}"
    )
    print(
        f"Generation time   : "
        f"{result.generation_time:.2f}s"
    )

    assert result.strategy == "teacher_plan"
    assert result.teacher_plan
    assert result.teacher_plan_verified is True
    assert len(result.strategy_trace) == 1
    assert (
        result.strategy_trace[0].name
        == "code_generation"
    )
    assert result.raw_output.strip()
    assert extracted_code.strip()

    assert evaluation.passed is True
    assert evaluation.status == "PASS"
    assert (
        evaluation.passed_tests
        == evaluation.total_tests
    )

    print()
    print(
        "[PASS] TeacherPlanStrategy + "
        "CodeExtractor + Evaluator "
        "integration test passed."
    )

if __name__ == "__main__":
    main()