"""SelfPlanningStrategy smoke test.

Usage:
    python -m src.strategies.test_self_plan \
        --model Qwen/Qwen2.5-Coder-3B-Instruct
"""

from __future__ import annotations

import argparse

from src.datasets.dataset_loader import DatasetLoader
from src.models.generator import ModelGenerator
from src.strategies.self_plan import (
    SelfPlanningStrategy,
)


SYSTEM_PROMPT = (
    "You are an expert competitive programming assistant. "
    "Analyze problems carefully and generate correct "
    "and efficient Python solutions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SelfPlanningStrategy smoke test."
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
        "--plan-max-new-tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--code-max-new-tokens",
        type=int,
        default=512,
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

    generator = ModelGenerator(
        model_name_or_path=args.model,
        dtype="bfloat16",
    )

    strategy = SelfPlanningStrategy(
        generator=generator,
        plan_prompt_path="prompts/self_plan_plan.txt",
        code_prompt_path="prompts/self_plan_code.txt",
        system_prompt=SYSTEM_PROMPT,
        plan_max_new_tokens=args.plan_max_new_tokens,
        code_max_new_tokens=args.code_max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    result = strategy.run(example)

    print("=" * 80)
    print("SelfPlanningStrategy Smoke Test")
    print("=" * 80)
    print(f"Problem ID : {example.problem_id}")
    print(f"Title      : {example.title}")
    print(f"Difficulty : {example.difficulty}")
    print(f"Strategy   : {result.strategy}")

    print()
    print("=" * 80)
    print("Generated Plan")
    print("=" * 80)
    print(result.strategy_trace[0].raw_output)

    print()
    print("=" * 80)
    print("Code Generation Prompt")
    print("=" * 80)
    print(result.formatted_prompt)

    print()
    print("=" * 80)
    print("Raw Code Output")
    print("=" * 80)
    print(result.raw_output)

    print()
    print("=" * 80)
    print("Generation Statistics")
    print("=" * 80)
    print(f"Total prompt tokens     : {result.prompt_tokens}")
    print(
        f"Total completion tokens : "
        f"{result.completion_tokens}"
    )
    print(
        f"Total generation time   : "
        f"{result.generation_time:.2f}s"
    )

    assert result.strategy == "self_plan"
    assert len(result.strategy_trace) == 2
    assert (
        result.strategy_trace[0].name
        == "plan_generation"
    )
    assert (
        result.strategy_trace[1].name
        == "code_generation"
    )
    assert result.strategy_trace[0].raw_output.strip()
    assert result.raw_output.strip()

    print()
    print(
        "[PASS] SelfPlanningStrategy "
        "smoke test passed."
    )


if __name__ == "__main__":
    main()