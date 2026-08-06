"""DirectStrategy smoke test.

Usage:
    python -m src.strategies.test_direct \
        --model Qwen/Qwen2.5-Coder-3B-Instruct
"""

from __future__ import annotations

import argparse

from src.datasets.dataset_loader import DatasetLoader
from src.models.generator import ModelGenerator
from src.strategies.direct import DirectStrategy


SYSTEM_PROMPT = (
    "You are an expert competitive programming assistant. "
    "Generate correct and efficient Python solutions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DirectStrategy smoke test."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model ID or local model path.",
    )

    parser.add_argument(
        "--prompt-path",
        default="prompts/direct.txt",
    )

    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=[
            "float16",
            "bfloat16",
            "float32",
        ],
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--problem-index",
        type=int,
        default=0,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader = DatasetLoader(
        dataset_name="livecodebench_v6",
        split="test",
        limit=args.problem_index + 1,
    )

    examples = loader.load()
    example = examples[args.problem_index]

    generator = ModelGenerator(
        model_name_or_path=args.model,
        dtype=args.dtype,
    )

    strategy = DirectStrategy(
        generator=generator,
        prompt_path=args.prompt_path,
        system_prompt=SYSTEM_PROMPT,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    prompt = strategy.build_prompt(example)

    print("=" * 80)
    print("DirectStrategy Smoke Test")
    print("=" * 80)
    print(f"Problem ID : {example.problem_id}")
    print(f"Title      : {example.title}")
    print(f"Difficulty : {example.difficulty}")
    print(f"Strategy   : {strategy.name}")

    print()
    print("=" * 80)
    print("Formatted Prompt")
    print("=" * 80)
    print(prompt)

    result = strategy.run(example)

    print()
    print("=" * 80)
    print("Generation Statistics")
    print("=" * 80)
    print(f"Strategy          : {result.strategy}")
    print(f"Prompt tokens     : {result.prompt_tokens}")
    print(f"Completion tokens : {result.completion_tokens}")
    print(f"Generation time   : {result.generation_time:.2f}s")

    print()
    print("=" * 80)
    print("Raw Model Output")
    print("=" * 80)
    print(result.raw_output)

    assert result.problem_id == example.problem_id
    assert result.strategy == "direct"
    assert result.prompt.strip()
    assert result.raw_output.strip()
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0
    assert result.generation_time > 0

    assert example.title in result.prompt
    assert example.prompt in result.prompt

    print()
    print("[PASS] DirectStrategy smoke test passed.")


if __name__ == "__main__":
    main()