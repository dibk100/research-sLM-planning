"""ModelGenerator smoke test.

Usage:
    python -m src.models.test_generator \
        --model Qwen/Qwen2.5-Coder-3B-Instruct
"""

from __future__ import annotations

import argparse

from src.datasets.dataset_loader import DatasetLoader
from src.models.generator import ModelGenerator


SYSTEM_PROMPT = (
    "You are an expert competitive programming assistant. "
    "Generate correct and efficient Python solutions."
)


def build_direct_prompt(
    title: str,
    problem: str,
    starter_code: str,
) -> str:
    starter_section = ""

    if starter_code.strip():
        starter_section = (
            "\n\nStarter Code:\n"
            f"{starter_code}"
        )

    return f"""
Solve the following competitive programming problem in Python.

Requirements:
- Return only the final Python code.
- Do not include explanations.
- Do not include Markdown code fences.
- Read input from standard input.
- Write output to standard output.
- Use an efficient algorithm.

Problem Title:
{title}

Problem:
{problem}
{starter_section}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test ModelGenerator with one LiveCodeBench problem."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model name or local model path.",
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
        default=1024,
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

    prompt = build_direct_prompt(
        title=example.title,
        problem=example.prompt,
        starter_code=example.starter_code,
    )

    print("=" * 80)
    print("Model Generator Smoke Test")
    print("=" * 80)
    print(f"Problem ID : {example.problem_id}")
    print(f"Title      : {example.title}")
    print(f"Difficulty : {example.difficulty}")
    print(f"Model      : {args.model}")

    generator = ModelGenerator(
        model_name_or_path=args.model,
        dtype=args.dtype,
    )

    result = generator.generate(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )

    print()
    print("=" * 80)
    print("Generation Statistics")
    print("=" * 80)
    print(f"Prompt tokens     : {result.prompt_tokens}")
    print(f"Completion tokens : {result.completion_tokens}")
    print(f"Generation time   : {result.generation_time:.2f}s")

    print()
    print("=" * 80)
    print("Raw Model Output")
    print("=" * 80)
    print(result.text)

    assert result.text.strip(), (
        "Generated output is empty."
    )

    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0
    assert result.generation_time > 0

    print()
    print("[PASS] ModelGenerator smoke test passed.")


if __name__ == "__main__":
    main()