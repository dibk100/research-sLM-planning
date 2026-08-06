"""DirectStrategy와 CodeExtractor 통합 smoke test.

Usage:
    python -m src.execution.test_extractor_with_generator \
        --model Qwen/Qwen2.5-Coder-3B-Instruct
"""

from __future__ import annotations

import argparse

from src.datasets.dataset_loader import DatasetLoader
from src.execution.code_extractor import CodeExtractor
from src.models.generator import ModelGenerator
from src.strategies.direct import DirectStrategy


SYSTEM_PROMPT = (
    "You are an expert competitive programming assistant. "
    "Generate correct and efficient Python solutions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--prompt-path",
        default="prompts/direct.txt",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader = DatasetLoader(
        dataset_name="livecodebench_v6",
        split="test",
        limit=1,
    )

    example = loader.load()[0]

    generator = ModelGenerator(
        model_name_or_path=args.model,
        dtype="bfloat16",
    )

    strategy = DirectStrategy(
        generator=generator,
        prompt_path=args.prompt_path,
        system_prompt=SYSTEM_PROMPT,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )

    strategy_output = strategy.run(example)

    extractor = CodeExtractor()
    code = extractor.extract(
        strategy_output.raw_output
    )

    print("=" * 80)
    print("Raw Output")
    print("=" * 80)
    print(strategy_output.raw_output)

    print()
    print("=" * 80)
    print("Extracted Code")
    print("=" * 80)
    print(code)

    assert code.strip()
    assert "```" not in code

    print()
    print(
        "[PASS] DirectStrategy + CodeExtractor "
        "integration test passed."
    )


if __name__ == "__main__":
    main()