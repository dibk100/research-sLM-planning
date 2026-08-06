"""
Direct Code Generation 통합 실행 스크립트

usage: python scripts/run_direct.py --config configs/direct.yaml


- create_benchmark_evaluator()는 선택한 데이터셋에 맞춰 별도로 연결해야함.
"""

import argparse
import random

import numpy as np
import torch

from src.datasets.dataset_loader import DatasetLoader
from src.execution.code_extractor import CodeExtractor
from src.execution.evaluator import Evaluator
from src.models.generator import ModelGenerator
from src.schemas import GenerationRecord
from src.strategies.direct import DirectStrategy
from src.utils.jsonl_logger import JSONLLogger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    loader = DatasetLoader(
        dataset_name=args.dataset,
        split=args.split,
        limit=args.limit,
    )
    examples = loader.load()

    generator = ModelGenerator(
        model_name_or_path=args.model,
        dtype=args.dtype,
    )

    strategy = DirectStrategy(
        generator=generator,
        prompt_path=args.prompt_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    extractor = CodeExtractor()

    # 선택한 벤치마크의 evaluator 구현체를 연결해야 한다.
    evaluator = Evaluator(
        benchmark_evaluator=create_benchmark_evaluator(
            args.dataset
        )
    )

    logger = JSONLLogger(args.output_path)
    completed_ids = logger.completed_problem_ids()

    for index, example in enumerate(examples, start=1):
        if example.problem_id in completed_ids:
            continue

        print(
            f"[{index}/{len(examples)}] "
            f"{example.problem_id}"
        )

        prompt, generation = strategy.run(example)

        try:
            code = extractor.extract(generation.text)
            evaluation = evaluator.evaluate(
                example=example,
                code=code,
            )
            error_message = evaluation.stderr or None

        except ValueError as error:
            code = ""
            evaluation = None
            error_message = str(error)

        record = GenerationRecord(
            problem_id=example.problem_id,
            dataset=args.dataset,
            strategy="direct",
            model_name=args.model,
            seed=args.seed,
            prompt=prompt,
            raw_output=generation.text,
            code=code,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            generation_time=generation.generation_time,
            passed=(
                evaluation.passed
                if evaluation is not None
                else False
            ),
            status=(
                evaluation.status
                if evaluation is not None
                else "EXTRACTION_ERROR"
            ),
            passed_tests=(
                evaluation.passed_tests
                if evaluation is not None
                else 0
            ),
            total_tests=(
                evaluation.total_tests
                if evaluation is not None
                else 0
            ),
            execution_time=(
                evaluation.execution_time
                if evaluation is not None
                else 0.0
            ),
            difficulty=example.difficulty,
            error_message=error_message,
        )

        logger.append(record.to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="humaneval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--prompt-path",
        default="prompts/direct.txt",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/direct/results.jsonl",
    )

    parser.add_argument(
        "--dtype",
        default="bfloat16",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    main(parser.parse_args())