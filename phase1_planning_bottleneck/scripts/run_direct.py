"""
Direct Code Generation 통합 실행 스크립트

usage: 

python scripts/run_direct.py --config configs/direct.yaml

python -m scripts.run_direct \
  --config configs/direct.yaml

python -m scripts.run_direct \
  --config configs/direct.yaml \
  --limit 3


저장 결과 확인 : wc -l outputs/direct/results.jsonl
head -n 1 outputs/direct/results.jsonl | python -m json.tool

"""
from __future__ import annotations

import argparse
import traceback
from dataclasses import asdict
from pathlib import Path

from src.datasets.dataset_loader import DatasetLoader
from src.execution.code_extractor import (
    CodeExtractionError,
    CodeExtractor,
)
from src.execution.evaluator import Evaluator
from src.models.generator import ModelGenerator
from src.schemas import (
    EvaluationResult,
    ExperimentRecord,
)
from src.strategies.direct import DirectStrategy
from src.utils.config import load_config
from src.utils.jsonl_logger import JSONLLogger
from src.utils.record_builder import (
    build_experiment_record,
)
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Direct code generation on "
            "LiveCodeBench v6."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Override dataset limit from config."
        ),
    )

    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Override output JSONL path."
        ),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Disable resume and run all selected problems."
        ),
    )

    return parser.parse_args()


def build_extraction_failure_evaluation(
    error: Exception,
) -> EvaluationResult:
    return EvaluationResult(
        passed=False,
        status="EXTRACTION_ERROR",
        passed_tests=0,
        total_tests=0,
        execution_time=0.0,
        test_results=[],
        error_message=str(error),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiment_config = config["experiment"]
    dataset_config = config["dataset"]
    model_config = config["model"]
    generation_config = config["generation"]
    strategy_config = config["strategy"]
    evaluation_config = config["evaluation"]
    output_config = config["output"]

    seed = int(experiment_config["seed"])
    set_seed(seed)

    dataset_limit = (
        args.limit
        if args.limit is not None
        else dataset_config.get("limit")
    )

    output_path = Path(
        args.output_path
        if args.output_path is not None
        else output_config["path"]
    )

    resume = (
        bool(output_config.get("resume", True))
        and not args.no_resume
    )

    print("=" * 80)
    print("Phase1 Direct Experiment")
    print("=" * 80)
    print(
        f"Experiment : "
        f"{experiment_config['name']}"
    )
    print(
        f"Dataset    : "
        f"{dataset_config['name']}"
    )
    print(
        f"Model      : "
        f"{model_config['name_or_path']}"
    )
    print(f"Seed       : {seed}")
    print(f"Limit      : {dataset_limit}")
    print(f"Output     : {output_path}")
    print(f"Resume     : {resume}")
    print()

    loader = DatasetLoader(
        dataset_name=dataset_config["name"],
        split=dataset_config["split"],
        limit=dataset_limit,
    )

    examples = loader.load()

    generator = ModelGenerator(
        model_name_or_path=model_config[
            "name_or_path"
        ],
        dtype=model_config.get(
            "dtype",
            "bfloat16",
        ),
        device_map=model_config.get(
            "device_map",
            "auto",
        ),
        trust_remote_code=model_config.get(
            "trust_remote_code",
            True,
        ),
    )

    strategy = DirectStrategy(
        generator=generator,
        prompt_path=strategy_config[
            "prompt_path"
        ],
        system_prompt=strategy_config.get(
            "system_prompt"
        ),
        max_new_tokens=generation_config[
            "max_new_tokens"
        ],
        temperature=generation_config[
            "temperature"
        ],
        top_p=generation_config[
            "top_p"
        ],
    )

    extractor = CodeExtractor()

    evaluator = Evaluator(
        timeout_seconds=evaluation_config[
            "timeout_seconds"
        ],
        include_public_tests=evaluation_config[
            "include_public_tests"
        ],
        include_private_tests=evaluation_config[
            "include_private_tests"
        ],
    )

    logger = JSONLLogger(output_path)

    completed_ids = (
        logger.completed_ids()
        if resume
        else set()
    )

    if completed_ids:
        print(
            f"[Resume] Loaded "
            f"{len(completed_ids)} completed problems."
        )

    total_selected = len(examples)
    processed_count = 0
    skipped_count = 0
    pass_count = 0

    for index, example in enumerate(
        examples,
        start=1,
    ):
        if example.problem_id in completed_ids:
            skipped_count += 1
            print(
                f"[{index}/{total_selected}] "
                f"[SKIP] {example.problem_id}"
            )
            continue

        print()
        print("-" * 80)
        print(
            f"[{index}/{total_selected}] "
            f"{example.problem_id} | "
            f"{example.difficulty} | "
            f"{example.title}"
        )
        print("-" * 80)

        try:
            strategy_output = strategy.run(
                example
            )

            try:
                extracted_code = extractor.extract(
                    strategy_output.raw_output
                )

                evaluation = evaluator.evaluate(
                    example=example,
                    code=extracted_code,
                )

            except CodeExtractionError as error:
                extracted_code = ""
                evaluation = (
                    build_extraction_failure_evaluation(
                        error
                    )
                )

            record = build_experiment_record(
                example=example,
                strategy_output=strategy_output,
                extracted_code=extracted_code,
                evaluation=evaluation,
                dataset_name=dataset_config[
                    "name"
                ],
                model_name=model_config[
                    "name_or_path"
                ],
                seed=seed,
            )

            logger.append(record.to_dict())

            processed_count += 1

            if evaluation.passed:
                pass_count += 1

            print(
                f"Status     : "
                f"{evaluation.status}"
            )
            print(
                f"Tests      : "
                f"{evaluation.passed_tests}/"
                f"{evaluation.total_tests}"
            )
            print(
                f"Gen tokens : "
                f"{strategy_output.completion_tokens}"
            )
            print(
                f"Gen time   : "
                f"{strategy_output.generation_time:.2f}s"
            )
            print(
                f"Exec time  : "
                f"{evaluation.execution_time:.4f}s"
            )

            if evaluation.error_message:
                print(
                    f"Error      : "
                    f"{evaluation.error_message[:500]}"
                )

        except Exception as error:
            print(
                f"[ERROR] {example.problem_id}: "
                f"{error}"
            )
            traceback.print_exc()
            raise

    print()
    print("=" * 80)
    print("Experiment Summary")
    print("=" * 80)
    print(f"Selected problems : {total_selected}")
    print(f"Processed         : {processed_count}")
    print(f"Skipped           : {skipped_count}")
    print(f"Passed            : {pass_count}")

    if processed_count > 0:
        pass_rate = (
            pass_count / processed_count
        )
        print(
            f"Current pass rate : "
            f"{pass_rate:.4f}"
        )

    print(f"Output            : {output_path}")
    print()
    print("[DONE] Direct experiment completed.")


if __name__ == "__main__":
    main()