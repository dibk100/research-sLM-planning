"""
Self-Planning Code Generation 통합 실행 스크립트.

Usage:

python -m scripts.run_self_plan \
  --config configs/self_plan.yaml

python -m scripts.run_self_plan \
  --config configs/self_plan.yaml \
  --limit 10

저장 결과 확인:

wc -l outputs/self_plan_stdin/results.jsonl

head -n 1 outputs/self_plan_stdin/results.jsonl \
  | python -m json.tool
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from src.datasets.dataset_loader import DatasetLoader
from src.execution.code_extractor import (
    CodeExtractionError,
    CodeExtractor,
)
from src.execution.evaluator import Evaluator
from src.models.generator import ModelGenerator
from src.schemas import EvaluationResult
from src.strategies.self_plan import SelfPlanningStrategy
from src.utils.config import load_config
from src.utils.jsonl_logger import JSONLLogger
from src.utils.record_builder import build_experiment_record
from src.utils.run_metadata import (
    save_run_config,
    save_run_metadata,
)
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Self-Planning code generation "
            "on LiveCodeBench v6."
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
        help="Override dataset limit from config.",
    )

    parser.add_argument(
        "--output-path",
        default=None,
        help="Override output JSONL path.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume.",
    )

    return parser.parse_args()


def build_failure_evaluation(
    *,
    status: str,
    error: Exception,
) -> EvaluationResult:
    """평가 불가능한 문제를 실패 결과로 변환한다."""
    return EvaluationResult(
        passed=False,
        status=status,
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

    # CLI override를 실제 실행 설정에 반영한다.
    dataset_config["limit"] = dataset_limit
    output_config["path"] = str(output_path)
    output_config["resume"] = resume

    output_dir = output_path.parent
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_config_path = save_run_config(
        config=config,
        output_dir=output_dir,
        overwrite=False,
    )

    run_metadata_path = save_run_metadata(
        config=config,
        output_dir=output_dir,
        overwrite=False,
    )

    print("=" * 80)
    print("Phase1 Self-Planning Experiment")
    print("=" * 80)
    print(f"Experiment : {experiment_config['name']}")
    print(f"Dataset    : {dataset_config['name']}")
    print(f"Model      : {model_config['name_or_path']}")
    print(f"Seed       : {seed}")
    print(f"Limit      : {dataset_limit}")
    print(f"Output     : {output_path}")
    print(f"Run config : {run_config_path}")
    print(f"Metadata   : {run_metadata_path}")
    print(f"Resume     : {resume}")
    print(
        f"Test type  : "
        f"{dataset_config.get('test_type', 'stdin')}"
    )
    print()

    loader = DatasetLoader(
        dataset_name=dataset_config["name"],
        split=dataset_config["split"],
        limit=dataset_limit,
        test_type=dataset_config.get(
            "test_type",
            "stdin",
        ),
        release_version=dataset_config.get(
            "release_version",
            "release_v6",
        ),
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

    strategy = SelfPlanningStrategy(
        generator=generator,
        plan_prompt_path=strategy_config[
            "plan_prompt_path"
        ],
        code_prompt_path=strategy_config[
            "code_prompt_path"
        ],
        system_prompt=strategy_config.get(
            "system_prompt"
        ),
        plan_max_new_tokens=generation_config[
            "plan_max_new_tokens"
        ],
        code_max_new_tokens=generation_config[
            "code_max_new_tokens"
        ],
        temperature=generation_config.get(
            "temperature",
            0.0,
        ),
        top_p=generation_config.get(
            "top_p",
            1.0,
        ),
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
            strategy_output = strategy.run(example)

            try:
                extracted_code = extractor.extract(
                    strategy_output.raw_output
                )

            except CodeExtractionError as error:
                extracted_code = ""

                evaluation = build_failure_evaluation(
                    status="EXTRACTION_ERROR",
                    error=error,
                )

            else:
                try:
                    evaluation = evaluator.evaluate(
                        example=example,
                        code=extracted_code,
                    )

                except ValueError as error:
                    if (
                        "Unsupported test type"
                        in str(error)
                    ):
                        evaluation = (
                            build_failure_evaluation(
                                status=(
                                    "UNSUPPORTED_TEST_TYPE"
                                ),
                                error=error,
                            )
                        )
                    else:
                        evaluation = (
                            build_failure_evaluation(
                                status="EVALUATION_ERROR",
                                error=error,
                            )
                        )

                except Exception as error:
                    evaluation = (
                        build_failure_evaluation(
                            status="EVALUATION_ERROR",
                            error=error,
                        )
                    )

            record = build_experiment_record(
                example=example,
                strategy_output=strategy_output,
                extracted_code=extracted_code,
                evaluation=evaluation,
                dataset_name=dataset_config["name"],
                model_name=model_config[
                    "name_or_path"
                ],
                seed=seed,
            )

            logger.append(record.to_dict())

            processed_count += 1

            if evaluation.passed:
                pass_count += 1

            plan_step = next(
                (
                    step
                    for step
                    in strategy_output.strategy_trace
                    if step.name == "plan_generation"
                ),
                None,
            )

            code_step = next(
                (
                    step
                    for step
                    in strategy_output.strategy_trace
                    if step.name == "code_generation"
                ),
                None,
            )

            print(f"Status     : {evaluation.status}")
            print(
                f"Tests      : "
                f"{evaluation.passed_tests}/"
                f"{evaluation.total_tests}"
            )

            if plan_step is not None:
                print(
                    f"Plan tokens: "
                    f"{plan_step.completion_tokens}"
                )
                print(
                    f"Plan time  : "
                    f"{plan_step.generation_time:.2f}s"
                )

            if code_step is not None:
                print(
                    f"Code tokens: "
                    f"{code_step.completion_tokens}"
                )
                print(
                    f"Code time  : "
                    f"{code_step.generation_time:.2f}s"
                )

            print(
                f"Total tokens: "
                f"{strategy_output.completion_tokens}"
            )
            print(
                f"Total time  : "
                f"{strategy_output.generation_time:.2f}s"
            )
            print(
                f"Exec time   : "
                f"{evaluation.execution_time:.4f}s"
            )

            if evaluation.error_message:
                print(
                    f"Error       : "
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
    print(
        "[DONE] Self-Planning experiment completed."
    )


if __name__ == "__main__":
    main()