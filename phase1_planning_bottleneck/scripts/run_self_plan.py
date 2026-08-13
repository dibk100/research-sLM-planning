# phase1_planning_bottleneck/scripts/run_self_plan.py
"""
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase1_planning_bottleneck/scripts/run_self_plan.py \
  --config phase1_planning_bottleneck/configs/self_plan_sanity.yaml \
  --limit 1 \
  --no-resume

"""
from __future__ import annotations

import argparse
from pathlib import Path

from phase1_planning_bottleneck.runner import Phase1Runner
from phase1_planning_bottleneck.strategies.self_plan import (
    SelfPlanningStrategy,
)

from src.datasets.dataset_loader import load_dataset
from src.execution.evaluator import Evaluator
from src.models.generator import ModelGenerator
from src.parsing.code_parser import CodeParser
from src.utils.config import load_config
from src.utils.run_metadata import (
    save_run_config,
    save_run_metadata,
)
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 Self-Planning code generation."
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

    # ------------------------------------------------------------------
    # Seed / CLI overrides
    # ------------------------------------------------------------------

    seed = int(
        experiment_config["seed"]
    )
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
        bool(
            output_config.get(
                "resume",
                True,
            )
        )
        and not args.no_resume
    )

    dataset_config["limit"] = dataset_limit
    output_config["path"] = str(
        output_path
    )
    output_config["resume"] = resume

    # ------------------------------------------------------------------
    # Run metadata
    # ------------------------------------------------------------------

    output_dir = output_path.parent

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

    # ------------------------------------------------------------------
    # Experiment header
    # ------------------------------------------------------------------

    print("=" * 80)
    print("Phase 1 Self-Planning Experiment")
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
        f"Data path  : "
        f"{dataset_config['path']}"
    )
    print(
        f"Model      : "
        f"{model_config['name_or_path']}"
    )
    print(f"Seed       : {seed}")
    print(f"Limit      : {dataset_limit}")
    print(f"Output     : {output_path}")
    print(f"Run config : {run_config_path}")
    print(f"Metadata   : {run_metadata_path}")
    print(f"Resume     : {resume}")
    print()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    examples = load_dataset(
        dataset_name=dataset_config["name"],
        data_path=dataset_config["path"],
        limit=dataset_limit,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    generator = ModelGenerator(
        model_name_or_path=(
            model_config["name_or_path"]
        ),
        dtype=model_config.get(
            "dtype",
            "bfloat16",
        ),
        device_map=model_config.get(
            "device_map",
            "auto",
        ),
        trust_remote_code=(
            model_config.get(
                "trust_remote_code",
                True,
            )
        ),
    )

    # ------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------

    strategy = SelfPlanningStrategy(
        generator=generator,
        plan_prompt_path=(
            strategy_config[
                "plan_prompt_path"
            ]
        ),
        code_prompt_path=(
            strategy_config[
                "code_prompt_path"
            ]
        ),
        system_prompt=(
            strategy_config.get(
                "system_prompt"
            )
        ),
        plan_max_new_tokens=(
            generation_config.get(
                "plan_max_new_tokens",
                512,
            )
        ),
        code_max_new_tokens=(
            generation_config.get(
                "code_max_new_tokens",
                1024,
            )
        ),
        temperature=(
            generation_config.get(
                "temperature",
                0.0,
            )
        ),
        top_p=(
            generation_config.get(
                "top_p",
                1.0,
            )
        ),
    )

    # ------------------------------------------------------------------
    # Parser
    # ------------------------------------------------------------------

    parser = CodeParser()

    # ------------------------------------------------------------------
    # Evaluator
    # ------------------------------------------------------------------
    evaluator = Evaluator(
        timeout_seconds=(
            evaluation_config.get(
                "timeout_seconds",
                6,
            )
        ),
        include_public_tests=(
            evaluation_config.get(
                "include_public_tests",
                True,
            )
        ),
        include_private_tests=(
            evaluation_config.get(
                "include_private_tests",
                True,
            )
        ),
        debug=(
            evaluation_config.get(
                "debug",
                False,
            )
        ),
    )
    # ------------------------------------------------------------------
    # Shared Phase 1 Runner
    # ------------------------------------------------------------------

    runner = Phase1Runner(
        strategy=strategy,
        evaluator=evaluator,
        parser=parser,
        output_path=output_path,
        model_name=(
            model_config["name_or_path"]
        ),
        seed=seed,
        resume=resume,
    )

    runner.run(examples)

    print()
    print(
        "[DONE] Self-Planning experiment completed."
    )


if __name__ == "__main__":
    main()