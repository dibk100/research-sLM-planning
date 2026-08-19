"""
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase3_coverage_analysis/a_planning_coverage/scripts/run_planning_coverage.py \
  --config phase3_coverage_analysis/a_planning_coverage/configs/planning_coverage_qwen25Coder3b.yaml
  
"""
# phase3_coverage_analysis/a_planning_coverage/
# scripts/run_planning_coverage.py

from __future__ import annotations

import argparse
from pathlib import Path

from phase3_coverage_analysis.a_planning_coverage.runner import (
    PlanningCoverageRunner,
)
from phase3_coverage_analysis.a_planning_coverage.strategies.planning_coverage import (
    PlanningCoverageStrategy,
)

from src.datasets.dataset_loader import load_dataset
from src.execution.evaluator import Evaluator
from src.models.generator import ModelGenerator
from src.parsing.code_parser import CodeParser
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 3-A planning coverage experiment."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to the Phase 3-A planning coverage "
            "YAML config."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional override for the number of "
            "benchmark problems."
        ),
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help=(
            "Optional override for the number of "
            "planning candidates N."
        ),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Disable resume from an existing "
            "results.jsonl."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --------------------------------------------------------------
    # 1. Config
    # --------------------------------------------------------------

    config = load_config(
        args.config
    )

    experiment_config = config[
        "experiment"
    ]
    dataset_config = config[
        "dataset"
    ]
    model_config = config[
        "model"
    ]
    sampling_config = config[
        "sampling"
    ]
    generation_config = config[
        "generation"
    ]
    strategy_config = config[
        "strategy"
    ]
    evaluation_config = config[
        "evaluation"
    ]
    output_config = config[
        "output"
    ]

    plan_generation_config = (
        generation_config[
            "plan"
        ]
    )

    code_generation_config = (
        generation_config[
            "code"
        ]
    )

    # --------------------------------------------------------------
    # 2. Seed / limit / N
    # --------------------------------------------------------------

    seed = int(
        experiment_config.get(
            "seed",
            42,
        )
    )

    set_seed(
        seed
    )

    dataset_limit = (
        args.limit
        if args.limit is not None
        else dataset_config.get(
            "limit"
        )
    )

    if dataset_limit is not None:
        dataset_limit = int(
            dataset_limit
        )

        if dataset_limit <= 0:
            raise ValueError(
                "dataset limit must be "
                "greater than 0."
            )

    num_samples = int(
        args.num_samples
        if args.num_samples is not None
        else sampling_config.get(
            "num_samples",
            8,
        )
    )

    if num_samples <= 0:
        raise ValueError(
            "num_samples must be "
            "greater than 0."
        )

    # --------------------------------------------------------------
    # 3. Validate Phase 3-A experimental condition
    # --------------------------------------------------------------

    plan_temperature = float(
        plan_generation_config.get(
            "temperature",
            0.7,
        )
    )

    code_temperature = float(
        code_generation_config.get(
            "temperature",
            0.0,
        )
    )

    if plan_temperature <= 0.0:
        raise ValueError(
            "Phase 3-A requires stochastic "
            "plan generation "
            "(plan temperature > 0)."
        )

    if code_temperature != 0.0:
        raise ValueError(
            "Phase 3-A requires deterministic "
            "code generation "
            "(code temperature = 0.0)."
        )

    # --------------------------------------------------------------
    # 4. Dataset
    # --------------------------------------------------------------

    examples = load_dataset(
        dataset_name=dataset_config[
            "name"
        ],
        data_path=dataset_config[
            "path"
        ],
        limit=dataset_limit,
    )

    if not examples:
        raise ValueError(
            "No benchmark problems were loaded."
        )

    # --------------------------------------------------------------
    # 5. Model
    # --------------------------------------------------------------

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
            False,
        ),
    )

    # --------------------------------------------------------------
    # 6. Strategy
    # --------------------------------------------------------------

    strategy = PlanningCoverageStrategy(
        generator=generator,

        plan_prompt_path=strategy_config[
            "plan_prompt_path"
        ],
        code_prompt_path=strategy_config[
            "code_prompt_path"
        ],

        base_seed=seed,

        plan_max_new_tokens=int(
            plan_generation_config.get(
                "max_new_tokens",
                512,
            )
        ),
        plan_temperature=(
            plan_temperature
        ),
        plan_top_p=float(
            plan_generation_config.get(
                "top_p",
                0.95,
            )
        ),

        code_max_new_tokens=int(
            code_generation_config.get(
                "max_new_tokens",
                1024,
            )
        ),
        code_temperature=(
            code_temperature
        ),
        code_top_p=float(
            code_generation_config.get(
                "top_p",
                1.0,
            )
        ),

        system_prompt=strategy_config.get(
            "system_prompt"
        ),
    )

    # --------------------------------------------------------------
    # 7. Parser / evaluator
    # --------------------------------------------------------------

    parser = CodeParser()

    evaluator = Evaluator(
        timeout_seconds=int(
            evaluation_config.get(
                "timeout_seconds",
                6,
            )
        ),
        debug=bool(
            evaluation_config.get(
                "debug",
                False,
            )
        ),
        include_public_tests=bool(
            evaluation_config.get(
                "include_public_tests",
                True,
            )
        ),
        include_private_tests=bool(
            evaluation_config.get(
                "include_private_tests",
                True,
            )
        ),
    )

    # --------------------------------------------------------------
    # 8. Output / resume
    # --------------------------------------------------------------

    output_path = Path(
        output_config[
            "path"
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resume = (
        not args.no_resume
        and bool(
            output_config.get(
                "resume",
                True,
            )
        )
    )

    store_prompts = bool(
        output_config.get(
            "store_prompts",
            False,
        )
    )

    store_test_results = bool(
        output_config.get(
            "store_test_results",
            False,
        )
    )

    # --------------------------------------------------------------
    # 9. Run configuration
    # --------------------------------------------------------------

    print("=" * 80)
    print(
        "Phase 3-A Planning Coverage Experiment"
    )
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
        f"Problems   : "
        f"{len(examples)}"
    )

    print(
        f"Model      : "
        f"{model_config['name_or_path']}"
    )

    print(
        f"Seed       : "
        f"{seed}"
    )

    print(
        f"N samples  : "
        f"{num_samples}"
    )

    print(
        "Plan gen   : "
        f"temperature="
        f"{plan_temperature}, "
        f"top_p="
        f"{plan_generation_config.get('top_p', 0.95)}, "
        f"max_new_tokens="
        f"{plan_generation_config.get('max_new_tokens', 512)}"
    )

    print(
        "Code gen   : "
        f"temperature="
        f"{code_temperature}, "
        f"top_p="
        f"{code_generation_config.get('top_p', 1.0)}, "
        f"max_new_tokens="
        f"{code_generation_config.get('max_new_tokens', 1024)}"
    )

    print(
        f"Plan prompt: "
        f"{strategy_config['plan_prompt_path']}"
    )

    print(
        f"Code prompt: "
        f"{strategy_config['code_prompt_path']}"
    )

    print(
        f"Store prompts      : "
        f"{store_prompts}"
    )

    print(
        f"Store test results : "
        f"{store_test_results}"
    )

    print(
        f"Output     : "
        f"{output_path}"
    )

    print(
        f"Resume     : "
        f"{resume}"
    )

    print()

    # --------------------------------------------------------------
    # 10. Runner
    # --------------------------------------------------------------

    runner = PlanningCoverageRunner(
        strategy=strategy,
        evaluator=evaluator,
        parser=parser,

        output_path=output_path,

        model_name=model_config[
            "name_or_path"
        ],
        dataset_name=dataset_config[
            "name"
        ],

        seed=seed,
        num_samples=num_samples,

        resume=resume,
        store_prompts=store_prompts,
        store_test_results=(
            store_test_results
        ),
    )

    runner.run(
        examples
    )


if __name__ == "__main__":
    main()