"""
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase3_coverage_analysis/b_code_coverage/scripts/run_code_coverage.py \
  --config phase3_coverage_analysis/b_code_coverage/configs/qwen25Coder3b.yaml
"""
# phase3_coverage_analysis/b_code_coverage/
# scripts/run_code_coverage.py

from __future__ import annotations

import argparse
from pathlib import Path

from phase3_coverage_analysis.b_code_coverage.fixed_plan_loader import (
    FixedPlanLoader,
)
from phase3_coverage_analysis.b_code_coverage.runner import (
    CodeCoverageRunner,
)
from phase3_coverage_analysis.b_code_coverage.strategies.code_coverage import (
    CodeCoverageStrategy,
)

from src.datasets.dataset_loader import (
    load_dataset,
)
from src.execution.evaluator import (
    Evaluator,
)
from src.models.generator import (
    ModelGenerator,
)
from src.parsing.code_parser import (
    CodeParser,
)
from src.utils.config import (
    load_config,
)
from src.utils.seed import (
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 3-B code coverage "
            "experiment using a fixed Phase 1 "
            "Self-Plan and stochastic code sampling."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to the Phase 3-B "
            "code-coverage YAML config."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of benchmark "
            "problems to process."
        ),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Disable resume from existing "
            "results.jsonl."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ----------------------------------------------------------
    # 1. Load config
    # ----------------------------------------------------------

    config = load_config(
        args.config
    )

    experiment_config = config[
        "experiment"
    ]
    
    sampling_config = config[
        "sampling"
    ]
    
    generation_config = config[
            "generation"
    ]

    dataset_config = config[
        "dataset"
    ]

    model_config = config[
        "model"
    ]

    phase1_config = config[
        "phase1"
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
    
    # plan_generation_config = (
    #     generation_config[
    #         "plan"
    #     ]
    # )

    code_generation_config = (
        generation_config[
            "code"
        ]
    )

    # ----------------------------------------------------------
    # 2. Seed / limit
    # ----------------------------------------------------------

    seed = int(
        experiment_config.get(
            "seed",
            42,
        )
    )

    set_seed(
        seed
    )

    limit = (
        args.limit
        if args.limit is not None
        else experiment_config.get(
            "limit"
        )
    )

    if limit is not None:
        limit = int(
            limit
        )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

    # ----------------------------------------------------------
    # 3. Number of candidates
    # ----------------------------------------------------------

    num_samples = int(
        sampling_config.get(
            "num_samples",
            16,
        )
    )

    if num_samples <= 0:
        raise ValueError(
            "num_samples must be greater than 0."
        )

    # ----------------------------------------------------------
    # 4. Load canonical benchmark
    # ----------------------------------------------------------

    dataset_limit = (
        limit
        if limit is not None
        else dataset_config.get(
            "limit"
        )
    )

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

    # ----------------------------------------------------------
    # 5. Load fixed Phase 1 Self-Plans
    # ----------------------------------------------------------

    phase1_result_path = Path(
        phase1_config[
            "self_plan_result_path"
        ]
    )

    fixed_plan_loader = FixedPlanLoader(
        results_path=phase1_result_path,
    )

    # ----------------------------------------------------------
    # 6. Build model
    # ----------------------------------------------------------

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
    
    

    # ----------------------------------------------------------
    # 7. Build Code Coverage strategy
    # ----------------------------------------------------------

    strategy = CodeCoverageStrategy(
        generator=generator,

        code_prompt_path=strategy_config[
            "code_prompt_path"
        ],

        base_seed=seed,

        code_max_new_tokens=int(
            code_generation_config.get(
                "max_new_tokens",
                1024,
            )
        ),

        code_temperature=float(
            code_generation_config.get(
                "temperature",
                0.7,
            )
        ),

        code_top_p=float(
            code_generation_config.get(
                "top_p",
                0.95,
            )
        ),

        system_prompt=strategy_config.get(
            "system_prompt"
        ),
    )

    # ----------------------------------------------------------
    # 8. Parser
    # ----------------------------------------------------------

    parser = CodeParser()

    # ----------------------------------------------------------
    # 9. Evaluator
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # 10. Output / resume
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # 11. Print experiment configuration
    # ----------------------------------------------------------

    print("=" * 80)
    print(
        "Phase 3-B Code Coverage Experiment"
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
        "Code gen   : "
        f"temperature="
        f"{strategy.code_temperature}, "
        f"top_p="
        f"{strategy.code_top_p}, "
        f"max_new_tokens="
        f"{strategy.code_max_new_tokens}"
    )

    print(
        f"Fixed plan : "
        f"{phase1_result_path}"
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

    # ----------------------------------------------------------
    # 12. Build runner
    # ----------------------------------------------------------

    runner = CodeCoverageRunner(
        strategy=strategy,

        fixed_plan_loader=(
            fixed_plan_loader
        ),

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

        store_prompts=(
            store_prompts
        ),

        store_test_results=(
            store_test_results
        ),
    )

    # ----------------------------------------------------------
    # 13. Run
    # ----------------------------------------------------------

    runner.run(
        examples=examples
    )


if __name__ == "__main__":
    main()