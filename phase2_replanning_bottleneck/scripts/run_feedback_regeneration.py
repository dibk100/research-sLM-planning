# phase2_replanning_bottleneck/scripts/run_feedback_regeneration.py

from __future__ import annotations

import argparse
from pathlib import Path

from phase2_replanning_bottleneck.runner import Phase2Runner
from phase2_replanning_bottleneck.strategies.feedback_regeneration import (
    FeedbackRegenerationStrategy,
)

from src.datasets.dataset_loader import load_dataset
from src.datasets.phase1_failure_loader import (
    load_phase1_failures,
)
from src.execution.evaluator import Evaluator
from src.models.generator import ModelGenerator
from src.parsing.code_parser import CodeParser
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 2 feedback-based code regeneration "
            "on Phase 1 Direct failures."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to the Phase 2 feedback-regeneration "
            "YAML config."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of Phase 1 failure cases "
            "to process."
        ),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume from existing results.jsonl.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------

    config = load_config(
        args.config
    )

    experiment_config = config[
        "experiment"
    ]
    dataset_config = config[
        "dataset"
    ]
    phase1_config = config[
        "phase1"
    ]
    model_config = config[
        "model"
    ]
    strategy_config = config[
        "strategy"
    ]
    
    generation_config = config[
        "generation"
    ]
    
    evaluation_config = config[
        "evaluation"
    ]
    output_config = config[
        "output"
    ]

    # ------------------------------------------------------------------
    # 2. Resolve seed / limit
    # ------------------------------------------------------------------

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

    if (
        limit is not None
        and int(limit) <= 0
    ):
        raise ValueError(
            "limit must be greater than 0."
        )

    if limit is not None:
        limit = int(limit)

    # ------------------------------------------------------------------
    # 3. Load canonical benchmark dataset
    # ------------------------------------------------------------------

    examples = load_dataset(
        dataset_name=dataset_config[
            "name"
        ],
        data_path=dataset_config[
            "path"
        ],
        limit=dataset_config.get(
            "limit"
        ),
    )

    if not examples:
        raise ValueError(
            "No benchmark problems were loaded."
        )

    # ------------------------------------------------------------------
    # 4. Load Phase 1 Direct failures
    # ------------------------------------------------------------------

    phase1_result_path = Path(
        phase1_config[
            "direct_result_path"
        ]
    )

    failures = load_phase1_failures(
        result_path=phase1_result_path,
        limit=limit,
    )

    if not failures:
        raise ValueError(
            "No refinable Phase 1 failures were loaded."
        )

    # ------------------------------------------------------------------
    # 5. Build model generator
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 6. Build strategy
    # ------------------------------------------------------------------

    strategy = FeedbackRegenerationStrategy(
        generator=generator,
        prompt_path=strategy_config[
            "prompt_path"
        ],
        system_prompt=strategy_config.get(
            "system_prompt"
        ),
        max_new_tokens=int(
            generation_config.get(
                "code_max_new_tokens",
                1024,
            )
        ),
        temperature=float(
            generation_config.get(
                "temperature",
                0.0,
            )
        ),
        top_p=float(
            generation_config.get(
                "top_p",
                1.0,
            )
        ),
    )

    # ------------------------------------------------------------------
    # 7. Build parser
    # ------------------------------------------------------------------

    parser = CodeParser()

    # ------------------------------------------------------------------
    # 8. Build evaluator
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 9. Output path
    # ------------------------------------------------------------------
    output_path = Path(
        output_config["path"]
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

    # ------------------------------------------------------------------
    # 10. Print run configuration
    # ------------------------------------------------------------------

    print("=" * 80)
    print(
        "Phase 2 Feedback-Regeneration Experiment"
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
        f"Model      : "
        f"{model_config['name_or_path']}"
    )

    print(
        f"Seed       : "
        f"{seed}"
    )

    print(
        f"Phase1 src : "
        f"{phase1_result_path}"
    )

    print(
        f"Failures   : "
        f"{len(failures)}"
    )

    print(
        f"Prompt     : "
        f"{strategy_config['prompt_path']}"
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

    # ------------------------------------------------------------------
    # 11. Run Phase 2
    # ------------------------------------------------------------------

    runner = Phase2Runner(
        strategy=strategy,
        evaluator=evaluator,
        parser=parser,
        output_path=output_path,
        model_name=model_config[
            "name_or_path"
        ],
        seed=seed,
        resume=resume,
    )

    runner.run(
        failures=failures,
        examples=examples,
    )


if __name__ == "__main__":
    main()