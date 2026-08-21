"""  
vanilla_planning_rlvr/
└── data/
    └── processed/
        ├── train.parquet
        ├── val.parquet
        └── dataset_manifest.json
        
PYTHONPATH="$HOME/workspace/project_sLM_planning" \
python \
  phase4_method_discovery/vanilla_planning_rlvr/scripts/smoke_test_reward.py \
  --input /mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/vanilla_planning_rlvr/train.parquet \
  --row-index 0 \
  --show-plan \
  --show-code

"""
# phase4_method_discovery/vanilla_planning_rlvr/scripts/smoke_test_reward.py

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ======================================================================
# Project root
# ======================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from phase4_method_discovery.vanilla_planning_rlvr.logging.rollout_logger import (
    RolloutLogger,
)
from phase4_method_discovery.vanilla_planning_rlvr.reward.planning_execution_reward import (
    compute_planning_execution_reward,
    initialize_reward_runtime,
    restore_problem_from_extra_info,
)
from src.models.generator import (
    ModelGenerator,
)
from src.schemas import (
    ProblemExample,
)


# ======================================================================
# Constants
# ======================================================================

DEFAULT_INPUT = Path(
    "/mnt/hdd/project_sLM_planning/data/"
    "deepcoder_taco/processed/"
    "vanilla_planning_rlvr/train.parquet"
)

DEFAULT_MODEL = (
    "Qwen/Qwen2.5-Coder-3B-Instruct"
)

DEFAULT_CODE_PROMPT = (
    PROJECT_ROOT
    / "prompt_templates"
    / "self_plan_code.txt"
)

DEFAULT_LOG_PATH = (
    PROJECT_ROOT
    / "phase4_method_discovery"
    / "archive"
    / "reward_smoke_test.jsonl"
)


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end smoke test for the production "
            "Vanilla Planning-RLVR reward pipeline."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="verl parquet dataset.",
    )

    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Parquet row index to evaluate.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Frozen coder model.",
    )

    parser.add_argument(
        "--code-prompt",
        type=str,
        default=str(DEFAULT_CODE_PROMPT),
        help="Plan-conditioned code prompt template.",
    )

    parser.add_argument(
        "--plan",
        type=str,
        default=None,
        help=(
            "Optional manual plan. "
            "If omitted, a deterministic smoke-test plan is used."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=6,
        help="Per-test execution timeout.",
    )

    parser.add_argument(
        "--coder-max-new-tokens",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--coder-temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--coder-top-p",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-reward-tests",
        type=int,
        default=15,
        help=(
            "Maximum number of reward tests. "
            "Use 0 to evaluate all tests."
        ),
    )

    parser.add_argument(
        "--log-output",
        type=str,
        default=str(DEFAULT_LOG_PATH),
    )

    parser.add_argument(
        "--show-code",
        action="store_true",
    )

    parser.add_argument(
        "--show-plan",
        action="store_true",
    )

    parser.add_argument(
        "--show-tests",
        action="store_true",
    )

    parser.add_argument(
        "--num-tests-to-show",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# ======================================================================
# Parquet helpers
# ======================================================================

def normalize_mapping(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any]:
    if isinstance(
        value,
        dict,
    ):
        return value

    if hasattr(
        value,
        "as_py",
    ):
        converted = value.as_py()

        if isinstance(
            converted,
            dict,
        ):
            return converted

    raise TypeError(
        f"{field_name} must be mapping-like, "
        f"got {type(value).__name__}"
    )


def load_parquet_row(
    *,
    input_path: Path,
    row_index: int,
) -> tuple[
    pd.Series,
    dict[str, Any],
    ProblemExample,
]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    df = pd.read_parquet(
        input_path,
        engine="pyarrow",
    )

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    if row_index < 0:
        raise ValueError(
            "row_index must be >= 0."
        )

    if row_index >= len(df):
        raise IndexError(
            f"row_index={row_index}, "
            f"dataset rows={len(df)}"
        )

    row = df.iloc[
        row_index
    ]

    data_source = row[
        "data_source"
    ]

    if data_source != "deepcoder_taco":
        raise ValueError(
            "Unexpected data_source: "
            f"{data_source!r}"
        )

    extra_info = normalize_mapping(
        row["extra_info"],
        field_name="extra_info",
    )

    problem = (
        restore_problem_from_extra_info(
            extra_info
        )
    )

    return (
        row,
        extra_info,
        problem,
    )


# ======================================================================
# Smoke-test plan
# ======================================================================

def build_default_plan(
    problem: ProblemExample,
) -> str:
    """
    Generic deterministic plan.

    This plan is intentionally not guaranteed to be correct.

    Smoke-test success means:
        reward pipeline completed end-to-end.

    It does NOT require reward == 1.
    """

    del problem

    return (
        "- Parse the input exactly according to the given format.\n"
        "- Identify the algorithm that directly satisfies the required output and constraints.\n"
        "- Maintain only the minimal state or data structures needed by that algorithm.\n"
        "- Handle boundary cases and special input configurations explicitly.\n"
        "- Produce output exactly in the required format.\n"
        "- Ensure the implementation fits the required time and space complexity."
    )


# ======================================================================
# Test preview
# ======================================================================

def print_test_preview(
    problem: ProblemExample,
    *,
    limit: int,
) -> None:
    if limit <= 0:
        return

    count = min(
        limit,
        len(problem.private_tests),
    )

    print()
    print("-" * 90)
    print("Private Test Preview")
    print("-" * 90)

    for test_index in range(
        count
    ):
        print()
        print(
            f"[test {test_index}]"
        )

        print(
            json.dumps(
                problem.private_tests[
                    test_index
                ],
                ensure_ascii=False,
                indent=2,
            )
        )

    remaining = (
        len(problem.private_tests)
        - count
    )

    if remaining > 0:
        print()
        print(
            f"... {remaining} more tests omitted"
        )


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Validate args
    # ------------------------------------------------------------------

    if args.timeout <= 0:
        raise ValueError(
            "--timeout must be > 0."
        )

    if args.coder_max_new_tokens <= 0:
        raise ValueError(
            "--coder-max-new-tokens must be > 0."
        )

    if args.coder_temperature < 0:
        raise ValueError(
            "--coder-temperature must be >= 0."
        )

    if not 0 < args.coder_top_p <= 1:
        raise ValueError(
            "--coder-top-p must be in (0, 1]."
        )

    if args.max_reward_tests < 0:
        raise ValueError(
            "--max-reward-tests must be >= 0."
        )

    if args.num_tests_to_show <= 0:
        raise ValueError(
            "--num-tests-to-show must be > 0."
        )

    input_path = Path(
        args.input
    )

    code_prompt_path = Path(
        args.code_prompt
    )

    log_output_path = Path(
        args.log_output
    )

    # ------------------------------------------------------------------
    # 1. Load one actual verl row
    # ------------------------------------------------------------------

    (
        row,
        extra_info,
        problem,
    ) = load_parquet_row(
        input_path=input_path,
        row_index=args.row_index,
    )

    print("=" * 90)
    print(
        "Vanilla Planning-RLVR "
        "Production Reward Smoke Test"
    )
    print("=" * 90)

    print(
        f"dataset            : "
        f"{input_path}"
    )

    print(
        f"row index          : "
        f"{args.row_index}"
    )

    print(
        f"problem id         : "
        f"{problem.problem_id}"
    )

    print(
        f"dataset source     : "
        f"{problem.dataset}"
    )

    print(
        f"evaluation type    : "
        f"{problem.evaluation_type}"
    )

    print(
        f"available tests    : "
        f"{len(problem.private_tests)}"
    )

    print(
        f"frozen coder       : "
        f"{args.model}"
    )

    # ------------------------------------------------------------------
    # 2. Plan
    # ------------------------------------------------------------------

    if args.plan is None:
        plan = build_default_plan(
            problem
        )

        plan_source = (
            "fixed_smoke_plan"
        )

    else:
        plan = args.plan.strip()

        if not plan:
            raise ValueError(
                "--plan must not be empty."
            )

        plan_source = (
            "manual"
        )

    if args.show_plan:
        print()
        print("-" * 90)
        print(
            f"Plan ({plan_source})"
        )
        print("-" * 90)
        print(plan)

    # ------------------------------------------------------------------
    # 3. Optional raw test preview
    # ------------------------------------------------------------------

    if args.show_tests:
        print_test_preview(
            problem,
            limit=(
                args.num_tests_to_show
            ),
        )

    # ------------------------------------------------------------------
    # 4. Load frozen coder
    # ------------------------------------------------------------------

    print()
    print(
        "[Runtime] loading frozen coder..."
    )

    frozen_coder = ModelGenerator(
        args.model,
        dtype="bfloat16",
        device_map="auto",
    )

    # ------------------------------------------------------------------
    # 5. Initialize production reward runtime
    # ------------------------------------------------------------------

    max_reward_tests: int | None

    if args.max_reward_tests == 0:
        max_reward_tests = None
    else:
        max_reward_tests = (
            args.max_reward_tests
        )

    initialize_reward_runtime(
        frozen_coder=frozen_coder,
        code_prompt_path=(
            code_prompt_path
        ),
        timeout_seconds=(
            args.timeout
        ),
        debug=False,
        coder_max_new_tokens=(
            args.coder_max_new_tokens
        ),
        coder_temperature=(
            args.coder_temperature
        ),
        coder_top_p=(
            args.coder_top_p
        ),
        max_reward_tests=(
            max_reward_tests
        ),
    )

    print(
        "[Runtime] reward runtime initialized."
    )

    # ------------------------------------------------------------------
    # 6. Execute actual production reward
    # ------------------------------------------------------------------

    print()
    print(
        "[Reward] running "
        "plan -> frozen coder -> code -> execution..."
    )

    reward_result = (
        compute_planning_execution_reward(
            problem=problem,
            plan=plan,
        )
    )

    # ------------------------------------------------------------------
    # 7. Rollout log
    # ------------------------------------------------------------------

    logger = RolloutLogger(
        output_path=(
            log_output_path
        ),
        flush_every_write=True,
    )

    rollout_record = (
        RolloutLogger.from_reward_result(
            reward_result=(
                reward_result
            ),
            global_step=0,
            group_id=(
                f"smoke_"
                f"{problem.problem_id}"
            ),
            sample_id=0,
            dataset=(
                problem.dataset
            ),
            model_name=(
                args.model
            ),
            seed=(
                args.seed
            ),
            plan_tokens=None,
            plan_token_ids=None,
            token_logprobs=None,
            extra={
                "smoke_test": True,
                "row_index": (
                    args.row_index
                ),
                "plan_source": (
                    plan_source
                ),
                "available_tests": (
                    reward_result.available_tests
                ),
                "reward_tests": (
                    reward_result.reward_tests
                ),
                "parquet_split": (
                    extra_info.get(
                        "split"
                    )
                ),
            },
        )
    )

    logger.log(
        rollout_record
    )

    # ------------------------------------------------------------------
    # 8. Print generated code
    # ------------------------------------------------------------------

    if args.show_code:
        print()
        print("-" * 90)
        print("Frozen Coder Raw Output")
        print("-" * 90)
        print(
            reward_result.raw_code_output
        )

        print()
        print("-" * 90)
        print("Extracted Code")
        print("-" * 90)
        print(
            reward_result.generated_code
        )

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 90)
    print("Smoke Test Result")
    print("=" * 90)

    print(
        f"problem id         : "
        f"{reward_result.problem_id}"
    )

    print(
        f"plan source        : "
        f"{plan_source}"
    )

    print(
        f"reward             : "
        f"{reward_result.reward:.1f}"
    )

    print(
        f"passed             : "
        f"{reward_result.passed}"
    )

    print(
        f"status             : "
        f"{reward_result.status}"
    )

    print(
        f"available tests    : "
        f"{reward_result.available_tests}"
    )

    print(
        f"reward tests       : "
        f"{reward_result.reward_tests}"
    )

    print(
        f"executed result    : "
        f"{reward_result.passed_tests}/"
        f"{reward_result.total_tests}"
    )

    print(
        f"coder prompt toks  : "
        f"{reward_result.coder_prompt_tokens}"
    )

    print(
        f"coder output toks  : "
        f"{reward_result.coder_completion_tokens}"
    )

    print(
        f"coder gen time     : "
        f"{reward_result.coder_generation_time:.3f}s"
    )

    print(
        f"execution time     : "
        f"{reward_result.execution_time:.3f}s"
    )

    print(
        f"error              : "
        f"{reward_result.error_message}"
    )

    print(
        f"rollout log        : "
        f"{log_output_path}"
    )

    print()
    print(
        "[PASS] Production reward pipeline "
        "completed end-to-end."
    )

    print(
        "reward=0 is valid for this smoke test; "
        "the goal is pipeline execution, not solving the problem."
    )

    print("=" * 90)


if __name__ == "__main__":
    main()