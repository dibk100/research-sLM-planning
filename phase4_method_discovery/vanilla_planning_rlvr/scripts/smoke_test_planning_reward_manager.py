"""
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/verl" \
python \
  phase4_method_discovery/vanilla_planning_rlvr/scripts/smoke_test_planning_reward_manager.py \
  --config phase4_method_discovery/vanilla_planning_rlvr/configs/vanilla_planning_rlvr_qwen25coder3b.yaml \
  --row-index 0 \
  --show-plan

"""
# phase4_method_discovery/vanilla_planning_rlvr/scripts/smoke_test_planning_reward_manager.py

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl import DataProto


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


from phase4_method_discovery.vanilla_planning_rlvr.reward.planning_execution_reward import (
    compute_score,
)
from phase4_method_discovery.vanilla_planning_rlvr.reward.planning_reward_manager import (
    PlanningRewardManager,
    load_planning_rlvr_config,
)


# ======================================================================
# Defaults
# ======================================================================

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "phase4_method_discovery"
    / "vanilla_planning_rlvr"
    / "configs"
    / "vanilla_planning_rlvr_qwen25coder3b.yaml"
)


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test the custom verl V1 PlanningRewardManager "
            "using a real DeepCoder TACO parquet row."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Planning-RLVR research YAML.",
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Optional parquet override. "
            "If omitted, data.train_path from YAML is used."
        ),
    )

    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--plan",
        type=str,
        default=None,
        help=(
            "Optional manual planner response. "
            "If omitted, a fixed smoke-test plan is used."
        ),
    )

    parser.add_argument(
        "--max-plan-tokens",
        type=int,
        default=None,
        help=(
            "Optional plan token truncation limit. "
            "Default: planner.generation.max_new_tokens from YAML."
        ),
    )

    parser.add_argument(
        "--show-plan",
        action="store_true",
    )

    parser.add_argument(
        "--show-extra-info",
        action="store_true",
    )

    return parser.parse_args()


# ======================================================================
# Generic helpers
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


def resolve_path(
    path: str | Path,
) -> Path:
    resolved = Path(
        path
    )

    if not resolved.is_absolute():
        resolved = (
            PROJECT_ROOT
            / resolved
        )

    return resolved


# ======================================================================
# Fixed planner response
# ======================================================================

def build_default_plan() -> str:
    """
    Generic plan used only to exercise the reward pipeline.

    reward=0 is perfectly valid for this smoke test.
    """

    return (
        "- Parse the input exactly according to the given format.\n"
        "- Identify the algorithm implied by the constraints and required output.\n"
        "- Maintain only the minimal state required for the computation.\n"
        "- Handle boundary cases explicitly.\n"
        "- Produce output exactly in the requested format.\n"
        "- Ensure the implementation satisfies the time and space constraints."
    )


# ======================================================================
# Dataset loading
# ======================================================================

def load_dataset_row(
    *,
    input_path: Path,
    row_index: int,
) -> tuple[
    str,
    dict[str, Any],
    dict[str, Any],
    str,
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
            "--row-index must be >= 0."
        )

    if row_index >= len(df):
        raise IndexError(
            f"row_index={row_index}, "
            f"dataset rows={len(df)}"
        )

    row = df.iloc[
        row_index
    ]

    data_source = str(
        row["data_source"]
    )

    if data_source != "deepcoder_taco":
        raise ValueError(
            "Expected data_source='deepcoder_taco', "
            f"got {data_source!r}."
        )

    reward_model = normalize_mapping(
        row["reward_model"],
        field_name="reward_model",
    )

    extra_info = normalize_mapping(
        row["extra_info"],
        field_name="extra_info",
    )

    problem_json = extra_info.get(
        "problem_json"
    )

    if not isinstance(
        problem_json,
        str,
    ):
        raise TypeError(
            "extra_info['problem_json'] must be str."
        )

    problem_payload = json.loads(
        problem_json
    )

    if not isinstance(
        problem_payload,
        dict,
    ):
        raise TypeError(
            "Decoded problem_json must be dict."
        )

    problem_id = str(
        problem_payload.get(
            "problem_id",
            "unknown",
        )
    )

    return (
        data_source,
        reward_model,
        extra_info,
        problem_id,
    )


# ======================================================================
# verl DataProto construction
# ======================================================================

def build_data_proto(
    *,
    tokenizer: Any,
    plan: str,
    data_source: str,
    reward_model: dict[str, Any],
    extra_info: dict[str, Any],
    max_plan_tokens: int,
) -> DataProto:
    """
    Construct the minimal DataProto required by
    PlanningRewardManager.run_single().

    Tensor fields used by the manager:
        responses
        attention_mask

    Non-tensor fields:
        data_source
        reward_model
        extra_info
    """

    if max_plan_tokens <= 0:
        raise ValueError(
            "max_plan_tokens must be > 0."
        )

    encoded = tokenizer(
        plan,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_plan_tokens,
    )

    response_ids = encoded[
        "input_ids"
    ]

    response_attention_mask = encoded[
        "attention_mask"
    ]

    if (
        response_ids.ndim != 2
        or response_ids.shape[0] != 1
    ):
        raise ValueError(
            "Expected planner response tokenization "
            "with shape [1, response_length]."
        )

    if response_ids.shape[1] == 0:
        raise ValueError(
            "Planner response encoded to zero tokens."
        )

    tensors = {
        "responses": (
            response_ids
        ),
        "attention_mask": (
            response_attention_mask
        ),
    }

    non_tensors = {
        "data_source": [
            data_source
        ],

        "reward_model": [
            dict(
                reward_model
            )
        ],

        "extra_info": [
            dict(
                extra_info
            )
        ],
    }

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
    )


# ======================================================================
# Minimal verl config
# ======================================================================

def build_minimal_verl_config() -> Any:
    """
    PlanningRewardManager no longer reads coder/reward values from
    verl's Hydra config.

    It reads the project's own Planning-RLVR YAML.

    RewardManagerBase still requires a config argument, so this smoke
    test passes an intentionally minimal config object.
    """

    return OmegaConf.create(
        {}
    )


# ======================================================================
# Async execution
# ======================================================================

async def run_manager(
    *,
    manager: PlanningRewardManager,
    data: DataProto,
) -> dict[str, Any]:
    return await manager.run_single(
        data
    )


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load research config
    # ------------------------------------------------------------------

    config_path = resolve_path(
        args.config
    )

    experiment_config = (
        load_planning_rlvr_config(
            config_path
        )
    )

    planner_model = str(
        experiment_config.planner.model_path
    )

    planner_max_tokens = int(
        experiment_config
        .planner
        .generation
        .max_new_tokens
    )

    if args.max_plan_tokens is None:
        max_plan_tokens = (
            planner_max_tokens
        )
    else:
        max_plan_tokens = int(
            args.max_plan_tokens
        )

        if max_plan_tokens <= 0:
            raise ValueError(
                "--max-plan-tokens must be > 0."
            )

    # ------------------------------------------------------------------
    # 2. Dataset path
    # ------------------------------------------------------------------

    if args.input is None:
        input_path = Path(
            str(
                experiment_config
                .data
                .train_path
            )
        )
    else:
        input_path = resolve_path(
            args.input
        )

    # ------------------------------------------------------------------
    # 3. Load one real parquet row
    # ------------------------------------------------------------------

    (
        data_source,
        reward_model,
        extra_info,
        problem_id,
    ) = load_dataset_row(
        input_path=input_path,
        row_index=args.row_index,
    )

    # ------------------------------------------------------------------
    # 4. Planner response
    # ------------------------------------------------------------------

    if args.plan is None:
        plan = (
            build_default_plan()
        )

        plan_source = (
            "fixed_smoke_plan"
        )

    else:
        plan = (
            args.plan.strip()
        )

        if not plan:
            raise ValueError(
                "--plan must not be empty."
            )

        plan_source = (
            "manual"
        )

    # ------------------------------------------------------------------
    # 5. Header
    # ------------------------------------------------------------------

    print("=" * 90)
    print(
        "PlanningRewardManager Smoke Test"
    )
    print("=" * 90)

    print(
        f"experiment       : "
        f"{experiment_config.experiment.name}"
    )

    print(
        f"config           : "
        f"{config_path}"
    )

    print(
        f"dataset          : "
        f"{input_path}"
    )

    print(
        f"row index        : "
        f"{args.row_index}"
    )

    print(
        f"problem id       : "
        f"{problem_id}"
    )

    print(
        f"data source      : "
        f"{data_source}"
    )

    print(
        f"planner model    : "
        f"{planner_model}"
    )

    print(
        f"frozen coder     : "
        f"{experiment_config.coder.model_path}"
    )

    print(
        f"reward max tests : "
        f"{experiment_config.reward.max_tests}"
    )

    print(
        f"reward workers   : "
        f"{experiment_config.reward.num_workers}"
    )

    print(
        f"plan source      : "
        f"{plan_source}"
    )

    # ------------------------------------------------------------------
    # 6. Tokenizer
    # ------------------------------------------------------------------

    print()
    print(
        "[Tokenizer] loading planner tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            planner_model,
            trust_remote_code=True,
        )
    )

    if (
        tokenizer.pad_token_id
        is None
    ):
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    # ------------------------------------------------------------------
    # 7. Build DataProto
    # ------------------------------------------------------------------

    data = build_data_proto(
        tokenizer=tokenizer,
        plan=plan,
        data_source=data_source,
        reward_model=reward_model,
        extra_info=extra_info,
        max_plan_tokens=(
            max_plan_tokens
        ),
    )

    response_token_count = int(
        data.batch[
            "responses"
        ].shape[-1]
    )

    print(
        f"[DataProto] response tokens="
        f"{response_token_count}"
    )

    # ------------------------------------------------------------------
    # 8. Optional debug
    # ------------------------------------------------------------------

    if args.show_plan:
        print()
        print("-" * 90)
        print("Planner Response")
        print("-" * 90)
        print(plan)

    if args.show_extra_info:
        print()
        print("-" * 90)
        print("extra_info")
        print("-" * 90)

        printable_extra = dict(
            extra_info
        )

        # problem_json can be very large.
        problem_json = printable_extra.get(
            "problem_json"
        )

        if isinstance(
            problem_json,
            str,
        ):
            printable_extra[
                "problem_json"
            ] = (
                problem_json[:1000]
                + (
                    "... <truncated>"
                    if len(problem_json) > 1000
                    else ""
                )
            )

        print(
            json.dumps(
                printable_extra,
                indent=2,
                ensure_ascii=False,
            )
        )

    # ------------------------------------------------------------------
    # 9. Construct PlanningRewardManager
    #
    # This should trigger:
    #
    # RewardManagerBase.__init__()
    #   -> PlanningRewardManager.init_class()
    #       -> load project YAML
    #       -> load frozen coder exactly once
    #       -> initialize_reward_runtime()
    # ------------------------------------------------------------------

    print()
    print(
        "[RewardManager] constructing..."
    )

    minimal_verl_config = (
        build_minimal_verl_config()
    )

    manager = (
        PlanningRewardManager(
            config=(
                minimal_verl_config
            ),
            tokenizer=(
                tokenizer
            ),
            compute_score=(
                compute_score
            ),
        )
    )

    print(
        "[RewardManager] initialized."
    )

    # ------------------------------------------------------------------
    # 10. Execute actual V1 run_single()
    # ------------------------------------------------------------------

    print()
    print(
        "[RewardManager] running "
        "DataProto -> run_single() -> compute_score()..."
    )

    result = asyncio.run(
        run_manager(
            manager=manager,
            data=data,
        )
    )

    # ------------------------------------------------------------------
    # 11. Validate return contract
    # ------------------------------------------------------------------

    if not isinstance(
        result,
        dict,
    ):
        raise TypeError(
            "PlanningRewardManager.run_single() "
            "must return dict."
        )

    if (
        "reward_score"
        not in result
    ):
        raise KeyError(
            "run_single() result is missing "
            "'reward_score'."
        )

    if (
        "reward_extra_info"
        not in result
    ):
        raise KeyError(
            "run_single() result is missing "
            "'reward_extra_info'."
        )

    reward_score = float(
        result[
            "reward_score"
        ]
    )

    if reward_score not in {
        0.0,
        1.0,
    }:
        raise ValueError(
            "Expected binary Planning-RLVR reward, "
            f"got {reward_score}."
        )

    # ------------------------------------------------------------------
    # 12. Final summary
    # ------------------------------------------------------------------

    print()
    print("=" * 90)
    print("Smoke Test Result")
    print("=" * 90)

    print(
        f"problem id        : "
        f"{problem_id}"
    )

    print(
        f"response tokens   : "
        f"{response_token_count}"
    )

    print(
        f"reward score      : "
        f"{reward_score:.1f}"
    )

    print(
        f"reward extra info : "
        f"{result['reward_extra_info']}"
    )

    print()
    print(
        "[PASS] PlanningRewardManager completed:"
    )

    print(
        "DataProto -> planner response decode "
        "-> frozen coder -> TACO execution -> reward"
    )

    print()
    print(
        "reward=0 is a valid smoke-test result. "
        "The success criterion is end-to-end execution."
    )

    print("=" * 90)


if __name__ == "__main__":
    main()