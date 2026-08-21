"""

PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/verl" \
python \
  phase4_method_discovery/vanilla_planning_rlvr/scripts/smoke_test_planning_reward_rpc.py \
  --config phase4_method_discovery/vanilla_planning_rlvr/configs/vanilla_planning_rlvr_qwen25coder3b.yaml \
  --row-index 0 \
  --show-plan \
  --show-worker-status

"""
# phase4_method_discovery/vanilla_planning_rlvr/scripts/smoke_test_planning_reward_rpc.py

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import ray
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
)
from phase4_method_discovery.vanilla_planning_rlvr.workers.frozen_coder_worker import (
    FrozenCoderWorker,
)


# ======================================================================
# Defaults
# ======================================================================

DEFAULT_CONFIG_PATH = (
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
            "End-to-end RPC smoke test for Vanilla Planning-RLVR reward."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Optional parquet path override. "
            "Default: data.train_path from YAML."
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
            "Optional fixed planner response. "
            "If omitted, a generic smoke-test plan is used."
        ),
    )

    parser.add_argument(
        "--show-plan",
        action="store_true",
    )

    parser.add_argument(
        "--show-worker-status",
        action="store_true",
    )

    parser.add_argument(
        "--ray-num-cpus",
        type=int,
        default=2,
    )

    return parser.parse_args()


# ======================================================================
# Helpers
# ======================================================================

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
        value = value.as_py()

        if isinstance(
            value,
            dict,
        ):
            return value

    raise TypeError(
        f"{field_name} must be mapping-like, "
        f"got {type(value).__name__}."
    )


def build_default_plan() -> str:
    return (
        "- Parse the input exactly according to the specified format.\n"
        "- Identify the core algorithm implied by the constraints.\n"
        "- Maintain only the necessary state and data structures.\n"
        "- Handle edge cases explicitly.\n"
        "- Produce output exactly in the required format.\n"
        "- Ensure the solution satisfies the required time and space complexity."
    )


# ======================================================================
# Dataset
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
            f"dataset rows={len(df)}."
        )

    row = df.iloc[
        row_index
    ]

    data_source = str(
        row["data_source"]
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

    payload = json.loads(
        problem_json
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            "Decoded problem_json must be dict."
        )

    problem_id = str(
        payload.get(
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
# DataProto
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

    if response_ids.shape[1] <= 0:
        raise ValueError(
            "Planner response encoded to zero tokens."
        )

    return DataProto.from_dict(
        tensors={
            "responses": (
                response_ids
            ),
            "attention_mask": (
                response_attention_mask
            ),
        },
        non_tensors={
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
        },
    )


# ======================================================================
# Async manager execution
# ======================================================================

async def run_reward_manager(
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
    # 1. Load experiment config
    # ------------------------------------------------------------------

    config_path = resolve_path(
        args.config
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}"
        )

    config = OmegaConf.load(
        config_path
    )

    planner_model_path = str(
        config.planner.model_path
    )

    planner_max_tokens = int(
        config.planner.generation.max_new_tokens
    )

    frozen_coder_model_path = str(
        config.coder.model_path
    )

    # ------------------------------------------------------------------
    # 2. Resolve dataset
    # ------------------------------------------------------------------

    if args.input is None:
        input_path = Path(
            str(
                config.data.train_path
            )
        )
    else:
        input_path = resolve_path(
            args.input
        )

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
    # 3. Planner response
    # ------------------------------------------------------------------

    if args.plan is None:
        plan = build_default_plan()
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

    # ------------------------------------------------------------------
    # 4. Header
    # ------------------------------------------------------------------

    print("=" * 90)
    print(
        "Planning-RLVR RPC Reward Smoke Test"
    )
    print("=" * 90)

    print(
        f"config            : {config_path}"
    )

    print(
        f"dataset           : {input_path}"
    )

    print(
        f"row index         : {args.row_index}"
    )

    print(
        f"problem id        : {problem_id}"
    )

    print(
        f"data source       : {data_source}"
    )

    print(
        f"planner model     : {planner_model_path}"
    )

    print(
        f"frozen coder      : {frozen_coder_model_path}"
    )

    print(
        f"reward max tests  : {config.reward.max_tests}"
    )

    print(
        f"timeout seconds   : {config.reward.timeout_seconds}"
    )

    print(
        f"plan source       : {plan_source}"
    )

    if args.show_plan:
        print()
        print("-" * 90)
        print(
            "Planner Response"
        )
        print("-" * 90)
        print(
            plan
        )

    # ------------------------------------------------------------------
    # 5. Planner tokenizer
    # ------------------------------------------------------------------

    print()
    print(
        "[Tokenizer] loading planner tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            planner_model_path,
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
    # 6. Build DataProto
    # ------------------------------------------------------------------

    data = build_data_proto(
        tokenizer=tokenizer,
        plan=plan,
        data_source=data_source,
        reward_model=reward_model,
        extra_info=extra_info,
        max_plan_tokens=(
            planner_max_tokens
        ),
    )

    response_tokens = int(
        data.batch[
            "responses"
        ].shape[-1]
    )

    print(
        f"[DataProto] response tokens={response_tokens}"
    )

    # ------------------------------------------------------------------
    # 7. Start local Ray runtime
    #
    # This smoke test intentionally does not use verl's placement group.
    # Its purpose is only to verify the RPC reward contract.
    #
    # Real training will create FrozenCoderWorker inside global_pool
    # from main_ppo_sync.py.
    # ------------------------------------------------------------------

    print()
    print(
        "[Ray] initializing local runtime..."
    )

    ray.init(
        num_cpus=args.ray_num_cpus,
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    frozen_coder_handle = None

    try:
        # --------------------------------------------------------------
        # 8. Create GPU FrozenCoderWorker actor
        #
        # For this standalone smoke test we request one full GPU.
        # This is intentionally different from training, where the
        # actor is colocated using a fractional GPU resource.
        # --------------------------------------------------------------

        print()
        print(
            "[FrozenCoderWorker] creating Ray actor..."
        )

        frozen_coder_cls = ray.remote(
            num_gpus=1,
        )(
            FrozenCoderWorker
        )

        frozen_coder_handle = (
            frozen_coder_cls.remote(
                str(
                    config_path
                )
            )
        )

        print(
            "[FrozenCoderWorker] initializing model..."
        )

        worker_status = ray.get(
            frozen_coder_handle
            .init_model
            .remote()
        )

        print(
            "[FrozenCoderWorker] initialized."
        )

        if args.show_worker_status:
            print()
            print("-" * 90)
            print(
                "FrozenCoderWorker Status"
            )
            print("-" * 90)

            print(
                json.dumps(
                    worker_status,
                    indent=2,
                    ensure_ascii=False,
                )
            )

        # --------------------------------------------------------------
        # 9. Create custom RewardManager
        #
        # No local model is loaded here.
        # The manager receives only the Ray actor handle.
        # --------------------------------------------------------------

        print()
        print(
            "[PlanningRewardManager] constructing..."
        )

        minimal_verl_config = (
            OmegaConf.create(
                {}
            )
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
                frozen_coder_handle=(
                    frozen_coder_handle
                ),
            )
        )

        print(
            "[PlanningRewardManager] initialized."
        )

        # --------------------------------------------------------------
        # 10. Run full reward RPC
        # --------------------------------------------------------------

        print()
        print(
            "[Reward] running:"
        )

        print(
            "plan -> PlanningRewardManager -> "
            "FrozenCoderWorker RPC -> TACO execution"
        )

        result = asyncio.run(
            run_reward_manager(
                manager=manager,
                data=data,
            )
        )

        # --------------------------------------------------------------
        # 11. Validate return contract
        # --------------------------------------------------------------

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
                "Missing reward_score."
            )

        if (
            "reward_extra_info"
            not in result
        ):
            raise KeyError(
                "Missing reward_extra_info."
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
                "Expected binary reward, "
                f"got {reward_score}."
            )

        reward_extra_info = (
            result[
                "reward_extra_info"
            ]
        )

        # --------------------------------------------------------------
        # 12. Summary
        # --------------------------------------------------------------

        print()
        print("=" * 90)
        print(
            "RPC Smoke Test Result"
        )
        print("=" * 90)

        print(
            f"problem id         : {problem_id}"
        )

        print(
            f"response tokens    : {response_tokens}"
        )

        print(
            f"reward score       : {reward_score:.1f}"
        )

        print(
            f"status             : "
            f"{reward_extra_info.get('status')}"
        )

        print(
            f"passed             : "
            f"{reward_extra_info.get('passed')}"
        )

        print(
            f"available tests    : "
            f"{reward_extra_info.get('available_tests')}"
        )

        print(
            f"reward tests       : "
            f"{reward_extra_info.get('reward_tests')}"
        )

        print(
            f"executed result    : "
            f"{reward_extra_info.get('passed_tests')}"
            f"/"
            f"{reward_extra_info.get('total_tests')}"
        )

        print(
            f"coder prompt toks  : "
            f"{reward_extra_info.get('coder_prompt_tokens')}"
        )

        print(
            f"coder output toks  : "
            f"{reward_extra_info.get('coder_completion_tokens')}"
        )

        print(
            f"coder gen time     : "
            f"{reward_extra_info.get('coder_generation_time')}"
        )

        print(
            f"execution time     : "
            f"{reward_extra_info.get('execution_time')}"
        )

        print(
            f"error              : "
            f"{reward_extra_info.get('error_message')}"
        )

        print()
        print(
            "[PASS] RPC reward pipeline completed end-to-end."
        )

        print(
            "reward=0 is valid; the smoke-test criterion "
            "is successful execution of the full RPC path."
        )

        print("=" * 90)

    finally:
        # --------------------------------------------------------------
        # 13. Cleanup Ray
        # --------------------------------------------------------------

        print()
        print(
            "[Ray] shutting down..."
        )

        ray.shutdown()


if __name__ == "__main__":
    main()