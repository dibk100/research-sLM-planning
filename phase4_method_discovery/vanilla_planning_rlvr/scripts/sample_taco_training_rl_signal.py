"""
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench:$HOME/workspace/verl" \
python phase4_method_discovery/vanilla_planning_rlvr/scripts/sample_taco_training_rl_signal.py \
  --config phase4_method_discovery/vanilla_planning_rlvr/configs/vanilla_planning_rlvr_qwen25coder3b.yaml \
  --verl-config phase4_method_discovery/vanilla_planning_rlvr/configs/verl_grpo_pilot_50step.yaml \
  --dataset-path /mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/vanilla_planning_rlvr/train.parquet \
  --checkpoint-label step25 \
  --adapter-path /mnt/hdd/project_sLM_planning/checkpoints/vanilla_planning_rlvr_lora_pilot50/exported/step25 \
  --num-problems 20 \
  --output-path phase4_method_discovery/vanilla_planning_rlvr/outputs/planning_rl_signal_sampling/taco_step25_pilot20_g16.jsonl \
  --overwrite
"""
# phase4_method_discovery/vanilla_planning_rlvr/scripts/
# sample_taco_training_rl_signal.py

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ray
import torch
from omegaconf import OmegaConf
from peft import PeftModel
from tensordict import TensorDict
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl import DataProto
from verl.experimental.reward_loop.reward_loop import RewardLoopManager

from phase4_method_discovery.vanilla_planning_rlvr.workers.frozen_coder_worker import (
    FrozenCoderWorker,
)


# =============================================================================
# Paths
# =============================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

DEFAULT_RESEARCH_CONFIG = (
    PROJECT_ROOT
    / "phase4_method_discovery"
    / "vanilla_planning_rlvr"
    / "configs"
    / "vanilla_planning_rlvr_qwen25coder3b.yaml"
)

DEFAULT_VERL_CONFIG = (
    PROJECT_ROOT
    / "phase4_method_discovery"
    / "vanilla_planning_rlvr"
    / "configs"
    / "verl_grpo_pilot_50step.yaml"
)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the actual TACO training-time Planning-RLVR reward "
            "distribution using the same verl RewardLoopWorker -> "
            "PlanningRewardManager -> compute_score path as GRPO training."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RESEARCH_CONFIG,
        help=(
            "Research config used by FrozenCoderWorker "
            "(vanilla_planning_rlvr_qwen25coder3b.yaml)."
        ),
    )

    parser.add_argument(
        "--verl-config",
        type=Path,
        default=DEFAULT_VERL_CONFIG,
        help=(
            "Actual verl GRPO pilot config. "
            "Used for rollout/reward settings."
        ),
    )

    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=(
            "Processed TACO train parquet. "
            "Defaults to data.train_files[0] from --verl-config."
        ),
    )

    parser.add_argument(
        "--checkpoint-label",
        type=str,
        default="step25",
    )

    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Exported PEFT LoRA adapter. Omit for base planner.",
    )

    parser.add_argument(
        "--num-problems",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help=(
            "Plans per problem. Defaults to actual "
            "actor_rollout_ref.rollout.n from verl config."
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--show-plans",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# Generic helpers
# =============================================================================


def resolve_path(path: Path | str) -> Path:
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def _pythonize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_pythonize(x) for x in value.tolist()]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(k): _pythonize(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_pythonize(v) for v in value]

    return value


def make_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(k): make_serializable(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_serializable(v)
            for v in value
        ]

    if isinstance(value, np.ndarray):
        return make_serializable(
            value.tolist()
        )

    if isinstance(value, np.generic):
        return value.item()

    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()

        return value.detach().cpu().tolist()

    return value


def append_jsonl(
    path: Path,
    record: dict[str, Any],
) -> None:
    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                make_serializable(record),
                ensure_ascii=False,
            )
            + "\n"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def population_variance(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    mean = sum(values) / len(values)

    return sum(
        (x - mean) ** 2
        for x in values
    ) / len(values)


def print_header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# =============================================================================
# Dataset helpers
# =============================================================================


def normalize_messages(
    value: Any,
) -> list[dict[str, str]]:
    value = _pythonize(value)

    if isinstance(value, str):
        stripped = value.strip()

        try:
            decoded = json.loads(
                stripped
            )
            value = decoded
        except json.JSONDecodeError:
            value = [
                {
                    "role": "user",
                    "content": stripped,
                }
            ]

    if not isinstance(value, list):
        raise TypeError(
            "prompt must be a list of chat messages; "
            f"got {type(value).__name__}"
        )

    messages: list[
        dict[str, str]
    ] = []

    for item in value:
        item = _pythonize(item)

        if not isinstance(item, dict):
            raise TypeError(
                "Each prompt message must be dict."
            )

        role = str(
            item.get(
                "role",
                "user",
            )
        )

        content = item.get(
            "content",
            "",
        )

        if isinstance(content, list):
            text_parts = []

            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                ):
                    text_parts.append(
                        str(
                            part.get(
                                "text",
                                "",
                            )
                        )
                    )
                else:
                    text_parts.append(
                        str(part)
                    )

            content = "\n".join(
                text_parts
            )

        messages.append(
            {
                "role": role,
                "content": str(content),
            }
        )

    if not messages:
        raise ValueError(
            "Empty planner prompt."
        )

    return messages


def normalize_extra_info(
    value: Any,
) -> dict[str, Any]:
    value = _pythonize(value)

    if isinstance(value, str):
        value = json.loads(
            value
        )

    if not isinstance(value, dict):
        raise TypeError(
            "extra_info must decode to dict; "
            f"got {type(value).__name__}"
        )

    return value


def normalize_reward_model(
    value: Any,
) -> dict[str, Any]:
    value = _pythonize(value)

    if isinstance(value, str):
        try:
            value = json.loads(
                value
            )
        except json.JSONDecodeError:
            value = {
                "ground_truth": value,
            }

    if not isinstance(value, dict):
        raise TypeError(
            "reward_model must decode to dict; "
            f"got {type(value).__name__}"
        )

    value.setdefault(
        "ground_truth",
        "",
    )

    return value


def get_problem_id(
    row: pd.Series,
    fallback_index: int,
) -> str:
    extra_info = normalize_extra_info(
        row["extra_info"]
    )

    if extra_info.get(
        "problem_id"
    ) is not None:
        return str(
            extra_info[
                "problem_id"
            ]
        )

    problem_json = extra_info.get(
        "problem_json"
    )

    if isinstance(
        problem_json,
        str,
    ):
        try:
            payload = json.loads(
                problem_json
            )

            if isinstance(
                payload,
                dict,
            ):
                for key in (
                    "problem_id",
                    "id",
                    "question_id",
                ):
                    if (
                        payload.get(key)
                        is not None
                    ):
                        return str(
                            payload[key]
                        )

        except json.JSONDecodeError:
            pass

    return f"row_{fallback_index}"


def load_dataset(
    path: Path,
    *,
    start_index: int,
    num_problems: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Training parquet not found: {path}"
        )

    df = pd.read_parquet(
        path
    )

    required = {
        "data_source",
        "prompt",
        "reward_model",
        "extra_info",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise KeyError(
            "Missing parquet columns: "
            f"{sorted(missing)}"
        )

    if start_index < 0:
        raise ValueError(
            "start-index must be >= 0"
        )

    if start_index >= len(df):
        raise IndexError(
            f"start-index={start_index} "
            f"outside dataset size {len(df)}"
        )

    stop = min(
        len(df),
        start_index
        + num_problems,
    )

    return (
        df.iloc[
            start_index:stop
        ]
        .copy()
        .reset_index(
            drop=False
        )
        .rename(
            columns={
                "index": "_dataset_index"
            }
        )
    )


# =============================================================================
# Planner
# =============================================================================


def load_planner(
    model_name: str,
    adapter_path: Path | None,
):
    print_header(
        "LOAD PLANNER"
    )

    print(
        f"Base model      : {model_name}"
    )

    print(
        f"LoRA adapter    : "
        f"{adapter_path or '<none>'}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
            device_map=None,
        )
    )

    if adapter_path is not None:
        adapter_path = resolve_path(
            adapter_path
        )

        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter not found: "
                f"{adapter_path}"
            )

        model = (
            PeftModel.from_pretrained(
                model,
                str(adapter_path),
                is_trainable=False,
            )
        )

        lora_count = sum(
            1
            for name, _
            in model.named_parameters()
            if "lora_" in name
        )

        print(
            f"LoRA tensors    : {lora_count}"
        )

        if lora_count == 0:
            raise RuntimeError(
                "Adapter supplied but "
                "no LoRA tensors were loaded."
            )

    model.eval()
    model.to("cuda")

    return (
        tokenizer,
        model,
    )


@torch.inference_mode()
def sample_plans(
    *,
    tokenizer,
    model,
    messages: list[dict[str, str]],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Sample one GRPO-style group.

    Important:
      verl pilot:
        n = 16
        temperature = 1.0
        top_p = 1.0
        top_k = -1

    HF generate uses top_k=50 by default when sampling,
    therefore top_k=0 is explicitly set to reproduce
    vLLM top_k=-1 (disabled).
    """

    prompt_text = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    encoded = {
        key: tensor.to(
            "cuda"
        )
        for key, tensor
        in encoded.items()
    }

    prompt_length = int(
        encoded[
            "input_ids"
        ].shape[1]
    )

    set_seed(
        seed
    )

    start = time.perf_counter()

    outputs = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=0,
        num_return_sequences=group_size,
        pad_token_id=(
            tokenizer.pad_token_id
        ),
        eos_token_id=(
            tokenizer.eos_token_id
        ),
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    samples: list[
        dict[str, Any]
    ] = []

    for sample_index, sequence in enumerate(
        outputs
    ):
        generated_ids = (
            sequence[
                prompt_length:
            ]
            .detach()
            .cpu()
        )

        plan = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            .strip()
        )

        samples.append(
            {
                "sample_index": (
                    sample_index
                ),
                "plan": plan,
                "response_ids": (
                    generated_ids
                ),
                "completion_tokens": int(
                    generated_ids.numel()
                ),
            }
        )

    if len(samples) != group_size:
        raise RuntimeError(
            f"Expected {group_size} plans, "
            f"got {len(samples)}."
        )

    print(
        f"  sampled {group_size} plans "
        f"in {elapsed:.2f}s"
    )

    return samples


def release_planner(
    model,
) -> None:
    print_header(
        "RELEASE PLANNER GPU"
    )

    try:
        model.to(
            "cpu"
        )
    except Exception:
        pass

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        print(
            "CUDA allocated : "
            f"{torch.cuda.memory_allocated() / 1024**3:.3f} GiB"
        )

        print(
            "CUDA reserved  : "
            f"{torch.cuda.memory_reserved() / 1024**3:.3f} GiB"
        )


# =============================================================================
# Frozen coder
# =============================================================================


def initialize_ray() -> None:
    if ray.is_initialized():
        return

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=True,
    )


def start_frozen_coder(
    research_config_path: Path,
):
    print_header(
        "START FROZEN CODER"
    )

    FrozenCoderActor = ray.remote(
        num_cpus=1,
        num_gpus=1,
    )(
        FrozenCoderWorker
    )

    actor = (
        FrozenCoderActor.remote(
            str(
                research_config_path
            )
        )
    )

    status = ray.get(
        actor.init_model.remote()
    )

    print(
        json.dumps(
            make_serializable(
                status
            ),
            indent=2,
            ensure_ascii=False,
        )
    )

    return actor


# =============================================================================
# DataProto construction
# =============================================================================


def build_reward_dataproto(
    *,
    tokenizer,
    row: pd.Series,
    response_ids: torch.Tensor,
) -> DataProto:
    """
    Build the one-item DataProto expected by the real
    PlanningRewardManager.

    PlanningRewardManager consumes:
      batch["responses"]
      batch["attention_mask"]

      non_tensor_batch["data_source"]
      non_tensor_batch["reward_model"]
      non_tensor_batch["extra_info"]

    raw_prompt is also preserved to match the normal verl
    rollout payload.
    """

    messages = normalize_messages(
        row["prompt"]
    )

    prompt_text = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )[
        "input_ids"
    ][0]

    response_ids = (
        response_ids
        .detach()
        .cpu()
        .long()
    )

    # --------------------------------------------------------------
    # The reward manager determines valid response length from:
    #
    # attention_mask[-response_length:].sum()
    #
    # Therefore concatenate prompt and response masks.
    # --------------------------------------------------------------

    prompt_mask = torch.ones(
        prompt_ids.shape[0],
        dtype=torch.long,
    )

    response_mask = torch.ones(
        response_ids.shape[0],
        dtype=torch.long,
    )

    attention_mask = torch.cat(
        [
            prompt_mask,
            response_mask,
        ],
        dim=0,
    )

    input_ids = torch.cat(
        [
            prompt_ids.long(),
            response_ids,
        ],
        dim=0,
    )

    position_ids = torch.arange(
        input_ids.shape[0],
        dtype=torch.long,
    )

    # One-item TensorDict.
    batch = TensorDict(
        {
            "prompts": (
                prompt_ids.unsqueeze(
                    0
                )
            ),
            "responses": (
                response_ids.unsqueeze(
                    0
                )
            ),
            "input_ids": (
                input_ids.unsqueeze(
                    0
                )
            ),
            "attention_mask": (
                attention_mask.unsqueeze(
                    0
                )
            ),
            "position_ids": (
                position_ids.unsqueeze(
                    0
                )
            ),
        },
        batch_size=[1],
    )

    data_source = str(
        row.get(
            "data_source",
            "deepcoder_taco",
        )
    )

    reward_model = (
        normalize_reward_model(
            row[
                "reward_model"
            ]
        )
    )

    extra_info = (
        normalize_extra_info(
            row[
                "extra_info"
            ]
        )
    )

    non_tensor_batch = {
        "raw_prompt": np.array(
            [messages],
            dtype=object,
        ),
        "data_source": np.array(
            [data_source],
            dtype=object,
        ),
        "reward_model": np.array(
            [reward_model],
            dtype=object,
        ),
        "extra_info": np.array(
            [extra_info],
            dtype=object,
        ),
        "__num_turns__": np.array(
            [1],
            dtype=object,
        ),
        "reward_scores": np.array(
            [{}],
            dtype=object,
        ),
    }

    return DataProto(
        batch=batch,
        non_tensor_batch=(
            non_tensor_batch
        ),
        meta_info={},
    )


# =============================================================================
# RewardLoop
# =============================================================================


def start_reward_loop_manager(
    *,
    verl_config,
    frozen_coder_handle,
) -> RewardLoopManager:
    print_header(
        "START ACTUAL VERL REWARD LOOP"
    )

    manager = RewardLoopManager(
        config=verl_config,
        rm_resource_pool=None,
        frozen_coder_handle=(
            frozen_coder_handle
        ),
    )

    workers = (
        manager
        .reward_loop_worker_handles
    )

    if not workers:
        raise RuntimeError(
            "RewardLoopManager created "
            "no reward workers."
        )

    print(
        f"Reward workers   : "
        f"{len(workers)}"
    )

    print(
        "Reward manager   : "
        f"{verl_config.reward.reward_manager.name}"
    )

    print(
        "Reward function  : "
        f"{verl_config.reward.custom_reward_function.name}"
    )

    return manager


def evaluate_group_via_reward_loop(
    *,
    reward_loop_manager: RewardLoopManager,
    data_items: list[DataProto],
) -> list[dict[str, Any]]:
    """
    Use the actual RewardLoopWorker.compute_score_batch().

    This reproduces the training reward execution topology:

      controller
        -> RewardLoopWorker Ray actor
        -> PlanningRewardManager.run_single()
        -> run_in_executor()
        -> compute_score()
        -> FrozenCoderWorker
        -> TACOEvaluator spawn

    Because reward.num_workers=1 in the pilot, all G trajectories
    are submitted to that one reward worker.

    PlanningRewardManager's asyncio.Lock serializes complete reward
    transactions inside the worker, matching training behavior.
    """

    if not data_items:
        return []

    # Concatenate one-item DataProto objects into one G-sized batch.
    group_data = DataProto.concat(
        data_items
    )

    workers = (
        reward_loop_manager
        .reward_loop_worker_handles
    )

    if len(workers) != 1:
        # General fallback matching RewardLoopManager's own
        # chunking behavior.
        chunks = group_data.chunk(
            len(workers)
        )

        nested = ray.get(
            [
                worker
                .compute_score_batch
                .remote(chunk)
                for worker, chunk
                in zip(
                    workers,
                    chunks,
                    strict=True,
                )
            ]
        )

        return [
            item
            for sublist in nested
            for item in sublist
        ]

    return ray.get(
        workers[0]
        .compute_score_batch
        .remote(
            group_data
        )
    )


# =============================================================================
# Reward diagnostics
# =============================================================================


def normalize_reward_output(
    output: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(
        output,
        dict,
    ):
        raise TypeError(
            "RewardLoopWorker output "
            "must be dict."
        )

    score = float(
        output[
            "reward_score"
        ]
    )

    info = output.get(
        "reward_extra_info",
        {},
    )

    if not isinstance(
        info,
        dict,
    ):
        info = dict(
            info
        )

    available_tests = int(
        info.get(
            "available_tests",
            0,
        )
        or 0
    )

    reward_tests = int(
        info.get(
            "reward_tests",
            0,
        )
        or 0
    )

    passed_tests = int(
        info.get(
            "passed_tests",
            0,
        )
        or 0
    )

    total_tests = int(
        info.get(
            "total_tests",
            0,
        )
        or 0
    )

    # --------------------------------------------------------------
    # IMPORTANT:
    #
    # TACOEvaluator is fail-fast.
    #
    # reward_test_progress:
    #   passed prefix / complete selected reward tests.
    #
    # This is NOT true independent K/N test-pass ratio.
    # It is a conservative/order-dependent progress diagnostic.
    #
    # executed_prefix_ratio:
    #   passed / actually executed prefix.
    # --------------------------------------------------------------

    reward_test_progress = (
        passed_tests
        / reward_tests
        if reward_tests > 0
        else 0.0
    )

    executed_prefix_ratio = (
        passed_tests
        / total_tests
        if total_tests > 0
        else 0.0
    )

    return {
        "reward": score,
        "passed": bool(
            info.get(
                "passed",
                score == 1.0,
            )
        ),
        "status": str(
            info.get(
                "status",
                "",
            )
        ),
        "available_tests": (
            available_tests
        ),
        "reward_tests": (
            reward_tests
        ),
        "passed_tests": (
            passed_tests
        ),
        "total_tests": (
            total_tests
        ),
        "reward_test_progress": float(
            reward_test_progress
        ),
        "executed_prefix_ratio": float(
            executed_prefix_ratio
        ),
        "execution_time": float(
            info.get(
                "execution_time",
                0.0,
            )
            or 0.0
        ),
        "coder_prompt_tokens": int(
            info.get(
                "coder_prompt_tokens",
                0,
            )
            or 0
        ),
        "coder_completion_tokens": int(
            info.get(
                "coder_completion_tokens",
                0,
            )
            or 0
        ),
        "coder_generation_time": float(
            info.get(
                "coder_generation_time",
                0.0,
            )
            or 0.0
        ),
        "error_message": str(
            info.get(
                "error_message",
                "",
            )
            or ""
        ),
        "reward_extra_info": (
            make_serializable(
                info
            )
        ),
    }


def classify_group(
    rewards: list[float],
) -> str:
    positives = sum(
        r > 0.0
        for r in rewards
    )

    if positives == 0:
        return "all_zero"

    if positives == len(
        rewards
    ):
        return "all_one"

    return "mixed"


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    research_config_path = (
        resolve_path(
            args.config
        )
    )

    verl_config_path = (
        resolve_path(
            args.verl_config
        )
    )

    if not research_config_path.exists():
        raise FileNotFoundError(
            f"Research config not found: "
            f"{research_config_path}"
        )

    if not verl_config_path.exists():
        raise FileNotFoundError(
            f"verl config not found: "
            f"{verl_config_path}"
        )

    verl_config = OmegaConf.load(
        verl_config_path
    )

    # -----------------------------------------------------------------
    # Actual pilot rollout settings are the authority.
    # -----------------------------------------------------------------

    rollout_cfg = (
        verl_config
        .actor_rollout_ref
        .rollout
    )

    group_size = (
        args.num_samples
        if args.num_samples
        is not None
        else int(
            rollout_cfg.n
        )
    )

    temperature = (
        args.temperature
        if args.temperature
        is not None
        else float(
            rollout_cfg.temperature
        )
    )

    top_p = (
        args.top_p
        if args.top_p
        is not None
        else float(
            rollout_cfg.top_p
        )
    )

    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens
        is not None
        else int(
            rollout_cfg.response_length
        )
    )

    model_name = str(
        verl_config
        .actor_rollout_ref
        .model
        .path
    )

    if args.dataset_path is not None:
        dataset_path = resolve_path(
            args.dataset_path
        )
    else:
        train_files = (
            verl_config
            .data
            .train_files
        )

        dataset_path = resolve_path(
            str(
                train_files[0]
            )
        )

    output_path = resolve_path(
        args.output_path
    )

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output exists: "
                f"{output_path}\n"
                "Use --overwrite."
            )

        output_path.unlink()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset(
        dataset_path,
        start_index=(
            args.start_index
        ),
        num_problems=(
            args.num_problems
        ),
    )

    print_header(
        "TACO TRAINING RL SIGNAL DIAGNOSTIC"
    )

    print(
        f"Checkpoint       : "
        f"{args.checkpoint_label}"
    )
    print(
        f"Dataset          : "
        f"{dataset_path}"
    )
    print(
        f"Problems         : "
        f"{len(df)}"
    )
    print(
        f"Group size       : "
        f"{group_size}"
    )
    print(
        f"Temperature      : "
        f"{temperature}"
    )
    print(
        f"Top-p            : "
        f"{top_p}"
    )
    print(
        f"Max plan tokens  : "
        f"{max_new_tokens}"
    )
    print(
        f"Reward workers   : "
        f"{verl_config.reward.num_workers}"
    )
    print(
        f"Output           : "
        f"{output_path}"
    )

    # =================================================================
    # Stage 1: planner
    # =================================================================

    tokenizer, planner = (
        load_planner(
            model_name=model_name,
            adapter_path=(
                args.adapter_path
            ),
        )
    )

    sampled_groups: list[
        dict[str, Any]
    ] = []

    print_header(
        "STAGE 1: SAMPLE PLANS"
    )

    for local_index, row in (
        df.iterrows()
    ):
        dataset_index = int(
            row[
                "_dataset_index"
            ]
        )

        problem_id = get_problem_id(
            row,
            dataset_index,
        )

        messages = normalize_messages(
            row[
                "prompt"
            ]
        )

        # Stable group-level seed.
        group_seed = (
            int(args.seed)
            + dataset_index
            * 100_000
        )

        print(
            f"[{local_index + 1:03d}/"
            f"{len(df):03d}] "
            f"row={dataset_index} "
            f"problem={problem_id}"
        )

        plans = sample_plans(
            tokenizer=tokenizer,
            model=planner,
            messages=messages,
            group_size=group_size,
            max_new_tokens=(
                max_new_tokens
            ),
            temperature=(
                temperature
            ),
            top_p=top_p,
            seed=group_seed,
        )

        if args.show_plans:
            for sample in plans:
                print()
                print(
                    f"--- sample "
                    f"{sample['sample_index']} ---"
                )
                print(
                    sample[
                        "plan"
                    ]
                )

        sampled_groups.append(
            {
                "dataset_index": (
                    dataset_index
                ),
                "problem_id": (
                    problem_id
                ),
                "row": row,
                "plans": plans,
            }
        )

    # =================================================================
    # Stage 2: release planner, start training reward runtime
    # =================================================================

    release_planner(
        planner
    )

    initialize_ray()

    frozen_coder = (
        start_frozen_coder(
            research_config_path
        )
    )

    reward_loop_manager = (
        start_reward_loop_manager(
            verl_config=(
                verl_config
            ),
            frozen_coder_handle=(
                frozen_coder
            ),
        )
    )

    # =================================================================
    # Stage 3: exact reward-loop evaluation
    # =================================================================

    print_header(
        "STAGE 3: ACTUAL TRAINING REWARD PATH"
    )

    group_counts = {
        "all_zero": 0,
        "mixed": 0,
        "all_one": 0,
    }

    total_samples = 0
    total_positive = 0

    binary_flat_progress_variable = 0

    all_rewards: list[
        float
    ] = []

    all_progress: list[
        float
    ] = []

    try:
        for group_index, group in enumerate(
            sampled_groups
        ):
            row = group[
                "row"
            ]

            problem_id = str(
                group[
                    "problem_id"
                ]
            )

            dataset_index = int(
                group[
                    "dataset_index"
                ]
            )

            print()
            print(
                f"[{group_index + 1:03d}/"
                f"{len(sampled_groups):03d}] "
                f"row={dataset_index} "
                f"problem={problem_id}"
            )

            data_items: list[
                DataProto
            ] = []

            for sample in group[
                "plans"
            ]:
                data_items.append(
                    build_reward_dataproto(
                        tokenizer=tokenizer,
                        row=row,
                        response_ids=(
                            sample[
                                "response_ids"
                            ]
                        ),
                    )
                )

            started = (
                time.perf_counter()
            )

            raw_outputs = (
                evaluate_group_via_reward_loop(
                    reward_loop_manager=(
                        reward_loop_manager
                    ),
                    data_items=(
                        data_items
                    ),
                )
            )

            group_wall_time = (
                time.perf_counter()
                - started
            )

            if len(raw_outputs) != len(
                group[
                    "plans"
                ]
            ):
                raise RuntimeError(
                    "Reward output count mismatch: "
                    f"{len(raw_outputs)} != "
                    f"{len(group['plans'])}"
                )

            trajectories: list[
                dict[str, Any]
            ] = []

            rewards: list[
                float
            ] = []

            progresses: list[
                float
            ] = []

            executed_ratios: list[
                float
            ] = []

            for sample, raw_output in zip(
                group["plans"],
                raw_outputs,
                strict=True,
            ):
                result = (
                    normalize_reward_output(
                        raw_output
                    )
                )

                reward = float(
                    result[
                        "reward"
                    ]
                )

                progress = float(
                    result[
                        "reward_test_progress"
                    ]
                )

                executed_ratio = float(
                    result[
                        "executed_prefix_ratio"
                    ]
                )

                rewards.append(
                    reward
                )

                progresses.append(
                    progress
                )

                executed_ratios.append(
                    executed_ratio
                )

                all_rewards.append(
                    reward
                )

                all_progress.append(
                    progress
                )

                total_samples += 1

                if reward > 0:
                    total_positive += 1

                trajectory = {
                    "sample_index": int(
                        sample[
                            "sample_index"
                        ]
                    ),
                    "plan": (
                        sample[
                            "plan"
                        ]
                    ),
                    "plan_completion_tokens": int(
                        sample[
                            "completion_tokens"
                        ]
                    ),
                    **result,
                }

                trajectories.append(
                    trajectory
                )

                print(
                    f"  sample "
                    f"{sample['sample_index'] + 1:02d}/"
                    f"{group_size:02d} "
                    f"R={reward:.0f} "
                    f"progress={progress:.4f} "
                    f"exec={executed_ratio:.4f} "
                    f"tests="
                    f"{result['passed_tests']}/"
                    f"{result['reward_tests']} "
                    f"executed="
                    f"{result['total_tests']} "
                    f"{result['status']}"
                )

            group_type = (
                classify_group(
                    rewards
                )
            )

            group_counts[
                group_type
            ] += 1

            reward_mean = (
                sum(rewards)
                / len(rewards)
            )

            reward_variance = (
                population_variance(
                    rewards
                )
            )

            progress_mean = (
                sum(progresses)
                / len(progresses)
            )

            progress_variance = (
                population_variance(
                    progresses
                )
            )

            executed_mean = (
                sum(executed_ratios)
                / len(executed_ratios)
            )

            executed_variance = (
                population_variance(
                    executed_ratios
                )
            )

            binary_flat = (
                reward_variance
                <= 1e-12
            )

            progress_variable = (
                progress_variance
                > 1e-12
            )

            hidden_partial_signal = (
                binary_flat
                and progress_variable
            )

            if hidden_partial_signal:
                binary_flat_progress_variable += 1

            num_positive = sum(
                reward > 0
                for reward in rewards
            )

            output_record = {
                "checkpoint": (
                    args.checkpoint_label
                ),
                "dataset_index": (
                    dataset_index
                ),
                "problem_id": (
                    problem_id
                ),
                "num_samples": (
                    len(rewards)
                ),
                "num_positive": (
                    num_positive
                ),
                "group_type": (
                    group_type
                ),
                "binary_reward_flat": (
                    binary_flat
                ),
                "binary_flat_progress_variable": (
                    hidden_partial_signal
                ),
                "reward_mean": (
                    reward_mean
                ),
                "reward_variance": (
                    reward_variance
                ),
                "reward_test_progress_mean": (
                    progress_mean
                ),
                "reward_test_progress_variance": (
                    progress_variance
                ),
                "executed_prefix_ratio_mean": (
                    executed_mean
                ),
                "executed_prefix_ratio_variance": (
                    executed_variance
                ),
                "group_reward_wall_time": (
                    group_wall_time
                ),
                "trajectories": (
                    trajectories
                ),
            }

            append_jsonl(
                output_path,
                output_record,
            )

            print(
                "  -> "
                f"{group_type} "
                f"positive={num_positive}/"
                f"{len(rewards)} "
                f"R_mean={reward_mean:.4f} "
                f"R_var={reward_variance:.4f} "
                f"progress_mean={progress_mean:.4f} "
                f"progress_var={progress_variance:.6f} "
                f"time={group_wall_time:.1f}s"
            )

            if hidden_partial_signal:
                print(
                    "     [hidden partial signal] "
                    "binary reward flat, "
                    "reward-test progress varies"
                )

    finally:
        # --------------------------------------------------------------
        # Cleanup
        # --------------------------------------------------------------

        try:
            for worker in (
                reward_loop_manager
                .reward_loop_worker_handles
                or []
            ):
                ray.kill(
                    worker,
                    no_restart=True,
                )
        except Exception:
            pass

        try:
            ray.kill(
                frozen_coder,
                no_restart=True,
            )
        except Exception:
            pass

        if ray.is_initialized():
            ray.shutdown()

    # =================================================================
    # Summary
    # =================================================================

    n_groups = len(
        sampled_groups
    )

    positive_rate = (
        total_positive
        / total_samples
        if total_samples
        else 0.0
    )

    mean_reward = (
        sum(all_rewards)
        / len(all_rewards)
        if all_rewards
        else 0.0
    )

    mean_progress = (
        sum(all_progress)
        / len(all_progress)
        if all_progress
        else 0.0
    )

    print_header(
        "TACO TRAINING RL SIGNAL SUMMARY"
    )

    print(
        f"Checkpoint       : "
        f"{args.checkpoint_label}"
    )
    print(
        f"Problems         : "
        f"{n_groups}"
    )
    print(
        f"Group size       : "
        f"{group_size}"
    )
    print(
        f"Trajectories     : "
        f"{total_samples}"
    )

    print()

    for key in (
        "all_zero",
        "mixed",
        "all_one",
    ):
        count = (
            group_counts[
                key
            ]
        )

        rate = (
            count / n_groups
            if n_groups
            else 0.0
        )

        print(
            f"{key:16s}: "
            f"{count}/{n_groups} "
            f"({100 * rate:.2f}%)"
        )

    print()

    print(
        "Flat-R + variable-progress: "
        f"{binary_flat_progress_variable}/"
        f"{n_groups} "
        f"("
        f"{100 * binary_flat_progress_variable / n_groups:.2f}%"
        f")"
        if n_groups
        else "Flat-R + variable-progress: 0/0"
    )

    print()

    print(
        f"Positive samples : "
        f"{total_positive}/"
        f"{total_samples} "
        f"({100 * positive_rate:.2f}%)"
    )

    print(
        f"Mean binary R    : "
        f"{mean_reward:.6f}"
    )

    print(
        f"Mean progress    : "
        f"{mean_progress:.6f}"
    )

    print()
    print(
        f"Output           : "
        f"{output_path}"
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\nInterrupted.",
            file=sys.stderr,
        )

        if ray.is_initialized():
            ray.shutdown()

        raise