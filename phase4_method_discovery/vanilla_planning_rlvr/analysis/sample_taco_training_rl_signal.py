"""

"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

from src.models.generator import ModelGenerator

# IMPORTANT:
# These imports should point to the SAME modules used by the actual
# Phase-4 RL training pipeline.
from phase4_method_discovery.vanilla_planning_rlvr.reward.planning_execution_reward import (
    compute_score,
)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample G plans per TACO training problem using a planner checkpoint, "
            "run the frozen coder, and measure binary reward + test-pass ratio "
            "under the actual training reward distribution."
        )
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the exact TACO/DeepCoder training dataset used for RL.",
    )

    parser.add_argument(
        "--planner-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-3B-Instruct",
    )

    parser.add_argument(
        "--coder-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-3B-Instruct",
    )

    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="PEFT adapter path. Omit for base checkpoint.",
    )

    parser.add_argument(
        "--checkpoint-label",
        type=str,
        default="step25",
    )

    parser.add_argument(
        "--num-problems",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "Use the SAME planner rollout temperature as RL training. "
            "Default=1.0 from the Phase-4 GRPO config."
        ),
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Use the same top_p as RL rollout if explicitly configured.",
    )

    parser.add_argument(
        "--plan-max-new-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--code-max-new-tokens",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
    )

    return parser.parse_args()


# =============================================================================
# Utilities
# =============================================================================


def population_variance(values: list[float]) -> float:
    if not values:
        return 0.0

    m = sum(values) / len(values)

    return sum(
        (x - m) ** 2
        for x in values
    ) / len(values)


def stable_sample_seed(
    base_seed: int,
    problem_index: int,
    sample_id: int,
) -> int:
    return (
        base_seed
        + problem_index * 100_000
        + sample_id
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_jsonl(
    path: Path,
    record: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


# =============================================================================
# Dataset
# =============================================================================


def load_taco_training_records(
    dataset_path: str,
    num_problems: int,
) -> list[dict[str, Any]]:
    """
    Load EXACTLY the same TACO training records consumed by Phase-4 RL.

    This loader intentionally supports parquet/json/jsonl, but it does NOT
    reinterpret the reward fields. The original record must be passed through
    to the actual training reward function.
    """

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)

        records = df.to_dict(
            orient="records"
        )

    elif path.suffix == ".jsonl":
        records = []

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()

                if line:
                    records.append(
                        json.loads(line)
                    )

    elif path.suffix == ".json":
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if isinstance(data, list):
            records = data
        else:
            raise ValueError(
                "Expected top-level JSON list."
            )

    else:
        raise ValueError(
            f"Unsupported dataset format: {path.suffix}"
        )

    return records[:num_problems]


# =============================================================================
# Prompt extraction
# =============================================================================


def extract_prompt(
    record: dict[str, Any],
) -> str:
    """
    IMPORTANT:
    Prefer the already-materialized prompt from the RL dataset.

    verl training datasets normally store the exact rollout prompt.
    We should not rebuild it from TACO problem text because that can
    introduce protocol drift.
    """

    prompt = record.get("prompt")

    if prompt is None:
        raise KeyError(
            "TACO training record has no 'prompt' field. "
            "Do not invent a new prompt here. Inspect the actual RL dataset "
            "schema and reuse the same prompt-building path as training."
        )

    # Some verl datasets store chat messages rather than plain text.
    if isinstance(prompt, str):
        return prompt

    if isinstance(prompt, list):
        # Return marker; actual formatting will be handled below.
        return prompt

    raise TypeError(
        f"Unsupported prompt type: {type(prompt)}"
    )


def build_planner_chat_prompt(
    generator: ModelGenerator,
    record: dict[str, Any],
) -> str:
    """
    Preserve the actual RL rollout prompt as closely as possible.
    """

    prompt = extract_prompt(record)

    if isinstance(prompt, str):
        # If the dataset already stores a fully rendered prompt, use it
        # directly only if that is what training does.
        #
        # If training stores the raw user prompt instead, replace this branch
        # with the exact training-side chat-template call.
        return prompt

    # verl-style list[dict] chat prompt
    tokenizer = generator.tokenizer

    return tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )


# =============================================================================
# Planner generation
# =============================================================================


@torch.inference_mode()
def generate_plan(
    generator: ModelGenerator,
    formatted_prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """
    Direct generation from the already-formatted training prompt.

    We intentionally do NOT call ModelGenerator.generate() here if that method
    applies another chat template, because double templating would invalidate
    the diagnostic.
    """

    tokenizer = generator.tokenizer
    model = generator.model

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )

    device = next(
        model.parameters()
    ).device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if temperature > 0.0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p

    output_ids = model.generate(
        **inputs,
        **kwargs,
    )

    prompt_length = inputs[
        "input_ids"
    ].shape[1]

    generated_ids = output_ids[
        0,
        prompt_length:,
    ]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )


# =============================================================================
# Frozen coder
# =============================================================================


def build_coder_prompt_from_training_record(
    record: dict[str, Any],
    plan: str,
) -> tuple[str, str | None]:
    """
    THIS FUNCTION MUST MATCH frozen_coder_worker.py.

    The exact Phase-4 worker should ideally expose its prompt builder as a
    reusable function. Until then, do NOT create a new prompt format here.

    Expected approach:
        1. retrieve original problem text from record / extra_info
        2. inject sampled plan
        3. use the same system/user prompt templates as FrozenCoderWorker

    Replace the field accesses below only if your training parquet uses
    different names.
    """

    extra_info = record.get(
        "extra_info",
        {},
    )

    problem = (
        extra_info.get("problem")
        or extra_info.get("question")
        or extra_info.get("problem_statement")
    )

    if problem is None:
        raise KeyError(
            "Could not find original problem text in record['extra_info']. "
            "Reuse the exact extraction logic from frozen_coder_worker.py."
        )

    user_prompt = f"""Solve the following programming problem using the provided plan.

Problem:
{problem}

Plan:
{plan}

Return only the final code.
"""

    system_prompt = (
        "You are an expert competitive programming assistant."
    )

    return (
        user_prompt,
        system_prompt,
    )


def generate_code(
    coder_generator: ModelGenerator,
    record: dict[str, Any],
    plan: str,
    *,
    max_new_tokens: int,
) -> str:
    """
    Greedy frozen-coder generation.

    IMPORTANT:
    build_coder_prompt_from_training_record() must be made byte/protocol
    equivalent to FrozenCoderWorker before using the results scientifically.
    """

    user_prompt, system_prompt = (
        build_coder_prompt_from_training_record(
            record,
            plan,
        )
    )

    return coder_generator.generate(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )


# =============================================================================
# Exact training reward
# =============================================================================


def evaluate_with_training_reward(
    record: dict[str, Any],
    code: str,
) -> dict[str, Any]:
    """
    Run the SAME reward function as Phase-4 RL training.

    The exact argument mapping depends on the current
    planning_execution_reward.compute_score() signature.

    Keep BOTH:
        binary_reward
        test_pass_ratio

    The reward must be exactly what GRPO sees.
    """

    reward_data = (
        record.get("reward_model")
        or record.get("reward_data")
    )

    if reward_data is None:
        raise KeyError(
            "No reward_model/reward_data found in TACO training record. "
            "Inspect the training parquet schema and map the exact field "
            "used by planning_execution_reward.py."
        )

    # ------------------------------------------------------------------
    # IMPORTANT ADAPTER POINT
    #
    # Change ONLY this call if compute_score() has a different signature.
    #
    # Do not replace the reward implementation itself.
    # ------------------------------------------------------------------

    result = compute_score(
        solution_str=code,
        ground_truth=reward_data,
    )

    # Support common return shapes.
    if isinstance(result, (int, float)):
        binary_reward = float(result)

        return {
            "reward": binary_reward,
            "test_pass_ratio": binary_reward,
            "status": (
                "PASS"
                if binary_reward >= 1.0
                else "FAIL"
            ),
            "raw_reward_result": result,
        }

    if not isinstance(result, dict):
        raise TypeError(
            "Unexpected compute_score() result type: "
            f"{type(result)}"
        )

    reward = result.get(
        "reward",
        result.get(
            "score",
            result.get(
                "acc",
                0.0,
            ),
        ),
    )

    tpr = result.get(
        "test_pass_ratio",
        result.get(
            "tpr",
            result.get(
                "pass_ratio",
                None,
            ),
        ),
    )

    if tpr is None:
        raise KeyError(
            "Training reward result does not expose test_pass_ratio. "
            "For this diagnostic, modify planning_execution_reward.py to "
            "optionally return passed_tests / total_tests in addition to "
            "the unchanged binary training reward."
        )

    return {
        "reward": float(reward),
        "test_pass_ratio": float(tpr),
        "status": str(
            result.get(
                "status",
                "PASS"
                if float(reward) >= 1.0
                else "FAIL",
            )
        ),
        "raw_reward_result": result,
    }


# =============================================================================
# Model loading
# =============================================================================


def load_generator(
    model_name: str,
    adapter_path: str | None = None,
) -> ModelGenerator:
    generator = ModelGenerator(
        model_name_or_path=model_name,
        dtype="bfloat16",
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path is not None:
        print(
            f"Loading planner adapter: {adapter_path}"
        )

        generator.model = PeftModel.from_pretrained(
            generator.model,
            adapter_path,
            is_trainable=False,
        )

        generator.model.eval()

        lora_count = sum(
            1
            for name, _ in generator.model.named_parameters()
            if "lora_" in name
        )

        print(
            f"LoRA parameter tensors: {lora_count}"
        )

    return generator


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    output_path = Path(
        args.output_path
    )

    if output_path.exists():
        output_path.unlink()

    print("=" * 100)
    print("TACO TRAINING REWARD DISTRIBUTION DIAGNOSTIC")
    print("=" * 100)

    print(
        f"Checkpoint       : {args.checkpoint_label}"
    )
    print(
        f"Dataset          : {args.dataset_path}"
    )
    print(
        f"Problems         : {args.num_problems}"
    )
    print(
        f"Group size       : {args.num_samples}"
    )
    print(
        f"Planner T        : {args.temperature}"
    )
    print(
        f"Planner top_p    : {args.top_p}"
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    records = load_taco_training_records(
        args.dataset_path,
        args.num_problems,
    )

    print(
        f"Loaded records   : {len(records)}"
    )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    print()
    print("Loading planner...")

    planner = load_generator(
        args.planner_model,
        args.adapter_path,
    )

    print()
    print("Loading frozen coder...")

    coder = load_generator(
        args.coder_model,
        None,
    )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    group_counts = {
        "all_zero": 0,
        "mixed": 0,
        "all_one": 0,
    }

    total_positive = 0
    total_samples = 0

    binary_flat_tpr_variable = 0

    for problem_index, record in enumerate(
        records
    ):
        problem_id = str(
            record.get(
                "problem_id",
                record.get(
                    "id",
                    record.get(
                        "index",
                        problem_index,
                    ),
                ),
            )
        )

        print()
        print(
            f"[{problem_index + 1}/{len(records)}] "
            f"{problem_id}"
        )

        planner_prompt = (
            build_planner_chat_prompt(
                planner,
                record,
            )
        )

        samples: list[
            dict[str, Any]
        ] = []

        rewards: list[float] = []
        tprs: list[float] = []

        for sample_id in range(
            args.num_samples
        ):
            seed = stable_sample_seed(
                args.seed,
                problem_index,
                sample_id,
            )

            set_seed(seed)

            start = time.time()

            plan = generate_plan(
                planner,
                planner_prompt,
                max_new_tokens=(
                    args.plan_max_new_tokens
                ),
                temperature=(
                    args.temperature
                ),
                top_p=args.top_p,
            )

            code = generate_code(
                coder,
                record,
                plan,
                max_new_tokens=(
                    args.code_max_new_tokens
                ),
            )

            evaluation = (
                evaluate_with_training_reward(
                    record,
                    code,
                )
            )

            elapsed = (
                time.time() - start
            )

            reward = float(
                evaluation["reward"]
            )

            tpr = float(
                evaluation[
                    "test_pass_ratio"
                ]
            )

            rewards.append(reward)
            tprs.append(tpr)

            total_samples += 1

            if reward > 0.0:
                total_positive += 1

            sample_record = {
                "sample_id": sample_id,
                "seed": seed,
                "plan": plan,
                "code": code,
                "reward": reward,
                "test_pass_ratio": tpr,
                "status": evaluation[
                    "status"
                ],
                "elapsed_sec": elapsed,
            }

            samples.append(
                sample_record
            )

            print(
                f"  sample "
                f"{sample_id + 1:02d}/"
                f"{args.num_samples:02d}: "
                f"R={reward:.0f} "
                f"TPR={tpr:.4f} "
                f"status={evaluation['status']}"
            )

        # ------------------------------------------------------------------
        # Group statistics
        # ------------------------------------------------------------------

        num_positive = sum(
            reward > 0.0
            for reward in rewards
        )

        reward_mean = (
            sum(rewards)
            / len(rewards)
        )

        reward_variance = (
            population_variance(
                rewards
            )
        )

        tpr_mean = (
            sum(tprs)
            / len(tprs)
        )

        tpr_variance = (
            population_variance(
                tprs
            )
        )

        if num_positive == 0:
            group_type = "all_zero"

        elif num_positive == len(
            rewards
        ):
            group_type = "all_one"

        else:
            group_type = "mixed"

        group_counts[
            group_type
        ] += 1

        if (
            reward_variance <= 1e-12
            and tpr_variance > 1e-12
        ):
            binary_flat_tpr_variable += 1

        group_record = {
            "checkpoint": (
                args.checkpoint_label
            ),
            "problem_index": (
                problem_index
            ),
            "problem_id": problem_id,
            "num_samples": (
                len(samples)
            ),
            "num_positive": (
                num_positive
            ),
            "group_type": (
                group_type
            ),
            "reward_mean": (
                reward_mean
            ),
            "reward_variance": (
                reward_variance
            ),
            "tpr_mean": (
                tpr_mean
            ),
            "tpr_variance": (
                tpr_variance
            ),
            "samples": samples,
        }

        write_jsonl(
            output_path,
            group_record,
        )

        print(
            "  -> "
            f"group={group_type}, "
            f"positive={num_positive}/"
            f"{len(samples)}, "
            f"R_mean={reward_mean:.4f}, "
            f"R_var={reward_variance:.4f}, "
            f"TPR_mean={tpr_mean:.4f}, "
            f"TPR_var={tpr_variance:.4f}"
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    n_groups = len(records)

    print()
    print("=" * 100)
    print(
        "TACO Training Reward Diagnostic Complete"
    )
    print("=" * 100)

    print(
        f"Checkpoint       : "
        f"{args.checkpoint_label}"
    )

    print(
        f"Groups analyzed  : "
        f"{n_groups}"
    )

    for group_type in (
        "all_zero",
        "mixed",
        "all_one",
    ):
        count = group_counts[
            group_type
        ]

        print(
            f"{group_type:16s}: "
            f"{count}/{n_groups} "
            f"({100.0 * count / n_groups:.2f}%)"
        )

    print(
        "Flat-R / variable-TPR groups: "
        f"{binary_flat_tpr_variable}/"
        f"{n_groups} "
        f"({100.0 * binary_flat_tpr_variable / n_groups:.2f}%)"
    )

    print(
        "Positive samples : "
        f"{total_positive}/"
        f"{total_samples} "
        f"({100.0 * total_positive / total_samples:.2f}%)"
    )

    print(
        f"Output           : "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()