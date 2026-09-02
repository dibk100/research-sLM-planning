"""
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase4_method_discovery/vanilla_planning_rlvr/scripts/sample_planning_rl_signal.py \
  --config phase1_planning_bottleneck/configs/self_plan_qwen25Coder3b.yaml \
  --checkpoint-label step25 \
  --adapter-path /mnt/hdd/project_sLM_planning/checkpoints/vanilla_planning_rlvr_lora_pilot50/exported/step25 \
  --limit 2 \
  --num-samples 4 \
  --output-path phase4_method_discovery/vanilla_planning_rlvr/outputs/planning_rl_signal_sampling/step25_smoke.jsonl

R=1: 모든 테스트 통과 → binary reward 1
TPR=1.0: test pass ratio 100%
H=0.4960: plan을 생성한 236개 토큰의 평균 entropy
tokens=236: plan 생성 토큰 수
PASS: 최종 코드가 전체 테스트 통과

"""
# phase4_method_discovery/vanilla_planning_rlvr/scripts/sample_planning_rl_signal.py

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel

from phase4_method_discovery.vanilla_planning_rlvr.evaluation.rl_planner_strategy import (
    RLPlannerStrategy,
)

from src.datasets.dataset_loader import load_dataset
from src.execution.evaluator import Evaluator
from src.models.generator import ModelGenerator
from src.parsing.code_parser import CodeParser
from src.utils.config import load_config
from src.utils.seed import set_seed


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample stochastic plans from Base / RL planner checkpoints "
            "and measure execution reward, test-pass ratio, and "
            "token-level policy entropy."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Phase 1 self-plan YAML config.",
    )

    parser.add_argument(
        "--checkpoint-label",
        type=str,
        required=True,
        choices=[
            "base",
            "step25",
            "step50",
        ],
        help="Planner checkpoint label.",
    )

    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help=(
            "Exported PEFT LoRA adapter path. "
            "Required for step25/step50, omitted for base."
        ),
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output JSONL path.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of problems to sample.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of sampled plans per problem.",
    )

    parser.add_argument(
        "--plan-temperature",
        type=float,
        default=0.7,
        help="Planner sampling temperature.",
    )

    parser.add_argument(
        "--plan-top-p",
        type=float,
        default=0.95,
        help="Planner top-p.",
    )

    parser.add_argument(
        "--plan-max-new-tokens",
        type=int,
        default=None,
        help=(
            "Override planner max_new_tokens. "
            "Defaults to config value."
        ),
    )

    parser.add_argument(
        "--code-max-new-tokens",
        type=int,
        default=None,
        help=(
            "Override coder max_new_tokens. "
            "Defaults to config value."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output JSONL.",
    )

    return parser.parse_args()


# ============================================================================
# Basic helpers
# ============================================================================


def set_sample_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def population_variance(values: list[float]) -> float:
    if not values:
        return 0.0

    mean = sum(values) / len(values)

    return sum(
        (value - mean) ** 2
        for value in values
    ) / len(values)


def classify_group(
    rewards: list[float],
) -> str:
    if not rewards:
        return "empty"

    num_positive = sum(
        reward > 0.0
        for reward in rewards
    )

    if num_positive == 0:
        return "all_zero"

    if num_positive == len(rewards):
        return "all_one"

    return "mixed"


def get_completed_problem_ids(
    output_path: Path,
) -> set[str]:
    completed: set[str] = set()

    if not output_path.exists():
        return completed

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            problem_id = record.get(
                "problem_id"
            )

            if problem_id:
                completed.add(
                    str(problem_id)
                )

    return completed


def append_jsonl(
    output_path: Path,
    record: dict[str, Any],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
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


# ============================================================================
# LoRA helpers
# ============================================================================


def count_lora_parameter_tensors(
    model: torch.nn.Module,
) -> int:
    return sum(
        1
        for name, _ in model.named_parameters()
        if "lora_" in name
    )


def attach_planner_adapter(
    *,
    planner_generator: ModelGenerator,
    checkpoint_label: str,
    adapter_path: str | None,
) -> None:
    if checkpoint_label == "base":
        if adapter_path is not None:
            raise ValueError(
                "--adapter-path must not be used "
                "with --checkpoint-label base."
            )

        lora_count = (
            count_lora_parameter_tensors(
                planner_generator.model
            )
        )

        if lora_count != 0:
            raise RuntimeError(
                "Base planner unexpectedly contains "
                f"{lora_count} LoRA parameter tensors."
            )

        print(
            "[Planner] base planner: "
            "no LoRA adapter attached."
        )

        return

    if adapter_path is None:
        raise ValueError(
            "--adapter-path is required for "
            f"{checkpoint_label}."
        )

    adapter = Path(adapter_path)

    if not adapter.exists():
        raise FileNotFoundError(
            f"Adapter path not found: {adapter}"
        )

    print(
        "[Planner] attaching RL LoRA adapter..."
    )

    planner_generator.model = (
        PeftModel.from_pretrained(
            planner_generator.model,
            str(adapter),
            is_trainable=False,
        )
    )

    planner_generator.model.eval()

    lora_count = (
        count_lora_parameter_tensors(
            planner_generator.model
        )
    )

    print(
        "[Planner] LoRA parameter tensors: "
        f"{lora_count}"
    )

    if lora_count <= 0:
        raise RuntimeError(
            "No LoRA parameters found after "
            "loading the adapter."
        )

    print(
        "[Planner] RL LoRA attached."
    )


# ============================================================================
# Stochastic planner generation
# ============================================================================


@torch.inference_mode()
def sample_plan(
    *,
    generator: ModelGenerator,
    prompt: str,
    system_prompt: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> dict[str, Any]:
    """
    Sample one plan using the same prompt/chat-template path as
    ModelGenerator.generate().

    Difference from ModelGenerator.generate():
    - stochastic sampling is explicitly used
    - generated token IDs are retained
    - raw-policy token entropy is measured with a second
      teacher-forced forward pass
    """

    if temperature <= 0:
        raise ValueError(
            "Planner diagnostic sampling requires "
            "temperature > 0."
        )

    if not 0 < top_p <= 1:
        raise ValueError(
            "top_p must be in (0, 1]."
        )

    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be > 0."
        )

    # ------------------------------------------------------------------
    # EXACT same chat formatting path as ModelGenerator.generate()
    # ------------------------------------------------------------------

    chat_formatted_prompt = (
        generator.build_chat_prompt(
            user_prompt=prompt,
            system_prompt=system_prompt,
        )
    )

    inputs = generator.tokenizer(
        chat_formatted_prompt,
        return_tensors="pt",
    )

    model_device = next(
        generator.model.parameters()
    ).device

    inputs = {
        key: value.to(model_device)
        for key, value in inputs.items()
    }

    prompt_length = int(
        inputs["input_ids"].shape[1]
    )

    # ------------------------------------------------------------------
    # Stochastic generation
    # ------------------------------------------------------------------

    set_sample_seed(seed)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": (
            generator.tokenizer.pad_token_id
        ),
        "eos_token_id": (
            generator.tokenizer.eos_token_id
        ),
    }

    generation_start = (
        time.perf_counter()
    )

    generated = generator.model.generate(
        **inputs,
        **generation_kwargs,
    )

    generation_seconds = (
        time.perf_counter()
        - generation_start
    )

    sequence = generated[0]

    generated_ids = sequence[
        prompt_length:
    ]

    plan = generator.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    if not plan:
        raise ValueError(
            "Planner generated an empty plan."
        )

    # ------------------------------------------------------------------
    # Teacher-forced raw-policy entropy
    #
    # We intentionally measure entropy from the model's raw logits,
    # rather than the temperature/top-p warped sampling distribution.
    #
    # For generated token y_t:
    #   logits[prompt_length - 1 + t]
    # predicts y_t.
    # ------------------------------------------------------------------

    full_input_ids = (
        sequence.unsqueeze(0)
    )

    attention_mask = (
        torch.ones_like(
            full_input_ids
        )
    )

    entropy_start = (
        time.perf_counter()
    )

    forward_output = generator.model(
        input_ids=full_input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    entropy_forward_seconds = (
        time.perf_counter()
        - entropy_start
    )

    logits = (
        forward_output.logits[0]
        .float()
    )

    token_diagnostics: list[
        dict[str, Any]
    ] = []

    entropies: list[float] = []
    selected_logprobs: list[float] = []

    num_generated_tokens = int(
        generated_ids.shape[0]
    )

    for token_index in range(
        num_generated_tokens
    ):
        prediction_position = (
            prompt_length
            - 1
            + token_index
        )

        token_logits = logits[
            prediction_position
        ]

        log_probs = torch.log_softmax(
            token_logits,
            dim=-1,
        )

        probs = torch.softmax(
            token_logits,
            dim=-1,
        )

        entropy = float(
            -torch.sum(
                probs * log_probs
            ).item()
        )

        token_id = int(
            generated_ids[
                token_index
            ].item()
        )

        selected_logprob = float(
            log_probs[
                token_id
            ].item()
        )

        top1_probability, top1_id = (
            torch.max(
                probs,
                dim=-1,
            )
        )

        top1_id_int = int(
            top1_id.item()
        )

        token_text = (
            generator.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )
        )

        top1_token_text = (
            generator.tokenizer.decode(
                [top1_id_int],
                skip_special_tokens=False,
            )
        )

        token_diagnostics.append(
            {
                "token_index": (
                    token_index
                ),
                "token_id": token_id,
                "token_text": (
                    token_text
                ),
                "entropy": entropy,
                "selected_logprob": (
                    selected_logprob
                ),
                "top1_token_id": (
                    top1_id_int
                ),
                "top1_token": (
                    top1_token_text
                ),
                "top1_probability": float(
                    top1_probability.item()
                ),
            }
        )

        entropies.append(
            entropy
        )

        selected_logprobs.append(
            selected_logprob
        )

    mean_entropy = (
        sum(entropies)
        / len(entropies)
        if entropies
        else 0.0
    )

    max_entropy = (
        max(entropies)
        if entropies
        else 0.0
    )

    min_entropy = (
        min(entropies)
        if entropies
        else 0.0
    )

    mean_logprob = (
        sum(selected_logprobs)
        / len(selected_logprobs)
        if selected_logprobs
        else 0.0
    )

    # Release the largest temporary tensors before coder generation.
    del forward_output
    del logits
    del full_input_ids
    del attention_mask
    del generated

    return {
        "plan": plan,
        "plan_token_count": (
            num_generated_tokens
        ),
        "mean_token_entropy": (
            mean_entropy
        ),
        "max_token_entropy": (
            max_entropy
        ),
        "min_token_entropy": (
            min_entropy
        ),
        "mean_token_logprob": (
            mean_logprob
        ),
        "token_diagnostics": (
            token_diagnostics
        ),
        "generation_seconds": (
            generation_seconds
        ),
        "entropy_forward_seconds": (
            entropy_forward_seconds
        ),
        "chat_formatted_prompt": (
            chat_formatted_prompt
        ),
    }


# ============================================================================
# Code parsing
# ============================================================================


def parse_generated_code(
    parser: CodeParser,
    raw_output: str,
) -> str:
    """
    Support the shared CodeParser while being tolerant to whether parse()
    returns a raw string or a structured parse result.
    """

    parsed = parser.parse(
        raw_output
    )

    if isinstance(parsed, str):
        return parsed

    for attribute in (
        "code",
        "parsed_code",
        "extracted_code",
        "text",
    ):
        if hasattr(
            parsed,
            attribute,
        ):
            value = getattr(
                parsed,
                attribute,
            )

            if isinstance(
                value,
                str,
            ):
                return value

    raise TypeError(
        "Unsupported CodeParser.parse() "
        f"return type: {type(parsed).__name__}"
    )


# ============================================================================
# Evaluation result helpers
# ============================================================================


def get_result_attr(
    result: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(
        result,
        dict,
    ):
        return result.get(
            name,
            default,
        )

    return getattr(
        result,
        name,
        default,
    )


def normalize_status(
    status: Any,
) -> str:
    if status is None:
        return "UNKNOWN"

    if hasattr(
        status,
        "value",
    ):
        return str(
            status.value
        )

    return str(status)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Config
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

    model_config = config[
        "model"
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

    base_seed = int(
        experiment_config.get(
            "seed",
            42,
        )
    )

    set_seed(base_seed)

    if args.limit <= 0:
        raise ValueError(
            "--limit must be > 0."
        )

    if args.num_samples <= 0:
        raise ValueError(
            "--num-samples must be > 0."
        )

    if args.plan_temperature <= 0:
        raise ValueError(
            "--plan-temperature must be > 0."
        )

    if not 0 < args.plan_top_p <= 1:
        raise ValueError(
            "--plan-top-p must be in (0, 1]."
        )

    plan_max_new_tokens = (
        args.plan_max_new_tokens
        if args.plan_max_new_tokens
        is not None
        else generation_config.get(
            "plan_max_new_tokens",
            512,
        )
    )

    code_max_new_tokens = (
        args.code_max_new_tokens
        if args.code_max_new_tokens
        is not None
        else generation_config.get(
            "code_max_new_tokens",
            1024,
        )
    )

    base_model_name = (
        model_config[
            "name_or_path"
        ]
    )

    output_path = Path(
        args.output_path
    )

    # ------------------------------------------------------------------
    # Dataset
    #
    # Same load_dataset() signature as evaluate_rl_planner.py.
    # ------------------------------------------------------------------

    examples = load_dataset(
        dataset_name=dataset_config[
            "name"
        ],
        data_path=dataset_config[
            "path"
        ],
        limit=args.limit,
    )

    print(
        "=" * 100
    )
    print(
        "Planning RL Signal Sampler"
    )
    print(
        "=" * 100
    )

    print(
        f"Config          : {args.config}"
    )

    print(
        f"Checkpoint      : "
        f"{args.checkpoint_label}"
    )

    print(
        f"Adapter         : "
        f"{args.adapter_path}"
    )

    print(
        f"Dataset         : "
        f"{dataset_config['name']}"
    )

    print(
        f"Data path       : "
        f"{dataset_config['path']}"
    )

    print(
        f"Base model      : "
        f"{base_model_name}"
    )

    print(
        f"Base seed       : "
        f"{base_seed}"
    )

    print(
        f"Problems        : "
        f"{len(examples)}"
    )

    print(
        f"Group size      : "
        f"{args.num_samples}"
    )

    print(
        "Plan decoding   : "
        f"T={args.plan_temperature}, "
        f"top_p={args.plan_top_p}, "
        f"max_new_tokens="
        f"{plan_max_new_tokens}"
    )

    print(
        "Coder decoding  : "
        "greedy, "
        f"max_new_tokens="
        f"{code_max_new_tokens}"
    )

    print(
        f"Output          : "
        f"{output_path}"
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    completed_ids: set[str] = set()

    if args.resume:
        completed_ids = (
            get_completed_problem_ids(
                output_path
            )
        )

        print(
            "[Resume] completed problems: "
            f"{len(completed_ids)}"
        )
    else:
        if output_path.exists():
            raise FileExistsError(
                "Output already exists. "
                "Delete it or use --resume: "
                f"{output_path}"
            )

    # ------------------------------------------------------------------
    # Planner
    # ------------------------------------------------------------------

    print(
        "\n[Dataset] loaded "
        f"{len(examples)} examples."
    )

    print(
        "\n[Planner] loading base model..."
    )

    planner_generator = ModelGenerator(
        model_name_or_path=(
            base_model_name
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

    attach_planner_adapter(
        planner_generator=(
            planner_generator
        ),
        checkpoint_label=(
            args.checkpoint_label
        ),
        adapter_path=(
            args.adapter_path
        ),
    )

    # ------------------------------------------------------------------
    # Frozen coder
    # ------------------------------------------------------------------

    print(
        "\n[Coder] loading frozen "
        "base model..."
    )

    coder_generator = ModelGenerator(
        model_name_or_path=(
            base_model_name
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

    coder_lora_count = (
        count_lora_parameter_tensors(
            coder_generator.model
        )
    )

    if coder_lora_count != 0:
        raise RuntimeError(
            "Frozen coder unexpectedly "
            "contains LoRA parameters: "
            f"{coder_lora_count}"
        )

    coder_generator.model.eval()

    print(
        "[Coder] frozen base coder loaded."
    )

    # ------------------------------------------------------------------
    # Exact same prompt builder as evaluation
    # ------------------------------------------------------------------

    strategy = RLPlannerStrategy(
        planner_generator=(
            planner_generator
        ),
        coder_generator=(
            coder_generator
        ),
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
            plan_max_new_tokens
        ),
        code_max_new_tokens=(
            code_max_new_tokens
        ),
        temperature=(
            args.plan_temperature
        ),
        top_p=(
            args.plan_top_p
        ),
    )

    print(
        "\n[Prompt] plan : "
        f"{strategy.plan_prompt_path}"
    )

    print(
        "[Prompt] code : "
        f"{strategy.code_prompt_path}"
    )

    # ------------------------------------------------------------------
    # Parser / evaluator
    # ------------------------------------------------------------------

    parser = CodeParser()

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
    # Global statistics
    # ------------------------------------------------------------------

    analyzed_groups = 0

    num_all_zero = 0
    num_mixed = 0
    num_all_one = 0

    total_samples = 0
    total_positive = 0

    all_tprs: list[float] = []

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    for problem_index, example in enumerate(
        examples
    ):
        problem_id = str(
            example.problem_id
        )

        if problem_id in completed_ids:
            print(
                f"[Skip] {problem_id}"
            )
            continue

        difficulty = getattr(
            example,
            "difficulty",
            None,
        )

        title = getattr(
            example,
            "title",
            None,
        )

        print()

        print(
            f"[{problem_index + 1}/"
            f"{len(examples)}] "
            f"{problem_id} "
            f"({difficulty})"
        )

        # --------------------------------------------------------------
        # IMPORTANT:
        # Exact same user-level plan prompt as RLPlannerStrategy.run().
        # --------------------------------------------------------------

        plan_formatted_prompt = (
            strategy.build_plan_prompt(
                example
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
            # Stable seed mapping across checkpoints.
            #
            # Same problem/sample IDs receive the same RNG seed.
            #
            # This controls randomness but should NOT be interpreted
            # as producing statistically paired trajectories across
            # different model distributions.
            sample_seed = (
                base_seed
                + problem_index * 100_000
                + sample_id
            )

            # ----------------------------------------------------------
            # Plan
            # ----------------------------------------------------------

            plan_result = sample_plan(
                generator=(
                    planner_generator
                ),
                prompt=(
                    plan_formatted_prompt
                ),
                system_prompt=(
                    strategy.system_prompt
                ),
                max_new_tokens=(
                    plan_max_new_tokens
                ),
                temperature=(
                    args.plan_temperature
                ),
                top_p=(
                    args.plan_top_p
                ),
                seed=sample_seed,
            )

            plan = plan_result[
                "plan"
            ]

            # ----------------------------------------------------------
            # Exact same code prompt builder as evaluation.
            # ----------------------------------------------------------

            code_formatted_prompt = (
                strategy.build_code_prompt(
                    example=example,
                    plan=plan,
                )
            )

            # ----------------------------------------------------------
            # Frozen coder: exact ModelGenerator.generate() path.
            #
            # Greedy decoding intentionally mirrors Phase 1 evaluation.
            # ----------------------------------------------------------

            code_start = (
                time.perf_counter()
            )

            code_generation = (
                coder_generator.generate(
                    prompt=(
                        code_formatted_prompt
                    ),
                    system_prompt=(
                        strategy.system_prompt
                    ),
                    max_new_tokens=(
                        code_max_new_tokens
                    ),
                    temperature=0.0,
                    top_p=1.0,
                )
            )

            code_generation_seconds = (
                time.perf_counter()
                - code_start
            )

            raw_code = (
                code_generation.text
            )

            # ----------------------------------------------------------
            # Parse
            # ----------------------------------------------------------

            code = parse_generated_code(
                parser,
                raw_code,
            )

            # ----------------------------------------------------------
            # Evaluate
            # ----------------------------------------------------------

            evaluation_start = (
                time.perf_counter()
            )

            evaluation_result = (
                evaluator.evaluate(
                    example,
                    code,
                )
            )

            evaluation_seconds = (
                time.perf_counter()
                - evaluation_start
            )

            passed = bool(
                get_result_attr(
                    evaluation_result,
                    "passed",
                    False,
                )
            )

            status = normalize_status(
                get_result_attr(
                    evaluation_result,
                    "status",
                    "UNKNOWN",
                )
            )

            test_pass_ratio = float(
                get_result_attr(
                    evaluation_result,
                    "test_pass_ratio",
                    1.0 if passed else 0.0,
                )
            )

            reward = (
                1.0
                if passed
                else 0.0
            )

            rewards.append(
                reward
            )

            tprs.append(
                test_pass_ratio
            )

            all_tprs.append(
                test_pass_ratio
            )

            total_samples += 1

            if reward > 0:
                total_positive += 1

            sample_record = {
                "sample_id": (
                    sample_id
                ),
                "seed": (
                    sample_seed
                ),

                # ------------------------------------------------------
                # Prompt protocol audit
                # ------------------------------------------------------

                "plan_formatted_prompt": (
                    plan_formatted_prompt
                ),

                "plan_chat_formatted_prompt": (
                    plan_result[
                        "chat_formatted_prompt"
                    ]
                ),

                # ------------------------------------------------------
                # Plan
                # ------------------------------------------------------

                "plan": plan,

                "plan_token_count": (
                    plan_result[
                        "plan_token_count"
                    ]
                ),

                "mean_token_entropy": (
                    plan_result[
                        "mean_token_entropy"
                    ]
                ),

                "max_token_entropy": (
                    plan_result[
                        "max_token_entropy"
                    ]
                ),

                "min_token_entropy": (
                    plan_result[
                        "min_token_entropy"
                    ]
                ),

                "mean_token_logprob": (
                    plan_result[
                        "mean_token_logprob"
                    ]
                ),

                "token_diagnostics": (
                    plan_result[
                        "token_diagnostics"
                    ]
                ),

                # ------------------------------------------------------
                # Code
                # ------------------------------------------------------

                "code_formatted_prompt": (
                    code_formatted_prompt
                ),

                "code_raw": (
                    raw_code
                ),

                "code": (
                    code
                ),

                # ------------------------------------------------------
                # Evaluation
                # ------------------------------------------------------

                "passed": (
                    passed
                ),

                "status": (
                    status
                ),

                "test_pass_ratio": (
                    test_pass_ratio
                ),

                "reward": (
                    reward
                ),

                # ------------------------------------------------------
                # Timing
                # ------------------------------------------------------

                "plan_generation_seconds": (
                    plan_result[
                        "generation_seconds"
                    ]
                ),

                "entropy_forward_seconds": (
                    plan_result[
                        "entropy_forward_seconds"
                    ]
                ),

                "code_generation_seconds": (
                    code_generation_seconds
                ),

                "evaluation_seconds": (
                    evaluation_seconds
                ),
            }

            samples.append(
                sample_record
            )

            print(
                "  sample "
                f"{sample_id + 1:02d}/"
                f"{args.num_samples}: "
                f"R={int(reward)} "
                f"TPR={test_pass_ratio:.4f} "
                f"H="
                f"{plan_result['mean_token_entropy']:.4f} "
                f"tokens="
                f"{plan_result['plan_token_count']} "
                f"status={status}"
            )

        # --------------------------------------------------------------
        # Group statistics
        # --------------------------------------------------------------

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

        group_type = classify_group(
            rewards
        )

        if group_type == "all_zero":
            num_all_zero += 1
        elif group_type == "mixed":
            num_mixed += 1
        elif group_type == "all_one":
            num_all_one += 1

        analyzed_groups += 1

        record = {
            "checkpoint": (
                args.checkpoint_label
            ),

            "problem_id": (
                problem_id
            ),

            "title": (
                title
            ),

            "difficulty": (
                difficulty
            ),

            "num_samples": (
                args.num_samples
            ),

            "num_positive": (
                num_positive
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

            "group_type": (
                group_type
            ),

            "samples": (
                samples
            ),
        }

        append_jsonl(
            output_path,
            record,
        )

        print(
            "  -> "
            f"group={group_type}, "
            f"positive="
            f"{num_positive}/"
            f"{args.num_samples}, "
            f"R_mean="
            f"{reward_mean:.4f}, "
            f"R_var="
            f"{reward_variance:.4f}, "
            f"TPR_mean="
            f"{tpr_mean:.4f}, "
            f"TPR_var="
            f"{tpr_variance:.4f}"
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print()

    print(
        "=" * 100
    )

    print(
        "Planning RL Signal Sampling Complete"
    )

    print(
        "=" * 100
    )

    print(
        f"Checkpoint       : "
        f"{args.checkpoint_label}"
    )

    print(
        f"Groups analyzed  : "
        f"{analyzed_groups}"
    )

    if analyzed_groups > 0:
        print(
            "All-zero groups  : "
            f"{num_all_zero} "
            f"({100.0 * num_all_zero / analyzed_groups:.2f}%)"
        )

        print(
            "Mixed groups     : "
            f"{num_mixed} "
            f"({100.0 * num_mixed / analyzed_groups:.2f}%)"
        )

        print(
            "All-one groups   : "
            f"{num_all_one} "
            f"({100.0 * num_all_one / analyzed_groups:.2f}%)"
        )

    if total_samples > 0:
        print(
            "Positive samples : "
            f"{total_positive}/"
            f"{total_samples} "
            f"({100.0 * total_positive / total_samples:.2f}%)"
        )

    mean_tpr = (
        sum(all_tprs)
        / len(all_tprs)
        if all_tprs
        else 0.0
    )

    print(
        f"Mean TPR         : "
        f"{mean_tpr:.6f}"
    )

    print(
        f"Output           : "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()