"""
PYTHONPATH="$PWD:$HOME/workspace/LiveCodeBench" \
python \
phase4_method_discovery/vanilla_planning_rlvr/scripts/evaluate_checkpoint.py \
  --mode base \
  --checkpoint-label base \
  --limit 3 \
  --output-path \
phase4_method_discovery/vanilla_planning_rlvr/outputs/checkpoint_eval/base_val3.jsonl \
  --no-resume
"""
# phase4_method_discovery/vanilla_planning_rlvr/scripts/evaluate_checkpoint.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.execution.taco_evaluator import TACOEvaluator

DEFAULT_MODEL = (
    "Qwen/Qwen2.5-Coder-3B-Instruct"
)

DEFAULT_VAL_PARQUET = (
    "/mnt/hdd/project_sLM_planning/"
    "data/deepcoder_taco/processed/"
    "vanilla_planning_rlvr_scale1000/"
    "val.parquet"
)

DEFAULT_PLAN_PROMPT = (
    "prompt_templates/self_plan_plan.txt"
)

DEFAULT_CODE_PROMPT = (
    "prompt_templates/self_plan_code.txt"
)

class FullTACOEvaluator:
    def __init__(
        self,
        *,
        timeout_seconds: int,
        debug: bool = False,
    ) -> None:
        self.backend = TACOEvaluator(
            timeout_seconds=timeout_seconds,
            debug=True,
        )

    def evaluate(
        self,
        problem,
        code,
    ):
        return self.backend.evaluate_non_fail_fast(
            problem=problem,
            code=code,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Base or RL-trained planner "
            "on the held-out Phase 4 validation set."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "base",
            "rl",
        ],
        required=True,
    )

    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help=(
            "PEFT adapter directory. "
            "Required when --mode rl."
        ),
    )

    parser.add_argument(
        "--checkpoint-label",
        type=str,
        required=True,
        help=(
            "Human-readable label, "
            "e.g. base or step900."
        ),
    )

    parser.add_argument(
        "--val-parquet",
        type=str,
        default=DEFAULT_VAL_PARQUET,
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--plan-prompt",
        type=str,
        default=DEFAULT_PLAN_PROMPT,
    )

    parser.add_argument(
        "--code-prompt",
        type=str,
        default=DEFAULT_CODE_PROMPT,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
    )

    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
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
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=6,
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    if args.mode == "rl":
        if not args.adapter_path:
            raise ValueError(
                "--adapter-path is required "
                "when --mode rl."
            )

        adapter_path = Path(
            args.adapter_path
        )

        if not adapter_path.exists():
            raise FileNotFoundError(
                "Adapter directory not found: "
                f"{adapter_path}"
            )

        for filename in [
            "adapter_config.json",
            "adapter_model.safetensors",
        ]:
            path = (
                adapter_path
                / filename
            )

            if not path.exists():
                raise FileNotFoundError(
                    "Required adapter file "
                    f"not found: {path}"
                )

    elif args.adapter_path:
        raise ValueError(
            "--adapter-path must not be "
            "provided in base mode."
        )


def build_generator(
    *,
    model_name: str,
    dtype: str,
    device_map: str,
):
    from src.models.generator import ModelGenerator

    return ModelGenerator(
        model_name_or_path=model_name,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )


def count_lora_tensors(
    generator,
) -> int:
    return sum(
        1
        for name, _ in generator.model.named_parameters()
        if "lora_" in name
    )


def main() -> None:
    from peft import PeftModel

    from phase1_planning_bottleneck.runner import (
        Phase1Runner,
    )

    from phase4_method_discovery.vanilla_planning_rlvr.evaluation.checkpoint_dataset import (
        load_checkpoint_eval_dataset,
    )

    from phase4_method_discovery.vanilla_planning_rlvr.evaluation.rl_planner_strategy import (
        RLPlannerStrategy,
    )

    from src.parsing.code_parser import CodeParser
    from src.utils.seed import set_seed

    args = parse_args()

    validate_args(args)

    set_seed(
        args.seed
    )

    output_path = Path(
        args.output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Evaluation dataset
    # --------------------------------------------------------------

    examples = (
        load_checkpoint_eval_dataset(
            args.val_parquet,
            limit=args.limit,
        )
    )

    # --------------------------------------------------------------
    # Header
    # --------------------------------------------------------------

    print("=" * 80)
    print(
        "Phase 4 Checkpoint Evaluation"
    )
    print("=" * 80)

    print(
        f"Mode            : "
        f"{args.mode}"
    )
    print(
        f"Checkpoint      : "
        f"{args.checkpoint_label}"
    )
    print(
        f"Base model      : "
        f"{args.base_model}"
    )
    print(
        f"Evaluation set  : "
        f"{args.val_parquet}"
    )
    print(
        f"Problems        : "
        f"{len(examples)}"
    )
    print(
        f"Planner decode  : "
        f"T={args.temperature}, "
        f"top_p={args.top_p}, "
        f"max={args.plan_max_new_tokens}"
    )
    print(
        f"Coder decode    : "
        f"T={args.temperature}, "
        f"top_p={args.top_p}, "
        f"max={args.code_max_new_tokens}"
    )
    print(
        f"Output          : "
        f"{output_path}"
    )

    # --------------------------------------------------------------
    # Planner
    # --------------------------------------------------------------

    print()
    print(
        "[Planner] loading base model..."
    )

    planner_generator = (
        build_generator(
            model_name=args.base_model,
            dtype=args.dtype,
            device_map=args.device_map,
        )
    )

    if args.mode == "rl":
        print(
            "[Planner] attaching "
            "RL LoRA..."
        )

        planner_generator.model = (
            PeftModel.from_pretrained(
                planner_generator.model,
                args.adapter_path,
                is_trainable=False,
            )
        )

        planner_generator.model.eval()

        lora_count = (
            count_lora_tensors(
                planner_generator
            )
        )

        print(
            "[Planner] LoRA tensors : "
            f"{lora_count}"
        )

        if lora_count == 0:
            raise RuntimeError(
                "RL planner contains no "
                "LoRA tensors."
            )

    else:
        planner_generator.model.eval()

        lora_count = (
            count_lora_tensors(
                planner_generator
            )
        )

        if lora_count != 0:
            raise RuntimeError(
                "Base planner unexpectedly "
                "contains LoRA tensors."
            )

        print(
            "[Planner] base policy loaded."
        )

    # --------------------------------------------------------------
    # Frozen coder
    # --------------------------------------------------------------

    print()
    print(
        "[Coder] loading frozen "
        "base model..."
    )

    coder_generator = (
        build_generator(
            model_name=args.base_model,
            dtype=args.dtype,
            device_map=args.device_map,
        )
    )

    coder_generator.model.eval()

    coder_lora_count = (
        count_lora_tensors(
            coder_generator
        )
    )

    if coder_lora_count != 0:
        raise RuntimeError(
            "Frozen coder unexpectedly "
            "contains LoRA tensors."
        )

    print(
        "[Coder] frozen base "
        "coder loaded."
    )

    # --------------------------------------------------------------
    # Strategy
    # --------------------------------------------------------------

    strategy = RLPlannerStrategy(
        planner_generator=(
            planner_generator
        ),
        coder_generator=(
            coder_generator
        ),
        plan_prompt_path=(
            args.plan_prompt
        ),
        code_prompt_path=(
            args.code_prompt
        ),
        system_prompt=None,
        plan_max_new_tokens=(
            args.plan_max_new_tokens
        ),
        code_max_new_tokens=(
            args.code_max_new_tokens
        ),
        temperature=(
            args.temperature
        ),
        top_p=(
            args.top_p
        ),
    )

    # --------------------------------------------------------------
    # Parser / evaluator
    # --------------------------------------------------------------

    parser = CodeParser()

    evaluator = FullTACOEvaluator(
        timeout_seconds=args.timeout_seconds,
        debug=False,
    )

    # --------------------------------------------------------------
    # Runner
    # --------------------------------------------------------------

    model_label = (
        args.base_model
        if args.mode == "base"
        else (
            f"{args.base_model}"
            f"+RL-LoRA-"
            f"{args.checkpoint_label}"
        )
    )

    runner = Phase1Runner(
        strategy=strategy,
        evaluator=evaluator,
        parser=parser,
        output_path=output_path,
        model_name=model_label,
        seed=args.seed,
        resume=(
            not args.no_resume
        ),
    )

    summary = runner.run(
        examples
    )

    # --------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "Checkpoint Evaluation Complete"
    )
    print("=" * 80)

    print(
        f"Mode       : {args.mode}"
    )
    print(
        f"Checkpoint : "
        f"{args.checkpoint_label}"
    )
    print(
        f"Selected   : "
        f"{summary.selected}"
    )
    print(
        f"Processed  : "
        f"{summary.processed}"
    )
    print(
        f"Skipped    : "
        f"{summary.skipped}"
    )
    print(
        f"Passed     : "
        f"{summary.passed}"
    )

    if summary.processed > 0:
        print(
            f"Pass rate  : "
            f"{summary.pass_rate:.4%}"
        )

    print(
        f"Results    : "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()