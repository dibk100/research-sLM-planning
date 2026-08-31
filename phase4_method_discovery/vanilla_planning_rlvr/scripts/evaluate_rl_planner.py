"""
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase4_method_discovery/vanilla_planning_rlvr/scripts/evaluate_rl_planner.py \
  --config phase1_planning_bottleneck/configs/self_plan_qwen25Coder3b.yaml \
  --adapter-path /mnt/hdd/project_sLM_planning/checkpoints/vanilla_planning_rlvr_lora_pilot50/exported/step50 \
  --checkpoint-label step50 \
  --output-path /mnt/hdd/project_sLM_planning/output/phase4_rl_planner_eval/step50/results.jsonl
  
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase4_method_discovery/vanilla_planning_rlvr/scripts/evaluate_rl_planner.py \
  --config phase1_planning_bottleneck/configs/self_plan_qwen25Coder3b.yaml \
  --adapter-path /mnt/hdd/project_sLM_planning/checkpoints/vanilla_planning_rlvr_lora_pilot50/exported/step25 \
  --checkpoint-label step25 \
  --output-path /mnt/hdd/project_sLM_planning/output/phase4_rl_planner_eval/step25/results.jsonl
"""
# phase4_method_discovery/vanilla_planning_rlvr/scripts/evaluate_rl_planner.py

from __future__ import annotations

import argparse
from pathlib import Path

from peft import PeftModel

from phase1_planning_bottleneck.runner import Phase1Runner

from phase4_method_discovery.vanilla_planning_rlvr.evaluation.rl_planner_strategy import (
    RLPlannerStrategy,
)

from src.datasets.dataset_loader import load_dataset
from src.execution.evaluator import Evaluator
from src.models.generator import ModelGenerator
from src.parsing.code_parser import CodeParser
from src.utils.config import load_config
from src.utils.run_metadata import (
    save_run_config,
    save_run_metadata,
)
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an RL-trained LoRA planner using the "
            "Phase 1 Self-Plan evaluation protocol."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Phase 1 Qwen2.5-Coder-3B Self-Plan YAML config. "
            "Dataset, prompts, decoding, and evaluator settings "
            "are reused from this config."
        ),
    )

    parser.add_argument(
        "--adapter-path",
        required=True,
        help=(
            "Path to the exported PEFT LoRA adapter "
            "(e.g. exported/step50)."
        ),
    )

    parser.add_argument(
        "--checkpoint-label",
        required=True,
        help=(
            "Human-readable checkpoint label, "
            "e.g. step25 or step50."
        ),
    )

    parser.add_argument(
        "--output-path",
        required=True,
        help="Output JSONL path.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override dataset limit from the Phase 1 config.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable JSONL resume.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    experiment_config = config["experiment"]
    dataset_config = config["dataset"]
    model_config = config["model"]
    generation_config = config["generation"]
    strategy_config = config["strategy"]
    evaluation_config = config["evaluation"]

    # ------------------------------------------------------------------
    # Paths / validation
    # ------------------------------------------------------------------

    adapter_path = Path(
        args.adapter_path
    ).resolve()

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_path}"
        )

    adapter_config_path = (
        adapter_path / "adapter_config.json"
    )
    adapter_model_path = (
        adapter_path / "adapter_model.safetensors"
    )

    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"adapter_config.json not found: "
            f"{adapter_config_path}"
        )

    if not adapter_model_path.exists():
        raise FileNotFoundError(
            f"adapter_model.safetensors not found: "
            f"{adapter_model_path}"
        )

    output_path = Path(
        args.output_path
    )

    # ------------------------------------------------------------------
    # Seed / CLI overrides
    # ------------------------------------------------------------------

    seed = int(
        experiment_config["seed"]
    )

    set_seed(seed)

    dataset_limit = (
        args.limit
        if args.limit is not None
        else dataset_config.get("limit")
    )

    output_config = config["output"]

    resume = (
        bool(
            output_config.get(
                "resume",
                True,
            )
        )
        and not args.no_resume
    )
    # ------------------------------------------------------------------
    # Model identity
    # ------------------------------------------------------------------

    base_model_name = (
        model_config["name_or_path"]
    )

    evaluation_model_name = (
        f"{base_model_name}"
        f"+RL-LoRA-{args.checkpoint_label}"
    )

    # ------------------------------------------------------------------
    # Evaluation header
    # ------------------------------------------------------------------

    print("=" * 80)
    print("Phase 4 RL Planner Evaluation")
    print("=" * 80)

    print(
        f"Phase 1 config : {args.config}"
    )
    print(
        f"Dataset        : "
        f"{dataset_config['name']}"
    )
    print(
        f"Data path      : "
        f"{dataset_config['path']}"
    )
    print(
        f"Base model     : "
        f"{base_model_name}"
    )
    print(
        f"RL adapter     : "
        f"{adapter_path}"
    )
    print(
        f"Checkpoint     : "
        f"{args.checkpoint_label}"
    )
    print(
        f"Seed           : {seed}"
    )
    print(
        f"Limit          : {dataset_limit}"
    )
    print(
        f"Output         : {output_path}"
    )
    print(
        f"Resume         : {resume}"
    )

    print()

    print(
        "Evaluation protocol:"
    )
    print(
        "  planner = base model + RL LoRA"
    )
    print(
        "  coder   = frozen base model"
    )
    print(
        "  prompts / decoding / evaluator "
        "= Phase 1 Self-Plan"
    )

    print()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    examples = load_dataset(
        dataset_name=dataset_config["name"],
        data_path=dataset_config["path"],
        limit=dataset_limit,
    )

    print(
        f"[Dataset] loaded {len(examples)} examples."
    )

    # ------------------------------------------------------------------
    # Planner generator
    # ------------------------------------------------------------------

    print()
    print(
        "[Planner] loading base model..."
    )

    planner_generator = ModelGenerator(
        model_name_or_path=base_model_name,
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
            True,
        ),
    )

    print(
        "[Planner] attaching RL LoRA adapter..."
    )

    planner_generator.model = (
        PeftModel.from_pretrained(
            planner_generator.model,
            str(adapter_path),
            is_trainable=False,
        )
    )

    planner_generator.model.eval()

    print(
        "[Planner] RL LoRA attached."
    )

    planner_lora_count = sum(
        1
        for name, _ in (
            planner_generator.model.named_parameters()
        )
        if "lora_" in name
    )

    print(
        f"[Planner] LoRA parameter tensors: "
        f"{planner_lora_count}"
    )

    if planner_lora_count == 0:
        raise RuntimeError(
            "No LoRA parameters were loaded into "
            "the planner."
        )

    # ------------------------------------------------------------------
    # Coder generator
    # ------------------------------------------------------------------

    print()
    print(
        "[Coder] loading frozen base model..."
    )

    coder_generator = ModelGenerator(
        model_name_or_path=base_model_name,
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
            True,
        ),
    )

    coder_generator.model.eval()

    coder_lora_count = sum(
        1
        for name, _ in (
            coder_generator.model.named_parameters()
        )
        if "lora_" in name
    )

    if coder_lora_count != 0:
        raise RuntimeError(
            "Frozen coder unexpectedly contains "
            f"{coder_lora_count} LoRA tensors."
        )

    print(
        "[Coder] frozen base coder loaded."
    )

    # ------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------

    strategy = RLPlannerStrategy(
        planner_generator=planner_generator,
        coder_generator=coder_generator,
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
            generation_config.get(
                "plan_max_new_tokens",
                512,
            )
        ),
        code_max_new_tokens=(
            generation_config.get(
                "code_max_new_tokens",
                1024,
            )
        ),
        temperature=(
            generation_config.get(
                "temperature",
                0.0,
            )
        ),
        top_p=(
            generation_config.get(
                "top_p",
                1.0,
            )
        ),
    )

    # ------------------------------------------------------------------
    # Parser
    # ------------------------------------------------------------------

    code_parser = CodeParser()

    # ------------------------------------------------------------------
    # Evaluator
    # ------------------------------------------------------------------

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
    # Save evaluation metadata
    # ------------------------------------------------------------------

    output_dir = output_path.parent
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_config_snapshot = dict(
        config
    )

    evaluation_config_snapshot[
        "phase4_rl_evaluation"
    ] = {
        "checkpoint_label": (
            args.checkpoint_label
        ),
        "adapter_path": str(
            adapter_path
        ),
        "planner_model": (
            evaluation_model_name
        ),
        "coder_model": (
            base_model_name
        ),
        "protocol": (
            "phase1_self_plan"
        ),
    }

    run_config_path = save_run_config(
        config=evaluation_config_snapshot,
        output_dir=output_dir,
        overwrite=False,
    )

    run_metadata_path = save_run_metadata(
        config=evaluation_config_snapshot,
        output_dir=output_dir,
        overwrite=False,
    )

    print(
        f"Run config     : {run_config_path}"
    )
    print(
        f"Metadata       : {run_metadata_path}"
    )

    # ------------------------------------------------------------------
    # Existing Phase 1 runner
    # ------------------------------------------------------------------

    runner = Phase1Runner(
        strategy=strategy,
        evaluator=evaluator,
        parser=code_parser,
        output_path=output_path,
        model_name=evaluation_model_name,
        seed=seed,
        resume=resume,
    )

    summary = runner.run(
        examples
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("RL Planner Evaluation Complete")
    print("=" * 80)

    print(
        f"Checkpoint : "
        f"{args.checkpoint_label}"
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
            f"{summary.pass_rate:.6f}"
        )

    print(
        f"Results    : "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()