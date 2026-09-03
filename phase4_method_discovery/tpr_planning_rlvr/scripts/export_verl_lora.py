#!/usr/bin/env python3
"""
PYTHONPATH="$HOME/workspace/project_sLM_planning" \
python \
  phase4_method_discovery/tpr_planning_rlvr/scripts/export_verl_lora.py \
  --checkpoint-dir \
  /mnt/hdd/project_sLM_planning/checkpoints/tpr_planning_rlvr_lora_pilot50_rerun/global_step_25 \
  --output-dir \
  /mnt/hdd/project_sLM_planning/checkpoints/tpr_planning_rlvr_lora_pilot50_rerun/exported/step25 \
  --base-model \
  Qwen/Qwen2.5-Coder-3B-Instruct
  
PYTHONPATH="$HOME/workspace/project_sLM_planning" \
python \
  phase4_method_discovery/tpr_planning_rlvr/scripts/export_verl_lora.py \
  --checkpoint-dir \
  /mnt/hdd/project_sLM_planning/checkpoints/tpr_planning_rlvr_lora_pilot50_rerun/global_step_25 \
  --output-dir \
  /mnt/hdd/project_sLM_planning/checkpoints/tpr_planning_rlvr_lora_pilot50_rerun/exported/step25 \
  --base-model \
  Qwen/Qwen2.5-Coder-3B-Instruct
        
        

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a verl FSDP LoRA checkpoint as a standard "
            "Hugging Face PEFT adapter."
        )
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help=(
            "Path to a verl global_step_* checkpoint directory. "
            "The directory must contain actor/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the PEFT adapter will be written.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Base Hugging Face model name or path.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help=(
            "LoRA rank. If omitted, read from actor/lora_train_meta.json."
        ),
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help=(
            "LoRA alpha. If omitted, read from actor/lora_train_meta.json."
        ),
    )

    return parser.parse_args()


def load_lora_meta(actor_dir: Path) -> dict:
    meta_path = actor_dir / "lora_train_meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"LoRA metadata not found: {meta_path}"
        )

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    return meta


def load_verl_state_dict(actor_dir: Path) -> Dict[str, torch.Tensor]:
    fsdp_config_path = actor_dir / "fsdp_config.json"

    if fsdp_config_path.exists():
        with fsdp_config_path.open("r", encoding="utf-8") as f:
            fsdp_config = json.load(f)

        world_size = int(fsdp_config.get("world_size", 1))

        if world_size != 1:
            raise NotImplementedError(
                "This exporter currently supports only world_size=1 "
                f"checkpoints, but checkpoint reports world_size={world_size}."
            )

    checkpoint_path = actor_dir / "model_world_size_1_rank_0.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    print(f"[1/5] Loading checkpoint:")
    print(f"      {checkpoint_path}")

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Expected checkpoint to contain a state dict, "
            f"but got {type(state_dict)}."
        )

    print(f"      tensors in checkpoint: {len(state_dict)}")

    return state_dict


def is_lora_key(key: str) -> bool:
    return (
        ".lora_A.default.weight" in key
        or ".lora_B.default.weight" in key
    )


def convert_verl_key_to_peft(key: str) -> str:
    """
    Convert verl/PEFT internal state-dict keys into the key format stored
    in adapter_model.safetensors.

    verl checkpoint:
        base_model.model.model.layers.0.self_attn.q_proj.
            lora_A.default.weight

    PEFT adapter file:
        base_model.model.model.layers.0.self_attn.q_proj.
            lora_A.weight

    PEFT's save_pretrained() removes the adapter name ("default") from
    serialized adapter keys. PeftModel.from_pretrained() restores the
    adapter namespace when loading.
    """

    if not is_lora_key(key):
        raise ValueError(f"Not a LoRA key: {key}")

    key = key.replace(
        ".lora_A.default.weight",
        ".lora_A.weight",
    )
    key = key.replace(
        ".lora_B.default.weight",
        ".lora_B.weight",
    )

    return key


def extract_lora_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    lora_state_dict: Dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        if not is_lora_key(key):
            continue

        new_key = convert_verl_key_to_peft(key)

        if new_key in lora_state_dict:
            raise RuntimeError(
                f"Duplicate converted key detected: {new_key}"
            )

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Expected tensor for {key}, got {type(tensor)}"
            )

        # safetensors requires contiguous tensors.
        lora_state_dict[new_key] = (
            tensor.detach()
            .cpu()
            .contiguous()
        )

    if not lora_state_dict:
        raise RuntimeError(
            "No LoRA parameters were found in the checkpoint."
        )

    return lora_state_dict


def infer_target_modules(
    lora_state_dict: Dict[str, torch.Tensor],
) -> list[str]:
    """
    Infer LoRA target module names from keys such as:

        ...self_attn.q_proj.lora_A.weight
        ...mlp.gate_proj.lora_A.weight

    -> q_proj, k_proj, v_proj, o_proj,
       gate_proj, up_proj, down_proj
    """

    target_modules = set()

    for key in lora_state_dict:
        if ".lora_A.weight" not in key:
            continue

        module_path = key.rsplit(".lora_A.weight", 1)[0]
        module_name = module_path.rsplit(".", 1)[-1]

        target_modules.add(module_name)

    if not target_modules:
        raise RuntimeError(
            "Could not infer LoRA target modules."
        )

    return sorted(target_modules)


def validate_lora_shapes(
    lora_state_dict: Dict[str, torch.Tensor],
    rank: int,
) -> None:
    """
    Basic structural validation.

    For standard LoRA:
        A: [r, in_features]
        B: [out_features, r]
    """

    num_a = 0
    num_b = 0

    for key, tensor in lora_state_dict.items():
        if tensor.ndim != 2:
            raise ValueError(
                f"Expected 2-D LoRA tensor for {key}, "
                f"got shape={tuple(tensor.shape)}"
            )

        if key.endswith(".lora_A.weight"):
            num_a += 1

            if tensor.shape[0] != rank:
                raise ValueError(
                    f"Unexpected LoRA-A rank for {key}: "
                    f"shape={tuple(tensor.shape)}, expected rank={rank}"
                )

        elif key.endswith(".lora_B.weight"):
            num_b += 1

            if tensor.shape[1] != rank:
                raise ValueError(
                    f"Unexpected LoRA-B rank for {key}: "
                    f"shape={tuple(tensor.shape)}, expected rank={rank}"
                )

    if num_a != num_b:
        raise ValueError(
            "LoRA A/B tensor count mismatch: "
            f"A={num_a}, B={num_b}"
        )

    print(f"      LoRA A tensors: {num_a}")
    print(f"      LoRA B tensors: {num_b}")


def build_adapter_config(
    base_model: str,
    rank: int,
    lora_alpha: int,
    target_modules: list[str],
) -> dict:
    """
    Configuration compatible with PEFT LoraConfig / from_pretrained().
    """

    return {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": base_model,
        "bias": "none",
        "corda_config": None,
        "eva_config": None,
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": lora_alpha,
        "lora_bias": False,
        "lora_dropout": 0.0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "qalora_group_size": 16,
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "trainable_token_indices": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }


def main() -> None:
    args = parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()
    actor_dir = checkpoint_dir / "actor"
    output_dir = args.output_dir.resolve()

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}"
        )

    if not actor_dir.exists():
        raise FileNotFoundError(
            f"Actor directory not found: {actor_dir}"
        )

    print("=" * 72)
    print("verl FSDP LoRA -> Hugging Face PEFT exporter")
    print("=" * 72)

    meta = load_lora_meta(actor_dir)

    rank = (
        args.rank
        if args.rank is not None
        else int(meta["r"])
    )
    lora_alpha = (
        args.lora_alpha
        if args.lora_alpha is not None
        else int(meta["lora_alpha"])
    )

    print(f"Checkpoint : {checkpoint_dir}")
    print(f"Output     : {output_dir}")
    print(f"Base model : {args.base_model}")
    print(f"LoRA rank  : {rank}")
    print(f"LoRA alpha : {lora_alpha}")
    print()

    state_dict = load_verl_state_dict(actor_dir)

    print("[2/5] Extracting LoRA tensors...")

    lora_state_dict = extract_lora_state_dict(state_dict)

    print(
        f"      exported LoRA tensors: "
        f"{len(lora_state_dict)}"
    )

    target_modules = infer_target_modules(lora_state_dict)

    print(
        "      target modules: "
        + ", ".join(target_modules)
    )

    print("[3/5] Validating LoRA tensors...")

    validate_lora_shapes(
        lora_state_dict,
        rank=rank,
    )

    adapter_config = build_adapter_config(
        base_model=args.base_model,
        rank=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    adapter_model_path = (
        output_dir / "adapter_model.safetensors"
    )
    adapter_config_path = (
        output_dir / "adapter_config.json"
    )

    print("[4/5] Writing PEFT adapter...")

    save_file(
        lora_state_dict,
        str(adapter_model_path),
        metadata={"format": "pt"},
    )

    with adapter_config_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            adapter_config,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    print("[5/5] Export complete.")
    print()
    print(f"  {adapter_model_path}")
    print(f"  {adapter_config_path}")
    print()
    print(
        f"LoRA tensors : {len(lora_state_dict)}"
    )
    print(
        f"Target modules: {target_modules}"
    )
    print()
    print("Load with:")
    print()
    print(
        "  base = AutoModelForCausalLM.from_pretrained("
        f'"{args.base_model}", ...)'
    )
    print(
        "  model = PeftModel.from_pretrained("
        f'base, "{output_dir}")'
    )


if __name__ == "__main__":
    main()