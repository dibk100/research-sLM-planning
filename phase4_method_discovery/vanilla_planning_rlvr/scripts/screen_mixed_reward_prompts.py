"""
Find DeepCoder-TACO prompts whose GRPO group has mixed binary rewards.

Screening pipeline
------------------
1. Load candidate rows from the Planning-RLVR training parquet.
2. Load the planner on GPU.
3. Sample N plans per problem (default: 4) with the same stochastic
   settings used by the GRPO smoke test.
4. Delete the planner completely and release GPU memory.
5. Start one FrozenCoderWorker Ray actor.
6. For each candidate problem, evaluate all sampled plans via the
   existing Planning-RLVR execution reward:
       plan -> frozen coder -> TACO -> {0, 1}
7. Stop at the first problem satisfying:
       0 < sum(rewards) < group_size
8. Save that one original parquet row as a dedicated GRPO signal-smoke
   parquet, plus a JSON diagnostics report.

Important
---------
This is a screening utility, not a training script.

For the initial LoRA smoke (fresh zero-initialized LoRA), the planner is
equivalent to the base Qwen checkpoint, so loading the base planner here
is appropriate. If you later want to screen with a trained LoRA adapter,
pass --planner-lora-adapter.

Phase A
planner GPU
→ 후보 문제 30개 × plan 4개 sampling
→ 모든 plan을 RAM에 저장
→ planner 완전히 제거
→ CUDA cache release

Phase B
FrozenCoderWorker
→ 저장된 plan을 문제별로 4개 평가
→ TACO binary rewards 계산

[0,0,0,0] → 계속 탐색
[1,1,1,1] → 계속 탐색
[0,1,0,0] → FOUND
[1,0,1,0] → FOUND
...


/mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/
vanilla_planning_rlvr/grpo_signal_smoke.parquet


PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/verl" \
/mnt/hdd/conda_envs/planning_rlvr/bin/python \
  phase4_method_discovery/vanilla_planning_rlvr/scripts/screen_mixed_reward_prompts.py \
  --start-index 0 \
  --num-candidates 20 \
  --group-size 4 \
  --num-trials 5 \
  --min-mixed-trials 2

20문제를 대상으로 각 문제를 5번씩 sampling하고, 최소 2/5회 이상 mixed reward가 발생한 문제만 후보로 인정
  --show-plans

자동 저장
phase4_method_discovery/vanilla_planning_rlvr/
outputs/mixed_reward_screen.json

"""
# phase4_method_discovery/vanilla_planning_rlvr/scripts/screen_mixed_reward_prompts.py
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


# ============================================================================
# Project paths
# ============================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

DEFAULT_TRAIN_PARQUET = Path(
    "/mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/"
    "vanilla_planning_rlvr/train.parquet"
)

DEFAULT_RESEARCH_CONFIG = (
    PROJECT_ROOT
    / "phase4_method_discovery"
    / "vanilla_planning_rlvr"
    / "configs"
    / "vanilla_planning_rlvr_qwen25coder3b.yaml"
)

DEFAULT_OUTPUT_PARQUET = Path(
    "/mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/"
    "vanilla_planning_rlvr/grpo_signal_smoke.parquet"
)

DEFAULT_REPORT_JSON = (
    PROJECT_ROOT
    / "phase4_method_discovery"
    / "vanilla_planning_rlvr"
    / "outputs"
    / "mixed_reward_screen.json"
)


# ============================================================================
# Local imports
# ============================================================================

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase4_method_discovery.vanilla_planning_rlvr.reward.planning_execution_reward import (  # noqa: E402
    compute_score,
)
from phase4_method_discovery.vanilla_planning_rlvr.workers.frozen_coder_worker import (  # noqa: E402
    FrozenCoderWorker,
)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screen Planning-RLVR training prompts for a mixed GRPO reward "
            "group such as [0, 1, 0, 0]."
        )
    )

    parser.add_argument(
        "--train-parquet",
        type=Path,
        default=DEFAULT_TRAIN_PARQUET,
        help="Source Planning-RLVR training parquet.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RESEARCH_CONFIG,
        help="Research config used by FrozenCoderWorker.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=DEFAULT_OUTPUT_PARQUET,
        help="One-row parquet written when a mixed-reward prompt is found.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_REPORT_JSON,
        help="Screening diagnostics JSON.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First parquet row to screen.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=30,
        help="Maximum number of rows to screen.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Number of sampled plans per problem.",
    )

    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help=(
            "Number of independent planner-sampling trials per problem. "
            "Use >1 to find prompts that repeatedly produce mixed reward groups."
        ),
    )
    parser.add_argument(
        "--min-mixed-trials",
        type=int,
        default=1,
        help=(
            "Minimum number of mixed-reward trials required for a problem "
            "to qualify as a robust signal-smoke candidate."
        ),
    )

    parser.add_argument(
        "--planner-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-3B-Instruct",
        help="Planner base checkpoint.",
    )
    parser.add_argument(
        "--planner-lora-adapter",
        type=str,
        default=None,
        help=(
            "Optional PEFT adapter path. Omit for the initial fresh-LoRA "
            "signal smoke."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Planner maximum response tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Planner sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Planner nucleus sampling top-p.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base planner sampling seed.",
    )

    parser.add_argument(
        "--save-all-results",
        action="store_true",
        help="Keep diagnostics for every screened row in the JSON report.",
    )
    parser.add_argument(
        "--show-plans",
        action="store_true",
        help="Print sampled plans.",
    )

    return parser.parse_args()


# ============================================================================
# Generic parquet helpers
# ============================================================================

def _pythonize(value: Any) -> Any:
    """Convert numpy/pandas container-like values into ordinary Python types."""
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def normalize_messages(value: Any) -> list[dict[str, str]]:
    """
    Restore the verl parquet `prompt` column as chat messages.

    Accepted forms:
      - list[dict]
      - numpy array of dicts
      - JSON string containing list[dict]
      - plain string (treated as one user message)
    """
    value = _pythonize(value)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("Empty prompt string.")

        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return [{"role": "user", "content": stripped}]

        value = decoded

    if isinstance(value, tuple):
        value = list(value)

    if not isinstance(value, list):
        raise TypeError(
            f"Unsupported prompt type: {type(value).__name__}. "
            "Expected chat-message list or string."
        )

    messages: list[dict[str, str]] = []
    for item in value:
        item = _pythonize(item)

        if not isinstance(item, dict):
            raise TypeError(
                "Each prompt message must be dict, "
                f"got {type(item).__name__}."
            )

        role = str(item.get("role", "user"))
        content = item.get("content", "")

        if isinstance(content, list):
            # Defensive handling for multimodal-style HF messages.
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                else:
                    text_parts.append(str(part))
            content = "\n".join(text_parts)

        messages.append(
            {
                "role": role,
                "content": str(content),
            }
        )

    if not messages:
        raise ValueError("Prompt contains no messages.")

    return messages


def normalize_extra_info(value: Any) -> dict[str, Any]:
    value = _pythonize(value)

    if isinstance(value, str):
        value = json.loads(value)

    if not isinstance(value, dict):
        raise TypeError(
            f"extra_info must decode to dict, got {type(value).__name__}."
        )

    return value


def get_problem_id(row: pd.Series) -> str:
    extra_info = normalize_extra_info(row["extra_info"])

    problem_json = extra_info.get("problem_json")
    if isinstance(problem_json, str):
        try:
            payload = json.loads(problem_json)
            if isinstance(payload, dict):
                for key in ("problem_id", "id", "question_id"):
                    if payload.get(key) is not None:
                        return str(payload[key])
        except json.JSONDecodeError:
            pass

    for key in ("problem_id", "id", "question_id"):
        if key in row and row[key] is not None:
            return str(row[key])

    return "<unknown>"


# ============================================================================
# Planner
# ============================================================================

def load_planner(
    model_name: str,
    lora_adapter: str | None,
):
    print()
    print("=" * 90)
    print("Load planner")
    print("=" * 90)
    print(f"base model        : {model_name}")
    print(f"LoRA adapter      : {lora_adapter or '<none>'}")
    print("dtype             : bfloat16")
    print("attention         : sdpa")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    if lora_adapter is not None:
        if PeftModel is None:
            raise RuntimeError(
                "PEFT is required for --planner-lora-adapter, "
                "but `peft` could not be imported."
            )
        model = PeftModel.from_pretrained(
            model,
            lora_adapter,
            is_trainable=False,
        )

    model.eval()
    model.to("cuda")

    return tokenizer, model


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
) -> list[str]:
    if group_size <= 0:
        raise ValueError("group_size must be > 0.")

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    encoded = {
        key: tensor.to("cuda")
        for key, tensor in encoded.items()
    }

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    generate_kwargs = dict(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=group_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Transformers uses top_k=50 by default for sampling. verl/vLLM smoke has
    # top_k=-1 (disabled), so explicitly disable top-k filtering here.
    generate_kwargs["top_k"] = 0

    outputs = model.generate(**generate_kwargs)

    prompt_len = int(encoded["input_ids"].shape[1])

    plans: list[str] = []
    for sequence in outputs:
        generated_ids = sequence[prompt_len:]
        plan = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        plans.append(plan)

    if len(plans) != group_size:
        raise RuntimeError(
            f"Expected {group_size} plans, got {len(plans)}."
        )

    return plans


def release_planner(model) -> None:
    print()
    print("[Planner] releasing GPU model before frozen coder screening...")

    try:
        model.to("cpu")
    except Exception:
        pass

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)

    print(f"[Planner] CUDA allocated after release: {allocated:.3f} GiB")
    print(f"[Planner] CUDA reserved  after release: {reserved:.3f} GiB")


# ============================================================================
# Frozen coder / reward
# ============================================================================

def start_frozen_coder(config_path: Path):
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=True,
        )

    FrozenCoderActor = ray.remote(
        num_cpus=1,
        num_gpus=1,
    )(FrozenCoderWorker)

    actor = FrozenCoderActor.remote(
        str(config_path.resolve())
    )

    status = ray.get(
        actor.init_model.remote()
    )

    print()
    print("=" * 90)
    print("FrozenCoderWorker")
    print("=" * 90)
    print(json.dumps(status, indent=2, ensure_ascii=False))

    return actor


def evaluate_plan(
    *,
    row: pd.Series,
    plan: str,
    frozen_coder_handle: Any,
) -> dict[str, Any]:
    data_source = str(row.get("data_source", "deepcoder_taco"))
    extra_info = normalize_extra_info(row["extra_info"])

    # compute_score ignores ground_truth in the current Planning-RLVR reward,
    # but preserve the parquet value when available.
    ground_truth = None

    reward_model = row.get("reward_model")
    reward_model = _pythonize(reward_model)

    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")

    result = compute_score(
        data_source=data_source,
        solution_str=plan,
        ground_truth=ground_truth,
        extra_info=extra_info,
        frozen_coder_handle=frozen_coder_handle,
    )

    if not isinstance(result, dict):
        raise TypeError(
            f"compute_score must return dict, got {type(result).__name__}."
        )

    return result


# ============================================================================
# Reports
# ============================================================================

def make_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return make_serializable(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): make_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_serializable(item) for item in value]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass

    if isinstance(value, Path):
        return str(value)

    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            make_serializable(payload),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()

    if args.group_size < 2:
        raise ValueError("group_size must be >= 2 for mixed-reward screening.")

    if args.num_candidates <= 0:
        raise ValueError("num_candidates must be > 0.")

    if args.num_trials <= 0:
        raise ValueError("num_trials must be > 0.")

    if args.min_mixed_trials <= 0:
        raise ValueError("min_mixed_trials must be > 0.")

    if args.min_mixed_trials > args.num_trials:
        raise ValueError(
            "min_mixed_trials must be <= num_trials."
        )

    train_path = args.train_parquet.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_path = args.output_parquet.expanduser().resolve()
    report_path = args.report_json.expanduser().resolve()

    if not train_path.exists():
        raise FileNotFoundError(f"Training parquet not found: {train_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Research config not found: {config_path}")

    df = pd.read_parquet(train_path)

    if "prompt" not in df.columns:
        raise KeyError(
            f"Parquet has no 'prompt' column. Columns: {list(df.columns)}"
        )

    if "extra_info" not in df.columns:
        raise KeyError(
            f"Parquet has no 'extra_info' column. Columns: {list(df.columns)}"
        )

    start = args.start_index
    stop = min(
        len(df),
        start + args.num_candidates,
    )

    if start < 0 or start >= len(df):
        raise IndexError(
            f"start-index={start} outside dataset of size {len(df)}."
        )

    candidate_indices = list(range(start, stop))

    print("=" * 90)
    print("Planning-RLVR Mixed Reward Prompt Screening")
    print("=" * 90)
    print(f"train parquet     : {train_path}")
    print(f"dataset rows      : {len(df)}")
    print(f"candidate range   : [{start}, {stop})")
    print(f"group size        : {args.group_size}")
    print(f"num trials        : {args.num_trials}")
    print(f"min mixed trials  : {args.min_mixed_trials}")
    print(f"planner           : {args.planner_model}")
    print(f"planner LoRA      : {args.planner_lora_adapter or '<none>'}")
    print(f"max new tokens    : {args.max_new_tokens}")
    print(f"temperature       : {args.temperature}")
    print(f"top_p             : {args.top_p}")
    print(f"base seed         : {args.seed}")
    print(f"output parquet    : {output_path}")
    print(f"report JSON       : {report_path}")
    print("=" * 90)

    # ------------------------------------------------------------------
    # Phase 1: sample all candidate plan groups while planner owns GPU.
    # ------------------------------------------------------------------

    tokenizer, planner = load_planner(
        args.planner_model,
        args.planner_lora_adapter,
    )

    sampled: dict[int, list[list[str]]] = {}
    sampling_errors: dict[str, str] = {}

    sampling_started = time.time()

    for ordinal, row_index in enumerate(candidate_indices, start=1):
        row = df.iloc[row_index]
        problem_id = get_problem_id(row)

        print()
        print(
            f"[Planner {ordinal:02d}/{len(candidate_indices):02d}] "
            f"row={row_index} problem={problem_id}"
        )

        try:
            messages = normalize_messages(row["prompt"])
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            sampling_errors[f"{row_index}:prompt"] = message
            print(f"  [SKIP] prompt parsing failed: {message}")
            continue

        trial_groups: list[list[str]] = []

        for trial_idx in range(args.num_trials):
            # Keep trial seeds deterministic and well separated across rows.
            trial_seed = (
                args.seed
                + row_index * 100_003
                + trial_idx * 1_009
            )

            try:
                plans = sample_plans(
                    tokenizer=tokenizer,
                    model=planner,
                    messages=messages,
                    group_size=args.group_size,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=trial_seed,
                )

                trial_groups.append(plans)

                lengths = [
                    len(tokenizer.encode(plan, add_special_tokens=False))
                    for plan in plans
                ]

                print(
                    f"  trial={trial_idx + 1}/{args.num_trials} "
                    f"seed={trial_seed} "
                    f"plan token lengths={lengths}"
                )

                if args.show_plans:
                    for sample_idx, plan in enumerate(plans):
                        print()
                        print(
                            f"  --- trial {trial_idx + 1} "
                            f"plan {sample_idx} ---"
                        )
                        print(plan)

            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                sampling_errors[
                    f"{row_index}:trial:{trial_idx}"
                ] = message
                print(
                    f"  [SKIP] trial={trial_idx + 1} "
                    f"planner sampling failed: {message}"
                )

        if trial_groups:
            sampled[row_index] = trial_groups

    planner_sampling_sec = time.time() - sampling_started

    # We do not need the planner again. Completely release it before the coder.
    del tokenizer
    release_planner(planner)

    # ------------------------------------------------------------------
    # Phase 2: evaluate cached plans using the existing reward pipeline.
    # ------------------------------------------------------------------

    frozen_coder = None
    all_results: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None

    try:
        frozen_coder = start_frozen_coder(config_path)

        reward_started = time.time()

        valid_row_indices = [
            idx for idx in candidate_indices
            if idx in sampled
        ]

        for ordinal, row_index in enumerate(valid_row_indices, start=1):
            row = df.iloc[row_index]
            problem_id = get_problem_id(row)
            trial_groups = sampled[row_index]

            print()
            print("=" * 90)
            print(
                f"[Reward {ordinal:02d}/{len(valid_row_indices):02d}] "
                f"row={row_index} problem={problem_id}"
            )
            print("=" * 90)

            trial_results: list[dict[str, Any]] = []
            mixed_trials = 0
            successful_trajectories = 0

            for trial_idx, plans in enumerate(trial_groups):
                print(
                    f"  [Trial {trial_idx + 1}/{len(trial_groups)}]"
                )

                trajectory_results: list[dict[str, Any]] = []
                rewards: list[float] = []

                for sample_idx, plan in enumerate(plans):
                    started = time.time()

                    try:
                        result = evaluate_plan(
                            row=row,
                            plan=plan,
                            frozen_coder_handle=frozen_coder,
                        )

                        score = float(result.get("score", 0.0))
                        rewards.append(score)

                        if score > 0:
                            successful_trajectories += 1

                        trajectory = {
                            "sample_index": sample_idx,
                            "reward": score,
                            "status": str(result.get("status", "")),
                            "passed": bool(result.get("passed", False)),
                            "passed_tests": int(result.get("passed_tests", 0)),
                            "total_tests": int(result.get("total_tests", 0)),
                            "coder_prompt_tokens": int(
                                result.get("coder_prompt_tokens", 0)
                            ),
                            "coder_completion_tokens": int(
                                result.get("coder_completion_tokens", 0)
                            ),
                            "coder_generation_time": float(
                                result.get("coder_generation_time", 0.0)
                            ),
                            "execution_time": float(
                                result.get("execution_time", 0.0)
                            ),
                            "error_message": str(
                                result.get("error_message", "")
                            ),
                            "plan": plan,
                            "wall_time": time.time() - started,
                        }

                        trajectory_results.append(trajectory)

                        print(
                            f"    sample={sample_idx} "
                            f"reward={score:.0f} "
                            f"status={trajectory['status']} "
                            f"tests={trajectory['passed_tests']}/"
                            f"{trajectory['total_tests']}"
                        )

                    except Exception as exc:
                        rewards.append(0.0)

                        trajectory_results.append(
                            {
                                "sample_index": sample_idx,
                                "reward": 0.0,
                                "status": "SCREEN_ERROR",
                                "passed": False,
                                "passed_tests": 0,
                                "total_tests": 0,
                                "error_message": (
                                    f"{type(exc).__name__}: {exc}"
                                ),
                                "plan": plan,
                                "wall_time": time.time() - started,
                            }
                        )

                        print(
                            f"    sample={sample_idx} reward=0 "
                            f"status=SCREEN_ERROR "
                            f"error={type(exc).__name__}: {exc}"
                        )

                reward_sum = float(sum(rewards))
                unique_rewards = sorted(set(rewards))
                mixed = 0.0 < reward_sum < float(args.group_size)

                if mixed:
                    mixed_trials += 1

                trial_result = {
                    "trial_index": trial_idx,
                    "rewards": rewards,
                    "reward_sum": reward_sum,
                    "unique_rewards": unique_rewards,
                    "mixed_reward": mixed,
                    "trajectories": trajectory_results,
                }
                trial_results.append(trial_result)

                print(f"    rewards: {rewards}")
                print(f"    mixed  : {mixed}")

            evaluated_trials = len(trial_results)
            mixed_rate = (
                mixed_trials / evaluated_trials
                if evaluated_trials > 0
                else 0.0
            )
            qualifies = (
                mixed_trials >= args.min_mixed_trials
            )

            row_result = {
                "row_index": row_index,
                "problem_id": problem_id,
                "num_trials_requested": args.num_trials,
                "num_trials_evaluated": evaluated_trials,
                "mixed_trials": mixed_trials,
                "mixed_rate": mixed_rate,
                "successful_trajectories": successful_trajectories,
                "qualifies": qualifies,
                "trials": trial_results,
            }

            # Keep every compact row summary in the report. Full trajectory
            # details are included only with --save-all-results.
            if args.save_all_results:
                all_results.append(row_result)
            else:
                all_results.append(
                    {
                        key: value
                        for key, value in row_result.items()
                        if key != "trials"
                    }
                )

            print(
                f"  summary: mixed_trials={mixed_trials}/"
                f"{evaluated_trials} "
                f"mixed_rate={mixed_rate:.3f} "
                f"successful_trajectories={successful_trajectories} "
                f"qualifies={qualifies}"
            )

            if qualifies:
                if best_candidate is None:
                    best_candidate = row_result
                else:
                    current_key = (
                        row_result["mixed_trials"],
                        row_result["mixed_rate"],
                        row_result["successful_trajectories"],
                    )
                    best_key = (
                        best_candidate["mixed_trials"],
                        best_candidate["mixed_rate"],
                        best_candidate["successful_trajectories"],
                    )

                    if current_key > best_key:
                        best_candidate = row_result

        reward_screening_sec = time.time() - reward_started

        if best_candidate is not None:
            best_row_index = int(best_candidate["row_index"])

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Preserve the original row/schema exactly.
            df.iloc[[best_row_index]].to_parquet(
                output_path,
                index=False,
            )

            print()
            print("*" * 90)
            print("[SELECTED] Robust mixed-reward GRPO prompt")
            print("*" * 90)
            print(
                f"row index         : {best_candidate['row_index']}"
            )
            print(
                f"problem id        : {best_candidate['problem_id']}"
            )
            print(
                f"mixed trials      : "
                f"{best_candidate['mixed_trials']}/"
                f"{best_candidate['num_trials_evaluated']}"
            )
            print(
                f"mixed rate        : "
                f"{best_candidate['mixed_rate']:.3f}"
            )
            print(
                f"successful traj.  : "
                f"{best_candidate['successful_trajectories']}"
            )
            print(f"saved parquet     : {output_path}")
            print("*" * 90)

    finally:
        if frozen_coder is not None:
            try:
                # Best-effort return to CPU before Ray shutdown.
                if hasattr(frozen_coder, "sleep"):
                    ray.get(frozen_coder.sleep.remote())
            except Exception:
                pass

        if ray.is_initialized():
            ray.shutdown()

    report = {
        "source_parquet": str(train_path),
        "candidate_start_index": start,
        "candidate_stop_index": stop,
        "num_candidates_requested": args.num_candidates,
        "num_candidates_sampled": len(sampled),
        "group_size": args.group_size,
        "num_trials": args.num_trials,
        "min_mixed_trials": args.min_mixed_trials,
        "planner_model": args.planner_model,
        "planner_lora_adapter": args.planner_lora_adapter,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "base_seed": args.seed,
        "planner_sampling_seconds": planner_sampling_sec,
        "reward_screening_seconds": locals().get(
            "reward_screening_sec",
            None,
        ),
        "sampling_errors": sampling_errors,
        "best_candidate": best_candidate,
        "output_parquet": (
            str(output_path)
            if best_candidate is not None
            else None
        ),
        "screened_results": all_results,
    }

    save_json(
        report_path,
        report,
    )

    print()
    print("=" * 90)
    print("Screening Finished")
    print("=" * 90)

    if best_candidate is None:
        print(
            "[NOT FOUND] No problem met the robust mixed-reward criterion "
            f"(mixed_trials >= {args.min_mixed_trials}) "
            f"in rows [{start}, {stop})."
        )
        print(
            "Try another range, e.g. --start-index "
            f"{stop} --num-candidates {args.num_candidates}."
        )
    else:
        print("[PASS] Robust mixed-reward prompt selected.")
        print(
            f"row index      : {best_candidate['row_index']}"
        )
        print(
            f"problem id     : {best_candidate['problem_id']}"
        )
        print(
            f"mixed trials   : {best_candidate['mixed_trials']}/"
            f"{best_candidate['num_trials_evaluated']}"
        )
        print(
            f"mixed rate     : {best_candidate['mixed_rate']:.3f}"
        )
        print(
            f"successful traj: {best_candidate['successful_trajectories']}"
        )
        print(f"signal parquet : {output_path}")

    print(f"report          : {report_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()