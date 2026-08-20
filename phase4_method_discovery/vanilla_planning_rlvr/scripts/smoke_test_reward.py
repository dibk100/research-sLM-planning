"""
PYTHONPATH=. python \
  phase4_method_discovery/vanilla_planning_rlvr/data/build_verl_dataset.py \
  --input /path/to/livecodebench_v6.jsonl \
  --output-dir phase4_method_discovery/vanilla_planning_rlvr/data/processed \
  --prompt-template prompt_templates/self_plan_plan.txt \
  --val-ratio 0.1 \
  --seed 42
  
  
vanilla_planning_rlvr/
└── data/
    └── processed/
        ├── train.parquet
        ├── val.parquet
        └── dataset_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ============================================================
# Project root setup
# ============================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.generator import ModelGenerator
from src.schemas import ProblemExample

from phase4_method_discovery.vanilla_planning_rlvr.logging.rollout_logger import (
    RolloutLogger,
)
from phase4_method_discovery.vanilla_planning_rlvr.reward.planning_execution_reward import (
    compute_planning_execution_reward,
    initialize_reward_runtime,
)


# ============================================================
# Helpers
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test for Vanilla Planning-RLVR "
            "reward + rollout logging pipeline."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Frozen coder model name or local path.",
    )

    parser.add_argument(
        "--problem-json",
        type=str,
        required=True,
        help="Path to one serialized ProblemExample JSON file.",
    )

    parser.add_argument(
        "--plan-file",
        type=str,
        required=True,
        help="Path to a text file containing one plan.",
    )

    parser.add_argument(
        "--code-prompt",
        type=str,
        default="prompt_templates/self_plan_code.txt",
        help="Plan-conditioned code-generation prompt template.",
    )

    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=(
            "phase4_method_discovery/"
            "vanilla_planning_rlvr/"
            "sample_log/smoke_test.jsonl"
        ),
        help="Output JSONL path for rollout logging.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=[
            "float16",
            "bfloat16",
            "float32",
        ],
    )

    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    return parser.parse_args()


def load_problem(
    path: str | Path,
) -> ProblemExample:
    problem_path = Path(path)

    if not problem_path.exists():
        raise FileNotFoundError(
            f"Problem JSON not found: {problem_path}"
        )

    with problem_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise TypeError(
            "Problem JSON must contain one JSON object."
        )

    return ProblemExample(**payload)


def load_plan(
    path: str | Path,
) -> str:
    plan_path = Path(path)

    if not plan_path.exists():
        raise FileNotFoundError(
            f"Plan file not found: {plan_path}"
        )

    plan = plan_path.read_text(
        encoding="utf-8",
    ).strip()

    if not plan:
        raise ValueError(
            f"Plan file is empty: {plan_path}"
        )

    return plan


def get_problem_text(
    problem: ProblemExample,
) -> str:
    if not isinstance(problem.problem, str):
        raise TypeError(
            "problem.problem must be str."
        )

    problem_text = problem.problem.strip()

    if not problem_text:
        raise ValueError(
            f"Problem text is empty: {problem.problem_id}"
        )

    return problem_text


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("Vanilla Planning-RLVR Reward Smoke Test")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Load problem / plan
    # --------------------------------------------------------

    problem = load_problem(
        args.problem_json
    )

    plan = load_plan(
        args.plan_file
    )

    problem_text = get_problem_text(
        problem
    )

    print(
        f"[Problem] id={problem.problem_id}"
    )

    print(
        f"[Problem] dataset={problem.dataset}"
    )

    print(
        f"[Problem] evaluation_type="
        f"{problem.evaluation_type}"
    )

    print(
        f"[Plan] chars={len(plan)}"
    )

    # --------------------------------------------------------
    # 2. Load frozen coder
    # --------------------------------------------------------

    print()
    print("[Runtime] Loading frozen coder...")

    frozen_coder = ModelGenerator(
        model_name_or_path=args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )

    # --------------------------------------------------------
    # 3. Initialize reward runtime
    # --------------------------------------------------------

    code_prompt_path = (
        PROJECT_ROOT
        / args.code_prompt
    )

    initialize_reward_runtime(
        frozen_coder=frozen_coder,
        code_prompt_path=code_prompt_path,
        timeout_seconds=args.timeout,
        debug=args.debug,
        coder_max_new_tokens=args.max_new_tokens,

        # Frozen coder is intentionally deterministic.
        coder_temperature=0.0,
        coder_top_p=1.0,
    )

    print(
        "[Runtime] Reward runtime initialized."
    )

    # --------------------------------------------------------
    # 4. Initialize rollout logger
    # --------------------------------------------------------

    output_jsonl = (
        PROJECT_ROOT
        / args.output_jsonl
    )

    logger = RolloutLogger(
        output_path=output_jsonl,
    )

    print(
        f"[Logger] output={output_jsonl}"
    )

    # --------------------------------------------------------
    # 5. Run complete reward pipeline
    # --------------------------------------------------------

    print()
    print("[Reward] Running pipeline...")

    result = (
        compute_planning_execution_reward(
            problem=problem,
            problem_text=problem_text,
            plan=plan,
        )
    )

    # --------------------------------------------------------
    # 6. Build + save rollout record
    # --------------------------------------------------------

    record = RolloutLogger.from_reward_result(
        reward_result=result,

        # Smoke test identity.
        global_step=0,
        group_id="smoke_test",
        sample_id=0,

        dataset=problem.dataset,
        model_name=args.model,
        seed=args.seed,

        # Token-wise fields intentionally omitted here.
        # These will later come from verl rollout tensors.
        plan_tokens=None,
        plan_token_ids=None,
        token_logprobs=None,

        extra={
            "title": problem.title,
            "difficulty": problem.difficulty,
            "rating": problem.rating,
            "evaluation_type": problem.evaluation_type,
        },
    )

    logger.log(record)

    print(
        "[Logger] rollout record saved."
    )

    # --------------------------------------------------------
    # 7. Print result
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Reward Result")
    print("=" * 80)

    print(
        f"problem_id              : "
        f"{result.problem_id}"
    )

    print(
        f"reward                  : "
        f"{result.reward}"
    )

    print(
        f"passed                  : "
        f"{result.passed}"
    )

    print(
        f"status                  : "
        f"{result.status}"
    )

    print(
        f"tests                    : "
        f"{result.passed_tests}"
        f"/{result.total_tests}"
    )

    print(
        f"code extraction         : "
        f"{result.code_extraction_method}"
    )

    print(
        f"coder prompt tokens     : "
        f"{result.coder_prompt_tokens}"
    )

    print(
        f"coder completion tokens : "
        f"{result.coder_completion_tokens}"
    )

    print(
        f"coder generation time   : "
        f"{result.coder_generation_time:.4f}s"
    )

    print(
        f"execution time          : "
        f"{result.execution_time:.4f}s"
    )

    if result.error_message:
        print(
            f"error                    : "
            f"{result.error_message}"
        )

    # --------------------------------------------------------
    # 8. Inspect generated artifacts
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Plan")
    print("=" * 80)
    print(result.plan)

    print()
    print("=" * 80)
    print("Raw Coder Output")
    print("=" * 80)
    print(result.raw_code_output)

    print()
    print("=" * 80)
    print("Parsed Code")
    print("=" * 80)
    print(result.generated_code)

    print()
    print("=" * 80)

    if result.reward == 1.0:
        print(
            "[PASS] Reward pipeline completed "
            "and generated code passed all tests."
        )
    else:
        print(
            "[DONE] Reward pipeline completed, "
            "but generated code did not pass all tests."
        )

    print(
        f"[LOG] {output_jsonl}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()