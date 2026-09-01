"""
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench:$HOME/workspace/verl" \
python phase4_method_discovery/vanilla_planning_rlvr/scripts/diagnose_taco_reward_granularity.py \
  --input phase4_method_discovery/vanilla_planning_rlvr/outputs/planning_rl_signal_sampling/taco_step25_pilot20_g16.jsonl \
  --train-parquet /mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/vanilla_planning_rlvr/train.parquet \
  --config phase4_method_discovery/vanilla_planning_rlvr/configs/vanilla_planning_rlvr_qwen25coder3b.yaml \
  --group-filter all_zero \
  --max-groups 2 \
  --output phase4_method_discovery/vanilla_planning_rlvr/outputs/planning_rl_signal_sampling/taco_step25_nonfail_smoke.jsonl \
  --overwrite
  
  
  
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench:$HOME/workspace/verl" \
python phase4_method_discovery/vanilla_planning_rlvr/scripts/diagnose_taco_reward_granularity.py \
  --input phase4_method_discovery/vanilla_planning_rlvr/outputs/planning_rl_signal_sampling/taco_step25_pilot20_g16.jsonl \
  --train-parquet /mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/vanilla_planning_rlvr/train.parquet \
  --config phase4_method_discovery/vanilla_planning_rlvr/configs/vanilla_planning_rlvr_qwen25coder3b.yaml \
  --group-filter all_zero \
  --output phase4_method_discovery/vanilla_planning_rlvr/outputs/planning_rl_signal_sampling/taco_step25_nonfail_allzero.jsonl \
  --overwrite
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ray
from omegaconf import OmegaConf

from src.execution.taco_evaluator import TACOEvaluator
from src.parsing.code_parser import CodeParser

from phase4_method_discovery.vanilla_planning_rlvr.reward.planning_execution_reward import (
    _generate_code_via_rpc,
    _problem_from_extra_info,
    build_code_prompt,
    select_reward_tests,
)
from phase4_method_discovery.vanilla_planning_rlvr.workers.frozen_coder_worker import (
    FrozenCoderWorker,
)


# =============================================================================
# Paths
# =============================================================================


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "phase4_method_discovery"
    / "vanilla_planning_rlvr"
    / "configs"
    / "vanilla_planning_rlvr_qwen25coder3b.yaml"
)

DEFAULT_TRAIN_PARQUET = Path(
    "/mnt/hdd/project_sLM_planning/data/deepcoder_taco/"
    "processed/vanilla_planning_rlvr/train.parquet"
)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate sampled TACO Planning-RLVR trajectories "
            "on all selected reward tests independently, removing "
            "the fail-fast effect from the partial-test diagnostic."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "JSONL produced by sample_taco_training_rl_signal.py."
        ),
    )

    parser.add_argument(
        "--train-parquet",
        type=Path,
        default=DEFAULT_TRAIN_PARQUET,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Research config used by FrozenCoderWorker.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--group-filter",
        choices=[
            "all_zero",
            "mixed",
            "all_one",
            "all",
        ],
        default="all_zero",
        help=(
            "Default=all_zero because the main question is whether "
            "binary-dead groups contain hidden partial correctness."
        ),
    )

    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Optional group limit for smoke testing.",
    )

    parser.add_argument(
        "--max-tests",
        type=int,
        default=None,
        help=(
            "Override reward test count. "
            "Default=config.reward.max_tests (15)."
        ),
    )

    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help=(
            "Override evaluator timeout. "
            "Default=config.reward.timeout_seconds."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--save-code",
        action="store_true",
        help=(
            "Store regenerated raw/code strings in the output. "
            "Useful for qualitative inspection but increases file size."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Utilities
# =============================================================================


def resolve_path(
    path: str | Path,
) -> Path:
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def population_variance(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    mean = sum(values) / len(values)

    return (
        sum(
            (value - mean) ** 2
            for value in values
        )
        / len(values)
    )


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
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def pythonize(
    value: Any,
) -> Any:
    if isinstance(value, np.ndarray):
        return [
            pythonize(x)
            for x in value.tolist()
        ]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(k): pythonize(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            pythonize(v)
            for v in value
        ]

    return value


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL line "
                    f"{line_number}: {exc}"
                ) from exc

    if not records:
        raise RuntimeError(
            f"No records found: {path}"
        )

    return records


# =============================================================================
# Frozen coder
# =============================================================================


def start_frozen_coder(
    config_path: Path,
):
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=True,
        )

    FrozenCoderActor = ray.remote(
        num_cpus=1,
        num_gpus=1,
    )(
        FrozenCoderWorker
    )

    actor = FrozenCoderActor.remote(
        str(config_path)
    )

    status = ray.get(
        actor.init_model.remote()
    )

    print()
    print("=" * 100)
    print("FrozenCoderWorker")
    print("=" * 100)

    print(
        json.dumps(
            status,
            indent=2,
            ensure_ascii=False,
        )
    )

    return actor


# =============================================================================
# CPU diagnostic worker
# =============================================================================
#
# Important:
#
# Keep TACOEvaluator inside a CPU Ray worker process.
#
# This mirrors the stable GRPO topology:
#
#   CPU reward worker
#       -> FrozenCoderWorker RPC
#       -> TACOEvaluator multiprocessing/spawn
#
# rather than running evaluator subprocesses directly from the
# controller/main process.
# =============================================================================


class NonFailFastDiagnosticWorker:
    def __init__(
        self,
        *,
        frozen_coder_handle: Any,
        timeout_seconds: int,
        max_tests: int,
    ) -> None:
        if frozen_coder_handle is None:
            raise ValueError(
                "frozen_coder_handle is required."
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be > 0."
            )

        if max_tests <= 0:
            raise ValueError(
                "max_tests must be > 0."
            )

        self.frozen_coder_handle = (
            frozen_coder_handle
        )

        self.timeout_seconds = int(
            timeout_seconds
        )

        self.max_tests = int(
            max_tests
        )

        self.parser = CodeParser()

        self.evaluator = TACOEvaluator(
            timeout_seconds=(
                self.timeout_seconds
            ),
            debug=False,
        )

    def evaluate_trajectory(
        self,
        *,
        extra_info: dict[str, Any],
        plan: str,
        original_reward: float,
        save_code: bool,
    ) -> dict[str, Any]:
        # ------------------------------------------------------------------
        # 1. Restore exact ProblemExample from the RL parquet.
        # ------------------------------------------------------------------

        problem = _problem_from_extra_info(
            extra_info
        )

        problem_text = problem.problem

        # ------------------------------------------------------------------
        # 2. Select the SAME reward-test subset used during training.
        #
        #    select_reward_tests:
        #        longest-input tests
        #        up to max_tests
        # ------------------------------------------------------------------

        reward_problem = (
            select_reward_tests(
                problem,
                max_tests=(
                    self.max_tests
                ),
            )
        )

        selected_tests = list(
            reward_problem.private_tests
        )

        reward_test_count = len(
            selected_tests
        )

        if reward_test_count <= 0:
            return {
                "original_reward": float(
                    original_reward
                ),
                "status": "NO_TESTS",
                "true_passed_tests": 0,
                "reward_tests": 0,
                "true_test_pass_ratio": 0.0,
                "true_binary_reward": 0.0,
                "binary_reward_match": (
                    float(original_reward)
                    == 0.0
                ),
                "per_test_results": [],
            }

        # ------------------------------------------------------------------
        # 3. Rebuild EXACT same plan -> code prompt.
        # ------------------------------------------------------------------

        coder_prompt = build_code_prompt(
            problem_text=problem_text,
            plan=plan,
            starter_code=(
                problem.starter_code
            ),
        )

        # ------------------------------------------------------------------
        # 4. Deterministic frozen coder.
        #
        # Same RPC helper used by training reward.
        # ------------------------------------------------------------------

        try:
            (
                raw_code_output,
                coder_prompt_tokens,
                coder_completion_tokens,
                coder_generation_time,
            ) = _generate_code_via_rpc(
                frozen_coder_handle=(
                    self.frozen_coder_handle
                ),
                prompt=coder_prompt,
            )

        except Exception as exc:
            return {
                "original_reward": float(
                    original_reward
                ),
                "status": (
                    "CODE_GENERATION_ERROR"
                ),
                "true_passed_tests": 0,
                "reward_tests": (
                    reward_test_count
                ),
                "true_test_pass_ratio": 0.0,
                "true_binary_reward": 0.0,
                "binary_reward_match": (
                    float(original_reward)
                    == 0.0
                ),
                "error_message": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                "per_test_results": [],
            }

        # ------------------------------------------------------------------
        # 5. Same CodeParser as training.
        # ------------------------------------------------------------------

        try:
            parse_result = (
                self.parser.parse(
                    raw_code_output
                )
            )

        except Exception as exc:
            return {
                "original_reward": float(
                    original_reward
                ),
                "status": (
                    "CODE_PARSING_ERROR"
                ),
                "true_passed_tests": 0,
                "reward_tests": (
                    reward_test_count
                ),
                "true_test_pass_ratio": 0.0,
                "true_binary_reward": 0.0,
                "binary_reward_match": (
                    float(original_reward)
                    == 0.0
                ),
                "error_message": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                "per_test_results": [],
            }

        if (
            parse_result.status
            != "SUCCESS"
        ):
            return {
                "original_reward": float(
                    original_reward
                ),
                "status": str(
                    parse_result.status
                ),
                "true_passed_tests": 0,
                "reward_tests": (
                    reward_test_count
                ),
                "true_test_pass_ratio": 0.0,
                "true_binary_reward": 0.0,
                "binary_reward_match": (
                    float(original_reward)
                    == 0.0
                ),
                "code_extraction_method": str(
                    parse_result.extraction_method
                ),
                "per_test_results": [],
            }

        generated_code = (
            parse_result.code
        )

        # ------------------------------------------------------------------
        # 6. NON-FAIL-FAST diagnostic.
        #
        # Important:
        # Do not send all 15 tests to TACOEvaluator at once.
        #
        # Instead:
        #
        #   test 0 -> TACOEvaluator(one test)
        #   test 1 -> TACOEvaluator(one test)
        #   ...
        #   test 14 -> TACOEvaluator(one test)
        #
        # Each evaluator invocation contains exactly one private test,
        # so a failure on one test cannot prevent the remaining tests
        # from being executed.
        # ------------------------------------------------------------------

        per_test_results: list[
            dict[str, Any]
        ] = []

        true_passed_tests = 0

        status_counter: Counter[
            str
        ] = Counter()

        total_execution_time = 0.0

        for test_index, test_case in enumerate(
            selected_tests
        ):
            one_test_problem = (
                copy.deepcopy(
                    reward_problem
                )
            )

            one_test_problem.private_tests = [
                copy.deepcopy(
                    test_case
                )
            ]

            try:
                evaluation = (
                    self.evaluator.evaluate(
                        problem=(
                            one_test_problem
                        ),
                        code=generated_code,
                    )
                )

                passed = bool(
                    evaluation.passed
                )

                status = str(
                    evaluation.status
                )

                execution_time = float(
                    evaluation.execution_time
                )

                error_message = (
                    ""
                    if evaluation.error_message
                    is None
                    else str(
                        evaluation.error_message
                    )
                )

            except Exception as exc:
                passed = False

                status = (
                    "EVALUATION_ERROR"
                )

                execution_time = 0.0

                error_message = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            if passed:
                true_passed_tests += 1

            status_counter[
                status
            ] += 1

            total_execution_time += (
                execution_time
            )

            per_test_results.append(
                {
                    "test_index": int(
                        test_index
                    ),
                    "passed": passed,
                    "status": status,
                    "execution_time": (
                        execution_time
                    ),
                    "error_message": (
                        error_message
                    ),
                }
            )

        # ------------------------------------------------------------------
        # 7. True K / selected-tests TPR.
        # ------------------------------------------------------------------

        true_tpr = (
            true_passed_tests
            / reward_test_count
        )

        true_binary_reward = (
            1.0
            if (
                true_passed_tests
                == reward_test_count
            )
            else 0.0
        )

        result = {
            "original_reward": float(
                original_reward
            ),

            "status": (
                "PASS"
                if true_binary_reward == 1.0
                else "PARTIAL_OR_FAIL"
            ),

            "true_passed_tests": int(
                true_passed_tests
            ),

            "reward_tests": int(
                reward_test_count
            ),

            "true_test_pass_ratio": float(
                true_tpr
            ),

            "true_binary_reward": float(
                true_binary_reward
            ),

            # Strong protocol sanity check.
            "binary_reward_match": bool(
                float(
                    original_reward
                )
                == true_binary_reward
            ),

            "per_test_status_counts": dict(
                status_counter
            ),

            "total_independent_test_execution_time": float(
                total_execution_time
            ),

            "coder_prompt_tokens": int(
                coder_prompt_tokens
            ),

            "coder_completion_tokens": int(
                coder_completion_tokens
            ),

            "coder_generation_time": float(
                coder_generation_time
            ),

            "code_extraction_method": str(
                parse_result.extraction_method
            ),

            "per_test_results": (
                per_test_results
            ),
        }

        if save_code:
            result[
                "raw_code_output"
            ] = raw_code_output

            result[
                "generated_code"
            ] = generated_code

        return result


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    input_path = resolve_path(
        args.input
    )

    train_path = resolve_path(
        args.train_parquet
    )

    config_path = resolve_path(
        args.config
    )

    output_path = resolve_path(
        args.output
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found: "
            f"{input_path}"
        )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Train parquet not found: "
            f"{train_path}"
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: "
            f"{config_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output exists: "
                f"{output_path}\n"
                "Use --overwrite."
            )

        output_path.unlink()

    config = OmegaConf.load(
        config_path
    )

    max_tests = (
        int(args.max_tests)
        if args.max_tests
        is not None
        else int(
            config.reward.max_tests
        )
    )

    timeout_seconds = (
        int(args.timeout_seconds)
        if args.timeout_seconds
        is not None
        else int(
            config.reward.timeout_seconds
        )
    )

    # ------------------------------------------------------------------
    # Existing sampled trajectories.
    # ------------------------------------------------------------------

    groups = load_jsonl(
        input_path
    )

    if args.group_filter != "all":
        groups = [
            group
            for group in groups
            if group.get(
                "group_type"
            )
            == args.group_filter
        ]

    if args.max_groups is not None:
        groups = groups[
            : args.max_groups
        ]

    if not groups:
        raise RuntimeError(
            "No groups matched "
            f"group_filter={args.group_filter!r}."
        )

    # ------------------------------------------------------------------
    # Original TACO parquet.
    # ------------------------------------------------------------------

    train_df = pd.read_parquet(
        train_path
    )

    print()
    print("=" * 100)
    print(
        "TACO NON-FAIL-FAST REWARD GRANULARITY DIAGNOSTIC"
    )
    print("=" * 100)

    print(
        f"Input            : "
        f"{input_path}"
    )

    print(
        f"Train parquet    : "
        f"{train_path}"
    )

    print(
        f"Group filter     : "
        f"{args.group_filter}"
    )

    print(
        f"Groups           : "
        f"{len(groups)}"
    )

    print(
        f"Reward tests     : "
        f"{max_tests}"
    )

    print(
        f"Timeout          : "
        f"{timeout_seconds}s"
    )

    print(
        f"Output           : "
        f"{output_path}"
    )

    # ------------------------------------------------------------------
    # Ray topology.
    # ------------------------------------------------------------------

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=True,
        )

    frozen_coder = None
    diagnostic_worker = None

    # ------------------------------------------------------------------
    # Global stats.
    # ------------------------------------------------------------------

    total_trajectories = 0

    total_true_partial_positive = 0
    total_original_positive = 0

    total_true_tpr = 0.0

    binary_mismatches = 0

    true_tpr_variable_groups = 0

    any_partial_signal_groups = 0

    try:
        # --------------------------------------------------------------
        # Frozen coder actor.
        # --------------------------------------------------------------

        frozen_coder = (
            start_frozen_coder(
                config_path
            )
        )

        # --------------------------------------------------------------
        # CPU diagnostic worker.
        # --------------------------------------------------------------

        DiagnosticActor = ray.remote(
            num_cpus=1,
        )(
            NonFailFastDiagnosticWorker
        )

        diagnostic_worker = (
            DiagnosticActor.remote(
                frozen_coder_handle=(
                    frozen_coder
                ),
                timeout_seconds=(
                    timeout_seconds
                ),
                max_tests=max_tests,
            )
        )

        # --------------------------------------------------------------
        # Groups.
        # --------------------------------------------------------------

        for group_index, group in enumerate(
            groups
        ):
            dataset_index = int(
                group[
                    "dataset_index"
                ]
            )

            problem_id = str(
                group[
                    "problem_id"
                ]
            )

            original_group_type = str(
                group.get(
                    "group_type",
                    "unknown",
                )
            )

            if (
                dataset_index < 0
                or dataset_index
                >= len(train_df)
            ):
                raise IndexError(
                    f"dataset_index="
                    f"{dataset_index} "
                    "outside train parquet."
                )

            row = train_df.iloc[
                dataset_index
            ]

            extra_info = pythonize(
                row[
                    "extra_info"
                ]
            )

            trajectories = group.get(
                "trajectories",
                []
            )

            if not isinstance(
                trajectories,
                list,
            ):
                raise TypeError(
                    "group['trajectories'] "
                    "must be list."
                )

            print()
            print(
                f"[{group_index + 1:02d}/"
                f"{len(groups):02d}] "
                f"row={dataset_index} "
                f"{problem_id} "
                f"original={original_group_type}"
            )

            group_results = []

            true_tprs: list[
                float
            ] = []

            original_rewards: list[
                float
            ] = []

            # ----------------------------------------------------------
            # Evaluate serially.
            #
            # The coder actor and execution stack were intentionally
            # serialized in the training reward path as well.
            # ----------------------------------------------------------

            for trajectory_index, trajectory in enumerate(
                trajectories
            ):
                plan = str(
                    trajectory.get(
                        "plan",
                        ""
                    )
                )

                original_reward = float(
                    trajectory.get(
                        "reward",
                        0.0,
                    )
                )

                if not plan.strip():
                    raise ValueError(
                        f"Empty saved plan: "
                        f"{problem_id}, "
                        f"trajectory="
                        f"{trajectory_index}"
                    )

                result = ray.get(
                    diagnostic_worker
                    .evaluate_trajectory
                    .remote(
                        extra_info=(
                            extra_info
                        ),
                        plan=plan,
                        original_reward=(
                            original_reward
                        ),
                        save_code=(
                            args.save_code
                        ),
                    )
                )

                true_tpr = float(
                    result[
                        "true_test_pass_ratio"
                    ]
                )

                true_tprs.append(
                    true_tpr
                )

                original_rewards.append(
                    original_reward
                )

                total_trajectories += 1

                total_original_positive += int(
                    original_reward > 0.0
                )

                total_true_partial_positive += int(
                    true_tpr > 0.0
                )

                total_true_tpr += (
                    true_tpr
                )

                if not bool(
                    result[
                        "binary_reward_match"
                    ]
                ):
                    binary_mismatches += 1

                output_trajectory = {
                    "sample_index": int(
                        trajectory.get(
                            "sample_index",
                            trajectory_index,
                        )
                    ),

                    "original_reward": (
                        original_reward
                    ),

                    "original_reward_test_progress": float(
                        trajectory.get(
                            "reward_test_progress",
                            0.0,
                        )
                    ),

                    "plan": plan,

                    **result,
                }

                group_results.append(
                    output_trajectory
                )

                print(
                    f"  sample "
                    f"{trajectory_index + 1:02d}/"
                    f"{len(trajectories):02d} "
                    f"R={original_reward:.0f} "
                    f"trueTPR="
                    f"{true_tpr:.4f} "
                    f"tests="
                    f"{result['true_passed_tests']}/"
                    f"{result['reward_tests']} "
                    f"binary_match="
                    f"{result['binary_reward_match']}"
                )

            # ----------------------------------------------------------
            # Group-level true partial reward.
            # ----------------------------------------------------------

            true_tpr_mean = (
                sum(true_tprs)
                / len(true_tprs)
                if true_tprs
                else 0.0
            )

            true_tpr_variance = (
                population_variance(
                    true_tprs
                )
            )

            true_tpr_min = (
                min(true_tprs)
                if true_tprs
                else 0.0
            )

            true_tpr_max = (
                max(true_tprs)
                if true_tprs
                else 0.0
            )

            partial_positive_count = sum(
                tpr > 0.0
                for tpr in true_tprs
            )

            binary_variance = (
                population_variance(
                    original_rewards
                )
            )

            binary_flat = (
                binary_variance
                <= 1e-12
            )

            true_tpr_variable = (
                true_tpr_variance
                > 1e-12
            )

            hidden_true_partial_signal = (
                binary_flat
                and true_tpr_variable
            )

            if true_tpr_variable:
                true_tpr_variable_groups += 1

            if hidden_true_partial_signal:
                any_partial_signal_groups += 1

            output_group = {
                "dataset_index": (
                    dataset_index
                ),

                "problem_id": (
                    problem_id
                ),

                "original_group_type": (
                    original_group_type
                ),

                "num_trajectories": (
                    len(group_results)
                ),

                "original_reward_variance": (
                    binary_variance
                ),

                "true_tpr_mean": (
                    true_tpr_mean
                ),

                "true_tpr_variance": (
                    true_tpr_variance
                ),

                "true_tpr_min": (
                    true_tpr_min
                ),

                "true_tpr_max": (
                    true_tpr_max
                ),

                "partial_positive_trajectories": (
                    partial_positive_count
                ),

                "binary_flat": (
                    binary_flat
                ),

                "true_tpr_variable": (
                    true_tpr_variable
                ),

                "hidden_true_partial_signal": (
                    hidden_true_partial_signal
                ),

                "trajectories": (
                    group_results
                ),
            }

            append_jsonl(
                output_path,
                output_group,
            )

            print(
                "  -> "
                f"trueTPR mean="
                f"{true_tpr_mean:.4f}, "
                f"var="
                f"{true_tpr_variance:.6f}, "
                f"range="
                f"[{true_tpr_min:.4f}, "
                f"{true_tpr_max:.4f}], "
                f"partial>0="
                f"{partial_positive_count}/"
                f"{len(true_tprs)}"
            )

            if hidden_true_partial_signal:
                print(
                    "     "
                    "[HIDDEN TRUE PARTIAL SIGNAL] "
                    "binary reward is flat, "
                    "but non-fail-fast TPR varies."
                )

    finally:
        if diagnostic_worker is not None:
            try:
                ray.kill(
                    diagnostic_worker,
                    no_restart=True,
                )
            except Exception:
                pass

        if frozen_coder is not None:
            try:
                ray.kill(
                    frozen_coder,
                    no_restart=True,
                )
            except Exception:
                pass

        if ray.is_initialized():
            ray.shutdown()

    # =========================================================================
    # Summary
    # =========================================================================

    mean_true_tpr = (
        total_true_tpr
        / total_trajectories
        if total_trajectories > 0
        else 0.0
    )

    print()
    print("=" * 100)
    print(
        "NON-FAIL-FAST REWARD GRANULARITY SUMMARY"
    )
    print("=" * 100)

    print(
        f"Groups evaluated             : "
        f"{len(groups)}"
    )

    print(
        f"Trajectories                 : "
        f"{total_trajectories}"
    )

    print()

    print(
        f"True-TPR-variable groups     : "
        f"{true_tpr_variable_groups}/"
        f"{len(groups)} "
        f"("
        f"{100.0 * true_tpr_variable_groups / len(groups):.2f}%"
        f")"
    )

    print(
        f"Binary-flat + true-TPR-var   : "
        f"{any_partial_signal_groups}/"
        f"{len(groups)} "
        f"("
        f"{100.0 * any_partial_signal_groups / len(groups):.2f}%"
        f")"
    )

    print()

    print(
        f"Original reward-positive     : "
        f"{total_original_positive}/"
        f"{total_trajectories} "
        f"("
        f"{100.0 * total_original_positive / total_trajectories:.2f}%"
        f")"
    )

    print(
        f"True partial-positive (TPR>0): "
        f"{total_true_partial_positive}/"
        f"{total_trajectories} "
        f"("
        f"{100.0 * total_true_partial_positive / total_trajectories:.2f}%"
        f")"
    )

    print(
        f"Mean true TPR                : "
        f"{mean_true_tpr:.6f}"
    )

    print()

    print(
        f"Binary sanity mismatches     : "
        f"{binary_mismatches}/"
        f"{total_trajectories}"
    )

    if binary_mismatches == 0:
        print(
            "[PASS] Regenerated deterministic coder outputs "
            "are binary-consistent with the original reward."
        )

    else:
        print(
            "[WARNING] Binary mismatches were observed. "
            "Inspect these cases before using the TPR diagnostic "
            "for method selection."
        )

    print()

    print(
        f"Output                       : "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()