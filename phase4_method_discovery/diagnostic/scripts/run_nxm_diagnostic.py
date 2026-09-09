"""
PYTHONPATH="$HOME/workspace/project_sLM_planning" \
python \
  phase4_method_discovery/diagnostic/scripts/run_nxm_diagnostic.py \
  --config \
  phase4_method_discovery/diagnostic/configs/nxm_qwen25coder3b.yaml \
  --overwrite
"""
# phase4_method_discovery/diagnostic/scripts/run_nxm_diagnostic.py

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.execution.taco_evaluator import TACOEvaluator
from src.models.generator import ModelGenerator
from src.parsing.code_parser import CodeParser

from phase1_planning_bottleneck.strategies.self_plan import (
    SelfPlanningStrategy,
)

from phase4_method_discovery.vanilla_planning_rlvr.evaluation.checkpoint_dataset import (
    load_checkpoint_eval_dataset,
)

from phase4_method_discovery.vanilla_planning_rlvr.reward.planning_execution_reward import (
    build_code_prompt,
    select_reward_tests,
)


# =============================================================================
# Seed utilities
# =============================================================================


def make_seed(
    base_seed: int,
    *parts: object,
) -> int:
    """
    Construct a deterministic seed from a base seed and arbitrary identifiers.

    Python's built-in hash() is intentionally avoided because it is affected by
    hash randomization between interpreter processes.

    Examples
    --------
    Plan:
        make_seed(
            base_seed,
            problem_id,
            "plan",
            plan_index,
        )

    Code:
        make_seed(
            base_seed,
            problem_id,
            "plan",
            plan_index,
            "code",
            code_index,
        )
    """

    text = "::".join(
        [str(base_seed)]
        + [str(part) for part in parts]
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    # 32-bit unsigned seed.
    return int.from_bytes(
        digest[:4],
        byteorder="big",
        signed=False,
    )


def set_generation_seed(
    seed: int,
) -> None:
    """
    Set PyTorch RNG state immediately before one generation call.

    For the current single-GPU diagnostic this is sufficient to make each
    plan/code sampling trajectory explicitly reproducible.
    """

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# =============================================================================
# I/O utilities
# =============================================================================


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


def prepare_output_path(
    path: Path,
    *,
    overwrite: bool,
) -> None:
    """
    Avoid accidentally mixing multiple diagnostic runs in one JSONL file.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not path.exists():
        return

    if overwrite:
        path.unlink()
        return

    raise FileExistsError(
        "Output file already exists: "
        f"{path}\n"
        "Use --overwrite to replace it."
    )


# =============================================================================
# Configuration validation
# =============================================================================


def validate_config(
    config: DictConfig,
) -> None:
    required_sections = (
        "experiment",
        "data",
        "model",
        "prompts",
        "planner",
        "coder",
        "evaluation",
        "output",
    )

    for section in required_sections:
        if not hasattr(
            config,
            section,
        ):
            raise ValueError(
                "Missing config section: "
                f"{section}"
            )

    if int(
        config.experiment.seed
    ) < 0:
        raise ValueError(
            "experiment.seed must be >= 0."
        )

    if int(
        config.data.limit
    ) <= 0:
        raise ValueError(
            "data.limit must be > 0."
        )

    if int(
        config.planner.num_plans
    ) <= 0:
        raise ValueError(
            "planner.num_plans must be > 0."
        )

    if int(
        config.coder.num_codes_per_plan
    ) <= 0:
        raise ValueError(
            "coder.num_codes_per_plan must be > 0."
        )

    if int(
        config.planner.max_new_tokens
    ) <= 0:
        raise ValueError(
            "planner.max_new_tokens must be > 0."
        )

    if int(
        config.coder.max_new_tokens
    ) <= 0:
        raise ValueError(
            "coder.max_new_tokens must be > 0."
        )

    if float(
        config.planner.temperature
    ) < 0:
        raise ValueError(
            "planner.temperature must be >= 0."
        )

    if float(
        config.coder.temperature
    ) < 0:
        raise ValueError(
            "coder.temperature must be >= 0."
        )

    if not (
        0.0
        < float(
            config.planner.top_p
        )
        <= 1.0
    ):
        raise ValueError(
            "planner.top_p must be in (0, 1]."
        )

    if not (
        0.0
        < float(
            config.coder.top_p
        )
        <= 1.0
    ):
        raise ValueError(
            "coder.top_p must be in (0, 1]."
        )

    if int(
        config.evaluation.max_reward_tests
    ) <= 0:
        raise ValueError(
            "evaluation.max_reward_tests must be > 0."
        )

    if int(
        config.evaluation.timeout_seconds
    ) <= 0:
        raise ValueError(
            "evaluation.timeout_seconds must be > 0."
        )


# =============================================================================
# Code-result construction helpers
# =============================================================================


def build_generation_failure_record(
    *,
    code_index: int,
    code_seed: int,
    scheduled_tests: int,
    error: Exception,
) -> dict[str, Any]:
    return {
        "code_index": int(
            code_index
        ),
        "code_seed": int(
            code_seed
        ),
        "raw_output": "",
        "generated_code": "",
        "code_extraction_method": "none",
        "status": "CODE_GENERATION_ERROR",
        "passed": False,
        "binary_reward": 0.0,
        "scheduled_tests": int(
            scheduled_tests
        ),
        "executed_tests": 0,
        "passed_tests": 0,
        "total_tests": 0,
        "test_pass_ratio": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "generation_time": 0.0,
        "execution_time": 0.0,
        "error_message": (
            f"{type(error).__name__}: "
            f"{error}"
        ),
        "per_test_results": [],
    }


def build_parsing_failure_record(
    *,
    code_index: int,
    code_seed: int,
    raw_output: str,
    generated_code: str,
    extraction_method: str,
    status: str,
    scheduled_tests: int,
    prompt_tokens: int,
    completion_tokens: int,
    generation_time: float,
    error_message: str,
) -> dict[str, Any]:
    return {
        "code_index": int(
            code_index
        ),
        "code_seed": int(
            code_seed
        ),
        "raw_output": str(
            raw_output
        ),
        "generated_code": str(
            generated_code
        ),
        "code_extraction_method": str(
            extraction_method
        ),
        "status": str(
            status
        ),
        "passed": False,
        "binary_reward": 0.0,
        "scheduled_tests": int(
            scheduled_tests
        ),
        "executed_tests": 0,
        "passed_tests": 0,
        "total_tests": 0,
        "test_pass_ratio": 0.0,
        "prompt_tokens": int(
            prompt_tokens
        ),
        "completion_tokens": int(
            completion_tokens
        ),
        "generation_time": float(
            generation_time
        ),
        "execution_time": 0.0,
        "error_message": str(
            error_message
        ),
        "per_test_results": [],
    }


def serialize_test_results(
    evaluation: Any,
) -> list[dict[str, Any]]:
    serialized: list[
        dict[str, Any]
    ] = []

    for result in (
        evaluation.test_results
        or []
    ):
        serialized.append(
            {
                "test_index": int(
                    result.test_index
                ),
                "passed": bool(
                    result.passed
                ),
                "status": str(
                    result.status
                ),
                "input_text": str(
                    result.input_text
                ),
                "expected_output": str(
                    result.expected_output
                ),
                "actual_output": str(
                    result.actual_output
                ),
                "execution_time": float(
                    result.execution_time
                ),
                "stderr": str(
                    result.stderr
                    or ""
                ),
                "metadata": dict(
                    result.metadata
                    or {}
                ),
            }
        )

    return serialized


# =============================================================================
# Core diagnostic
# =============================================================================


def run(
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    config_path = Path(
        config_path
    )

    if not config_path.exists():
        raise FileNotFoundError(
            "Diagnostic config not found: "
            f"{config_path}"
        )

    config = OmegaConf.load(
        config_path
    )

    validate_config(
        config
    )

    output_path = Path(
        str(
            config.output.path
        )
    )

    prepare_output_path(
        output_path,
        overwrite=overwrite,
    )

    # =========================================================================
    # 1. Dataset
    # =========================================================================

    examples = (
        load_checkpoint_eval_dataset(
            parquet_path=str(
                config.data.parquet_path
            ),
            limit=int(
                config.data.limit
            ),
        )
    )

    if not examples:
        raise RuntimeError(
            "No diagnostic problems loaded."
        )

    # =========================================================================
    # 2. Shared base generator
    #
    # Initial diagnostic:
    #
    #   planner = Qwen2.5-Coder-3B-Instruct
    #   coder   = same frozen base checkpoint
    #
    # Only one model instance is loaded to avoid unnecessary GPU duplication.
    # =========================================================================

    generator = ModelGenerator(
        model_name_or_path=str(
            config.model.name_or_path
        ),
        dtype=str(
            config.model.dtype
        ),
        device_map=str(
            config.model.device_map
        ),
        trust_remote_code=bool(
            config.model.trust_remote_code
        ),
    )

    # =========================================================================
    # 3. Planner prompt builder
    #
    # We intentionally do NOT call SelfPlanningStrategy.run().
    #
    # The diagnostic needs:
    #
    #       N plans
    #          x
    #       M codes per fixed plan
    #
    # whereas SelfPlanningStrategy.run() is a 1-plan -> 1-code strategy.
    # =========================================================================

    prompt_strategy = (
        SelfPlanningStrategy(
            generator=generator,
            plan_prompt_path=str(
                config.prompts.plan_prompt_path
            ),
            code_prompt_path=str(
                config.prompts.code_prompt_path
            ),
        )
    )

    code_parser = (
        CodeParser()
    )

    evaluator = (
        TACOEvaluator(
            timeout_seconds=int(
                config.evaluation.timeout_seconds
            ),
            debug=False,
        )
    )

    # =========================================================================
    # 4. Experiment parameters
    # =========================================================================

    base_seed = int(
        config.experiment.seed
    )

    num_plans = int(
        config.planner.num_plans
    )

    num_codes_per_plan = int(
        config.coder.num_codes_per_plan
    )

    max_reward_tests = int(
        config.evaluation.max_reward_tests
    )

    print(
        "=" * 88
    )
    print(
        "N x M Planning Diagnostic"
    )
    print(
        "=" * 88
    )

    print(
        f"experiment      : "
        f"{config.experiment.name}"
    )

    print(
        f"problems        : "
        f"{len(examples)}"
    )

    print(
        f"N plans         : "
        f"{num_plans}"
    )

    print(
        f"M codes / plan  : "
        f"{num_codes_per_plan}"
    )

    print(
        f"trajectories    : "
        f"{len(examples) * num_plans * num_codes_per_plan}"
    )

    print(
        f"planner temp    : "
        f"{float(config.planner.temperature)}"
    )

    print(
        f"coder temp      : "
        f"{float(config.coder.temperature)}"
    )

    print(
        f"max reward tests: "
        f"{max_reward_tests}"
    )

    print(
        f"output          : "
        f"{output_path}"
    )

    print(
        "=" * 88
    )

    # =========================================================================
    # 5. Problem loop
    # =========================================================================

    for (
        problem_index,
        example,
    ) in enumerate(
        examples
    ):
        print()
        print(
            "=" * 88
        )
        print(
            f"[Problem "
            f"{problem_index + 1}/"
            f"{len(examples)}] "
            f"{example.problem_id}"
        )
        print(
            "=" * 88
        )

        # ---------------------------------------------------------------------
        # Planner prompt is identical for all N sampled plans.
        # ---------------------------------------------------------------------

        plan_prompt = (
            prompt_strategy
            .build_plan_prompt(
                example
            )
        )

        # ---------------------------------------------------------------------
        # Reward-test subset is selected ONCE per problem.
        #
        # Therefore every plan and every code realization for this problem
        # is scored on exactly the same selected tests.
        # ---------------------------------------------------------------------

        reward_problem = (
            select_reward_tests(
                example,
                max_tests=(
                    max_reward_tests
                ),
            )
        )

        reward_tests = len(
            reward_problem.private_tests
        )

        if reward_tests <= 0:
            raise RuntimeError(
                "No reward tests selected: "
                f"{example.problem_id}"
            )

        print(
            f"Available tests : "
            f"{len(example.private_tests)}"
        )

        print(
            f"Reward tests    : "
            f"{reward_tests}"
        )

        # =====================================================================
        # 6. Plan loop
        # =====================================================================

        for plan_index in range(
            num_plans
        ):
            plan_seed = (
                make_seed(
                    base_seed,
                    example.problem_id,
                    "plan",
                    plan_index,
                )
            )

            set_generation_seed(
                plan_seed
            )

            # -----------------------------------------------------------------
            # Generate one stochastic plan.
            # -----------------------------------------------------------------

            try:
                plan_generation = (
                    generator.generate(
                        prompt=plan_prompt,
                        max_new_tokens=int(
                            config.planner.max_new_tokens
                        ),
                        temperature=float(
                            config.planner.temperature
                        ),
                        top_p=float(
                            config.planner.top_p
                        ),
                    )
                )

            except Exception as exc:
                raise RuntimeError(
                    "Plan generation failed: "
                    f"problem={example.problem_id}, "
                    f"plan_index={plan_index}, "
                    f"seed={plan_seed}, "
                    f"error={type(exc).__name__}: "
                    f"{exc}"
                ) from exc

            plan = (
                plan_generation
                .text
                .strip()
            )

            if not plan:
                raise RuntimeError(
                    "Generated empty plan: "
                    f"problem={example.problem_id}, "
                    f"plan_index={plan_index}"
                )

            print()
            print(
                "-" * 88
            )

            print(
                f"PLAN "
                f"{plan_index + 1}/"
                f"{num_plans} "
                f"seed={plan_seed}"
            )

            print(
                "-" * 88
            )

            print(
                plan
            )

            # -----------------------------------------------------------------
            # IMPORTANT:
            #
            # Construct the code prompt exactly ONCE for this plan.
            #
            # The M coder realizations therefore vary ONLY in sampling RNG.
            #
            # Same:
            #   - problem
            #   - plan
            #   - prompt
            #   - frozen model
            #   - reward tests
            #
            # Different:
            #   - code sampling seed
            # -----------------------------------------------------------------

            code_prompt = (
                build_code_prompt(
                    problem_text=(
                        example.problem
                    ),
                    plan=plan,
                    starter_code=(
                        example.starter_code
                    ),
                )
            )

            code_records: list[
                dict[str, Any]
            ] = []

            # =================================================================
            # 7. Code realization loop
            # =================================================================

            for code_index in range(
                num_codes_per_plan
            ):
                code_seed = (
                    make_seed(
                        base_seed,
                        example.problem_id,
                        "plan",
                        plan_index,
                        "code",
                        code_index,
                    )
                )

                set_generation_seed(
                    code_seed
                )

                # -------------------------------------------------------------
                # 7-1. Frozen coder generation
                # -------------------------------------------------------------

                try:
                    code_generation = (
                        generator.generate(
                            prompt=code_prompt,
                            max_new_tokens=int(
                                config.coder.max_new_tokens
                            ),
                            temperature=float(
                                config.coder.temperature
                            ),
                            top_p=float(
                                config.coder.top_p
                            ),
                        )
                    )

                except Exception as exc:
                    code_record = (
                        build_generation_failure_record(
                            code_index=(
                                code_index
                            ),
                            code_seed=(
                                code_seed
                            ),
                            scheduled_tests=(
                                reward_tests
                            ),
                            error=exc,
                        )
                    )

                    code_records.append(
                        code_record
                    )

                    print(
                        f"  CODE "
                        f"{code_index + 1}/"
                        f"{num_codes_per_plan} "
                        f"seed={code_seed} "
                        f"status=CODE_GENERATION_ERROR"
                    )

                    continue

                raw_output = (
                    code_generation.text
                )

                # -------------------------------------------------------------
                # 7-2. Parse generated code
                # -------------------------------------------------------------

                try:
                    parse_result = (
                        code_parser.parse(
                            raw_output
                        )
                    )

                except Exception as exc:
                    code_record = (
                        build_parsing_failure_record(
                            code_index=(
                                code_index
                            ),
                            code_seed=(
                                code_seed
                            ),
                            raw_output=(
                                raw_output
                            ),
                            generated_code="",
                            extraction_method="none",
                            status=(
                                "CODE_PARSING_ERROR"
                            ),
                            scheduled_tests=(
                                reward_tests
                            ),
                            prompt_tokens=int(
                                code_generation
                                .prompt_tokens
                            ),
                            completion_tokens=int(
                                code_generation
                                .completion_tokens
                            ),
                            generation_time=float(
                                code_generation
                                .generation_time
                            ),
                            error_message=(
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            ),
                        )
                    )

                    code_records.append(
                        code_record
                    )

                    print(
                        f"  CODE "
                        f"{code_index + 1}/"
                        f"{num_codes_per_plan} "
                        f"seed={code_seed} "
                        f"status=CODE_PARSING_ERROR"
                    )

                    continue

                if (
                    parse_result.status
                    != "SUCCESS"
                ):
                    code_record = (
                        build_parsing_failure_record(
                            code_index=(
                                code_index
                            ),
                            code_seed=(
                                code_seed
                            ),
                            raw_output=(
                                raw_output
                            ),
                            generated_code=(
                                parse_result.code
                            ),
                            extraction_method=(
                                parse_result
                                .extraction_method
                            ),
                            status=str(
                                parse_result.status
                            ),
                            scheduled_tests=(
                                reward_tests
                            ),
                            prompt_tokens=int(
                                code_generation
                                .prompt_tokens
                            ),
                            completion_tokens=int(
                                code_generation
                                .completion_tokens
                            ),
                            generation_time=float(
                                code_generation
                                .generation_time
                            ),
                            error_message=(
                                "Code parser returned "
                                f"{parse_result.status}."
                            ),
                        )
                    )

                    code_records.append(
                        code_record
                    )

                    print(
                        f"  CODE "
                        f"{code_index + 1}/"
                        f"{num_codes_per_plan} "
                        f"seed={code_seed} "
                        f"status="
                        f"{parse_result.status}"
                    )

                    continue

                generated_code = (
                    parse_result.code
                )

                # -------------------------------------------------------------
                # 7-3. Non-fail-fast TACO evaluation
                #
                # One evaluation supplies BOTH:
                #
                #   binary reward:
                #       all selected tests pass
                #
                #   dense reward:
                #       passed_tests / total_tests
                #
                # We use EvaluationResult.passed/status directly rather than
                # reconstructing evaluator semantics inside the diagnostic.
                # -------------------------------------------------------------

                try:
                    evaluation = (
                        evaluator
                        .evaluate_non_fail_fast(
                            problem=(
                                reward_problem
                            ),
                            code=(
                                generated_code
                            ),
                        )
                    )

                    passed = bool(
                        evaluation.passed
                    )

                    status = str(
                        evaluation.status
                    )

                    passed_tests = int(
                        evaluation.passed_tests
                    )

                    total_tests = int(
                        evaluation.total_tests
                    )

                    # Non-fail-fast evaluation should always account for
                    # every selected reward test.
                    if (
                        total_tests
                        != reward_tests
                    ):
                        raise RuntimeError(
                            "Unexpected non-fail-fast "
                            "test count: "
                            f"problem="
                            f"{example.problem_id}, "
                            f"plan_index="
                            f"{plan_index}, "
                            f"code_index="
                            f"{code_index}, "
                            f"expected="
                            f"{reward_tests}, "
                            f"actual="
                            f"{total_tests}"
                        )

                    if not (
                        0
                        <= passed_tests
                        <= total_tests
                    ):
                        raise RuntimeError(
                            "Invalid evaluator counts: "
                            f"passed_tests="
                            f"{passed_tests}, "
                            f"total_tests="
                            f"{total_tests}"
                        )

                    test_pass_ratio = (
                        passed_tests
                        / total_tests
                    )

                    binary_reward = (
                        1.0
                        if passed
                        else 0.0
                    )

                    # Defensive consistency checks.
                    if (
                        passed
                        and passed_tests
                        != total_tests
                    ):
                        raise RuntimeError(
                            "Evaluator inconsistency: "
                            "passed=True but not all "
                            "tests passed."
                        )

                    if (
                        not passed
                        and passed_tests
                        == total_tests
                    ):
                        raise RuntimeError(
                            "Evaluator inconsistency: "
                            "all tests passed but "
                            "passed=False."
                        )

                    error_message = (
                        ""
                        if (
                            evaluation
                            .error_message
                            is None
                        )
                        else str(
                            evaluation
                            .error_message
                        )
                    )

                    per_test_results = (
                        serialize_test_results(
                            evaluation
                        )
                    )

                    code_record = {
                        "code_index": int(
                            code_index
                        ),

                        "code_seed": int(
                            code_seed
                        ),

                        "raw_output": str(
                            raw_output
                        ),

                        "generated_code": str(
                            generated_code
                        ),

                        "code_extraction_method": str(
                            parse_result
                            .extraction_method
                        ),

                        "status": status,

                        "passed": bool(
                            passed
                        ),

                        "binary_reward": float(
                            binary_reward
                        ),

                        # Number of tests selected for this
                        # diagnostic trajectory.
                        "scheduled_tests": int(
                            reward_tests
                        ),

                        # Code reached the evaluator and the
                        # non-fail-fast evaluator returned one result
                        # for every selected test.
                        "executed_tests": int(
                            total_tests
                        ),

                        "passed_tests": int(
                            passed_tests
                        ),

                        "total_tests": int(
                            total_tests
                        ),

                        "test_pass_ratio": float(
                            test_pass_ratio
                        ),

                        "prompt_tokens": int(
                            code_generation
                            .prompt_tokens
                        ),

                        "completion_tokens": int(
                            code_generation
                            .completion_tokens
                        ),

                        "generation_time": float(
                            code_generation
                            .generation_time
                        ),

                        # Current evaluator exposes this field but the
                        # underlying rLLM wrapper does not provide useful
                        # aggregate timing, so it is currently expected
                        # to remain 0.0.
                        "execution_time": float(
                            evaluation
                            .execution_time
                        ),

                        "error_message": (
                            error_message
                        ),

                        "per_test_results": (
                            per_test_results
                        ),
                    }

                except Exception as exc:
                    # ---------------------------------------------------------
                    # Infrastructure / evaluator-level failure.
                    #
                    # For reward-style aggregation this realization receives
                    # binary=0 and TPR=0.
                    #
                    # But scheduled_tests/executed_tests remain separated so
                    # the failure can later be distinguished from an actual
                    # 0/N algorithmic result.
                    # ---------------------------------------------------------

                    code_record = {
                        "code_index": int(
                            code_index
                        ),

                        "code_seed": int(
                            code_seed
                        ),

                        "raw_output": str(
                            raw_output
                        ),

                        "generated_code": str(
                            generated_code
                        ),

                        "code_extraction_method": str(
                            parse_result
                            .extraction_method
                        ),

                        "status": (
                            "EVALUATION_ERROR"
                        ),

                        "passed": False,

                        "binary_reward": 0.0,

                        "scheduled_tests": int(
                            reward_tests
                        ),

                        "executed_tests": 0,

                        "passed_tests": 0,

                        "total_tests": 0,

                        "test_pass_ratio": 0.0,

                        "prompt_tokens": int(
                            code_generation
                            .prompt_tokens
                        ),

                        "completion_tokens": int(
                            code_generation
                            .completion_tokens
                        ),

                        "generation_time": float(
                            code_generation
                            .generation_time
                        ),

                        "execution_time": 0.0,

                        "error_message": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),

                        "per_test_results": [],
                    }

                code_records.append(
                    code_record
                )

                print(
                    f"  CODE "
                    f"{code_index + 1}/"
                    f"{num_codes_per_plan} "
                    f"seed={code_seed} "
                    f"status="
                    f"{code_record['status']} "
                    f"tests="
                    f"{code_record['passed_tests']}/"
                    f"{code_record['scheduled_tests']} "
                    f"TPR="
                    f"{code_record['test_pass_ratio']:.4f}"
                )

            # =================================================================
            # 8. Plan-level aggregation
            # =================================================================

            if (
                len(code_records)
                != num_codes_per_plan
            ):
                raise RuntimeError(
                    "Internal diagnostic error: "
                    f"expected "
                    f"{num_codes_per_plan} "
                    "code records, got "
                    f"{len(code_records)}."
                )

            num_passed_codes = sum(
                1
                for item
                in code_records
                if bool(
                    item["passed"]
                )
            )

            plan_success_rate = (
                num_passed_codes
                / num_codes_per_plan
            )

            mean_test_pass_ratio = (
                sum(
                    float(
                        item[
                            "test_pass_ratio"
                        ]
                    )
                    for item
                    in code_records
                )
                / num_codes_per_plan
            )

            num_generation_errors = sum(
                1
                for item
                in code_records
                if (
                    item["status"]
                    == "CODE_GENERATION_ERROR"
                )
            )

            num_parsing_errors = sum(
                1
                for item
                in code_records
                if (
                    item["status"]
                    in {
                        "CODE_PARSING_ERROR",
                        "NO_CODE_FOUND",
                        "EMPTY_CODE",
                    }
                )
            )

            num_evaluation_errors = sum(
                1
                for item
                in code_records
                if (
                    item["status"]
                    == "EVALUATION_ERROR"
                )
            )

            # -----------------------------------------------------------------
            # One JSONL record = one (problem, plan).
            # -----------------------------------------------------------------

            record: dict[
                str,
                Any,
            ] = {
                "experiment_name": str(
                    config.experiment.name
                ),

                "base_seed": int(
                    base_seed
                ),

                "planner_model": str(
                    config.model.name_or_path
                ),

                "coder_model": str(
                    config.model.name_or_path
                ),

                "problem_index": int(
                    problem_index
                ),

                "problem_id": str(
                    example.problem_id
                ),

                "dataset": str(
                    example.dataset
                ),

                "evaluation_type": str(
                    example.evaluation_type
                ),

                "plan_index": int(
                    plan_index
                ),

                "plan_seed": int(
                    plan_seed
                ),

                "plan": str(
                    plan
                ),

                "plan_prompt": str(
                    plan_prompt
                ),

                "plan_prompt_tokens": int(
                    plan_generation
                    .prompt_tokens
                ),

                "plan_completion_tokens": int(
                    plan_generation
                    .completion_tokens
                ),

                "plan_generation_time": float(
                    plan_generation
                    .generation_time
                ),

                "coder_prompt": str(
                    code_prompt
                ),

                "num_codes": int(
                    num_codes_per_plan
                ),

                "num_passed_codes": int(
                    num_passed_codes
                ),

                # q_i^{binary}
                "plan_success_rate": float(
                    plan_success_rate
                ),

                # q_i^{TPR}
                "mean_test_pass_ratio": float(
                    mean_test_pass_ratio
                ),

                "available_tests": int(
                    len(
                        example.private_tests
                    )
                ),

                "reward_tests": int(
                    reward_tests
                ),

                "num_generation_errors": int(
                    num_generation_errors
                ),

                "num_parsing_errors": int(
                    num_parsing_errors
                ),

                "num_evaluation_errors": int(
                    num_evaluation_errors
                ),

                "planner_sampling": {
                    "temperature": float(
                        config.planner
                        .temperature
                    ),
                    "top_p": float(
                        config.planner
                        .top_p
                    ),
                    "max_new_tokens": int(
                        config.planner
                        .max_new_tokens
                    ),
                },

                "coder_sampling": {
                    "temperature": float(
                        config.coder
                        .temperature
                    ),
                    "top_p": float(
                        config.coder
                        .top_p
                    ),
                    "max_new_tokens": int(
                        config.coder
                        .max_new_tokens
                    ),
                },

                "codes": (
                    code_records
                ),
            }

            write_jsonl(
                output_path,
                record,
            )

            print(
                "  -> PLAN SUMMARY "
                f"pass="
                f"{num_passed_codes}/"
                f"{num_codes_per_plan} "
                f"q_binary="
                f"{plan_success_rate:.4f} "
                f"q_tpr="
                f"{mean_test_pass_ratio:.4f}"
            )

    # =========================================================================
    # 9. Final run summary
    # =========================================================================

    expected_plan_records = (
        len(examples)
        * num_plans
    )

    expected_code_records = (
        expected_plan_records
        * num_codes_per_plan
    )

    print()
    print(
        "=" * 88
    )
    print(
        "Diagnostic completed"
    )
    print(
        "=" * 88
    )

    print(
        f"Problems     : "
        f"{len(examples)}"
    )

    print(
        f"Plan records : "
        f"{expected_plan_records}"
    )

    print(
        f"Code records : "
        f"{expected_code_records}"
    )

    print(
        f"Output       : "
        f"{output_path}"
    )

    print(
        "=" * 88
    )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = (
        argparse.ArgumentParser(
            description=(
                "Run Phase 4 N-plans x "
                "M-frozen-coder-codes diagnostic."
            )
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to N x M diagnostic YAML config."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing output JSONL file."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run(
        config_path=args.config,
        overwrite=bool(
            args.overwrite
        ),
    )