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

from omegaconf import DictConfig, OmegaConf

from src.execution.taco_evaluator import TACOEvaluator
from src.parsing.code_parser import CodeParser

from phase4_method_discovery.vanilla_planning_rlvr.reward.planning_reward_utils import (
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
    Build a stable 32-bit seed from the base seed and trajectory identifiers.

    Python's built-in hash() is intentionally avoided because its result can
    change across interpreter processes.
    """

    text = "::".join(
        [str(base_seed)]
        + [str(part) for part in parts]
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="big",
        signed=False,
    )


def set_generation_seed(
    seed: int,
) -> None:
    # Local import is intentional.
    #
    # TACOEvaluator uses multiprocessing "spawn".
    # A spawned evaluator process re-imports this main module.
    # Keeping torch out of module-level imports prevents every
    # evaluator child from importing the full PyTorch stack.
    import torch

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )
# =============================================================================
# JSON helpers
# =============================================================================


def _json_safe(
    value: Any,
) -> Any:
    """
    Recursively convert common non-JSON-native objects into safe values.

    This is mainly defensive for evaluator metadata.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    if hasattr(
        value,
        "item",
    ):
        try:
            return value.item()
        except Exception:
            pass

    if hasattr(
        value,
        "tolist",
    ):
        try:
            return value.tolist()
        except Exception:
            pass

    return str(value)


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
                _json_safe(record),
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
    Prevent accidental mixing of multiple experimental runs.
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
# Config validation
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
# Evaluation serialization
# =============================================================================


def serialize_test_results(
    evaluation: Any,
) -> list[dict[str, Any]]:
    records: list[
        dict[str, Any]
    ] = []

    for result in (
        evaluation.test_results
        or []
    ):
        records.append(
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
                "metadata": _json_safe(
                    result.metadata
                    or {}
                ),
            }
        )

    return records


# =============================================================================
# Failure-record builders
# =============================================================================


def build_code_generation_failure(
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


def build_code_parsing_failure(
    *,
    code_index: int,
    code_seed: int,
    raw_output: str,
    generated_code: str,
    extraction_method: str,
    parser_status: str,
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
            parser_status
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


def build_evaluation_failure(
    *,
    code_index: int,
    code_seed: int,
    raw_output: str,
    generated_code: str,
    extraction_method: str,
    scheduled_tests: int,
    prompt_tokens: int,
    completion_tokens: int,
    generation_time: float,
    error: Exception,
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

        "status": "EVALUATION_ERROR",

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

        "error_message": (
            f"{type(error).__name__}: "
            f"{error}"
        ),

        "per_test_results": [],
    }


# =============================================================================
# Core N x M diagnostic
# =============================================================================

def run(
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> None:

    # -------------------------------------------------------------------------
    # Heavy imports are intentionally local.
    #
    # TACOEvaluator uses multiprocessing with the "spawn" start method.
    # Every evaluator child re-imports this script as __mp_main__.
    #
    # Keeping torch / transformers / ModelGenerator out of module-level
    # imports prevents evaluator children from loading the LLM inference
    # stack and consuming excessive host RAM.
    # -------------------------------------------------------------------------

    from src.models.generator import (
        ModelGenerator,
    )

    from phase1_planning_bottleneck.strategies.self_plan import (
        SelfPlanningStrategy,
    )

    from phase4_method_discovery.vanilla_planning_rlvr.evaluation.checkpoint_dataset import (
        load_checkpoint_eval_dataset,
    )
    
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
            "No diagnostic examples loaded."
        )

    # =========================================================================
    # 2. Shared model
    #
    # Initial baseline diagnostic:
    #
    #     planner = Qwen2.5-Coder-3B-Instruct
    #     coder   = same base checkpoint
    #
    # We intentionally load ONE model instance.
    #
    # The model is conceptually frozen during the entire diagnostic because
    # ModelGenerator runs under torch.inference_mode().
    # =========================================================================

    generator = (
        ModelGenerator(
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
    )

    # =========================================================================
    # 3. Planner-prompt builder
    #
    # SelfPlanningStrategy.run() is NOT used.
    #
    # We reuse only build_plan_prompt() so that the planner prompt remains
    # consistent with the existing project pipeline.
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

    num_problems = len(
        examples
    )

    expected_plan_records = (
        num_problems
        * num_plans
    )

    expected_code_records = (
        expected_plan_records
        * num_codes_per_plan
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
        f"{num_problems}"
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
        f"{expected_code_records}"
    )

    print(
        f"planner temp    : "
        f"{float(config.planner.temperature)}"
    )

    print(
        f"planner top_p   : "
        f"{float(config.planner.top_p)}"
    )

    print(
        f"coder temp      : "
        f"{float(config.coder.temperature)}"
    )

    print(
        f"coder top_p     : "
        f"{float(config.coder.top_p)}"
    )

    print(
        f"max reward tests: "
        f"{max_reward_tests}"
    )

    print(
        f"timeout         : "
        f"{int(config.evaluation.timeout_seconds)} sec"
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
            f"{num_problems}] "
            f"{example.problem_id}"
        )

        print(
            "=" * 88
        )

        # ---------------------------------------------------------------------
        # Same planner prompt for all N plans.
        # ---------------------------------------------------------------------

        plan_prompt = (
            prompt_strategy
            .build_plan_prompt(
                example
            )
        )

        # ---------------------------------------------------------------------
        # Select reward tests ONCE per problem.
        #
        # Every plan/code realization for the same problem therefore sees the
        # same evaluation subset.
        #
        # This is the same selection policy used by Phase 4 RLVR.
        # ---------------------------------------------------------------------

        reward_problem = (
            select_reward_tests(
                example,
                max_tests=(
                    max_reward_tests
                ),
            )
        )

        available_tests = len(
            example.private_tests
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
            f"{available_tests}"
        )

        print(
            f"Reward tests    : "
            f"{reward_tests}"
        )

        # =====================================================================
        # 6. N sampled plans
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
            # 6-1. Generate one stochastic plan.
            # -----------------------------------------------------------------

            try:
                plan_generation = (
                    generator.generate(
                        prompt=(
                            plan_prompt
                        ),
                        max_new_tokens=int(
                            config.planner
                            .max_new_tokens
                        ),
                        temperature=float(
                            config.planner
                            .temperature
                        ),
                        top_p=float(
                            config.planner
                            .top_p
                        ),
                    )
                )

            except Exception as exc:
                raise RuntimeError(
                    "Plan generation failed: "
                    f"problem="
                    f"{example.problem_id}, "
                    f"plan_index="
                    f"{plan_index}, "
                    f"seed="
                    f"{plan_seed}, "
                    f"error="
                    f"{type(exc).__name__}: "
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
                    f"problem="
                    f"{example.problem_id}, "
                    f"plan_index="
                    f"{plan_index}"
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
            # 6-2. Construct code prompt ONCE.
            #
            # This is critical for the N x M decomposition.
            #
            # For one fixed p_i:
            #
            #   same problem
            #   same plan
            #   same exact coder prompt
            #   same frozen model
            #   same evaluation tests
            #
            # Only the code sampling RNG changes across j=1...M.
            # -----------------------------------------------------------------

            code_prompt = (
                build_code_prompt(
                    problem_text=(
                        example.problem
                    ),
                    plan=(
                        plan
                    ),
                    starter_code=(
                        example.starter_code
                    ),
                )
            )

            code_records: list[
                dict[str, Any]
            ] = []

            # =================================================================
            # 7. M coder realizations for fixed plan
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
                # 7-1. Stochastic frozen-coder generation
                # -------------------------------------------------------------

                try:
                    code_generation = (
                        generator.generate(
                            prompt=(
                                code_prompt
                            ),
                            max_new_tokens=int(
                                config.coder
                                .max_new_tokens
                            ),
                            temperature=float(
                                config.coder
                                .temperature
                            ),
                            top_p=float(
                                config.coder
                                .top_p
                            ),
                        )
                    )

                except Exception as exc:
                    code_record = (
                        build_code_generation_failure(
                            code_index=(
                                code_index
                            ),
                            code_seed=(
                                code_seed
                            ),
                            scheduled_tests=(
                                reward_tests
                            ),
                            error=(
                                exc
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
                        f"CODE_GENERATION_ERROR"
                    )

                    continue

                raw_output = str(
                    code_generation.text
                )

                # -------------------------------------------------------------
                # 7-2. Parse code
                # -------------------------------------------------------------

                try:
                    parse_result = (
                        code_parser.parse(
                            raw_output
                        )
                    )

                except Exception as exc:
                    code_record = (
                        build_code_parsing_failure(
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
                            extraction_method=(
                                "none"
                            ),
                            parser_status=(
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
                        f"status="
                        f"CODE_PARSING_ERROR"
                    )

                    continue

                if (
                    parse_result.status
                    != "SUCCESS"
                ):
                    code_record = (
                        build_code_parsing_failure(
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
                                or ""
                            ),
                            extraction_method=str(
                                parse_result
                                .extraction_method
                            ),
                            parser_status=str(
                                parse_result
                                .status
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

                generated_code = str(
                    parse_result.code
                )

                # -------------------------------------------------------------
                # 7-3. Non-fail-fast execution
                #
                # One execution gives both:
                #
                # binary:
                #   all selected tests pass
                #
                # TPR:
                #   passed_tests / total_tests
                #
                # EvaluationResult is the authority for pass/status.
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

                    # ---------------------------------------------------------
                    # Non-fail-fast evaluator must account for all selected
                    # reward tests.
                    # ---------------------------------------------------------

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

                    # ---------------------------------------------------------
                    # Consistency checks
                    # ---------------------------------------------------------

                    if (
                        passed
                        and (
                            passed_tests
                            != total_tests
                        )
                    ):
                        raise RuntimeError(
                            "Evaluator inconsistency: "
                            "passed=True but "
                            "passed_tests != "
                            "total_tests."
                        )

                    if (
                        (
                            not passed
                        )
                        and (
                            passed_tests
                            == total_tests
                        )
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

                        "raw_output": (
                            raw_output
                        ),

                        "generated_code": (
                            generated_code
                        ),

                        "code_extraction_method": str(
                            parse_result
                            .extraction_method
                        ),

                        "status": (
                            status
                        ),

                        "passed": bool(
                            passed
                        ),

                        "binary_reward": float(
                            binary_reward
                        ),

                        "scheduled_tests": int(
                            reward_tests
                        ),

                        # Non-fail-fast successful invocation accounts for
                        # every selected test.
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

                        # Current rLLM wrapper does not expose useful aggregate
                        # timing, so this is normally 0.0.
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
                    code_record = (
                        build_evaluation_failure(
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
                                generated_code
                            ),
                            extraction_method=str(
                                parse_result
                                .extraction_method
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
                            error=(
                                exc
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

            num_evaluation_errors = sum(
                1
                for item
                in code_records
                if (
                    item["status"]
                    == "EVALUATION_ERROR"
                )
            )

            # Parsing failures are identified through executed_tests=0 while
            # the failure occurred after code generation but before evaluator.
            num_parsing_errors = sum(
                1
                for item
                in code_records
                if (
                    item["executed_tests"]
                    == 0
                    and item["status"]
                    not in {
                        "CODE_GENERATION_ERROR",
                        "EVALUATION_ERROR",
                    }
                )
            )

            # Number of coder realizations that actually reached the evaluator.
            num_evaluated_codes = sum(
                1
                for item
                in code_records
                if (
                    item["executed_tests"]
                    > 0
                )
            )

            # -----------------------------------------------------------------
            # One JSONL line = one (problem, plan).
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
                    config.model
                    .name_or_path
                ),

                "coder_model": str(
                    config.model
                    .name_or_path
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

                "plan": (
                    plan
                ),

                # Store prompts during diagnostic development.
                # These can be removed later if JSONL size becomes large.
                "plan_prompt": (
                    plan_prompt
                ),

                "coder_prompt": (
                    code_prompt
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

                "available_tests": int(
                    available_tests
                ),

                "reward_tests": int(
                    reward_tests
                ),

                "num_codes": int(
                    num_codes_per_plan
                ),

                "num_evaluated_codes": int(
                    num_evaluated_codes
                ),

                "num_passed_codes": int(
                    num_passed_codes
                ),

                # -------------------------------------------------------------
                # q_i^(binary)
                #
                # Empirical probability that a frozen-coder realization
                # conditioned on plan p_i solves every selected test.
                # -------------------------------------------------------------
                "plan_success_rate": float(
                    plan_success_rate
                ),

                # -------------------------------------------------------------
                # q_i^(TPR)
                #
                # Mean partial-test utility over M code realizations.
                # -------------------------------------------------------------
                "mean_test_pass_ratio": float(
                    mean_test_pass_ratio
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
                f"{mean_test_pass_ratio:.4f} "
                f"evaluated="
                f"{num_evaluated_codes}/"
                f"{num_codes_per_plan}"
            )

    # =========================================================================
    # 9. Final summary
    # =========================================================================

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
        f"{num_problems}"
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
                "Phase 4 N-plans x M-frozen-coder-codes "
                "diagnostic."
            )
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to the N x M diagnostic YAML config."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing diagnostic JSONL output."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run(
        config_path=(
            args.config
        ),
        overwrite=bool(
            args.overwrite
        ),
    )