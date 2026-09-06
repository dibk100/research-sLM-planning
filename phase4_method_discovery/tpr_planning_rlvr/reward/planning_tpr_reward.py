# phase4_method_discovery/tpr_planning_rlvr/reward/planning_tpr_reward.py

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.execution.taco_evaluator import TACOEvaluator
from src.parsing.code_parser import CodeParser

from phase4_method_discovery.vanilla_planning_rlvr.reward.planning_execution_reward import (
    _generate_code_via_rpc,
    _problem_from_extra_info,
    build_code_prompt,
    select_reward_tests,
)


# =============================================================================
# Paths / configuration
# =============================================================================

THIS_FILE = Path(__file__).resolve()

PROJECT_ROOT = THIS_FILE.parents[3]

DEFAULT_EXPERIMENT_CONFIG_PATH = (
    PROJECT_ROOT
    / "phase4_method_discovery"
    / "tpr_planning_rlvr"
    / "configs"
    / "tpr_planning_rlvr_qwen25coder3b.yaml"
)


# =============================================================================
# Process-local TPR runtime
# =============================================================================

_CODE_PARSER: CodeParser | None = None
_EVALUATOR: TACOEvaluator | None = None

_MAX_REWARD_TESTS: int | None = None

_RUNTIME_INITIALIZED: bool = False


def _initialize_tpr_runtime() -> None:
    """
    Initialize lightweight TPR reward-side components once per
    RewardLoopWorker process.

    The frozen coder model is NOT loaded here. It is managed by the
    shared FrozenCoderWorker used by Vanilla Planning-RLVR.

    The TPR runtime differs from Vanilla only in reward semantics:

        Vanilla:
            fail-fast execution
            -> binary all-tests-pass reward

        TPR:
            non-fail-fast execution
            -> passed_tests / total_tests
    """

    global _CODE_PARSER
    global _EVALUATOR
    global _MAX_REWARD_TESTS
    global _RUNTIME_INITIALIZED

    if _RUNTIME_INITIALIZED:
        return

    # -------------------------------------------------------------------------
    # 1. Load TPR experiment configuration
    # -------------------------------------------------------------------------

    if not DEFAULT_EXPERIMENT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            "TPR Planning-RLVR config not found: "
            f"{DEFAULT_EXPERIMENT_CONFIG_PATH}"
        )

    config = OmegaConf.load(
        DEFAULT_EXPERIMENT_CONFIG_PATH
    )

    if not hasattr(config, "reward"):
        raise ValueError(
            "TPR Planning-RLVR config must contain "
            "a 'reward' section."
        )

    reward_cfg = config.reward

    # -------------------------------------------------------------------------
    # 2. Validate TPR reward settings
    # -------------------------------------------------------------------------

    if str(reward_cfg.type) != "test_pass_ratio":
        raise ValueError(
            "TPR Planning-RLVR requires "
            "reward.type=test_pass_ratio, "
            f"got {reward_cfg.type!r}."
        )

    max_reward_tests = int(
        reward_cfg.max_tests
    )

    if max_reward_tests <= 0:
        raise ValueError(
            "reward.max_tests must be > 0."
        )

    timeout_seconds = int(
        reward_cfg.timeout_seconds
    )

    if timeout_seconds <= 0:
        raise ValueError(
            "reward.timeout_seconds must be > 0."
        )

    # -------------------------------------------------------------------------
    # 3. Parser / evaluator
    # -------------------------------------------------------------------------

    _CODE_PARSER = CodeParser()

    _EVALUATOR = TACOEvaluator(
        timeout_seconds=timeout_seconds,
        debug=False,
    )

    _MAX_REWARD_TESTS = (
        max_reward_tests
    )

    _RUNTIME_INITIALIZED = True


def _ensure_tpr_runtime() -> None:
    if not _RUNTIME_INITIALIZED:
        _initialize_tpr_runtime()

    if _CODE_PARSER is None:
        raise RuntimeError(
            "TPR CodeParser is not initialized."
        )

    if _EVALUATOR is None:
        raise RuntimeError(
            "TPR TACOEvaluator is not initialized."
        )

    if _MAX_REWARD_TESTS is None:
        raise RuntimeError(
            "TPR reward test limit is not initialized."
        )


# =============================================================================
# Reference non-fail-fast TPR evaluation
# =============================================================================

def evaluate_tpr_non_fail_fast(
    *,
    problem: Any,
    code: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """
    Reference implementation of TPR evaluation.

    Every selected private test is evaluated independently:

        test_1 -> evaluator
        test_2 -> evaluator
        ...
        test_N -> evaluator

    This function is intentionally NOT used in the production training
    reward path.

    Its purpose is to serve as a correctness oracle for validating the
    optimized native:

        TACOEvaluator.evaluate_non_fail_fast()

    implementation.

    Returns
    -------
    dict
        {
            "passed_tests": int,
            "total_tests": int,
            "test_pass_ratio": float,
            "all_tests_passed": bool,
            "per_test_results": list[dict],
        }
    """

    selected_tests = list(
        problem.private_tests
    )

    total_tests = len(
        selected_tests
    )

    if total_tests == 0:
        return {
            "passed_tests": 0,
            "total_tests": 0,
            "test_pass_ratio": 0.0,
            "all_tests_passed": False,
            "per_test_results": [],
        }

    evaluator = TACOEvaluator(
        timeout_seconds=int(
            timeout_seconds
        ),
        debug=False,
    )

    passed_tests = 0

    per_test_results: list[
        dict[str, Any]
    ] = []

    for test_index, test_case in enumerate(
        selected_tests
    ):
        one_test_problem = copy.deepcopy(
            problem
        )

        one_test_problem.private_tests = [
            copy.deepcopy(
                test_case
            )
        ]

        try:
            evaluation = evaluator.evaluate(
                problem=one_test_problem,
                code=code,
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
                if evaluation.error_message is None
                else str(
                    evaluation.error_message
                )
            )

        except Exception as exc:
            passed = False
            status = "EVALUATION_ERROR"
            execution_time = 0.0

            error_message = (
                f"{type(exc).__name__}: {exc}"
            )

        if passed:
            passed_tests += 1

        per_test_results.append(
            {
                "test_index": int(
                    test_index
                ),
                "passed": bool(
                    passed
                ),
                "status": status,
                "execution_time": float(
                    execution_time
                ),
                "error_message": (
                    error_message
                ),
            }
        )

    test_pass_ratio = (
        passed_tests / total_tests
    )

    all_tests_passed = (
        passed_tests == total_tests
    )

    return {
        "passed_tests": int(
            passed_tests
        ),
        "total_tests": int(
            total_tests
        ),
        "test_pass_ratio": float(
            test_pass_ratio
        ),
        "all_tests_passed": bool(
            all_tests_passed
        ),
        "per_test_results": (
            per_test_results
        ),
    }


# =============================================================================
# Core TPR reward
# =============================================================================

def compute_tpr_reward(
    *,
    plan: str,
    extra_info: dict[str, Any],
    frozen_coder_handle: Any,
) -> dict[str, Any]:
    """
    Compute dense test-pass-ratio reward for one sampled plan.

    Reward
    ------
        R_TPR = K / N

    where:
        K = number of selected reward tests passed
        N = number of selected reward tests

    Experimental control
    --------------------
    TPR uses the same:

        - problem reconstruction,
        - reward-test selection,
        - plan -> code prompt,
        - frozen coder,
        - code parser,

    as Vanilla Planning-RLVR.

    The intentional methodological differences are:

        Vanilla:
            fail-fast execution
            reward = 1 iff all selected tests pass

        TPR:
            non-fail-fast execution
            reward = passed_tests / total_tests

    Therefore reward=1.0 has the same all-selected-tests-pass
    success semantics as Vanilla.
    """

    _ensure_tpr_runtime()

    if frozen_coder_handle is None:
        raise ValueError(
            "frozen_coder_handle is required."
        )

    # -------------------------------------------------------------------------
    # 1. Restore the exact RL problem
    # -------------------------------------------------------------------------

    problem = _problem_from_extra_info(
        extra_info
    )

    problem_text = (
        problem.problem
    )

    if (
        not isinstance(
            problem_text,
            str,
        )
        or not problem_text.strip()
    ):
        raise ValueError(
            "ProblemExample.problem "
            "must be non-empty."
        )

    available_tests = len(
        problem.private_tests
    )

    # -------------------------------------------------------------------------
    # 2. Validate planner output
    # -------------------------------------------------------------------------

    if (
        not isinstance(
            plan,
            str,
        )
        or not plan.strip()
    ):
        return {
            "score": 0.0,
            "reward": 0.0,
            "test_pass_ratio": 0.0,
            "passed_tests": 0,
            "reward_tests": 0,
            "available_tests": int(
                available_tests
            ),
            "all_tests_passed": False,
            "binary_reward": 0.0,
            "status": "EMPTY_PLAN",
            "error_message": (
                "Generated plan is empty."
            ),
            "per_test_results": [],
        }

    # -------------------------------------------------------------------------
    # 3. Build exactly the same plan -> code prompt as Vanilla
    # -------------------------------------------------------------------------

    try:
        coder_prompt = build_code_prompt(
            problem_text=problem_text,
            plan=plan,
            starter_code=(
                problem.starter_code
            ),
        )

    except Exception as exc:
        return {
            "score": 0.0,
            "reward": 0.0,
            "test_pass_ratio": 0.0,
            "passed_tests": 0,
            "reward_tests": 0,
            "available_tests": int(
                available_tests
            ),
            "all_tests_passed": False,
            "binary_reward": 0.0,
            "status": "CODE_PROMPT_ERROR",
            "error_message": (
                f"{type(exc).__name__}: {exc}"
            ),
            "per_test_results": [],
        }

    # -------------------------------------------------------------------------
    # 4. Generate code with exactly the same frozen coder RPC as Vanilla
    # -------------------------------------------------------------------------

    try:
        (
            raw_code_output,
            coder_prompt_tokens,
            coder_completion_tokens,
            coder_generation_time,
        ) = _generate_code_via_rpc(
            frozen_coder_handle=(
                frozen_coder_handle
            ),
            prompt=coder_prompt,
        )

    except Exception as exc:
        return {
            "score": 0.0,
            "reward": 0.0,
            "test_pass_ratio": 0.0,
            "passed_tests": 0,
            "reward_tests": 0,
            "available_tests": int(
                available_tests
            ),
            "all_tests_passed": False,
            "binary_reward": 0.0,
            "status": (
                "CODE_GENERATION_ERROR"
            ),
            "error_message": (
                f"{type(exc).__name__}: {exc}"
            ),
            "per_test_results": [],
        }

    # -------------------------------------------------------------------------
    # 5. Parse code
    # -------------------------------------------------------------------------

    assert _CODE_PARSER is not None

    try:
        parse_result = (
            _CODE_PARSER.parse(
                raw_code_output
            )
        )

    except Exception as exc:
        return {
            "score": 0.0,
            "reward": 0.0,
            "test_pass_ratio": 0.0,
            "passed_tests": 0,
            "reward_tests": 0,
            "available_tests": int(
                available_tests
            ),
            "all_tests_passed": False,
            "binary_reward": 0.0,
            "status": (
                "CODE_PARSING_ERROR"
            ),
            "error_message": (
                f"{type(exc).__name__}: {exc}"
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
            "per_test_results": [],
        }

    if parse_result.status != "SUCCESS":
        return {
            "score": 0.0,
            "reward": 0.0,
            "test_pass_ratio": 0.0,
            "passed_tests": 0,
            "reward_tests": 0,
            "available_tests": int(
                available_tests
            ),
            "all_tests_passed": False,
            "binary_reward": 0.0,
            "status": str(
                parse_result.status
            ),
            "error_message": (
                "Code parsing failed: "
                f"{parse_result.status}"
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
            "per_test_results": [],
        }

    generated_code = (
        parse_result.code
    )

    # -------------------------------------------------------------------------
    # 6. Select exactly the same reward-test subset as Vanilla
    # -------------------------------------------------------------------------

    assert _MAX_REWARD_TESTS is not None

    reward_problem = select_reward_tests(
        problem,
        max_tests=_MAX_REWARD_TESTS,
    )

    reward_tests = len(
        reward_problem.private_tests
    )

    if reward_tests <= 0:
        return {
            "score": 0.0,
            "reward": 0.0,
            "test_pass_ratio": 0.0,
            "passed_tests": 0,
            "reward_tests": 0,
            "available_tests": int(
                available_tests
            ),
            "all_tests_passed": False,
            "binary_reward": 0.0,
            "status": "NO_TESTS",
            "error_message": (
                "No TACO tests available."
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
            "per_test_results": [],
        }

    # -------------------------------------------------------------------------
    # 7. Execute ALL selected reward tests
    #
    # This is the key execution difference from Vanilla.
    #
    # Vanilla:
    #     _EVALUATOR.evaluate(...)
    #
    # TPR:
    #     _EVALUATOR.evaluate_non_fail_fast(...)
    #
    # All selected tests must contribute to the TPR denominator.
    # -------------------------------------------------------------------------

    assert _EVALUATOR is not None

    try:
        evaluation = (
            _EVALUATOR.evaluate_non_fail_fast(
                problem=reward_problem,
                code=generated_code,
            )
        )

    except Exception as exc:
        return {
            "score": 0.0,
            "reward": 0.0,
            "test_pass_ratio": 0.0,
            "passed_tests": 0,
            "reward_tests": int(
                reward_tests
            ),
            "available_tests": int(
                available_tests
            ),
            "all_tests_passed": False,
            "binary_reward": 0.0,
            "status": "EVALUATION_ERROR",
            "error_message": (
                f"{type(exc).__name__}: {exc}"
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
            "per_test_results": [],
        }

    # -------------------------------------------------------------------------
    # 8. Validate complete non-fail-fast result
    # -------------------------------------------------------------------------

    passed_tests = int(
        evaluation.passed_tests
    )

    total_tests = int(
        evaluation.total_tests
    )

    # TPR denominator must remain exactly the selected reward-test count.
    # Missing execution results must never silently reduce N.
    if total_tests != reward_tests:
        raise RuntimeError(
            "TPR evaluator returned an unexpected "
            "test count: "
            f"expected={reward_tests}, "
            f"actual={total_tests}."
        )

    if total_tests <= 0:
        raise RuntimeError(
            "TPR evaluator returned zero tests "
            "after non-empty reward-test selection."
        )

    if not (
        0 <= passed_tests <= total_tests
    ):
        raise RuntimeError(
            "Invalid TPR test counts: "
            f"passed_tests={passed_tests}, "
            f"total_tests={total_tests}."
        )

    # -------------------------------------------------------------------------
    # 9. Dense TPR reward
    # -------------------------------------------------------------------------

    test_pass_ratio = (
        passed_tests / total_tests
    )

    reward = float(
        test_pass_ratio
    )

    if not (
        0.0 <= reward <= 1.0
    ):
        raise RuntimeError(
            "TPR reward outside [0, 1]: "
            f"{reward}."
        )

    all_tests_passed = (
        passed_tests == total_tests
    )

    binary_reward = (
        1.0
        if all_tests_passed
        else 0.0
    )

    status = (
        "PASS"
        if all_tests_passed
        else (
            "PARTIAL_PASS"
            if passed_tests > 0
            else "FAIL"
        )
    )

    # -------------------------------------------------------------------------
    # 10. Return reward-manager-compatible result
    #
    # `score` is consumed by verl/GRPO.
    #
    # `binary_reward` allows direct diagnostic comparison against the
    # Vanilla all-tests-pass reward on exactly the same selected tests.
    # -------------------------------------------------------------------------

    return {
        "score": float(
            reward
        ),

        "reward": float(
            reward
        ),

        "test_pass_ratio": float(
            test_pass_ratio
        ),

        "passed_tests": int(
            passed_tests
        ),

        "reward_tests": int(
            total_tests
        ),

        "available_tests": int(
            available_tests
        ),

        "all_tests_passed": bool(
            all_tests_passed
        ),

        "binary_reward": float(
            binary_reward
        ),

        "status": status,

        "error_message": "",

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

        "per_test_results": [
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
                "execution_time": float(
                    result.execution_time
                ),
                "error_message": str(
                    result.stderr
                    or ""
                ),
            }
            for result in evaluation.test_results
        ],
    }


# =============================================================================
# verl custom reward entry point
# =============================================================================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    frozen_coder_handle: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    verl-compatible TPR Planning-RLVR reward entry point.

    Important
    ---------
    `solution_str` is the generated PLAN, not generated code.

    Reward trajectory:

        solution_str
            = plan
            ->
        FrozenCoderWorker
            ->
        code
            ->
        selected DeepCoder/TACO reward tests
            ->
        non-fail-fast execution
            ->
        test-pass-ratio reward in [0, 1]

    `ground_truth` is intentionally unused.
    Unit-test execution is the correctness authority.
    """

    del ground_truth
    del kwargs

    # -------------------------------------------------------------------------
    # 1. Dataset validation
    # -------------------------------------------------------------------------

    if data_source != "deepcoder_taco":
        raise ValueError(
            "Unsupported data_source: "
            f"{data_source!r}. "
            "Expected 'deepcoder_taco'."
        )

    if extra_info is None:
        raise ValueError(
            "extra_info is required."
        )

    if frozen_coder_handle is None:
        raise ValueError(
            "frozen_coder_handle is required."
        )

    # -------------------------------------------------------------------------
    # 2. Execute TPR reward trajectory
    # -------------------------------------------------------------------------

    return compute_tpr_reward(
        plan=solution_str,
        extra_info=extra_info,
        frozen_coder_handle=(
            frozen_coder_handle
        ),
    )