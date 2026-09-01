from __future__ import annotations

import copy
from typing import Any

from src.execution.taco_evaluator import TACOEvaluator
from src.parsing.code_parser import CodeParser

from phase4_method_discovery.vanilla_planning_rlvr.reward.planning_execution_reward import (
    _generate_code_via_rpc,
    _problem_from_extra_info,
    build_code_prompt,
    select_reward_tests,
)


# =============================================================================
# Constants
# =============================================================================


DEFAULT_MAX_TESTS = 15
DEFAULT_TIMEOUT_SECONDS = 6


# =============================================================================
# Non-fail-fast TPR evaluation
# =============================================================================


def evaluate_tpr_non_fail_fast(
    *,
    problem: Any,
    code: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """
    Evaluate generated code on every selected private test independently.

    Why independent evaluation?
    ---------------------------
    The existing TACO evaluator follows the benchmark's fail-fast execution
    behavior. That is appropriate for binary all-tests-pass reward, but it
    cannot provide a true test-pass ratio because tests after the first
    failure are not executed.

    For TPR reward, every selected reward test must contribute independently.

    Therefore this function evaluates each selected private test separately:

        test_1 -> evaluator
        test_2 -> evaluator
        ...
        test_N -> evaluator

    Each evaluator invocation contains exactly one private test, so failure
    on one test cannot prevent subsequent tests from being evaluated.

    NOTE:
        This implementation prioritizes correctness and exact correspondence
        with the validated diagnostic protocol.

        It may later be replaced by a more efficient native non-fail-fast
        execution backend, provided that reward semantics remain identical.

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
        timeout_seconds=timeout_seconds,
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
                "execution_time": (
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
# TPR reward
# =============================================================================


def compute_tpr_reward(
    *,
    plan: str,
    extra_info: dict[str, Any],
    frozen_coder_handle: Any,
    max_tests: int = DEFAULT_MAX_TESTS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Compute dense test-pass-ratio reward for a sampled plan.

    Reward
    ------
        R_TPR = K / N

    where:
        K = number of selected reward tests passed
        N = number of selected reward tests

    The reward is always in [0, 1].

    A reward of 1.0 has exactly the same success semantics as the vanilla
    binary reward: all selected reward tests must pass.
    """

    if frozen_coder_handle is None:
        raise ValueError(
            "frozen_coder_handle is required."
        )

    if max_tests <= 0:
        raise ValueError(
            "max_tests must be > 0."
        )

    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds must be > 0."
        )

    # -------------------------------------------------------------------------
    # 1. Restore the exact RL problem.
    # -------------------------------------------------------------------------

    problem = _problem_from_extra_info(
        extra_info
    )

    problem_text = problem.problem

    # -------------------------------------------------------------------------
    # 2. Select exactly the same reward-test subset as vanilla RLVR.
    #
    # Vanilla behavior:
    #   - private tests only
    #   - descending input length
    #   - stable original-index tie break
    #   - at most max_tests
    # -------------------------------------------------------------------------

    reward_problem = select_reward_tests(
        problem,
        max_tests=max_tests,
    )

    reward_tests = len(
        reward_problem.private_tests
    )

    if reward_tests == 0:
        return {
            "score": 0.0,
            "reward": 0.0,
            "test_pass_ratio": 0.0,
            "passed_tests": 0,
            "reward_tests": 0,
            "all_tests_passed": False,
            "status": "NO_TESTS",
            "error_message": (
                "No private reward tests available."
            ),
            "per_test_results": [],
        }

    # -------------------------------------------------------------------------
    # 3. Build exactly the same plan -> code prompt as vanilla.
    # -------------------------------------------------------------------------

    coder_prompt = build_code_prompt(
        problem_text=problem_text,
        plan=plan,
        starter_code=(
            problem.starter_code
        ),
    )

    # -------------------------------------------------------------------------
    # 4. Generate code using the same frozen coder RPC.
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
            "reward_tests": int(
                reward_tests
            ),
            "all_tests_passed": False,
            "status": (
                "CODE_GENERATION_ERROR"
            ),
            "error_message": (
                f"{type(exc).__name__}: {exc}"
            ),
            "per_test_results": [],
        }

    # -------------------------------------------------------------------------
    # 5. Parse code using the same parser as vanilla.
    # -------------------------------------------------------------------------

    parser = CodeParser()

    try:
        parse_result = parser.parse(
            raw_code_output
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
            "all_tests_passed": False,
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
            "reward_tests": int(
                reward_tests
            ),
            "all_tests_passed": False,
            "status": str(
                parse_result.status
            ),
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
            "per_test_results": [],
        }

    generated_code = parse_result.code

    # -------------------------------------------------------------------------
    # 6. Execute ALL selected reward tests.
    #
    # Unlike vanilla binary evaluation, this path must not fail fast.
    # -------------------------------------------------------------------------

    evaluation = evaluate_tpr_non_fail_fast(
        problem=reward_problem,
        code=generated_code,
        timeout_seconds=timeout_seconds,
    )

    passed_tests = int(
        evaluation[
            "passed_tests"
        ]
    )

    total_tests = int(
        evaluation[
            "total_tests"
        ]
    )

    test_pass_ratio = float(
        evaluation[
            "test_pass_ratio"
        ]
    )

    all_tests_passed = bool(
        evaluation[
            "all_tests_passed"
        ]
    )

    # -------------------------------------------------------------------------
    # 7. Dense TPR reward.
    # -------------------------------------------------------------------------

    reward = test_pass_ratio

    if not (
        0.0 <= reward <= 1.0
    ):
        raise RuntimeError(
            "TPR reward outside [0, 1]: "
            f"{reward}"
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
    # 8. Debug logging for TPR reward propagation.
    #
    # Temporary instrumentation for the integration smoke test.
    # This makes fractional rewards directly observable in the Ray worker log.
    # -------------------------------------------------------------------------

    print(
        "[TPR Reward] "
        f"passed={passed_tests}/{total_tests} "
        f"score={reward:.6f} "
        f"binary={1 if all_tests_passed else 0} "
        f"status={status}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # 9. Return reward-manager-compatible result.
    #
    # `score` is the value consumed by verl reward handling.
    #
    # Keep `reward` as an explicit duplicate for logging/analysis.
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

        "all_tests_passed": bool(
            all_tests_passed
        ),

        # Useful for direct comparison with vanilla reward.
        "binary_reward": (
            1.0
            if all_tests_passed
            else 0.0
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

        "per_test_results": (
            evaluation[
                "per_test_results"
            ]
        ),
    }


# =============================================================================
# verl custom reward entry point
# =============================================================================


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    verl-compatible reward entry point.

    Parameters
    ----------
    data_source
        Dataset source identifier. Kept for compatibility with verl.

    solution_str
        Planner rollout decoded by the reward manager.
        In this experiment, this is the generated PLAN, not generated code.

    ground_truth
        Unused for TPR execution reward. Kept for verl compatibility.

    extra_info
        TACO problem information stored in the RL parquet.

    kwargs
        Must contain `frozen_coder_handle`.

        Optional:
            max_tests
            timeout_seconds

    Returns
    -------
    dict
        Must contain:
            score: float in [0, 1]

        Additional fields are exposed through reward_extra_info for
        diagnostics and training analysis.
    """

    del data_source
    del ground_truth

    if extra_info is None:
        raise ValueError(
            "extra_info is required for "
            "TPR planning reward."
        )

    frozen_coder_handle = kwargs.get(
        "frozen_coder_handle"
    )

    if frozen_coder_handle is None:
        raise ValueError(
            "frozen_coder_handle was not "
            "provided to compute_score()."
        )

    max_tests = int(
        kwargs.get(
            "max_tests",
            DEFAULT_MAX_TESTS,
        )
    )

    timeout_seconds = int(
        kwargs.get(
            "timeout_seconds",
            DEFAULT_TIMEOUT_SECONDS,
        )
    )

    return compute_tpr_reward(
        plan=solution_str,
        extra_info=extra_info,
        frozen_coder_handle=(
            frozen_coder_handle
        ),
        max_tests=max_tests,
        timeout_seconds=(
            timeout_seconds
        ),
    )