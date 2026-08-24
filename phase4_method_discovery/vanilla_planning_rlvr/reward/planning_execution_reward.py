# phase4_method_discovery/vanilla_planning_rlvr/reward/planning_execution_reward.py

from __future__ import annotations

import copy
import json

from pathlib import Path
from typing import Any

import ray
from omegaconf import OmegaConf

from src.execution.taco_evaluator import (
    TACOEvaluator,
)
from src.parsing.code_parser import (
    CodeParser,
)
from src.schemas import (
    ProblemExample,
)


# ======================================================================
# Paths
# ======================================================================

THIS_FILE = Path(__file__).resolve()

PROJECT_ROOT = THIS_FILE.parents[3]

DEFAULT_EXPERIMENT_CONFIG_PATH = (
    PROJECT_ROOT
    / "phase4_method_discovery"
    / "vanilla_planning_rlvr"
    / "configs"
    / "vanilla_planning_rlvr_qwen25coder3b.yaml"
)

DEFAULT_CODE_PROMPT_PATH = (
    PROJECT_ROOT
    / "prompt_templates"
    / "self_plan_code.txt"
)


# ======================================================================
# Process-local CPU reward runtime
# ======================================================================
#
# Important:
#
# The frozen coder model is NOT stored here anymore.
#
# It lives in:
#
#   FrozenCoderWorker
#
# as a separate GPU Ray actor.
#
# The objects below are lightweight CPU-side objects and are created
# lazily once in each RewardLoopWorker process.
# ======================================================================

_CODE_PARSER: CodeParser | None = None

_EVALUATOR: TACOEvaluator | None = None

_CODE_PROMPT_TEMPLATE: str | None = None

_MAX_REWARD_TESTS: int | None = None

_RUNTIME_INITIALIZED: bool = False


# ======================================================================
# Reward result
# ======================================================================

class PlanningRewardResult:
    """
    Detailed result for one Vanilla Planning-RLVR trajectory.

    GRPO consumes only `reward`.

    The remaining fields are useful for rollout diagnostics.
    """

    def __init__(
        self,
        *,
        reward: float,
        problem_id: str,
        plan: str,
        raw_code_output: str,
        generated_code: str,
        code_extraction_method: str,
        passed: bool,
        status: str,
        available_tests: int,
        reward_tests: int,
        passed_tests: int,
        total_tests: int,
        execution_time: float,
        coder_prompt_tokens: int = 0,
        coder_completion_tokens: int = 0,
        coder_generation_time: float = 0.0,
        error_message: str | None = None,
    ) -> None:
        self.reward = float(reward)

        self.problem_id = str(problem_id)
        self.plan = str(plan)

        self.raw_code_output = str(raw_code_output)
        self.generated_code = str(generated_code)
        self.code_extraction_method = str(
            code_extraction_method
        )

        self.passed = bool(passed)
        self.status = str(status)

        self.available_tests = int(
            available_tests
        )
        self.reward_tests = int(
            reward_tests
        )

        self.passed_tests = int(
            passed_tests
        )
        self.total_tests = int(
            total_tests
        )

        self.execution_time = float(
            execution_time
        )

        self.coder_prompt_tokens = int(
            coder_prompt_tokens
        )
        self.coder_completion_tokens = int(
            coder_completion_tokens
        )
        self.coder_generation_time = float(
            coder_generation_time
        )

        self.error_message = (
            None
            if error_message is None
            else str(error_message)
        )


# ======================================================================
# CPU runtime initialization
# ======================================================================

def _initialize_cpu_runtime() -> None:
    """
    Initialize lightweight reward-side components once per
    RewardLoopWorker process.

    This function does NOT load any language model.
    """

    global _CODE_PARSER
    global _EVALUATOR
    global _CODE_PROMPT_TEMPLATE
    global _MAX_REWARD_TESTS
    global _RUNTIME_INITIALIZED

    if _RUNTIME_INITIALIZED:
        return

    # ------------------------------------------------------------------
    # 1. Load research configuration
    # ------------------------------------------------------------------

    if not DEFAULT_EXPERIMENT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            "Planning-RLVR config not found: "
            f"{DEFAULT_EXPERIMENT_CONFIG_PATH}"
        )

    config = OmegaConf.load(
        DEFAULT_EXPERIMENT_CONFIG_PATH
    )

    # ------------------------------------------------------------------
    # 2. Reward settings
    # ------------------------------------------------------------------

    reward_cfg = config.reward

    if str(
        reward_cfg.type
    ) != "binary_execution":
        raise ValueError(
            "Planning-RLVR requires "
            "reward.type=binary_execution, "
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

    # ------------------------------------------------------------------
    # 3. Code prompt
    # ------------------------------------------------------------------

    if not DEFAULT_CODE_PROMPT_PATH.exists():
        raise FileNotFoundError(
            "Self-plan code prompt not found: "
            f"{DEFAULT_CODE_PROMPT_PATH}"
        )

    code_prompt_template = (
        DEFAULT_CODE_PROMPT_PATH.read_text(
            encoding="utf-8",
        )
    )

    if not code_prompt_template.strip():
        raise ValueError(
            "Code prompt template is empty."
        )

    # ------------------------------------------------------------------
    # 4. Parser / evaluator
    # ------------------------------------------------------------------

    _CODE_PARSER = CodeParser()

    _EVALUATOR = TACOEvaluator(
        timeout_seconds=timeout_seconds,
        debug=False,
    )

    _CODE_PROMPT_TEMPLATE = (
        code_prompt_template
    )

    _MAX_REWARD_TESTS = (
        max_reward_tests
    )

    _RUNTIME_INITIALIZED = True


def _ensure_cpu_runtime() -> None:
    if not _RUNTIME_INITIALIZED:
        _initialize_cpu_runtime()

    if _CODE_PARSER is None:
        raise RuntimeError(
            "CodeParser is not initialized."
        )

    if _EVALUATOR is None:
        raise RuntimeError(
            "TACOEvaluator is not initialized."
        )

    if _CODE_PROMPT_TEMPLATE is None:
        raise RuntimeError(
            "Code prompt template is not initialized."
        )

    if _MAX_REWARD_TESTS is None:
        raise RuntimeError(
            "Reward test limit is not initialized."
        )


# ======================================================================
# Prompt construction
# ======================================================================
def build_code_prompt(
    *,
    problem_text: str,
    plan: str,
    starter_code: str = "",
) -> str:
    """
    Build the Self-Plan -> Code prompt.

    Supported template placeholders:
        {problem}
        {plan}
        {starter_code}
        {starter_code_section}
    """

    _ensure_cpu_runtime()

    if not isinstance(problem_text, str):
        raise TypeError(
            "problem_text must be str, "
            f"got {type(problem_text).__name__}."
        )

    if not problem_text.strip():
        raise ValueError(
            "problem_text must not be empty."
        )

    if not isinstance(plan, str):
        raise TypeError(
            "plan must be str, "
            f"got {type(plan).__name__}."
        )

    if not plan.strip():
        raise ValueError(
            "plan must not be empty."
        )

    if starter_code is None:
        starter_code = ""

    if not isinstance(starter_code, str):
        raise TypeError(
            "starter_code must be str, "
            f"got {type(starter_code).__name__}."
        )

    starter_code = starter_code.strip()

    if starter_code:
        starter_code_section = (
            "\nStarter Code:\n"
            "```python\n"
            f"{starter_code}\n"
            "```\n"
        )
    else:
        starter_code_section = ""

    assert _CODE_PROMPT_TEMPLATE is not None

    try:
        prompt = _CODE_PROMPT_TEMPLATE.format(
            problem=problem_text,
            plan=plan,
            starter_code=starter_code,
            starter_code_section=starter_code_section,
        )

    except KeyError as exc:
        raise KeyError(
            "Code prompt template placeholder mismatch. "
            "Supported placeholders: "
            "{problem}, {plan}, {starter_code}, "
            "{starter_code_section}. "
            f"Missing key: {exc}"
        ) from exc

    if not prompt.strip():
        raise RuntimeError(
            "Constructed code prompt is empty."
        )

    return prompt


# ======================================================================
# DeepCoder reward-test sampling
# ======================================================================

def _test_input_length(
    test_input: Any,
) -> int:
    """
    Compute a stable input-length proxy for DeepCoder-style
    hardest-test selection.

    DeepCoder samples tests with the longest input strings.

    TACO stdin inputs are not guaranteed to be plain strings, so:
      - str       -> direct string length
      - list      -> newline-joined representation
      - otherwise -> string representation
    """

    if isinstance(
        test_input,
        str,
    ):
        return len(
            test_input
        )

    if isinstance(
        test_input,
        list,
    ):
        text = "\n".join(
            str(item)
            for item in test_input
        )

        return len(
            text
        )

    return len(
        str(
            test_input
        )
    )


def select_reward_tests(
    problem: ProblemExample,
    *,
    max_tests: int,
) -> ProblemExample:
    """
    Select the DeepCoder-style reward subset.

    Policy:
        choose up to `max_tests` tests with the longest inputs.

    The returned ProblemExample is a copy.
    The original dataset object is never modified.
    """

    if not isinstance(
        problem,
        ProblemExample,
    ):
        raise TypeError(
            "problem must be ProblemExample, "
            f"got {type(problem).__name__}."
        )

    if max_tests <= 0:
        raise ValueError(
            "max_tests must be > 0."
        )

    private_tests = list(
        problem.private_tests
    )

    if not private_tests:
        return copy.deepcopy(
            problem
        )

    if len(
        private_tests
    ) <= max_tests:
        return copy.deepcopy(
            problem
        )

    ranked_tests = sorted(
        enumerate(
            private_tests
        ),
        key=lambda pair: (
            -_test_input_length(
                pair[1].get(
                    "input",
                    "",
                )
            ),
            pair[0],
        ),
    )

    selected_indices = [
        index
        for index, _
        in ranked_tests[
            :max_tests
        ]
    ]

    # Preserve deterministic ranking order used for selection.
    selected_tests = [
        copy.deepcopy(
            private_tests[index]
        )
        for index in selected_indices
    ]

    reward_problem = copy.deepcopy(
        problem
    )

    reward_problem.private_tests = (
        selected_tests
    )

    return reward_problem


# ======================================================================
# Frozen coder RPC
# ======================================================================
def _generate_code_via_rpc(
    *,
    frozen_coder_handle: Any,
    prompt: str,
) -> tuple[
    str,
    int,
    int,
    float,
]:
    """
    Request one deterministic completion from FrozenCoderWorker.

    Lifecycle
    ---------
    1. Wake frozen coder on GPU.
    2. Generate one code completion.
    3. Move frozen coder back to CPU in all cases.

    FrozenCoderWorker.generate_code() returns:

        {
            "text": str,
            "prompt_tokens": int,
            "completion_tokens": int,
            "generation_time": float,
        }

    Notes
    -----
    This per-request wake/sleep lifecycle is intentionally conservative
    for the single-GPU integration smoke test.

    For full GRPO training, this should later be promoted to a
    batch-level lifecycle:

        wake once
        -> score all plans in the reward batch
        -> sleep once

    to avoid repeated CPU <-> GPU transfers.
    """

    if frozen_coder_handle is None:
        raise ValueError(
            "frozen_coder_handle is required."
        )

    if (
        not isinstance(prompt, str)
        or not prompt.strip()
    ):
        raise ValueError(
            "prompt must be a non-empty string."
        )

    # ------------------------------------------------------------------
    # 1. Wake frozen coder on GPU
    # ------------------------------------------------------------------

    try:
        wake_status = ray.get(
            frozen_coder_handle
            .wake_up
            .remote()
        )

    except Exception as exc:
        raise RuntimeError(
            "Failed to wake frozen coder on GPU: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if isinstance(wake_status, dict):
        device = wake_status.get(
            "device"
        )

        if device != "cuda":
            raise RuntimeError(
                "Frozen coder wake_up() completed, "
                f"but reported device={device!r}."
            )

    # ------------------------------------------------------------------
    # 2. Generate
    #
    # Always return the coder to CPU after generation, even when
    # generation raises.
    # ------------------------------------------------------------------

    generation_error: Exception | None = None

    try:
        response = ray.get(
            frozen_coder_handle
            .generate_code
            .remote(
                prompt
            )
        )

    except Exception as exc:
        generation_error = exc
        response = None

    # ------------------------------------------------------------------
    # 3. Sleep frozen coder
    # ------------------------------------------------------------------

    sleep_error: Exception | None = None

    try:
        sleep_status = ray.get(
            frozen_coder_handle
            .sleep
            .remote()
        )

        if isinstance(
            sleep_status,
            dict,
        ):
            device = sleep_status.get(
                "device"
            )

            if device != "cpu":
                raise RuntimeError(
                    "Frozen coder sleep() completed, "
                    f"but reported device={device!r}."
                )

    except Exception as exc:
        sleep_error = exc

    # Preserve generation failure as the primary error.
    if generation_error is not None:
        if sleep_error is not None:
            raise RuntimeError(
                "Frozen coder generation failed, "
                "and subsequent sleep also failed. "
                f"generation_error="
                f"{type(generation_error).__name__}: "
                f"{generation_error}; "
                f"sleep_error="
                f"{type(sleep_error).__name__}: "
                f"{sleep_error}"
            ) from generation_error

        raise RuntimeError(
            "Frozen coder generation failed: "
            f"{type(generation_error).__name__}: "
            f"{generation_error}"
        ) from generation_error

    # A failure to release GPU memory is an infrastructure failure.
    if sleep_error is not None:
        raise RuntimeError(
            "Frozen coder generated successfully, "
            "but failed to return to CPU: "
            f"{type(sleep_error).__name__}: "
            f"{sleep_error}"
        ) from sleep_error

    # ------------------------------------------------------------------
    # 4. Validate RPC response
    # ------------------------------------------------------------------

    if not isinstance(
        response,
        dict,
    ):
        raise TypeError(
            "FrozenCoderWorker.generate_code() "
            "must return dict, "
            f"got {type(response).__name__}."
        )

    raw_output = response.get(
        "text"
    )

    if not isinstance(
        raw_output,
        str,
    ):
        raise TypeError(
            "Frozen coder response['text'] "
            "must be str."
        )

    if not raw_output.strip():
        raise RuntimeError(
            "Frozen coder returned empty output."
        )

    return (
        raw_output,
        int(
            response.get(
                "prompt_tokens",
                0,
            )
        ),
        int(
            response.get(
                "completion_tokens",
                0,
            )
        ),
        float(
            response.get(
                "generation_time",
                0.0,
            )
        ),
    )

# ======================================================================
# Core reward trajectory
# ======================================================================

def compute_planning_execution_reward(
    *,
    problem: ProblemExample,
    problem_text: str,
    plan: str,
    frozen_coder_handle: Any,
) -> PlanningRewardResult:
    """
    Execute one Vanilla Planning-RLVR trajectory.

        problem
            ->
        planner-generated plan
            ->
        FrozenCoderWorker RPC
            ->
        generated code
            ->
        CodeParser
            ->
        DeepCoder/rLLM-compatible TACO evaluator
            ->
        binary execution reward {0, 1}
    """

    _ensure_cpu_runtime()

    if not isinstance(
        problem,
        ProblemExample,
    ):
        raise TypeError(
            "problem must be ProblemExample, "
            f"got {type(problem).__name__}."
        )

    available_tests = len(
        problem.private_tests
    )

    # ------------------------------------------------------------------
    # 1. Validate plan
    # ------------------------------------------------------------------

    if (
        not isinstance(
            plan,
            str,
        )
        or not plan.strip()
    ):
        return PlanningRewardResult(
            reward=0.0,

            problem_id=(
                problem.problem_id
            ),

            plan=(
                plan
                if isinstance(
                    plan,
                    str,
                )
                else ""
            ),

            raw_code_output="",
            generated_code="",
            code_extraction_method="none",

            passed=False,
            status="EMPTY_PLAN",

            available_tests=(
                available_tests
            ),
            reward_tests=0,

            passed_tests=0,
            total_tests=0,

            execution_time=0.0,

            error_message=(
                "Generated plan is empty."
            ),
        )

    # ------------------------------------------------------------------
    # 2. Build code-generation prompt
    # ------------------------------------------------------------------

    try:
        coder_prompt = build_code_prompt(
            problem_text=problem_text,
            plan=plan,
            starter_code=problem.starter_code,
        )

    except Exception as exc:
        return PlanningRewardResult(
            reward=0.0,
            problem_id=problem.problem_id,
            plan=plan,

            raw_code_output="",
            generated_code="",
            code_extraction_method="none",

            passed=False,
            status="CODE_PROMPT_ERROR",

            available_tests=available_tests,
            reward_tests=0,

            passed_tests=0,
            total_tests=0,

            execution_time=0.0,

            error_message=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )

    # ------------------------------------------------------------------
    # 3. Frozen coder RPC
    # ------------------------------------------------------------------

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
        return PlanningRewardResult(
            reward=0.0,
            problem_id=problem.problem_id,
            plan=plan,

            raw_code_output="",
            generated_code="",
            code_extraction_method="none",

            passed=False,
            status="CODE_GENERATION_ERROR",

            available_tests=available_tests,
            reward_tests=0,

            passed_tests=0,
            total_tests=0,

            execution_time=0.0,

            error_message=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )

    # ------------------------------------------------------------------
    # 4. Code extraction
    # ------------------------------------------------------------------

    assert (
        _CODE_PARSER
        is not None
    )

    try:
        parse_result = (
            _CODE_PARSER.parse(
                raw_code_output
            )
        )

    except Exception as exc:
        return PlanningRewardResult(
            reward=0.0,
            problem_id=problem.problem_id,
            plan=plan,

            raw_code_output=(
                raw_code_output
            ),

            generated_code="",
            code_extraction_method="none",

            passed=False,
            status="CODE_PARSING_ERROR",

            available_tests=available_tests,
            reward_tests=0,

            passed_tests=0,
            total_tests=0,

            execution_time=0.0,

            coder_prompt_tokens=(
                coder_prompt_tokens
            ),
            coder_completion_tokens=(
                coder_completion_tokens
            ),
            coder_generation_time=(
                coder_generation_time
            ),

            error_message=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )

    if (
        parse_result.status
        != "SUCCESS"
    ):
        return PlanningRewardResult(
            reward=0.0,
            problem_id=problem.problem_id,
            plan=plan,

            raw_code_output=(
                raw_code_output
            ),

            generated_code=(
                parse_result.code
            ),

            code_extraction_method=(
                parse_result.extraction_method
            ),

            passed=False,
            status=parse_result.status,

            available_tests=available_tests,
            reward_tests=0,

            passed_tests=0,
            total_tests=0,

            execution_time=0.0,

            coder_prompt_tokens=(
                coder_prompt_tokens
            ),
            coder_completion_tokens=(
                coder_completion_tokens
            ),
            coder_generation_time=(
                coder_generation_time
            ),

            error_message=(
                "Code parsing failed: "
                f"{parse_result.status}"
            ),
        )

    generated_code = (
        parse_result.code
    )

    # ------------------------------------------------------------------
    # 5. DeepCoder-style reward test selection
    # ------------------------------------------------------------------

    assert (
        _MAX_REWARD_TESTS
        is not None
    )

    reward_problem = select_reward_tests(
        problem,
        max_tests=_MAX_REWARD_TESTS,
    )

    reward_tests = len(
        reward_problem.private_tests
    )

    if reward_tests <= 0:
        return PlanningRewardResult(
            reward=0.0,
            problem_id=problem.problem_id,
            plan=plan,

            raw_code_output=(
                raw_code_output
            ),
            generated_code=(
                generated_code
            ),
            code_extraction_method=(
                parse_result.extraction_method
            ),

            passed=False,
            status="NO_TESTS",

            available_tests=available_tests,
            reward_tests=0,

            passed_tests=0,
            total_tests=0,

            execution_time=0.0,

            coder_prompt_tokens=(
                coder_prompt_tokens
            ),
            coder_completion_tokens=(
                coder_completion_tokens
            ),
            coder_generation_time=(
                coder_generation_time
            ),

            error_message=(
                "No TACO tests available."
            ),
        )

    # ------------------------------------------------------------------
    # 6. TACO evaluation
    # ------------------------------------------------------------------

    assert (
        _EVALUATOR
        is not None
    )

    try:
        evaluation = (
            _EVALUATOR.evaluate(
                problem=reward_problem,
                code=generated_code,
            )
        )

    except Exception as exc:
        return PlanningRewardResult(
            reward=0.0,
            problem_id=problem.problem_id,
            plan=plan,

            raw_code_output=(
                raw_code_output
            ),
            generated_code=(
                generated_code
            ),
            code_extraction_method=(
                parse_result.extraction_method
            ),

            passed=False,
            status="EVALUATION_ERROR",

            available_tests=available_tests,
            reward_tests=reward_tests,

            passed_tests=0,
            total_tests=0,

            execution_time=0.0,

            coder_prompt_tokens=(
                coder_prompt_tokens
            ),
            coder_completion_tokens=(
                coder_completion_tokens
            ),
            coder_generation_time=(
                coder_generation_time
            ),

            error_message=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )

    # ------------------------------------------------------------------
    # 7. Sparse ORM reward
    #
    # No partial K/N reward.
    # ------------------------------------------------------------------

    reward = (
        1.0
        if evaluation.passed
        else 0.0
    )

    return PlanningRewardResult(
        reward=reward,

        problem_id=(
            problem.problem_id
        ),

        plan=plan,

        raw_code_output=(
            raw_code_output
        ),

        generated_code=(
            generated_code
        ),

        code_extraction_method=(
            parse_result.extraction_method
        ),

        passed=bool(
            evaluation.passed
        ),

        status=str(
            evaluation.status
        ),

        available_tests=(
            available_tests
        ),

        reward_tests=(
            reward_tests
        ),

        passed_tests=int(
            evaluation.passed_tests
        ),

        total_tests=int(
            evaluation.total_tests
        ),

        execution_time=float(
            evaluation.execution_time
        ),

        coder_prompt_tokens=(
            coder_prompt_tokens
        ),

        coder_completion_tokens=(
            coder_completion_tokens
        ),

        coder_generation_time=(
            coder_generation_time
        ),

        error_message=(
            evaluation.error_message
        ),
    )


# ======================================================================
# Dataset reconstruction
# ======================================================================

def _problem_from_extra_info(
    extra_info: dict[str, Any],
) -> ProblemExample:
    """
    Restore ProblemExample from the verl parquet `extra_info`.

    Current dataset builder stores evaluator-only information as
    JSON under:

        extra_info["problem_json"]
    """

    if not isinstance(
        extra_info,
        dict,
    ):
        raise TypeError(
            "extra_info must be dict, "
            f"got {type(extra_info).__name__}."
        )

    problem_json = (
        extra_info.get(
            "problem_json"
        )
    )

    if not isinstance(
        problem_json,
        str,
    ):
        raise TypeError(
            "extra_info['problem_json'] "
            "must be str."
        )

    try:
        payload = json.loads(
            problem_json
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Failed to decode "
            "extra_info['problem_json']."
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            "Decoded problem_json "
            "must be dict."
        )

    problem = ProblemExample(
        **payload
    )

    return problem


# ======================================================================
# verl custom reward entry point
# ======================================================================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    frozen_coder_handle: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    verl-compatible Planning-RLVR reward function.

    Important:
        solution_str is the PLANNER output, not code.

    Reward trajectory:

        solution_str
            = plan
            ->
        FrozenCoderWorker
            ->
        code
            ->
        DeepCoder TACO execution
            ->
        sparse {0, 1} reward

    `ground_truth` is intentionally not used for correctness.
    The unit-test execution result is the correctness authority.
    """

    del ground_truth
    del kwargs

    # ------------------------------------------------------------------
    # 1. Dataset validation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 2. Restore problem
    # ------------------------------------------------------------------

    problem = _problem_from_extra_info(
        extra_info
    )

    problem_text = (
        problem.problem
    )

    if not isinstance(
        problem_text,
        str,
    ) or not problem_text.strip():
        raise ValueError(
            "ProblemExample.problem "
            "must be non-empty."
        )

    # ------------------------------------------------------------------
    # 3. Execute reward trajectory
    # ------------------------------------------------------------------

    result = (
        compute_planning_execution_reward(
            problem=problem,
            problem_text=problem_text,
            plan=solution_str,
            frozen_coder_handle=(
                frozen_coder_handle
            ),
        )
    )

    # ------------------------------------------------------------------
    # 4. Return scalar reward + diagnostics to verl
    # ------------------------------------------------------------------
    #
    # PlanningRewardManager recognizes a dict result and uses:
    #
    #   result["score"]
    #
    # as the GRPO reward.
    #
    # Remaining scalar/string values become reward_extra_info.
    # ------------------------------------------------------------------

    return {
        "score": float(
            result.reward
        ),

        "acc": float(
            result.reward
        ),

        "passed": bool(
            result.passed
        ),

        "status": str(
            result.status
        ),

        "problem_id": str(
            result.problem_id
        ),

        "available_tests": int(
            result.available_tests
        ),

        "reward_tests": int(
            result.reward_tests
        ),

        "passed_tests": int(
            result.passed_tests
        ),

        "total_tests": int(
            result.total_tests
        ),

        "execution_time": float(
            result.execution_time
        ),

        "coder_prompt_tokens": int(
            result.coder_prompt_tokens
        ),

        "coder_completion_tokens": int(
            result.coder_completion_tokens
        ),

        "coder_generation_time": float(
            result.coder_generation_time
        ),

        "error_message": (
            ""
            if result.error_message is None
            else str(
                result.error_message
            )
        ),
    }