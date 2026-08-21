# phase4_method_discovery/vanilla_planning_rlvr/reward/planning_execution_reward.py
from __future__ import annotations

import copy
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.execution.taco_evaluator import (
    TACOEvaluator,
)
from src.models.generator import (
    ModelGenerator,
)
from src.parsing.code_parser import (
    CodeParser,
)
from src.schemas import (
    ProblemExample,
)


# ======================================================================
# Runtime state
# ======================================================================

_FROZEN_CODER: ModelGenerator | None = None
_CODE_PARSER: CodeParser | None = None
_EVALUATOR: TACOEvaluator | None = None
_CODE_PROMPT_TEMPLATE: str | None = None

_CODER_MAX_NEW_TOKENS: int = 1024
_CODER_TEMPERATURE: float = 0.0
_CODER_TOP_P: float = 1.0

# DeepCoder training recipe:
# use at most 15 challenging tests for reward computation.
#
# Set to None to evaluate all available tests.
_MAX_REWARD_TESTS: int | None = 15


# ======================================================================
# Reward result
# ======================================================================

@dataclass
class PlanningRewardResult:
    """
    Detailed result for one Vanilla Planning-RLVR trajectory.

    Only `reward` is required by GRPO.

    The remaining fields are retained for:
    - rollout logging
    - credit-assignment analysis
    - failure analysis
    - token-level analysis after verl integration
    """

    reward: float

    problem_id: str
    plan: str

    raw_code_output: str
    generated_code: str
    code_extraction_method: str

    passed: bool
    status: str

    passed_tests: int
    total_tests: int

    execution_time: float

    coder_prompt_tokens: int = 0
    coder_completion_tokens: int = 0
    coder_generation_time: float = 0.0

    available_tests: int = 0
    reward_tests: int = 0

    error_message: str | None = None


# ======================================================================
# Runtime initialization
# ======================================================================

def initialize_reward_runtime(
    *,
    frozen_coder: ModelGenerator,
    code_prompt_path: str | Path,
    timeout_seconds: int = 6,
    debug: bool = False,
    coder_max_new_tokens: int = 1024,
    coder_temperature: float = 0.0,
    coder_top_p: float = 1.0,
    max_reward_tests: int | None = 15,
) -> None:
    """
    Initialize reusable Phase 4 reward components.

    Must normally be called once per reward worker/process.

    Important
    ---------
    The frozen coder must NOT be reloaded inside compute_score().

    Parameters
    ----------
    frozen_coder:
        Frozen downstream code generator.

    code_prompt_path:
        self_plan_code.txt or compatible template.

    max_reward_tests:
        Maximum number of TACO tests used to compute one rollout reward.

        Default = 15, following the DeepCoder training recipe.

        If a problem contains more than this number, tests with the
        longest serialized inputs are selected.

        Set None to evaluate all tests.
    """

    global _FROZEN_CODER
    global _CODE_PARSER
    global _EVALUATOR
    global _CODE_PROMPT_TEMPLATE

    global _CODER_MAX_NEW_TOKENS
    global _CODER_TEMPERATURE
    global _CODER_TOP_P
    global _MAX_REWARD_TESTS

    if not isinstance(
        frozen_coder,
        ModelGenerator,
    ):
        raise TypeError(
            "frozen_coder must be ModelGenerator, "
            f"got {type(frozen_coder).__name__}"
        )

    prompt_path = Path(
        code_prompt_path
    )

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Code prompt template not found: "
            f"{prompt_path}"
        )

    if coder_max_new_tokens <= 0:
        raise ValueError(
            "coder_max_new_tokens must be > 0."
        )

    if coder_temperature < 0:
        raise ValueError(
            "coder_temperature must be >= 0."
        )

    if not 0 < coder_top_p <= 1:
        raise ValueError(
            "coder_top_p must be in (0, 1]."
        )

    if (
        max_reward_tests is not None
        and max_reward_tests <= 0
    ):
        raise ValueError(
            "max_reward_tests must be > 0 or None."
        )

    _FROZEN_CODER = frozen_coder

    _CODE_PROMPT_TEMPLATE = (
        prompt_path.read_text(
            encoding="utf-8",
        )
    )

    if not _CODE_PROMPT_TEMPLATE.strip():
        raise ValueError(
            "Code prompt template is empty."
        )

    _CODE_PARSER = CodeParser()

    _EVALUATOR = TACOEvaluator(
        timeout_seconds=timeout_seconds,
        debug=debug,
    )

    _CODER_MAX_NEW_TOKENS = (
        coder_max_new_tokens
    )

    _CODER_TEMPERATURE = (
        coder_temperature
    )

    _CODER_TOP_P = (
        coder_top_p
    )

    _MAX_REWARD_TESTS = (
        max_reward_tests
    )


def _ensure_runtime_initialized() -> None:
    if _FROZEN_CODER is None:
        raise RuntimeError(
            "Frozen coder is not initialized. "
            "Call initialize_reward_runtime(...) first."
        )

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


# ======================================================================
# Prompt construction
# ======================================================================

def _build_starter_code_section(
    starter_code: str,
) -> str:
    if not isinstance(
        starter_code,
        str,
    ):
        raise TypeError(
            "starter_code must be str."
        )

    if not starter_code.strip():
        return ""

    return (
        "\n\nStarter Code:\n"
        f"{starter_code.strip()}"
    )


def build_code_prompt(
    *,
    problem: ProblemExample,
    plan: str,
) -> str:
    """
    Build the plan-conditioned code-generation prompt used by
    the frozen coder.

    Supported template placeholders:
        {problem}
        {title}
        {plan}
        {starter_code}
        {starter_code_section}
    """

    _ensure_runtime_initialized()

    if not isinstance(
        problem,
        ProblemExample,
    ):
        raise TypeError(
            "problem must be ProblemExample."
        )

    if not problem.problem.strip():
        raise ValueError(
            "problem.problem must not be empty."
        )

    if (
        not isinstance(plan, str)
        or not plan.strip()
    ):
        raise ValueError(
            "plan must be a non-empty string."
        )

    assert _CODE_PROMPT_TEMPLATE is not None

    starter_code_section = (
        _build_starter_code_section(
            problem.starter_code
        )
    )

    try:
        prompt = (
            _CODE_PROMPT_TEMPLATE.format(
                problem=problem.problem,
                title=problem.title,
                plan=plan,
                starter_code=(
                    problem.starter_code
                ),
                starter_code_section=(
                    starter_code_section
                ),
            )
        )

    except KeyError as exc:
        raise KeyError(
            "Code prompt template placeholder mismatch. "
            "Supported placeholders: "
            "{problem}, {title}, {plan}, "
            "{starter_code}, {starter_code_section}. "
            f"Missing placeholder: {exc}"
        ) from exc

    prompt = prompt.strip()

    if not prompt:
        raise ValueError(
            "Built code prompt is empty."
        )

    return prompt


# ======================================================================
# Frozen coder generation
# ======================================================================

def _generate_code_output(
    *,
    problem: ProblemExample,
    plan: str,
) -> tuple[
    str,
    int,
    int,
    float,
]:
    """
    Generate one plan-conditioned code completion.

    Returns
    -------
    raw_output
    prompt_tokens
    completion_tokens
    generation_time
    """

    _ensure_runtime_initialized()

    prompt = build_code_prompt(
        problem=problem,
        plan=plan,
    )

    assert _FROZEN_CODER is not None

    generation = (
        _FROZEN_CODER.generate(
            prompt,
            max_new_tokens=(
                _CODER_MAX_NEW_TOKENS
            ),
            temperature=(
                _CODER_TEMPERATURE
            ),
            top_p=(
                _CODER_TOP_P
            ),
        )
    )

    raw_output = (
        generation.text
    )

    if not raw_output.strip():
        raise RuntimeError(
            "Frozen coder returned empty output."
        )

    return (
        raw_output,
        generation.prompt_tokens,
        generation.completion_tokens,
        generation.generation_time,
    )


# ======================================================================
# DeepCoder-style reward-test selection
# ======================================================================

def _serialized_input_length(
    test_case: dict[str, Any],
) -> int:
    """
    Compute a stable input-length proxy for DeepCoder-style
    challenging-test selection.

    TACO stdin inputs can be:
    - str
    - list[str]
    - other JSON-compatible values

    We therefore preserve the original representation while using
    its serialized length only for ranking.
    """

    value = test_case.get(
        "input"
    )

    if isinstance(value, str):
        return len(value)

    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    except (TypeError, ValueError):
        serialized = str(
            value
        )

    return len(
        serialized
    )


def select_reward_problem(
    problem: ProblemExample,
    *,
    max_tests: int | None,
) -> ProblemExample:
    """
    Create the ProblemExample used specifically for reward execution.

    The original problem is NOT modified.

    If max_tests is None or the problem already contains <= max_tests
    private tests, the original problem object is returned.

    Otherwise, the tests with the longest inputs are selected.

    This keeps dataset storage and exhaustive sanity evaluation separate
    from rollout-time reward computation.
    """

    if not isinstance(
        problem,
        ProblemExample,
    ):
        raise TypeError(
            "problem must be ProblemExample."
        )

    if max_tests is None:
        return problem

    if max_tests <= 0:
        raise ValueError(
            "max_tests must be > 0 or None."
        )

    tests = (
        problem.private_tests
    )

    if len(tests) <= max_tests:
        return problem

    indexed_tests = list(
        enumerate(tests)
    )

    ranked = sorted(
        indexed_tests,
        key=lambda pair: (
            _serialized_input_length(
                pair[1]
            ),
            -pair[0],
        ),
        reverse=True,
    )

    selected = [
        test_case
        for _, test_case
        in ranked[:max_tests]
    ]

    # Prevent reward-time modification from touching the
    # original dataset object.
    reward_problem = copy.copy(
        problem
    )

    reward_problem.private_tests = (
        selected
    )

    return reward_problem


# ======================================================================
# Common error result
# ======================================================================

def _failure_result(
    *,
    problem: ProblemExample,
    plan: str,
    status: str,
    error_message: str | None,
    raw_code_output: str = "",
    generated_code: str = "",
    code_extraction_method: str = "none",
    coder_prompt_tokens: int = 0,
    coder_completion_tokens: int = 0,
    coder_generation_time: float = 0.0,
    available_tests: int | None = None,
    reward_tests: int = 0,
) -> PlanningRewardResult:
    if available_tests is None:
        available_tests = len(
            problem.private_tests
        )

    return PlanningRewardResult(
        reward=0.0,

        problem_id=(
            problem.problem_id
        ),

        plan=(
            plan
            if isinstance(plan, str)
            else ""
        ),

        raw_code_output=(
            raw_code_output
        ),

        generated_code=(
            generated_code
        ),

        code_extraction_method=(
            code_extraction_method
        ),

        passed=False,
        status=status,

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

        available_tests=(
            available_tests
        ),

        reward_tests=(
            reward_tests
        ),

        error_message=(
            error_message
        ),
    )


# ======================================================================
# Core Planning-RLVR reward trajectory
# ======================================================================

def compute_planning_execution_reward(
    *,
    problem: ProblemExample,
    plan: str,
) -> PlanningRewardResult:
    """
    Execute one Vanilla Planning-RLVR trajectory.

        problem x
            ->
        planner-generated plan P
            ->
        frozen coder C ~ pi_coder(C | x, P)
            ->
        CodeParser
            ->
        DeepCoder/rLLM-compatible TACO execution
            ->
        R(C) in {0, 1}

    Only the planner is intended to be optimized by RL.
    The code generator is frozen.
    """

    _ensure_runtime_initialized()

    if not isinstance(
        problem,
        ProblemExample,
    ):
        raise TypeError(
            "problem must be ProblemExample, "
            f"got {type(problem).__name__}"
        )

    if (
        problem.dataset
        != "deepcoder_taco"
    ):
        raise ValueError(
            "Vanilla Planning-RLVR reward currently "
            "supports deepcoder_taco only, got "
            f"{problem.dataset!r}."
        )

    if (
        problem.evaluation_type
        != "stdin"
    ):
        raise ValueError(
            "Vanilla Planning-RLVR currently supports "
            "TACO stdin problems only."
        )

    available_tests = len(
        problem.private_tests
    )

    # ------------------------------------------------------------------
    # 1. Validate planner response
    # ------------------------------------------------------------------

    if (
        not isinstance(plan, str)
        or not plan.strip()
    ):
        return _failure_result(
            problem=problem,
            plan=(
                plan
                if isinstance(plan, str)
                else ""
            ),
            status="EMPTY_PLAN",
            error_message=(
                "Generated plan is empty."
            ),
            available_tests=(
                available_tests
            ),
        )

    # ------------------------------------------------------------------
    # 2. Frozen plan-conditioned code generation
    # ------------------------------------------------------------------

    try:
        (
            raw_code_output,
            coder_prompt_tokens,
            coder_completion_tokens,
            coder_generation_time,
        ) = _generate_code_output(
            problem=problem,
            plan=plan,
        )

    except Exception as exc:
        return _failure_result(
            problem=problem,
            plan=plan,
            status=(
                "CODE_GENERATION_ERROR"
            ),
            error_message=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            available_tests=(
                available_tests
            ),
        )

    # ------------------------------------------------------------------
    # 3. Parse Python code
    # ------------------------------------------------------------------

    assert _CODE_PARSER is not None

    try:
        parse_result = (
            _CODE_PARSER.parse(
                raw_code_output
            )
        )

    except Exception as exc:
        return _failure_result(
            problem=problem,
            plan=plan,

            status=(
                "CODE_PARSING_ERROR"
            ),

            error_message=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),

            raw_code_output=(
                raw_code_output
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

            available_tests=(
                available_tests
            ),
        )

    if (
        parse_result.status
        != "SUCCESS"
    ):
        return _failure_result(
            problem=problem,
            plan=plan,

            status=(
                parse_result.status
            ),

            error_message=(
                "Code parsing failed: "
                f"{parse_result.status}"
            ),

            raw_code_output=(
                raw_code_output
            ),

            generated_code=(
                parse_result.code
            ),

            code_extraction_method=(
                parse_result.extraction_method
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

            available_tests=(
                available_tests
            ),
        )

    generated_code = (
        parse_result.code
    )

    # ------------------------------------------------------------------
    # 4. Select reward tests
    # ------------------------------------------------------------------

    try:
        reward_problem = (
            select_reward_problem(
                problem,
                max_tests=(
                    _MAX_REWARD_TESTS
                ),
            )
        )

    except Exception as exc:
        return _failure_result(
            problem=problem,
            plan=plan,

            status=(
                "TEST_SELECTION_ERROR"
            ),

            error_message=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),

            raw_code_output=(
                raw_code_output
            ),

            generated_code=(
                generated_code
            ),

            code_extraction_method=(
                parse_result.extraction_method
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

            available_tests=(
                available_tests
            ),
        )

    reward_test_count = len(
        reward_problem.private_tests
    )

    # ------------------------------------------------------------------
    # 5. TACO execution
    # ------------------------------------------------------------------

    assert _EVALUATOR is not None

    try:
        evaluation = (
            _EVALUATOR.evaluate(
                problem=reward_problem,
                code=generated_code,
            )
        )

    except Exception as exc:
        return _failure_result(
            problem=problem,
            plan=plan,

            status="EVALUATION_ERROR",

            error_message=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),

            raw_code_output=(
                raw_code_output
            ),

            generated_code=(
                generated_code
            ),

            code_extraction_method=(
                parse_result.extraction_method
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

            available_tests=(
                available_tests
            ),

            reward_tests=(
                reward_test_count
            ),
        )

    # ------------------------------------------------------------------
    # 6. Sparse binary execution reward
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

        passed=(
            evaluation.passed
        ),

        status=(
            evaluation.status
        ),

        passed_tests=(
            evaluation.passed_tests
        ),

        total_tests=(
            evaluation.total_tests
        ),

        execution_time=(
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

        available_tests=(
            available_tests
        ),

        reward_tests=(
            reward_test_count
        ),

        error_message=(
            evaluation.error_message
        ),
    )


# ======================================================================
# Parquet payload restoration
# ======================================================================

def restore_problem_from_extra_info(
    extra_info: dict[str, Any],
) -> ProblemExample:
    """
    Restore the original ProblemExample serialized by
    build_verl_dataset.py.

    Expected:
        extra_info["problem_json"] = JSON string
    """

    if not isinstance(
        extra_info,
        dict,
    ):
        raise TypeError(
            "extra_info must be dict."
        )

    if (
        "problem_json"
        not in extra_info
    ):
        raise KeyError(
            "extra_info['problem_json'] is required."
        )

    raw_problem = (
        extra_info[
            "problem_json"
        ]
    )

    if not isinstance(
        raw_problem,
        str,
    ):
        raise TypeError(
            "extra_info['problem_json'] "
            "must be str."
        )

    try:
        payload = json.loads(
            raw_problem
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
            "Decoded problem_json must "
            "be a dict."
        )

    try:
        problem = ProblemExample(
            **payload
        )

    except TypeError as exc:
        raise TypeError(
            "Failed to reconstruct "
            "ProblemExample from problem_json."
        ) from exc

    if (
        "problem_id"
        in extra_info
        and str(
            extra_info[
                "problem_id"
            ]
        )
        != problem.problem_id
    ):
        raise ValueError(
            "problem_id mismatch between "
            "extra_info and problem_json."
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
) -> float:
    """
    verl-compatible Vanilla Planning-RLVR reward function.

    Parameters
    ----------
    data_source:
        Must be "deepcoder_taco".

    solution_str:
        Planner-generated response.
        In this experiment, this is the PLAN, not code.

    ground_truth:
        Intentionally unused.

        Planning quality is evaluated indirectly through:

            plan
              -> frozen coder
              -> generated code
              -> execution tests
              -> 0/1

    extra_info:
        Must contain the serialized ProblemExample under
        extra_info["problem_json"].

    Returns
    -------
    float
        1.0 if generated code passes the reward tests,
        otherwise 0.0.
    """

    del ground_truth

    if (
        data_source
        != "deepcoder_taco"
    ):
        raise ValueError(
            "Unsupported data_source: "
            f"{data_source!r}. "
            "Expected 'deepcoder_taco'."
        )

    if extra_info is None:
        raise ValueError(
            "extra_info is required."
        )

    problem = (
        restore_problem_from_extra_info(
            extra_info
        )
    )

    result = (
        compute_planning_execution_reward(
            problem=problem,
            plan=solution_str,
        )
    )

    return float(
        result.reward
    )