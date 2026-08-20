from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.execution.livecodebench_evaluator import (
    LiveCodeBenchEvaluator,
)
from src.models.generator import ModelGenerator
from src.parsing.code_parser import (
    CodeParser,
    CodeParsingError,
)
from src.schemas import ProblemExample


# ============================================================
# Runtime state
# ============================================================

_FROZEN_CODER: ModelGenerator | None = None
_CODE_PARSER: CodeParser | None = None
_EVALUATOR: LiveCodeBenchEvaluator | None = None
_CODE_PROMPT_TEMPLATE: str | None = None

_CODER_MAX_NEW_TOKENS: int = 1024
_CODER_TEMPERATURE: float = 0.0
_CODER_TOP_P: float = 1.0


@dataclass
class PlanningRewardResult:
    """
    Detailed result for one Vanilla Planning-RLVR trajectory.

    Only `reward` is used by GRPO.
    The remaining fields are intended for rollout logging
    and later diagnostic analysis.
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

    error_message: str | None = None


# ============================================================
# Runtime initialization
# ============================================================

def initialize_reward_runtime(
    *,
    frozen_coder: ModelGenerator,
    code_prompt_path: str | Path,
    timeout_seconds: int = 6,
    debug: bool = False,
    coder_max_new_tokens: int = 1024,
    coder_temperature: float = 0.0,
    coder_top_p: float = 1.0,
) -> None:
    """
    Initialize reusable reward-runtime components.

    Must be called once per worker/process.

    IMPORTANT:
    The frozen coder must NOT be reloaded inside compute_score().
    """

    global _FROZEN_CODER
    global _CODE_PARSER
    global _EVALUATOR
    global _CODE_PROMPT_TEMPLATE

    global _CODER_MAX_NEW_TOKENS
    global _CODER_TEMPERATURE
    global _CODER_TOP_P

    if not isinstance(frozen_coder, ModelGenerator):
        raise TypeError(
            "frozen_coder must be ModelGenerator, "
            f"got {type(frozen_coder).__name__}"
        )

    prompt_path = Path(code_prompt_path)

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Code prompt template not found: {prompt_path}"
        )

    if coder_max_new_tokens <= 0:
        raise ValueError(
            "coder_max_new_tokens must be greater than 0."
        )

    if coder_temperature < 0:
        raise ValueError(
            "coder_temperature must be >= 0."
        )

    if not 0 < coder_top_p <= 1:
        raise ValueError(
            "coder_top_p must be in (0, 1]."
        )

    _FROZEN_CODER = frozen_coder

    _CODE_PROMPT_TEMPLATE = prompt_path.read_text(
        encoding="utf-8",
    )

    _CODE_PARSER = CodeParser()

    _EVALUATOR = LiveCodeBenchEvaluator(
        timeout_seconds=timeout_seconds,
        debug=debug,
        include_public_tests=True,
        include_private_tests=True,
    )

    _CODER_MAX_NEW_TOKENS = coder_max_new_tokens
    _CODER_TEMPERATURE = coder_temperature
    _CODER_TOP_P = coder_top_p


def _ensure_runtime_initialized() -> None:
    if _FROZEN_CODER is None:
        raise RuntimeError(
            "Frozen coder is not initialized. "
            "Call initialize_reward_runtime(...) first."
        )

    if _CODE_PARSER is None:
        raise RuntimeError(
            "Code parser is not initialized."
        )

    if _EVALUATOR is None:
        raise RuntimeError(
            "LiveCodeBench evaluator is not initialized."
        )

    if _CODE_PROMPT_TEMPLATE is None:
        raise RuntimeError(
            "Code prompt template is not initialized."
        )


# ============================================================
# Prompt construction
# ============================================================

def build_code_prompt(
    *,
    problem_text: str,
    plan: str,
) -> str:
    """
    Build the same plan-conditioned code-generation prompt
    used in the previous planning experiments.

    Expected template placeholders:

        {problem}
        {plan}
    """

    _ensure_runtime_initialized()

    if not isinstance(problem_text, str):
        raise TypeError(
            "problem_text must be str, "
            f"got {type(problem_text).__name__}"
        )

    if not problem_text.strip():
        raise ValueError(
            "problem_text must not be empty."
        )

    if not isinstance(plan, str):
        raise TypeError(
            "plan must be str, "
            f"got {type(plan).__name__}"
        )

    if not plan.strip():
        raise ValueError(
            "plan must not be empty."
        )

    assert _CODE_PROMPT_TEMPLATE is not None

    try:
        return _CODE_PROMPT_TEMPLATE.format(
            problem=problem_text,
            plan=plan,
        )

    except KeyError as exc:
        raise KeyError(
            "Code prompt template placeholder mismatch. "
            "Expected {problem} and {plan}. "
            f"Missing placeholder: {exc}"
        ) from exc


# ============================================================
# Frozen coder generation
# ============================================================

def _generate_code_output(
    *,
    problem_text: str,
    plan: str,
) -> tuple[str, int, int, float]:
    """
    Generate raw plan-conditioned model output.

    Returns:
        raw_output
        prompt_tokens
        completion_tokens
        generation_time
    """

    _ensure_runtime_initialized()

    prompt = build_code_prompt(
        problem_text=problem_text,
        plan=plan,
    )

    assert _FROZEN_CODER is not None

    generation = _FROZEN_CODER.generate(
        prompt,
        max_new_tokens=_CODER_MAX_NEW_TOKENS,
        temperature=_CODER_TEMPERATURE,
        top_p=_CODER_TOP_P,
    )

    raw_output = generation.text

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


# ============================================================
# Core Planning-RLVR reward pipeline
# ============================================================

def compute_planning_execution_reward(
    *,
    problem: ProblemExample,
    problem_text: str,
    plan: str,
) -> PlanningRewardResult:
    """
    Run one Vanilla Planning-RLVR reward trajectory.

        problem x
            ->
        self-generated plan P
            ->
        frozen coder C ~ pi_coder(C | x, P)
            ->
        CodeParser
            ->
        official LiveCodeBench evaluation
            ->
        R(C) in {0, 1}
    """

    _ensure_runtime_initialized()

    if not isinstance(problem, ProblemExample):
        raise TypeError(
            "problem must be ProblemExample, "
            f"got {type(problem).__name__}"
        )

    # --------------------------------------------------------
    # 1. Validate plan
    # --------------------------------------------------------

    if not isinstance(plan, str) or not plan.strip():
        return PlanningRewardResult(
            reward=0.0,
            problem_id=problem.problem_id,
            plan=plan if isinstance(plan, str) else "",
            raw_code_output="",
            generated_code="",
            code_extraction_method="none",
            passed=False,
            status="EMPTY_PLAN",
            passed_tests=0,
            total_tests=0,
            execution_time=0.0,
            error_message="Generated plan is empty.",
        )

    # --------------------------------------------------------
    # 2. Plan-conditioned code generation
    # --------------------------------------------------------

    try:
        (
            raw_code_output,
            coder_prompt_tokens,
            coder_completion_tokens,
            coder_generation_time,
        ) = _generate_code_output(
            problem_text=problem_text,
            plan=plan,
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
            passed_tests=0,
            total_tests=0,
            execution_time=0.0,
            error_message=(
                f"{type(exc).__name__}: {exc}"
            ),
        )

    # --------------------------------------------------------
    # 3. Parse Python code
    # --------------------------------------------------------

    assert _CODE_PARSER is not None

    try:
        parse_result = _CODE_PARSER.parse(
            raw_code_output
        )

    except Exception as exc:
        return PlanningRewardResult(
            reward=0.0,
            problem_id=problem.problem_id,
            plan=plan,
            raw_code_output=raw_code_output,
            generated_code="",
            code_extraction_method="none",
            passed=False,
            status="CODE_PARSING_ERROR",
            passed_tests=0,
            total_tests=0,
            execution_time=0.0,
            coder_prompt_tokens=coder_prompt_tokens,
            coder_completion_tokens=coder_completion_tokens,
            coder_generation_time=coder_generation_time,
            error_message=(
                f"{type(exc).__name__}: {exc}"
            ),
        )

    if parse_result.status != "SUCCESS":
        return PlanningRewardResult(
            reward=0.0,
            problem_id=problem.problem_id,
            plan=plan,
            raw_code_output=raw_code_output,
            generated_code=parse_result.code,
            code_extraction_method=(
                parse_result.extraction_method
            ),
            passed=False,
            status=parse_result.status,
            passed_tests=0,
            total_tests=0,
            execution_time=0.0,
            coder_prompt_tokens=coder_prompt_tokens,
            coder_completion_tokens=coder_completion_tokens,
            coder_generation_time=coder_generation_time,
            error_message=(
                f"Code parsing failed: "
                f"{parse_result.status}"
            ),
        )

    generated_code = parse_result.code

    # --------------------------------------------------------
    # 4. Official LiveCodeBench evaluation
    # --------------------------------------------------------

    assert _EVALUATOR is not None

    try:
        evaluation = _EVALUATOR.evaluate(
            problem=problem,
            code=generated_code,
        )

    except Exception as exc:
        return PlanningRewardResult(
            reward=0.0,
            problem_id=problem.problem_id,
            plan=plan,
            raw_code_output=raw_code_output,
            generated_code=generated_code,
            code_extraction_method=(
                parse_result.extraction_method
            ),
            passed=False,
            status="EVALUATION_ERROR",
            passed_tests=0,
            total_tests=0,
            execution_time=0.0,
            coder_prompt_tokens=coder_prompt_tokens,
            coder_completion_tokens=coder_completion_tokens,
            coder_generation_time=coder_generation_time,
            error_message=(
                f"{type(exc).__name__}: {exc}"
            ),
        )

    # --------------------------------------------------------
    # 5. Sparse binary execution reward
    # --------------------------------------------------------

    reward = (
        1.0
        if evaluation.passed
        else 0.0
    )

    return PlanningRewardResult(
        reward=reward,
        problem_id=problem.problem_id,
        plan=plan,
        raw_code_output=raw_code_output,
        generated_code=generated_code,
        code_extraction_method=(
            parse_result.extraction_method
        ),
        passed=evaluation.passed,
        status=evaluation.status,
        passed_tests=evaluation.passed_tests,
        total_tests=evaluation.total_tests,
        execution_time=evaluation.execution_time,
        coder_prompt_tokens=coder_prompt_tokens,
        coder_completion_tokens=coder_completion_tokens,
        coder_generation_time=coder_generation_time,
        error_message=evaluation.error_message,
    )


# ============================================================
# verl custom reward entry point
# ============================================================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """
    verl-compatible reward function.

    For Vanilla Planning-RLVR:

        solution_str == generated PLAN

    The final scalar reward is obtained indirectly:

        plan
        -> frozen coder
        -> generated code
        -> execution
        -> 0 / 1

    `ground_truth` is intentionally unused because correctness
    is determined by the LiveCodeBench execution evaluator.
    """

    del ground_truth

    if data_source not in {
        "livecodebench",
        "livecodebench_v6",
    }:
        raise ValueError(
            f"Unsupported data_source: {data_source}"
        )

    if extra_info is None:
        raise ValueError(
            "extra_info is required."
        )

    if "problem" not in extra_info:
        raise KeyError(
            "extra_info['problem'] is required."
        )

    if "problem_text" not in extra_info:
        raise KeyError(
            "extra_info['problem_text'] is required."
        )

    problem_payload = extra_info["problem"]

    if isinstance(
        problem_payload,
        ProblemExample,
    ):
        problem = problem_payload

    elif isinstance(
        problem_payload,
        dict,
    ):
        problem = ProblemExample(
            **problem_payload
        )

    else:
        raise TypeError(
            "extra_info['problem'] must be "
            "ProblemExample or dict, "
            f"got {type(problem_payload).__name__}"
        )

    result = compute_planning_execution_reward(
        problem=problem,
        problem_text=extra_info["problem_text"],
        plan=solution_str,
    )

    return float(result.reward)