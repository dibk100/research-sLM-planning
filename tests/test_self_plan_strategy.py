# tests/test_self_plan_strategy.py

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from phase1_planning_bottleneck.strategies.self_plan import (
    SelfPlanningStrategy,
)
from src.schemas import (
    GenerationOutput,
    ProblemExample,
    StrategyOutput,
)


@pytest.fixture
def problem() -> ProblemExample:
    return ProblemExample(
        problem_id="abc400_a",
        title="ABC400 Party",
        problem="Given A, find B such that A * B = 400.",
        starter_code="",
        dataset="livecodebench_v6",
        platform="atcoder",
        difficulty="easy",
        rating=None,
        contest_date="2025-04-05T00:00:00",
        evaluation_type="stdin",
        function_name=None,
        public_tests=[],
        private_tests=[],
        time_limit=None,
        memory_limit=None,
    )


@pytest.fixture
def functional_problem() -> ProblemExample:
    return ProblemExample(
        problem_id="3773",
        title="minimum-pair-removal-to-sort-array-i",
        problem="Given an array nums, return the minimum operations.",
        starter_code=(
            "class Solution:\n"
            "    def minimumPairRemoval("
            "self, nums: List[int]) -> int:\n"
            "        "
        ),
        dataset="livecodebench_v6",
        platform="leetcode",
        difficulty="easy",
        rating=None,
        contest_date="2025-04-05T19:30:00",
        evaluation_type="functional",
        function_name="minimumPairRemoval",
        public_tests=[],
        private_tests=[],
        time_limit=None,
        memory_limit=None,
    )


@pytest.fixture
def plan_prompt_file(
    tmp_path: Path,
) -> Path:
    path = tmp_path / "self_plan_plan.txt"

    path.write_text(
        (
            "Title:\n"
            "{title}\n\n"
            "Problem:\n"
            "{problem}\n\n"
            "{starter_code_section}\n\n"
            "Create a concise algorithmic plan."
        ),
        encoding="utf-8",
    )

    return path


@pytest.fixture
def code_prompt_file(
    tmp_path: Path,
) -> Path:
    path = tmp_path / "self_plan_code.txt"

    path.write_text(
        (
            "Title:\n"
            "{title}\n\n"
            "Problem:\n"
            "{problem}\n\n"
            "Plan:\n"
            "{plan}\n\n"
            "{starter_code_section}\n\n"
            "Implement the plan in Python."
        ),
        encoding="utf-8",
    )

    return path


@pytest.fixture
def generator() -> MagicMock:
    generator = MagicMock()

    plan_generation = GenerationOutput(
        text=(
            "Check whether 400 is divisible by A. "
            "If so, output 400 // A; otherwise output -1."
        ),
        prompt_tokens=100,
        completion_tokens=40,
        generation_time=0.3,
    )

    code_generation = GenerationOutput(
        text=(
            "```python\n"
            "a = int(input())\n"
            "print(400 // a if 400 % a == 0 else -1)\n"
            "```"
        ),
        prompt_tokens=160,
        completion_tokens=35,
        generation_time=0.5,
    )

    generator.generate.side_effect = [
        plan_generation,
        code_generation,
    ]

    return generator


def test_strategy_name():
    assert SelfPlanningStrategy.name == "self_plan"


def test_missing_plan_prompt_file_raises(
    generator: MagicMock,
    code_prompt_file: Path,
    tmp_path: Path,
):
    with pytest.raises(
        FileNotFoundError,
        match="Plan prompt template not found",
    ):
        SelfPlanningStrategy(
            generator=generator,
            plan_prompt_path=tmp_path / "missing.txt",
            code_prompt_path=code_prompt_file,
        )


def test_missing_code_prompt_file_raises(
    generator: MagicMock,
    plan_prompt_file: Path,
    tmp_path: Path,
):
    with pytest.raises(
        FileNotFoundError,
        match="Code prompt template not found",
    ):
        SelfPlanningStrategy(
            generator=generator,
            plan_prompt_path=plan_prompt_file,
            code_prompt_path=tmp_path / "missing.txt",
        )


@pytest.mark.parametrize(
    "plan_max_new_tokens",
    [0, -1],
)
def test_invalid_plan_max_new_tokens_raises(
    generator: MagicMock,
    plan_prompt_file: Path,
    code_prompt_file: Path,
    plan_max_new_tokens: int,
):
    with pytest.raises(
        ValueError,
        match="plan_max_new_tokens",
    ):
        SelfPlanningStrategy(
            generator=generator,
            plan_prompt_path=plan_prompt_file,
            code_prompt_path=code_prompt_file,
            plan_max_new_tokens=plan_max_new_tokens,
        )


@pytest.mark.parametrize(
    "code_max_new_tokens",
    [0, -1],
)
def test_invalid_code_max_new_tokens_raises(
    generator: MagicMock,
    plan_prompt_file: Path,
    code_prompt_file: Path,
    code_max_new_tokens: int,
):
    with pytest.raises(
        ValueError,
        match="code_max_new_tokens",
    ):
        SelfPlanningStrategy(
            generator=generator,
            plan_prompt_path=plan_prompt_file,
            code_prompt_path=code_prompt_file,
            code_max_new_tokens=code_max_new_tokens,
        )


def test_negative_temperature_raises(
    generator: MagicMock,
    plan_prompt_file: Path,
    code_prompt_file: Path,
):
    with pytest.raises(
        ValueError,
        match="temperature",
    ):
        SelfPlanningStrategy(
            generator=generator,
            plan_prompt_path=plan_prompt_file,
            code_prompt_path=code_prompt_file,
            temperature=-0.1,
        )


@pytest.mark.parametrize(
    "top_p",
    [0.0, -0.1, 1.1],
)
def test_invalid_top_p_raises(
    generator: MagicMock,
    plan_prompt_file: Path,
    code_prompt_file: Path,
    top_p: float,
):
    with pytest.raises(
        ValueError,
        match="top_p",
    ):
        SelfPlanningStrategy(
            generator=generator,
            plan_prompt_path=plan_prompt_file,
            code_prompt_path=code_prompt_file,
            top_p=top_p,
        )


def test_missing_plan_placeholder_raises(
    generator: MagicMock,
    tmp_path: Path,
    code_prompt_file: Path,
):
    bad_plan_prompt = tmp_path / "bad_plan.txt"

    bad_plan_prompt.write_text(
        "{title}\n{starter_code_section}",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"\{problem\}",
    ):
        SelfPlanningStrategy(
            generator=generator,
            plan_prompt_path=bad_plan_prompt,
            code_prompt_path=code_prompt_file,
        )


def test_missing_code_plan_placeholder_raises(
    generator: MagicMock,
    plan_prompt_file: Path,
    tmp_path: Path,
):
    bad_code_prompt = tmp_path / "bad_code.txt"

    bad_code_prompt.write_text(
        (
            "{title}\n"
            "{problem}\n"
            "{starter_code_section}"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"\{plan\}",
    ):
        SelfPlanningStrategy(
            generator=generator,
            plan_prompt_path=plan_prompt_file,
            code_prompt_path=bad_code_prompt,
        )


def test_build_plan_prompt_for_stdin_problem(
    generator: MagicMock,
    plan_prompt_file: Path,
    code_prompt_file: Path,
    problem: ProblemExample,
):
    strategy = SelfPlanningStrategy(
        generator=generator,
        plan_prompt_path=plan_prompt_file,
        code_prompt_path=code_prompt_file,
    )

    prompt = strategy.build_plan_prompt(problem)

    assert problem.title in prompt
    assert problem.problem in prompt
    assert "Starter Code:" not in prompt
    assert "Create a concise algorithmic plan." in prompt


def test_build_plan_prompt_includes_starter_code(
    generator: MagicMock,
    plan_prompt_file: Path,
    code_prompt_file: Path,
    functional_problem: ProblemExample,
):
    strategy = SelfPlanningStrategy(
        generator=generator,
        plan_prompt_path=plan_prompt_file,
        code_prompt_path=code_prompt_file,
    )

    prompt = strategy.build_plan_prompt(
        functional_problem
    )

    assert "Starter Code:" in prompt
    assert (
        functional_problem.starter_code.strip()
        in prompt
    )


def test_build_code_prompt_contains_plan(
    generator: MagicMock,
    plan_prompt_file: Path,
    code_prompt_file: Path,
    problem: ProblemExample,
):
    strategy = SelfPlanningStrategy(
        generator=generator,
        plan_prompt_path=plan_prompt_file,
        code_prompt_path=code_prompt_file,
    )

    plan = "Use divisibility and compute 400 // A."

    prompt = strategy.build_code_prompt(
        example=problem,
        plan=plan,
    )

    assert problem.title in prompt
    assert problem.problem in prompt
    assert plan in prompt


def test_build_code_prompt_rejects_empty_plan(
    generator: MagicMock,
    plan_prompt_file: Path,
    code_prompt_file: Path,
    problem: ProblemExample,
):
    strategy = SelfPlanningStrategy(
        generator=generator,
        plan_prompt_path=plan_prompt_file,
        code_prompt_path=code_prompt_file,
    )

    with pytest.raises(
        ValueError,
        match="Generated plan must not be empty",
    ):
        strategy.build_code_prompt(
            example=problem,
            plan="   ",
        )


def test_build_code_prompt_rejects_non_string_plan(
    generator: MagicMock,
    plan_prompt_file: Path,
    code_prompt_file: Path,
    problem: ProblemExample,
):
    strategy = SelfPlanningStrategy(
        generator=generator,
        plan_prompt_path=plan_prompt_file,
        code_prompt_path=code_prompt_file,
    )

    with pytest.raises(TypeError):
        strategy.build_code_prompt(
            example=problem,
            plan=None,  # type: ignore[arg-type]
        )


def test_run_calls_generator_twice_in_order(
    generator: MagicMock,
    plan_prompt_file: Path,
    code_prompt_file: Path,
    problem: ProblemExample,
):
    strategy = SelfPlanningStrategy(
        generator=generator,
        plan_prompt_path=plan_prompt_file,
        code_prompt_path=code_prompt_file,
        system_prompt="You are a coding assistant.",
        plan_max_new_tokens=256,
        code_max_new_tokens=768,
        temperature=0.7,
        top_p=0.95,
    )

    plan_prompt = strategy.build_plan_prompt(problem)

    expected_plan = (
        "Check whether 400 is divisible by A. "
        "If so, output 400 // A; otherwise output -1."
    )

    code_prompt = strategy.build_code_prompt(
        example=problem,
        plan=expected_plan,
    )

    strategy.run(problem)

    assert generator.generate.call_count == 2

    assert generator.generate.call_args_list == [
        call(
            prompt=plan_prompt,
            system_prompt="You are a coding assistant.",
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
        ),
        call(
            prompt=code_prompt,
            system_prompt="You are a coding assistant.",
            max_new_tokens=768,
            temperature=0.7,
            top_p=0.95,
        ),
    ]

def test_generated_plan_is_passed_to_code_prompt(
    generator: MagicMock,
    plan_prompt_file: Path,
    code_prompt_file: Path,
    problem: ProblemExample,
):
    strategy = SelfPlanningStrategy(
        generator=generator,
        plan_prompt_path=plan_prompt_file,
        code_prompt_path=code_prompt_file,
    )

    expected_plan = (
        "Check whether 400 is divisible by A. "
        "If so, output 400 // A; otherwise output -1."
    )

    result = strategy.run(problem)

    assert expected_plan in result.formatted_prompt


def test_run_rejects_empty_generated_plan(
    plan_prompt_file: Path,
    code_prompt_file: Path,
    problem: ProblemExample,
):
    generator = MagicMock()

    generator.generate.return_value = GenerationOutput(
        text="   ",
        prompt_tokens=10,
        completion_tokens=1,
        generation_time=0.1,
    )

    strategy = SelfPlanningStrategy(
        generator=generator,
        plan_prompt_path=plan_prompt_file,
        code_prompt_path=code_prompt_file,
    )

    with pytest.raises(
        ValueError,
        match="Empty plan generated",
    ):
        strategy.run(problem)

    # Code-generation step must never be reached.
    assert generator.generate.call_count == 1


def test_run_returns_strategy_output(
    generator: MagicMock,
    plan_prompt_file: Path,
    code_prompt_file: Path,
    problem: ProblemExample,
):
    strategy = SelfPlanningStrategy(
        generator=generator,
        plan_prompt_path=plan_prompt_file,
        code_prompt_path=code_prompt_file,
    )

    result = strategy.run(problem)

    assert isinstance(result, StrategyOutput)

    assert result.problem_id == problem.problem_id
    assert result.strategy == "self_plan"

    assert result.raw_output.startswith("```python")
    assert result.raw_output.endswith("```")


def test_strategy_cost_is_sum_of_two_generations(
    generator: MagicMock,
    plan_prompt_file: Path,
    code_prompt_file: Path,
    problem: ProblemExample,
):
    strategy = SelfPlanningStrategy(
        generator=generator,
        plan_prompt_path=plan_prompt_file,
        code_prompt_path=code_prompt_file,
    )

    result = strategy.run(problem)

    assert result.prompt_tokens == 260
    assert result.completion_tokens == 75

    assert result.generation_time == pytest.approx(
        0.8
    )


def test_strategy_trace_has_two_steps_in_order(
    generator: MagicMock,
    plan_prompt_file: Path,
    code_prompt_file: Path,
    problem: ProblemExample,
):
    strategy = SelfPlanningStrategy(
        generator=generator,
        plan_prompt_path=plan_prompt_file,
        code_prompt_path=code_prompt_file,
    )

    result = strategy.run(problem)

    assert len(result.strategy_trace) == 2

    plan_step = result.strategy_trace[0]
    code_step = result.strategy_trace[1]

    assert plan_step.name == "plan_generation"
    assert code_step.name == "code_generation"

    assert plan_step.raw_output.startswith(
        "Check whether 400"
    )

    assert code_step.raw_output.startswith(
        "```python"
    )


def test_strategy_trace_cost_matches_total_cost(
    generator: MagicMock,
    plan_prompt_file: Path,
    code_prompt_file: Path,
    problem: ProblemExample,
):
    strategy = SelfPlanningStrategy(
        generator=generator,
        plan_prompt_path=plan_prompt_file,
        code_prompt_path=code_prompt_file,
    )

    result = strategy.run(problem)

    trace_prompt_tokens = sum(
        step.prompt_tokens
        for step in result.strategy_trace
    )

    trace_completion_tokens = sum(
        step.completion_tokens
        for step in result.strategy_trace
    )

    trace_generation_time = sum(
        step.generation_time
        for step in result.strategy_trace
    )

    assert trace_prompt_tokens == result.prompt_tokens
    assert (
        trace_completion_tokens
        == result.completion_tokens
    )
    assert trace_generation_time == pytest.approx(
        result.generation_time
    )


def test_final_fields_correspond_to_code_generation(
    generator: MagicMock,
    plan_prompt_file: Path,
    code_prompt_file: Path,
    problem: ProblemExample,
):
    strategy = SelfPlanningStrategy(
        generator=generator,
        plan_prompt_path=plan_prompt_file,
        code_prompt_path=code_prompt_file,
    )

    result = strategy.run(problem)

    code_step = result.strategy_trace[1]

    assert (
        result.formatted_prompt
        == code_step.formatted_prompt
    )
    assert result.raw_output == code_step.raw_output


def test_teacher_plan_fields_are_none(
    generator: MagicMock,
    plan_prompt_file: Path,
    code_prompt_file: Path,
    problem: ProblemExample,
):
    strategy = SelfPlanningStrategy(
        generator=generator,
        plan_prompt_path=plan_prompt_file,
        code_prompt_path=code_prompt_file,
    )

    result = strategy.run(problem)

    assert result.teacher_plan is None
    assert result.teacher_plan_source is None
    assert result.teacher_plan_version is None
    assert result.teacher_plan_verified is None