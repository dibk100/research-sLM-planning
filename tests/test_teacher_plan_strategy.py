# tests/test_teacher_plan_strategy.py

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from phase1_planning_bottleneck.strategies.teacher_plan import (
    TeacherPlanStrategy,
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
def prompt_file(tmp_path: Path) -> Path:
    path = tmp_path / "teacher_plan_code.txt"

    path.write_text(
        (
            "Title:\n"
            "{title}\n\n"
            "Problem:\n"
            "{problem}\n\n"
            "Teacher Plan:\n"
            "{teacher_plan}\n\n"
            "{starter_code_section}\n\n"
            "Implement the plan in Python."
        ),
        encoding="utf-8",
    )

    return path


@pytest.fixture
def plan_record():
    return SimpleNamespace(
        teacher_plan=(
            "Check whether 400 is divisible by A. "
            "If divisible, output 400 // A; otherwise output -1."
        ),
        teacher_model="claude-opus-4.1",
        plan_version="v1",
        verified=True,
    )


@pytest.fixture
def plan_store(
    plan_record,
) -> MagicMock:
    store = MagicMock()
    store.get.return_value = plan_record
    return store


@pytest.fixture
def generator() -> MagicMock:
    generator = MagicMock()

    generator.generate.return_value = GenerationOutput(
        text=(
            "```python\n"
            "a = int(input())\n"
            "print(400 // a if 400 % a == 0 else -1)\n"
            "```"
        ),
        prompt_tokens=150,
        completion_tokens=35,
        generation_time=0.45,
    )

    return generator


def test_strategy_name():
    assert TeacherPlanStrategy.name == "teacher_plan"


def test_missing_prompt_file_raises(
    generator: MagicMock,
    plan_store: MagicMock,
    tmp_path: Path,
):
    with pytest.raises(
        FileNotFoundError,
        match="Teacher-plan code prompt not found",
    ):
        TeacherPlanStrategy(
            generator=generator,
            plan_store=plan_store,
            code_prompt_path=tmp_path / "missing.txt",
        )


@pytest.mark.parametrize(
    "max_tokens",
    [0, -1],
)
def test_invalid_code_max_new_tokens_raises(
    generator: MagicMock,
    plan_store: MagicMock,
    prompt_file: Path,
    max_tokens: int,
):
    with pytest.raises(
        ValueError,
        match="code_max_new_tokens",
    ):
        TeacherPlanStrategy(
            generator=generator,
            plan_store=plan_store,
            code_prompt_path=prompt_file,
            code_max_new_tokens=max_tokens,
        )


def test_negative_temperature_raises(
    generator: MagicMock,
    plan_store: MagicMock,
    prompt_file: Path,
):
    with pytest.raises(
        ValueError,
        match="temperature",
    ):
        TeacherPlanStrategy(
            generator=generator,
            plan_store=plan_store,
            code_prompt_path=prompt_file,
            temperature=-0.1,
        )


@pytest.mark.parametrize(
    "top_p",
    [0.0, -0.1, 1.1],
)
def test_invalid_top_p_raises(
    generator: MagicMock,
    plan_store: MagicMock,
    prompt_file: Path,
    top_p: float,
):
    with pytest.raises(
        ValueError,
        match="top_p",
    ):
        TeacherPlanStrategy(
            generator=generator,
            plan_store=plan_store,
            code_prompt_path=prompt_file,
            top_p=top_p,
        )


@pytest.mark.parametrize(
    "missing_placeholder",
    [
        "{title}",
        "{problem}",
        "{teacher_plan}",
        "{starter_code_section}",
    ],
)
def test_missing_prompt_placeholder_raises(
    generator: MagicMock,
    plan_store: MagicMock,
    tmp_path: Path,
    missing_placeholder: str,
):
    template = (
        "{title}\n"
        "{problem}\n"
        "{teacher_plan}\n"
        "{starter_code_section}"
    )

    template = template.replace(
        missing_placeholder,
        "",
    )

    path = tmp_path / "bad_prompt.txt"
    path.write_text(
        template,
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Missing teacher-plan prompt",
    ):
        TeacherPlanStrategy(
            generator=generator,
            plan_store=plan_store,
            code_prompt_path=path,
        )


def test_build_code_prompt_contains_teacher_plan(
    generator: MagicMock,
    plan_store: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
    plan_record,
):
    strategy = TeacherPlanStrategy(
        generator=generator,
        plan_store=plan_store,
        code_prompt_path=prompt_file,
    )

    prompt = strategy.build_code_prompt(
        example=problem,
        teacher_plan=plan_record.teacher_plan,
    )

    assert problem.title in prompt
    assert problem.problem in prompt
    assert plan_record.teacher_plan in prompt

    assert "Starter Code:" not in prompt


def test_build_prompt_includes_starter_code(
    generator: MagicMock,
    plan_store: MagicMock,
    prompt_file: Path,
    functional_problem: ProblemExample,
    plan_record,
):
    strategy = TeacherPlanStrategy(
        generator=generator,
        plan_store=plan_store,
        code_prompt_path=prompt_file,
    )

    prompt = strategy.build_code_prompt(
        example=functional_problem,
        teacher_plan=plan_record.teacher_plan,
    )

    assert "Starter Code:" in prompt

    assert (
        functional_problem.starter_code.strip()
        in prompt
    )


def test_build_code_prompt_rejects_empty_teacher_plan(
    generator: MagicMock,
    plan_store: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
):
    strategy = TeacherPlanStrategy(
        generator=generator,
        plan_store=plan_store,
        code_prompt_path=prompt_file,
    )

    with pytest.raises(
        ValueError,
        match="Teacher plan must not be empty",
    ):
        strategy.build_code_prompt(
            example=problem,
            teacher_plan="   ",
        )


def test_build_code_prompt_rejects_non_string_plan(
    generator: MagicMock,
    plan_store: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
):
    strategy = TeacherPlanStrategy(
        generator=generator,
        plan_store=plan_store,
        code_prompt_path=prompt_file,
    )

    with pytest.raises(TypeError):
        strategy.build_code_prompt(
            example=problem,
            teacher_plan=None,  # type: ignore[arg-type]
        )


def test_run_retrieves_plan_by_problem_id(
    generator: MagicMock,
    plan_store: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
):
    strategy = TeacherPlanStrategy(
        generator=generator,
        plan_store=plan_store,
        code_prompt_path=prompt_file,
    )

    strategy.run(problem)

    plan_store.get.assert_called_once_with(
        problem.problem_id
    )


def test_run_calls_generator_once(
    generator: MagicMock,
    plan_store: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
    plan_record,
):
    strategy = TeacherPlanStrategy(
        generator=generator,
        plan_store=plan_store,
        code_prompt_path=prompt_file,
        system_prompt="You are a coding assistant.",
        code_max_new_tokens=768,
        temperature=0.7,
        top_p=0.95,
    )

    expected_prompt = strategy.build_code_prompt(
        example=problem,
        teacher_plan=plan_record.teacher_plan,
    )

    strategy.run(problem)

    generator.generate.assert_called_once_with(
        prompt=expected_prompt,
        system_prompt="You are a coding assistant.",
        max_new_tokens=768,
        temperature=0.7,
        top_p=0.95,
    )


def test_teacher_plan_is_passed_to_code_prompt(
    generator: MagicMock,
    plan_store: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
    plan_record,
):
    strategy = TeacherPlanStrategy(
        generator=generator,
        plan_store=plan_store,
        code_prompt_path=prompt_file,
    )

    strategy.run(problem)

    call_kwargs = (
        generator.generate.call_args.kwargs
    )

    assert (
        plan_record.teacher_plan
        in call_kwargs["prompt"]
    )


def test_run_rejects_empty_plan_from_store(
    generator: MagicMock,
    plan_store: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
):
    plan_store.get.return_value = SimpleNamespace(
        teacher_plan="   ",
        teacher_model="teacher",
        plan_version="v1",
        verified=True,
    )

    strategy = TeacherPlanStrategy(
        generator=generator,
        plan_store=plan_store,
        code_prompt_path=prompt_file,
    )

    with pytest.raises(
        ValueError,
        match="Empty teacher plan",
    ):
        strategy.run(problem)

    generator.generate.assert_not_called()


def test_run_returns_strategy_output(
    generator: MagicMock,
    plan_store: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
):
    strategy = TeacherPlanStrategy(
        generator=generator,
        plan_store=plan_store,
        code_prompt_path=prompt_file,
    )

    result = strategy.run(problem)

    assert isinstance(
        result,
        StrategyOutput,
    )

    assert result.problem_id == problem.problem_id
    assert result.strategy == "teacher_plan"

    assert result.raw_output.startswith(
        "```python"
    )
    assert result.raw_output.endswith(
        "```"
    )


def test_teacher_plan_metadata_is_preserved(
    generator: MagicMock,
    plan_store: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
    plan_record,
):
    strategy = TeacherPlanStrategy(
        generator=generator,
        plan_store=plan_store,
        code_prompt_path=prompt_file,
    )

    result = strategy.run(problem)

    assert (
        result.teacher_plan
        == plan_record.teacher_plan
    )
    assert (
        result.teacher_plan_source
        == plan_record.teacher_model
    )
    assert (
        result.teacher_plan_version
        == plan_record.plan_version
    )
    assert (
        result.teacher_plan_verified
        is plan_record.verified
    )


def test_strategy_cost_contains_only_student_generation(
    generator: MagicMock,
    plan_store: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
):
    strategy = TeacherPlanStrategy(
        generator=generator,
        plan_store=plan_store,
        code_prompt_path=prompt_file,
    )

    result = strategy.run(problem)

    assert result.prompt_tokens == 150
    assert result.completion_tokens == 35
    assert result.generation_time == pytest.approx(
        0.45
    )


def test_strategy_trace_has_only_code_generation(
    generator: MagicMock,
    plan_store: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
):
    strategy = TeacherPlanStrategy(
        generator=generator,
        plan_store=plan_store,
        code_prompt_path=prompt_file,
    )

    result = strategy.run(problem)

    assert len(result.strategy_trace) == 1

    step = result.strategy_trace[0]

    assert step.name == "code_generation"
    assert (
        step.formatted_prompt
        == result.formatted_prompt
    )
    assert step.raw_output == result.raw_output

    assert step.prompt_tokens == 150
    assert step.completion_tokens == 35
    assert step.generation_time == pytest.approx(
        0.45
    )


def test_strategy_trace_does_not_fake_teacher_generation(
    generator: MagicMock,
    plan_store: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
):
    """
    Teacher plan is externally supplied and must not appear
    as a model generation step.
    """

    strategy = TeacherPlanStrategy(
        generator=generator,
        plan_store=plan_store,
        code_prompt_path=prompt_file,
    )

    result = strategy.run(problem)

    step_names = [
        step.name
        for step in result.strategy_trace
    ]

    assert "plan_generation" not in step_names
    assert step_names == ["code_generation"]


def test_final_fields_match_code_generation_step(
    generator: MagicMock,
    plan_store: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
):
    strategy = TeacherPlanStrategy(
        generator=generator,
        plan_store=plan_store,
        code_prompt_path=prompt_file,
    )

    result = strategy.run(problem)

    code_step = result.strategy_trace[0]

    assert (
        result.formatted_prompt
        == code_step.formatted_prompt
    )
    assert (
        result.raw_output
        == code_step.raw_output
    )