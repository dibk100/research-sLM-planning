# tests/test_direct_strategy.py

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from phase1_planning_bottleneck.strategies.direct import (
    DirectStrategy,
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
    path = tmp_path / "direct.txt"

    path.write_text(
        (
            "Title:\n"
            "{title}\n\n"
            "Problem:\n"
            "{problem}\n\n"
            "{starter_code_section}\n\n"
            "Return Python code only."
        ),
        encoding="utf-8",
    )

    return path


@pytest.fixture
def generator() -> MagicMock:
    generator = MagicMock()

    generator.generate.return_value = GenerationOutput(
        text=(
            "```python\n"
            "a = int(input())\n"
            "print(400 // a)\n"
            "```"
        ),
        prompt_tokens=120,
        completion_tokens=24,
        generation_time=0.35,
    )

    return generator


def test_strategy_name():
    assert DirectStrategy.name == "direct"


def test_missing_prompt_file_raises(
    generator: MagicMock,
    tmp_path: Path,
):
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(
        FileNotFoundError,
        match="Prompt template not found",
    ):
        DirectStrategy(
            generator=generator,
            prompt_path=missing_path,
        )


def test_invalid_max_new_tokens_raises(
    generator: MagicMock,
    prompt_file: Path,
):
    with pytest.raises(
        ValueError,
        match="max_new_tokens",
    ):
        DirectStrategy(
            generator=generator,
            prompt_path=prompt_file,
            max_new_tokens=0,
        )


def test_negative_temperature_raises(
    generator: MagicMock,
    prompt_file: Path,
):
    with pytest.raises(
        ValueError,
        match="temperature",
    ):
        DirectStrategy(
            generator=generator,
            prompt_path=prompt_file,
            temperature=-0.1,
        )


@pytest.mark.parametrize(
    "top_p",
    [
        0.0,
        -0.1,
        1.1,
    ],
)
def test_invalid_top_p_raises(
    generator: MagicMock,
    prompt_file: Path,
    top_p: float,
):
    with pytest.raises(
        ValueError,
        match="top_p",
    ):
        DirectStrategy(
            generator=generator,
            prompt_path=prompt_file,
            top_p=top_p,
        )


def test_missing_title_placeholder_raises(
    generator: MagicMock,
    tmp_path: Path,
):
    path = tmp_path / "direct.txt"

    path.write_text(
        (
            "{problem}\n"
            "{starter_code_section}"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"\{title\}",
    ):
        DirectStrategy(
            generator=generator,
            prompt_path=path,
        )


def test_missing_problem_placeholder_raises(
    generator: MagicMock,
    tmp_path: Path,
):
    path = tmp_path / "direct.txt"

    path.write_text(
        (
            "{title}\n"
            "{starter_code_section}"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"\{problem\}",
    ):
        DirectStrategy(
            generator=generator,
            prompt_path=path,
        )


def test_missing_starter_code_placeholder_raises(
    generator: MagicMock,
    tmp_path: Path,
):
    path = tmp_path / "direct.txt"

    path.write_text(
        (
            "{title}\n"
            "{problem}"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"\{starter_code_section\}",
    ):
        DirectStrategy(
            generator=generator,
            prompt_path=path,
        )


def test_build_prompt_for_stdin_problem(
    generator: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
):
    strategy = DirectStrategy(
        generator=generator,
        prompt_path=prompt_file,
    )

    prompt = strategy.build_prompt(problem)

    assert "ABC400 Party" in prompt
    assert problem.problem in prompt

    # stdin example has no starter code.
    assert "Starter Code:" not in prompt

    assert "Return Python code only." in prompt


def test_build_prompt_includes_starter_code(
    generator: MagicMock,
    prompt_file: Path,
    functional_problem: ProblemExample,
):
    strategy = DirectStrategy(
        generator=generator,
        prompt_path=prompt_file,
    )

    prompt = strategy.build_prompt(
        functional_problem
    )

    assert functional_problem.title in prompt
    assert functional_problem.problem in prompt

    assert "Starter Code:" in prompt
    assert (
        functional_problem.starter_code.strip()
        in prompt
    )


def test_run_calls_generator_with_expected_arguments(
    generator: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
):
    strategy = DirectStrategy(
        generator=generator,
        prompt_path=prompt_file,
        system_prompt="You are a coding assistant.",
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )

    expected_prompt = strategy.build_prompt(
        problem
    )

    strategy.run(problem)

    generator.generate.assert_called_once_with(
        prompt=expected_prompt,
        system_prompt="You are a coding assistant.",
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )


def test_run_returns_strategy_output(
    generator: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
):
    strategy = DirectStrategy(
        generator=generator,
        prompt_path=prompt_file,
    )

    result = strategy.run(problem)

    assert isinstance(
        result,
        StrategyOutput,
    )

    assert result.problem_id == problem.problem_id
    assert result.strategy == "direct"

    assert result.raw_output == (
        generator.generate.return_value.text
    )

    assert result.prompt_tokens == 120
    assert result.completion_tokens == 24

    assert result.generation_time == pytest.approx(
        0.35
    )


def test_strategy_output_preserves_raw_output(
    generator: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
):
    """
    DirectStrategy must not parse or modify model output.
    """

    strategy = DirectStrategy(
        generator=generator,
        prompt_path=prompt_file,
    )

    result = strategy.run(problem)

    assert result.raw_output.startswith(
        "```python"
    )

    assert result.raw_output.endswith(
        "```"
    )


def test_strategy_trace_contains_code_generation(
    generator: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
):
    strategy = DirectStrategy(
        generator=generator,
        prompt_path=prompt_file,
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

    assert step.prompt_tokens == 120
    assert step.completion_tokens == 24

    assert step.generation_time == pytest.approx(
        0.35
    )


def test_formatted_prompt_matches_build_prompt(
    generator: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
):
    strategy = DirectStrategy(
        generator=generator,
        prompt_path=prompt_file,
    )

    expected_prompt = strategy.build_prompt(
        problem
    )

    result = strategy.run(problem)

    assert (
        result.formatted_prompt
        == expected_prompt
    )


def test_teacher_plan_fields_are_none(
    generator: MagicMock,
    prompt_file: Path,
    problem: ProblemExample,
):
    strategy = DirectStrategy(
        generator=generator,
        prompt_path=prompt_file,
    )

    result = strategy.run(problem)

    assert result.teacher_plan is None
    assert result.teacher_plan_source is None
    assert result.teacher_plan_version is None
    assert result.teacher_plan_verified is None