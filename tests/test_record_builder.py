# tests/test_record_builder.py

from dataclasses import asdict

import pytest

from src.execution.evaluator import (
    CombinedEvaluationResult,
)
from src.schemas import (
    CodeParseResult,
    EvaluationResult,
    GenerationStep,
    ProblemExample,
    StrategyOutput,
    TestCaseResult,
)
from src.utils.record_builder import (
    build_experiment_record,
)


@pytest.fixture
def example() -> ProblemExample:
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
        public_tests=[
            {
                "input": "10",
                "output": "40",
                "testtype": "stdin",
            }
        ],
        private_tests=[
            {
                "input": "20",
                "output": "20",
                "testtype": "stdin",
            }
        ],
        time_limit=None,
        memory_limit=None,
    )


@pytest.fixture
def strategy_output() -> StrategyOutput:
    step = GenerationStep(
        name="code_generation",
        formatted_prompt="<chat>Write code</chat>",
        raw_output=(
            "```python\n"
            "a = int(input())\n"
            "print(400 // a if 400 % a == 0 else -1)\n"
            "```"
        ),
        prompt_tokens=100,
        completion_tokens=30,
        generation_time=0.5,
    )

    return StrategyOutput(
        problem_id="abc400_a",
        strategy="direct",
        formatted_prompt="<chat>Write code</chat>",
        raw_output=step.raw_output,
        prompt_tokens=100,
        completion_tokens=30,
        generation_time=0.5,
        strategy_trace=[step],
    )


@pytest.fixture
def parse_result() -> CodeParseResult:
    return CodeParseResult(
        code=(
            "a = int(input())\n"
            "print(400 // a if 400 % a == 0 else -1)"
        ),
        status="SUCCESS",
        extraction_method="python_code_block",
    )


@pytest.fixture
def official_result() -> EvaluationResult:
    return EvaluationResult(
        passed=True,
        status="PASS",
        passed_tests=2,
        total_tests=2,
        execution_time=0.12,
        test_results=[],
        error_message=None,
    )


@pytest.fixture
def diagnostic_result() -> EvaluationResult:
    return EvaluationResult(
        passed=True,
        status="PASS",
        passed_tests=2,
        total_tests=2,
        execution_time=0.20,
        test_results=[
            TestCaseResult(
                test_index=0,
                passed=True,
                status="PASS",
                input_text="10",
                expected_output="40",
                actual_output="40",
                execution_time=0.01,
            ),
            TestCaseResult(
                test_index=1,
                passed=True,
                status="PASS",
                input_text="20",
                expected_output="20",
                actual_output="20",
                execution_time=0.01,
            ),
        ],
        error_message=None,
    )


def test_build_record_with_official_and_diagnostic(
    example: ProblemExample,
    strategy_output: StrategyOutput,
    parse_result: CodeParseResult,
    official_result: EvaluationResult,
    diagnostic_result: EvaluationResult,
):
    evaluation = CombinedEvaluationResult(
        official=official_result,
        diagnostic=diagnostic_result,
    )

    record = build_experiment_record(
        example=example,
        strategy_output=strategy_output,
        parse_result=parse_result,
        evaluation=evaluation,
        model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
        seed=42,
    )

    # Identity
    assert record.problem_id == "abc400_a"
    assert record.dataset == "livecodebench_v6"
    assert record.strategy == "direct"
    assert record.model_name == (
        "Qwen/Qwen2.5-Coder-3B-Instruct"
    )
    assert record.seed == 42

    # Problem metadata
    assert record.title == "ABC400 Party"
    assert record.platform == "atcoder"
    assert record.difficulty == "easy"
    assert record.rating is None
    assert record.contest_date == (
        "2025-04-05T00:00:00"
    )

    # Generation / parsing
    assert record.problem == example.problem
    assert (
        record.formatted_prompt
        == strategy_output.formatted_prompt
    )
    assert record.raw_output == strategy_output.raw_output
    assert record.extracted_code == parse_result.code
    assert record.parse_status == "SUCCESS"
    assert (
        record.extraction_method
        == "python_code_block"
    )

    assert record.prompt_tokens == 100
    assert record.completion_tokens == 30
    assert record.generation_time == pytest.approx(0.5)

    # Official evaluation
    assert record.passed is True
    assert record.status == "PASS"
    assert record.execution_time == pytest.approx(0.12)
    assert record.error_message is None

    # Diagnostic evaluation
    assert record.diagnostic_status == "PASS"
    assert record.diagnostic_passed_tests == 2
    assert record.diagnostic_total_tests == 2
    assert (
        record.diagnostic_test_pass_ratio
        == pytest.approx(1.0)
    )
    assert (
        record.diagnostic_execution_time
        == pytest.approx(0.20)
    )

    assert len(record.diagnostic_test_results) == 2
    assert (
        record.diagnostic_test_results[0]["status"]
        == "PASS"
    )

    # Trace
    assert len(record.strategy_trace) == 1
    assert (
        record.strategy_trace[0]["name"]
        == "code_generation"
    )


def test_build_record_without_diagnostic(
    example: ProblemExample,
    strategy_output: StrategyOutput,
    parse_result: CodeParseResult,
    official_result: EvaluationResult,
):
    evaluation = CombinedEvaluationResult(
        official=official_result,
        diagnostic=None,
    )

    record = build_experiment_record(
        example=example,
        strategy_output=strategy_output,
        parse_result=parse_result,
        evaluation=evaluation,
        model_name="test-model",
        seed=1,
    )

    assert record.passed is True
    assert record.status == "PASS"

    assert record.diagnostic_status is None
    assert record.diagnostic_passed_tests is None
    assert record.diagnostic_total_tests is None
    assert record.diagnostic_test_pass_ratio is None
    assert record.diagnostic_execution_time is None
    assert record.diagnostic_test_results == []


def test_diagnostic_ratio_for_partial_pass(
    example: ProblemExample,
    strategy_output: StrategyOutput,
    parse_result: CodeParseResult,
):
    official = EvaluationResult(
        passed=False,
        status="WRONG_ANSWER",
        passed_tests=1,
        total_tests=2,
        execution_time=0.1,
    )

    diagnostic = EvaluationResult(
        passed=False,
        status="WRONG_ANSWER",
        passed_tests=3,
        total_tests=10,
        execution_time=0.3,
    )

    evaluation = CombinedEvaluationResult(
        official=official,
        diagnostic=diagnostic,
    )

    record = build_experiment_record(
        example=example,
        strategy_output=strategy_output,
        parse_result=parse_result,
        evaluation=evaluation,
        model_name="test-model",
        seed=42,
    )

    # Final correctness follows official result.
    assert record.passed is False
    assert record.status == "WRONG_ANSWER"

    # Detailed ratio follows diagnostic result.
    assert record.diagnostic_passed_tests == 3
    assert record.diagnostic_total_tests == 10
    assert (
        record.diagnostic_test_pass_ratio
        == pytest.approx(0.3)
    )


def test_zero_diagnostic_tests_ratio(
    example: ProblemExample,
    strategy_output: StrategyOutput,
    parse_result: CodeParseResult,
):
    official = EvaluationResult(
        passed=False,
        status="EMPTY_CODE",
        passed_tests=0,
        total_tests=0,
        execution_time=0.0,
    )

    diagnostic = EvaluationResult(
        passed=False,
        status="EMPTY_CODE",
        passed_tests=0,
        total_tests=0,
        execution_time=0.0,
    )

    record = build_experiment_record(
        example=example,
        strategy_output=strategy_output,
        parse_result=parse_result,
        evaluation=CombinedEvaluationResult(
            official=official,
            diagnostic=diagnostic,
        ),
        model_name="test-model",
        seed=42,
    )

    assert record.diagnostic_test_pass_ratio == 0.0


def test_problem_id_mismatch_raises(
    example: ProblemExample,
    strategy_output: StrategyOutput,
    parse_result: CodeParseResult,
    official_result: EvaluationResult,
):
    strategy_output.problem_id = "different_problem"

    evaluation = CombinedEvaluationResult(
        official=official_result,
        diagnostic=None,
    )

    with pytest.raises(
        ValueError,
        match="Problem ID mismatch",
    ):
        build_experiment_record(
            example=example,
            strategy_output=strategy_output,
            parse_result=parse_result,
            evaluation=evaluation,
            model_name="test-model",
            seed=42,
        )


def test_parse_information_is_preserved(
    example: ProblemExample,
    strategy_output: StrategyOutput,
    official_result: EvaluationResult,
):
    parse_result = CodeParseResult(
        code="print(0)",
        status="SUCCESS",
        extraction_method="plain_text",
    )

    record = build_experiment_record(
        example=example,
        strategy_output=strategy_output,
        parse_result=parse_result,
        evaluation=CombinedEvaluationResult(
            official=official_result,
            diagnostic=None,
        ),
        model_name="test-model",
        seed=42,
    )

    assert record.extracted_code == "print(0)"
    assert record.parse_status == "SUCCESS"
    assert record.extraction_method == "plain_text"


def test_teacher_plan_metadata_is_preserved(
    example: ProblemExample,
    parse_result: CodeParseResult,
    official_result: EvaluationResult,
):
    strategy_output = StrategyOutput(
        problem_id=example.problem_id,
        strategy="teacher_plan",
        formatted_prompt="prompt",
        raw_output="print(1)",
        prompt_tokens=10,
        completion_tokens=5,
        generation_time=0.1,
        teacher_plan="Use divisibility of 400.",
        teacher_plan_source="teacher-model",
        teacher_plan_version="v1",
        teacher_plan_verified=True,
    )

    record = build_experiment_record(
        example=example,
        strategy_output=strategy_output,
        parse_result=parse_result,
        evaluation=CombinedEvaluationResult(
            official=official_result,
            diagnostic=None,
        ),
        model_name="test-model",
        seed=42,
    )

    assert (
        record.teacher_plan
        == "Use divisibility of 400."
    )
    assert record.teacher_plan_source == "teacher-model"
    assert record.teacher_plan_version == "v1"
    assert record.teacher_plan_verified is True


def test_record_is_serializable_to_dict(
    example: ProblemExample,
    strategy_output: StrategyOutput,
    parse_result: CodeParseResult,
    official_result: EvaluationResult,
    diagnostic_result: EvaluationResult,
):
    record = build_experiment_record(
        example=example,
        strategy_output=strategy_output,
        parse_result=parse_result,
        evaluation=CombinedEvaluationResult(
            official=official_result,
            diagnostic=diagnostic_result,
        ),
        model_name="test-model",
        seed=42,
    )

    record_dict = record.to_dict()

    assert isinstance(record_dict, dict)

    assert record_dict["problem_id"] == "abc400_a"
    assert record_dict["passed"] is True

    assert isinstance(
        record_dict["strategy_trace"],
        list,
    )

    assert isinstance(
        record_dict["diagnostic_test_results"],
        list,
    )


def test_strategy_trace_matches_dataclass_conversion(
    example: ProblemExample,
    strategy_output: StrategyOutput,
    parse_result: CodeParseResult,
    official_result: EvaluationResult,
):
    record = build_experiment_record(
        example=example,
        strategy_output=strategy_output,
        parse_result=parse_result,
        evaluation=CombinedEvaluationResult(
            official=official_result,
            diagnostic=None,
        ),
        model_name="test-model",
        seed=42,
    )

    expected_trace = [
        asdict(step)
        for step in strategy_output.strategy_trace
    ]

    assert record.strategy_trace == expected_trace