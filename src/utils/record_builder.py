# src/utils/record_builder.py

from __future__ import annotations

from dataclasses import asdict

from src.execution.evaluator import (
    CombinedEvaluationResult,
)
from src.schemas import (
    CodeParseResult,
    ExperimentRecord,
    ProblemExample,
    StrategyOutput,
)


def build_experiment_record(
    *,
    example: ProblemExample,
    strategy_output: StrategyOutput,
    parse_result: CodeParseResult,
    evaluation: CombinedEvaluationResult,
    model_name: str,
    seed: int,
) -> ExperimentRecord:
    """
    Combine problem, generation, parsing, and evaluation outputs
    into one serializable experiment record.

    Final correctness always follows the official evaluator.
    Diagnostic evaluation is stored separately for analysis.
    """

    if (
        example.problem_id
        != strategy_output.problem_id
    ):
        raise ValueError(
            "Problem ID mismatch: "
            f"example={example.problem_id}, "
            "strategy_output="
            f"{strategy_output.problem_id}"
        )

    official = evaluation.official
    diagnostic = evaluation.diagnostic

    if diagnostic is not None:
        diagnostic_test_pass_ratio = (
            diagnostic.passed_tests
            / diagnostic.total_tests
            if diagnostic.total_tests > 0
            else 0.0
        )

        diagnostic_test_results = [
            asdict(test_result)
            for test_result
            in diagnostic.test_results
        ]

        diagnostic_status = diagnostic.status
        diagnostic_passed_tests = (
            diagnostic.passed_tests
        )
        diagnostic_total_tests = (
            diagnostic.total_tests
        )
        diagnostic_execution_time = (
            diagnostic.execution_time
        )

    else:
        diagnostic_test_pass_ratio = None
        diagnostic_test_results = []

        diagnostic_status = None
        diagnostic_passed_tests = None
        diagnostic_total_tests = None
        diagnostic_execution_time = None

    return ExperimentRecord(
        # Experiment identity
        problem_id=example.problem_id,
        dataset=example.dataset,
        strategy=strategy_output.strategy,
        model_name=model_name,
        seed=seed,

        # Problem metadata
        title=example.title,
        platform=example.platform,
        contest_date=example.contest_date,
        difficulty=example.difficulty,
        rating=example.rating,

        # Problem / prompt
        problem=example.problem,
        formatted_prompt=(
            strategy_output.formatted_prompt
        ),

        # Generation / parsing
        raw_output=strategy_output.raw_output,
        extracted_code=parse_result.code,
        parse_status=parse_result.status,
        extraction_method=(
            parse_result.extraction_method
        ),

        prompt_tokens=(
            strategy_output.prompt_tokens
        ),
        completion_tokens=(
            strategy_output.completion_tokens
        ),
        generation_time=(
            strategy_output.generation_time
        ),

        # Official evaluation
        passed=official.passed,
        status=official.status,
        execution_time=(
            official.execution_time
        ),
        error_message=(
            official.error_message
        ),

        # Diagnostic evaluation
        diagnostic_status=(
            diagnostic_status
        ),
        diagnostic_passed_tests=(
            diagnostic_passed_tests
        ),
        diagnostic_total_tests=(
            diagnostic_total_tests
        ),
        diagnostic_test_pass_ratio=(
            diagnostic_test_pass_ratio
        ),
        diagnostic_execution_time=(
            diagnostic_execution_time
        ),
        diagnostic_test_results=(
            diagnostic_test_results
        ),

        # Strategy trace
        strategy_trace=[
            asdict(step)
            for step
            in strategy_output.strategy_trace
        ],

        # Teacher plan
        teacher_plan=(
            strategy_output.teacher_plan
        ),
        teacher_plan_source=(
            strategy_output.teacher_plan_source
        ),
        teacher_plan_version=(
            strategy_output.teacher_plan_version
        ),
        teacher_plan_verified=(
            strategy_output.teacher_plan_verified
        ),
    )