# src/utils/record_builder.py

from __future__ import annotations

from dataclasses import asdict

from src.schemas import (
    CodeParseResult,
    EvaluationResult,
    ExperimentRecord,
    ProblemExample,
    StrategyOutput,
)


def build_experiment_record(
    *,
    example: ProblemExample,
    strategy_output: StrategyOutput,
    parse_result: CodeParseResult,
    evaluation: EvaluationResult,
    model_name: str,
    seed: int,
) -> ExperimentRecord:
    """
    Combine problem, generation, parsing, and exhaustive evaluation
    outputs into one serializable experiment record.

    Final correctness is determined by exhaustive evaluation:
    a problem passes only when all selected test cases pass.
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

    test_pass_ratio = (
        evaluation.passed_tests
        / evaluation.total_tests
        if evaluation.total_tests > 0
        else 0.0
    )

    test_results = [
        asdict(test_result)
        for test_result
        in evaluation.test_results
    ]

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

        # Exhaustive evaluation
        passed=evaluation.passed,
        status=evaluation.status,

        passed_tests=(
            evaluation.passed_tests
        ),
        total_tests=(
            evaluation.total_tests
        ),
        test_pass_ratio=(
            test_pass_ratio
        ),

        execution_time=(
            evaluation.execution_time
        ),
        error_message=(
            evaluation.error_message
        ),

        test_results=(
            test_results
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