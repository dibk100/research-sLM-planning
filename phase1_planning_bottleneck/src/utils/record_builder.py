"""
Experiment record construction utilities.

현재 컴포넌트는 스키마(src/schemas.py)에서 정의한 객체(ProblemExample, StrategyOutput, EvaluationResult)를 사용하여 실험 기록을 생성한다.
이를 하나의 ExperimentRecord로 합치는 함수 build_experiment_record()를 제공한다.

"""

from __future__ import annotations

from dataclasses import asdict

from src.schemas import (
    EvaluationResult,
    ExperimentRecord,
    ProblemExample,
    StrategyOutput,
)


def build_experiment_record(
    *,
    example: ProblemExample,
    strategy_output: StrategyOutput,
    extracted_code: str,
    evaluation: EvaluationResult,
    dataset_name: str,
    model_name: str,
    seed: int,
) -> ExperimentRecord:
    """파이프라인 출력을 저장 가능한 단일 레코드로 결합한다."""
    if example.problem_id != strategy_output.problem_id:
        raise ValueError(
            "Problem ID mismatch: "
            f"example={example.problem_id}, "
            f"strategy_output={strategy_output.problem_id}"
        )

    return ExperimentRecord(
        problem_id=example.problem_id,
        dataset=dataset_name,
        strategy=strategy_output.strategy,
        model_name=model_name,
        seed=seed,

        title=example.title,
        platform=example.platform,
        contest_id=example.contest_id,
        contest_date=example.contest_date,
        difficulty=example.difficulty,

        prompt=strategy_output.prompt,
        raw_output=strategy_output.raw_output,
        extracted_code=extracted_code,

        prompt_tokens=strategy_output.prompt_tokens,
        completion_tokens=strategy_output.completion_tokens,
        generation_time=strategy_output.generation_time,

        passed=evaluation.passed,
        status=evaluation.status,
        passed_tests=evaluation.passed_tests,
        total_tests=evaluation.total_tests,
        execution_time=evaluation.execution_time,

        error_message=evaluation.error_message,
        test_results=[
            asdict(test_result)
            for test_result in evaluation.test_results
        ],
    )