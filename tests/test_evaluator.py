# tests/test_evaluator.py

from pathlib import Path

import pytest

from src.datasets.dataset_loader import load_dataset
from src.execution.evaluator import (
    CombinedEvaluationResult,
    Evaluator,
)
from src.schemas import (
    EvaluationResult,
    ProblemExample,
)


STDIN_DATA_PATH = Path(
    "/mnt/hdd/project_sLM_planning/data/livecodebench_v6/stdin"
)


@pytest.fixture(scope="module")
def stdin_problem() -> ProblemExample:
    problems = load_dataset(
        dataset_name="livecodebench_v6_stdin",
        data_path=STDIN_DATA_PATH,
        limit=1,
    )

    assert len(problems) == 1

    return problems[0]


@pytest.fixture(scope="module")
def official_only_evaluator() -> Evaluator:
    return Evaluator(
        official_timeout_seconds=6,
        enable_diagnostic=False,
        debug=False,
    )


@pytest.fixture(scope="module")
def diagnostic_enabled_evaluator() -> Evaluator:
    return Evaluator(
        official_timeout_seconds=6,
        diagnostic_timeout_seconds=6,
        enable_diagnostic=True,
        debug=False,
    )


def test_official_only_evaluation(
    official_only_evaluator: Evaluator,
    stdin_problem: ProblemExample,
):
    """
    Diagnostic disabled by default:
    only the official result should be returned.
    """

    result = official_only_evaluator.evaluate(
        stdin_problem,
        "print(0)",
    )

    assert isinstance(
        result,
        CombinedEvaluationResult,
    )

    assert isinstance(
        result.official,
        EvaluationResult,
    )

    assert result.diagnostic is None

    assert result.passed == result.official.passed
    assert result.status == result.official.status

    assert (
        result.diagnostic_test_pass_ratio
        is None
    )

    assert result.evaluation_mismatch is None


def test_diagnostic_enabled_evaluation(
    diagnostic_enabled_evaluator: Evaluator,
    stdin_problem: ProblemExample,
):
    """
    When diagnostic evaluation is enabled,
    both official and diagnostic results should exist.
    """

    result = diagnostic_enabled_evaluator.evaluate(
        stdin_problem,
        "print(0)",
    )

    assert isinstance(
        result.official,
        EvaluationResult,
    )

    assert isinstance(
        result.diagnostic,
        EvaluationResult,
    )

    # Final decision must always follow official evaluation.
    assert result.passed == result.official.passed
    assert result.status == result.official.status

    expected_total_tests = (
        len(stdin_problem.public_tests)
        + len(stdin_problem.private_tests)
    )

    assert (
        result.diagnostic.total_tests
        == expected_total_tests
    )

    assert (
        len(result.diagnostic.test_results)
        == expected_total_tests
    )


def test_diagnostic_test_pass_ratio(
    diagnostic_enabled_evaluator: Evaluator,
    stdin_problem: ProblemExample,
):
    result = diagnostic_enabled_evaluator.evaluate(
        stdin_problem,
        "print(0)",
    )

    diagnostic = result.diagnostic

    assert diagnostic is not None

    expected_ratio = (
        diagnostic.passed_tests
        / diagnostic.total_tests
    )

    assert (
        result.diagnostic_test_pass_ratio
        == pytest.approx(expected_ratio)
    )

    assert (
        0.0
        <= result.diagnostic_test_pass_ratio
        <= 1.0
    )


def test_per_call_enable_diagnostic(
    official_only_evaluator: Evaluator,
    stdin_problem: ProblemExample,
):
    """
    Evaluator-wide diagnostic=False can be overridden
    for an individual call.
    """

    result = official_only_evaluator.evaluate(
        stdin_problem,
        "print(0)",
        run_diagnostic=True,
    )

    assert result.diagnostic is not None

    assert (
        len(result.diagnostic.test_results)
        == result.diagnostic.total_tests
    )


def test_per_call_disable_diagnostic(
    diagnostic_enabled_evaluator: Evaluator,
    stdin_problem: ProblemExample,
):
    """
    Evaluator-wide diagnostic=True can be disabled
    for an individual call.
    """

    result = diagnostic_enabled_evaluator.evaluate(
        stdin_problem,
        "print(0)",
        run_diagnostic=False,
    )

    assert result.diagnostic is None


def test_evaluate_official_only_method(
    official_only_evaluator: Evaluator,
    stdin_problem: ProblemExample,
):
    result = (
        official_only_evaluator.evaluate_official(
            stdin_problem,
            "print(0)",
        )
    )

    assert isinstance(
        result,
        EvaluationResult,
    )

    assert result.passed is False


def test_evaluate_diagnostic_only_method(
    official_only_evaluator: Evaluator,
    stdin_problem: ProblemExample,
):
    result = (
        official_only_evaluator.evaluate_diagnostic(
            stdin_problem,
            "print(0)",
        )
    )

    assert isinstance(
        result,
        EvaluationResult,
    )

    expected_total_tests = (
        len(stdin_problem.public_tests)
        + len(stdin_problem.private_tests)
    )

    assert result.total_tests == expected_total_tests

    assert (
        len(result.test_results)
        == expected_total_tests
    )


def test_evaluation_mismatch_property(
    stdin_problem: ProblemExample,
):
    """
    Test CombinedEvaluationResult mismatch logic directly.

    This avoids relying on a naturally occurring evaluator mismatch.
    """

    official = EvaluationResult(
        passed=False,
        status="WRONG_ANSWER",
        passed_tests=0,
        total_tests=1,
        execution_time=0.0,
    )

    diagnostic = EvaluationResult(
        passed=True,
        status="PASS",
        passed_tests=1,
        total_tests=1,
        execution_time=0.0,
    )

    result = CombinedEvaluationResult(
        official=official,
        diagnostic=diagnostic,
    )

    assert result.passed is False
    assert result.status == "WRONG_ANSWER"

    assert result.evaluation_mismatch is True


def test_no_evaluation_mismatch(
    stdin_problem: ProblemExample,
):
    official = EvaluationResult(
        passed=False,
        status="WRONG_ANSWER",
        passed_tests=0,
        total_tests=1,
        execution_time=0.0,
    )

    diagnostic = EvaluationResult(
        passed=False,
        status="WRONG_ANSWER",
        passed_tests=0,
        total_tests=1,
        execution_time=0.0,
    )

    result = CombinedEvaluationResult(
        official=official,
        diagnostic=diagnostic,
    )

    assert result.evaluation_mismatch is False


def test_empty_code_final_decision_follows_official(
    diagnostic_enabled_evaluator: Evaluator,
    stdin_problem: ProblemExample,
):
    result = diagnostic_enabled_evaluator.evaluate(
        stdin_problem,
        "",
    )

    assert result.official.passed is False
    assert result.official.status == "EMPTY_CODE"

    assert result.passed is False
    assert result.status == "EMPTY_CODE"


def test_invalid_problem_type(
    official_only_evaluator: Evaluator,
):
    with pytest.raises(TypeError):
        official_only_evaluator.evaluate(
            problem="not-a-problem",  # type: ignore
            code="print(0)",
        )