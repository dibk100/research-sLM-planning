# tests/test_diagnostic_evaluator.py
"""
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python -m pytest -v tests/test_diagnostic_evaluator.py
"""

from pathlib import Path

import pytest

from src.datasets.dataset_loader import load_dataset
from src.execution.diagnostic_evaluator import (
    DiagnosticEvaluator,
)
from src.schemas import ProblemExample


STDIN_DATA_PATH = Path(
    "/mnt/hdd/project_sLM_planning/data/livecodebench_v6/stdin"
)

FUNCTIONAL_DATA_PATH = Path(
    "/mnt/hdd/project_sLM_planning/data/livecodebench_v6/functional"
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
def functional_problem() -> ProblemExample:
    problems = load_dataset(
        dataset_name="livecodebench_v6_functional",
        data_path=FUNCTIONAL_DATA_PATH,
        limit=1,
    )

    assert len(problems) == 1

    return problems[0]


@pytest.fixture(scope="module")
def evaluator() -> DiagnosticEvaluator:
    return DiagnosticEvaluator(
        timeout_seconds=6,
        debug=False,
    )


def test_stdin_problem_loaded_correctly(
    stdin_problem: ProblemExample,
):
    assert stdin_problem.dataset == "livecodebench_v6"
    assert stdin_problem.evaluation_type == "stdin"

    assert stdin_problem.public_tests
    assert stdin_problem.private_tests


def test_functional_problem_loaded_correctly(
    functional_problem: ProblemExample,
):
    assert functional_problem.dataset == "livecodebench_v6"
    assert functional_problem.evaluation_type == "functional"

    assert functional_problem.function_name
    assert functional_problem.starter_code

    assert functional_problem.public_tests
    assert functional_problem.private_tests


def test_empty_code(
    evaluator: DiagnosticEvaluator,
    stdin_problem: ProblemExample,
):
    result = evaluator.evaluate(
        stdin_problem,
        "",
    )

    assert result.passed is False
    assert result.status == "EMPTY_CODE"
    assert result.passed_tests == 0
    assert result.total_tests == 0


def test_total_tests_matches_problem(
    evaluator: DiagnosticEvaluator,
    stdin_problem: ProblemExample,
):
    wrong_code = """
print(0)
"""

    result = evaluator.evaluate(
        stdin_problem,
        wrong_code,
    )

    expected_total_tests = (
        len(stdin_problem.public_tests)
        + len(stdin_problem.private_tests)
    )

    assert result.total_tests == expected_total_tests

    # Diagnostic evaluator must actually produce one result
    # for every enabled test case.
    assert len(result.test_results) == expected_total_tests


def test_wrong_answer_runs_all_tests(
    evaluator: DiagnosticEvaluator,
    stdin_problem: ProblemExample,
):
    """
    Unlike the official fail-fast evaluator, the diagnostic
    evaluator must evaluate every test independently.
    """

    wrong_code = """
print(0)
"""

    result = evaluator.evaluate(
        stdin_problem,
        wrong_code,
    )

    assert result.passed is False

    assert result.total_tests == len(
        result.test_results
    )

    assert result.passed_tests < result.total_tests

    assert all(
        test_result.status
        in {
            "PASS",
            "WRONG_ANSWER",
            "RUNTIME_ERROR",
            "TIME_LIMIT_EXCEEDED",
            "TEST_RUNNER_ERROR",
            "FAILED",
        }
        for test_result in result.test_results
    )


def test_test_pass_ratio_is_valid(
    evaluator: DiagnosticEvaluator,
    stdin_problem: ProblemExample,
):
    wrong_code = """
print(0)
"""

    result = evaluator.evaluate(
        stdin_problem,
        wrong_code,
    )

    ratio = evaluator.test_pass_ratio(result)

    assert 0.0 <= ratio <= 1.0

    assert ratio == pytest.approx(
        result.passed_tests
        / result.total_tests
    )


def test_runtime_error_on_all_tests(
    evaluator: DiagnosticEvaluator,
    stdin_problem: ProblemExample,
):
    runtime_error_code = """
raise RuntimeError("intentional diagnostic test")
"""

    result = evaluator.evaluate(
        stdin_problem,
        runtime_error_code,
    )

    assert result.passed is False

    assert len(result.test_results) == result.total_tests

    assert all(
        test_result.passed is False
        for test_result in result.test_results
    )

    assert all(
        test_result.status == "RUNTIME_ERROR"
        for test_result in result.test_results
    )

    assert result.status == "RUNTIME_ERROR"


def test_timeout_on_all_tests(
    stdin_problem: ProblemExample,
):
    evaluator = DiagnosticEvaluator(
        timeout_seconds=1,
        debug=False,
    )

    timeout_code = """
while True:
    pass
"""

    result = evaluator.evaluate(
        stdin_problem,
        timeout_code,
    )

    assert result.passed is False

    assert len(result.test_results) == result.total_tests

    assert all(
        test_result.passed is False
        for test_result in result.test_results
    )

    assert all(
        test_result.status == "TIME_LIMIT_EXCEEDED"
        for test_result in result.test_results
    )

    assert result.status == "TIME_LIMIT_EXCEEDED"


def test_functional_wrong_answer_runs_all_tests(
    evaluator: DiagnosticEvaluator,
    functional_problem: ProblemExample,
):
    function_name = functional_problem.function_name

    wrong_code = f"""
class Solution:
    def {function_name}(self, *args):
        return None
"""

    result = evaluator.evaluate(
        functional_problem,
        wrong_code,
    )

    expected_total_tests = (
        len(functional_problem.public_tests)
        + len(functional_problem.private_tests)
    )

    assert result.passed is False

    assert result.total_tests == expected_total_tests

    assert len(result.test_results) == expected_total_tests

    assert result.passed_tests < result.total_tests


def test_public_only_mode(
    stdin_problem: ProblemExample,
):
    evaluator = DiagnosticEvaluator(
        timeout_seconds=6,
        debug=False,
        include_public_tests=True,
        include_private_tests=False,
    )

    result = evaluator.evaluate(
        stdin_problem,
        "print(0)",
    )

    assert result.total_tests == len(
        stdin_problem.public_tests
    )

    assert len(result.test_results) == len(
        stdin_problem.public_tests
    )


def test_private_only_mode(
    stdin_problem: ProblemExample,
):
    evaluator = DiagnosticEvaluator(
        timeout_seconds=6,
        debug=False,
        include_public_tests=False,
        include_private_tests=True,
    )

    result = evaluator.evaluate(
        stdin_problem,
        "print(0)",
    )

    assert result.total_tests == len(
        stdin_problem.private_tests
    )

    assert len(result.test_results) == len(
        stdin_problem.private_tests
    )