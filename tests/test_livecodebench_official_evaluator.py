# tests/test_livecodebench_official_evaluator.py

"""

PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python -m pytest -v tests/test_livecodebench_official_evaluator.py

"""


from pathlib import Path

import pytest

from src.datasets.dataset_loader import load_dataset
from src.execution.livecodebench_evaluator import (
    LiveCodeBenchEvaluator,
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
    """
    Use one stdin problem from the diagnostic benchmark.
    """
    problems = load_dataset(
        dataset_name="livecodebench_v6_stdin",
        data_path=STDIN_DATA_PATH,
        limit=1,
    )

    assert len(problems) == 1

    return problems[0]


@pytest.fixture(scope="module")
def functional_problem() -> ProblemExample:
    """
    Use one functional problem from the diagnostic benchmark.
    """
    problems = load_dataset(
        dataset_name="livecodebench_v6_functional",
        data_path=FUNCTIONAL_DATA_PATH,
        limit=1,
    )

    assert len(problems) == 1

    return problems[0]


@pytest.fixture(scope="module")
def evaluator() -> LiveCodeBenchEvaluator:
    return LiveCodeBenchEvaluator(
        timeout_seconds=6,
        debug=False,
    )


def test_stdin_problem_loaded_correctly(
    stdin_problem: ProblemExample,
):
    assert stdin_problem.dataset == "livecodebench_v6"
    assert stdin_problem.platform == "atcoder"
    assert stdin_problem.evaluation_type == "stdin"

    assert stdin_problem.problem_id
    assert stdin_problem.problem

    assert len(stdin_problem.public_tests) > 0
    assert len(stdin_problem.private_tests) > 0


def test_functional_problem_loaded_correctly(
    functional_problem: ProblemExample,
):
    assert functional_problem.dataset == "livecodebench_v6"
    assert functional_problem.platform == "leetcode"
    assert functional_problem.evaluation_type == "functional"

    assert functional_problem.problem_id
    assert functional_problem.problem
    assert functional_problem.starter_code
    assert functional_problem.function_name

    assert len(functional_problem.public_tests) > 0
    assert len(functional_problem.private_tests) > 0


def test_stdin_empty_code(
    evaluator: LiveCodeBenchEvaluator,
    stdin_problem: ProblemExample,
):
    result = evaluator.evaluate(
        stdin_problem,
        "",
    )

    assert result.passed is False
    assert result.status == "EMPTY_CODE"


def test_stdin_wrong_answer(
    evaluator: LiveCodeBenchEvaluator,
    stdin_problem: ProblemExample,
):
    """
    A trivial program should fail on a normal stdin problem.
    """

    wrong_code = """
print(0)
"""

    result = evaluator.evaluate(
        stdin_problem,
        wrong_code,
    )

    assert result.passed is False

    assert result.status in {
        "WRONG_ANSWER",
        "RUNTIME_ERROR",
        "TIME_LIMIT_EXCEEDED",
        "FAILED",
    }


def test_stdin_runtime_error(
    evaluator: LiveCodeBenchEvaluator,
    stdin_problem: ProblemExample,
):
    runtime_error_code = """
raise RuntimeError("intentional sanity-check error")
"""

    result = evaluator.evaluate(
        stdin_problem,
        runtime_error_code,
    )

    assert result.passed is False

    assert result.status == "RUNTIME_ERROR"


def test_stdin_timeout(
    stdin_problem: ProblemExample,
):
    evaluator = LiveCodeBenchEvaluator(
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

    assert result.status == "TIME_LIMIT_EXCEEDED"


def test_functional_wrong_answer(
    evaluator: LiveCodeBenchEvaluator,
    functional_problem: ProblemExample,
):
    """
    Verify that the official evaluator actually enters
    call-based / functional evaluation.
    """

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

    assert result.passed is False

    assert result.status in {
        "WRONG_ANSWER",
        "RUNTIME_ERROR",
        "FAILED",
    }


def test_result_total_tests_matches_problem(
    evaluator: LiveCodeBenchEvaluator,
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