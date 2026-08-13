# tests/test_livecodebench_stdin_loader.py
"""
python -m pytest -v tests/test_livecodebench_stdin_loader.py
"""
from collections import Counter
from pathlib import Path

import pytest

from src.datasets.dataset_loader import load_dataset
from src.schemas import ProblemExample


DATA_PATH = Path(
    "/mnt/hdd/project_sLM_planning/data/livecodebench_v6/stdin"
)

EXPECTED_NUM_PROBLEMS = 300

EXPECTED_DIFFICULTY_COUNTS = {
    "easy": 100,
    "medium": 100,
    "hard": 100,
}


@pytest.fixture(scope="module")
def problems() -> list[ProblemExample]:
    """
    Load the complete LiveCodeBench-v6 stdin diagnostic benchmark once.
    """
    return load_dataset(
        dataset_name="livecodebench_v6_stdin",
        data_path=DATA_PATH,
        limit=None,
    )


def test_dataset_path_exists():
    """Diagnostic benchmark must exist on disk."""
    assert DATA_PATH.exists(), (
        f"Dataset path does not exist: {DATA_PATH}"
    )


def test_num_problems(problems):
    """The stdin diagnostic benchmark must contain exactly 300 problems."""
    assert len(problems) == EXPECTED_NUM_PROBLEMS, (
        f"Expected {EXPECTED_NUM_PROBLEMS} problems, "
        f"but loaded {len(problems)}"
    )


def test_problem_schema(problems):
    """Every loaded item must be normalized to ProblemExample."""
    assert all(isinstance(problem, ProblemExample) for problem in problems)


def test_problem_ids_are_present_and_unique(problems):
    """Every problem must have a non-empty and unique problem_id."""
    problem_ids = [problem.problem_id for problem in problems]

    assert all(problem_ids), "Found an empty problem_id"

    assert len(problem_ids) == len(set(problem_ids)), (
        "Duplicate problem_id detected"
    )


def test_required_problem_fields(problems):
    """
    Fields required by the planning experiments must not be empty.
    """
    for problem in problems:
        assert problem.problem_id
        assert problem.title
        assert problem.problem
        assert problem.platform
        assert problem.contest_date
        assert problem.difficulty


def test_dataset_identity(problems):
    """All examples must come from the expected benchmark."""
    for problem in problems:
        assert problem.dataset == "livecodebench_v6"


def test_platform_is_atcoder(problems):
    """
    The diagnostic stdin subset was constructed from AtCoder problems only.
    """
    platforms = {problem.platform.lower() for problem in problems}

    assert platforms == {"atcoder"}, (
        f"Unexpected platforms found: {platforms}"
    )


def test_evaluation_type_is_stdin(problems):
    """Every problem in this subset must use stdin evaluation."""
    evaluation_types = {
        problem.evaluation_type for problem in problems
    }

    assert evaluation_types == {"stdin"}, (
        f"Unexpected evaluation types: {evaluation_types}"
    )


def test_difficulty_distribution(problems):
    """
    Phase 0 diagnostic benchmark construction selected
    100 Easy / 100 Medium / 100 Hard problems.
    """
    counts = Counter(
        problem.difficulty.lower()
        for problem in problems
    )

    assert dict(counts) == EXPECTED_DIFFICULTY_COUNTS, (
        "Unexpected difficulty distribution: "
        f"{dict(counts)}"
    )


def test_public_tests_exist(problems):
    """Every problem should contain parsed public test cases."""
    for problem in problems:
        assert isinstance(problem.public_tests, list)
        assert len(problem.public_tests) > 0, (
            f"{problem.problem_id}: no public tests"
        )


def test_private_tests_exist(problems):
    """
    Every diagnostic problem must contain decoded private tests,
    because these are used for correctness evaluation.
    """
    for problem in problems:
        assert isinstance(problem.private_tests, list)
        assert len(problem.private_tests) > 0, (
            f"{problem.problem_id}: no private tests"
        )


def test_public_test_schema(problems):
    """Public stdin tests must contain input/output pairs."""
    for problem in problems:
        for index, test_case in enumerate(problem.public_tests):
            assert isinstance(test_case, dict), (
                f"{problem.problem_id} public test {index} "
                "is not a dictionary"
            )

            assert "input" in test_case, (
                f"{problem.problem_id} public test {index}: "
                "missing input"
            )

            assert "output" in test_case, (
                f"{problem.problem_id} public test {index}: "
                "missing output"
            )

            if "testtype" in test_case:
                assert test_case["testtype"] == "stdin", (
                    f"{problem.problem_id} public test {index}: "
                    f"unexpected testtype={test_case['testtype']}"
                )


def test_private_test_schema(problems):
    """Decoded private tests must contain input/output pairs."""
    for problem in problems:
        for index, test_case in enumerate(problem.private_tests):
            assert isinstance(test_case, dict), (
                f"{problem.problem_id} private test {index} "
                "is not a dictionary"
            )

            assert "input" in test_case, (
                f"{problem.problem_id} private test {index}: "
                "missing input"
            )

            assert "output" in test_case, (
                f"{problem.problem_id} private test {index}: "
                "missing output"
            )


def test_starter_code_is_string(problems):
    """
    stdin problems normally have empty starter_code,
    but the common schema should consistently expose a string.
    """
    for problem in problems:
        assert isinstance(problem.starter_code, str)