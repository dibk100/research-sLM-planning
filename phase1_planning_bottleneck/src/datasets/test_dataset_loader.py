"""
LiveCodeBench v6 DatasetLoader 동작 확인

Usage:
python -m src.datasets.test_dataset_loader


최초 실행 시 데이터셋을 내려받으므로 시간이 조금 걸릴 수 있음. 이후에는 캐시를 사용하므로 빠르게 로드됨.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.datasets.dataset_loader import DatasetLoader
from src.schemas import ProblemExample


def compute_fingerprint(
    examples: list[ProblemExample],
) -> str:
    """로드된 문제의 핵심 필드로 재현성 확인용 fingerprint를 계산한다."""
    records = [
        {
            "problem_id": example.problem_id,
            "title": example.title,
            "prompt": example.prompt,
            "platform": example.platform,
            "contest_id": example.contest_id,
            "contest_date": example.contest_date,
            "difficulty": example.difficulty,
        }
        for example in examples
    ]

    serialized = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def validate_test_case(
    test_case: dict[str, Any],
    *,
    problem_id: str,
    test_group: str,
) -> None:
    """디코딩된 테스트 케이스의 기본 구조를 확인한다."""
    assert isinstance(test_case, dict), (
        f"{test_group} test must be a dict: "
        f"{problem_id}, got {type(test_case).__name__}"
    )

    assert "input" in test_case, (
        f"Missing input in {test_group} test: {problem_id}"
    )

    assert "output" in test_case, (
        f"Missing output in {test_group} test: {problem_id}"
    )

    assert isinstance(test_case["input"], str), (
        f"Test input must be str: {problem_id}"
    )

    assert isinstance(test_case["output"], str), (
        f"Test output must be str: {problem_id}"
    )


def validate_example(example: ProblemExample) -> None:
    """ProblemExample 하나의 필드와 타입을 확인한다."""
    assert isinstance(example, ProblemExample)

    assert example.problem_id
    assert example.title.strip()
    assert example.prompt.strip()
    assert example.platform.strip()
    assert example.contest_id
    assert example.contest_date

    assert example.difficulty in {
        "easy",
        "medium",
        "hard",
    }

    assert isinstance(example.starter_code, str)
    assert isinstance(example.public_tests, list)
    assert isinstance(example.private_tests, list)
    assert isinstance(example.metadata, dict)

    assert len(example.public_tests) > 0
    assert len(example.private_tests) > 0

    for test_case in example.public_tests:
        validate_test_case(
            test_case,
            problem_id=example.problem_id,
            test_group="public",
        )

    for test_case in example.private_tests:
        validate_test_case(
            test_case,
            problem_id=example.problem_id,
            test_group="private",
        )


def print_example(
    example: ProblemExample,
    index: int,
) -> None:
    """검사 결과를 사람이 확인할 수 있도록 출력한다."""
    print()
    print("=" * 80)
    print(f"Example {index}")
    print("=" * 80)

    print(f"problem_id    : {example.problem_id}")
    print(f"title         : {example.title}")
    print(f"platform      : {example.platform}")
    print(f"contest_id    : {example.contest_id}")
    print(f"contest_date  : {example.contest_date}")
    print(f"difficulty    : {example.difficulty}")
    print(f"starter_code  : {bool(example.starter_code.strip())}")
    print(f"public_tests  : {len(example.public_tests)}")
    print(f"private_tests : {len(example.private_tests)}")
    print(f"metadata      : {example.metadata}")

    print()
    print("-" * 80)
    print("Prompt preview")
    print("-" * 80)
    print(example.prompt[:500])

    print()
    print("-" * 80)
    print("First public test")
    print("-" * 80)
    print(
        json.dumps(
            example.public_tests[0],
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("-" * 80)
    print("First private test")
    print("-" * 80)
    print(
        json.dumps(
            example.private_tests[0],
            ensure_ascii=False,
            indent=2,
        )
    )


def test_limit() -> list[ProblemExample]:
    """limit=3이 정확히 적용되는지 확인한다."""
    loader = DatasetLoader(
        dataset_name="livecodebench_v6",
        split="test",
        limit=3,
    )

    examples = loader.load()

    assert len(examples) == 3, (
        f"Expected 3 examples, got {len(examples)}"
    )

    return examples


def test_unique_problem_ids(
    examples: list[ProblemExample],
) -> None:
    """로드된 문제 ID가 중복되지 않는지 확인한다."""
    problem_ids = [
        example.problem_id
        for example in examples
    ]

    assert len(problem_ids) == len(set(problem_ids)), (
        "Duplicated problem IDs detected."
    )


def test_deterministic_order() -> None:
    """동일한 설정으로 다시 로드했을 때 순서가 같은지 확인한다."""
    loader_a = DatasetLoader(
        dataset_name="livecodebench_v6",
        split="test",
        limit=3,
    )
    loader_b = DatasetLoader(
        dataset_name="livecodebench_v6",
        split="test",
        limit=3,
    )

    examples_a = loader_a.load()
    examples_b = loader_b.load()

    ids_a = [
        example.problem_id
        for example in examples_a
    ]
    ids_b = [
        example.problem_id
        for example in examples_b
    ]

    assert ids_a == ids_b, (
        "Problem order changed between repeated loads."
    )

    fingerprint_a = compute_fingerprint(examples_a)
    fingerprint_b = compute_fingerprint(examples_b)

    assert fingerprint_a == fingerprint_b, (
        "Dataset fingerprint changed between repeated loads."
    )


def test_invalid_dataset_name() -> None:
    """지원하지 않는 데이터셋 이름에 오류가 발생하는지 확인한다."""
    loader = DatasetLoader(
        dataset_name="unknown_dataset",
        split="test",
        limit=3,
    )

    try:
        loader.load()
    except ValueError as error:
        assert "Unsupported dataset" in str(error)
    else:
        raise AssertionError(
            "Unsupported dataset did not raise ValueError."
        )


def test_invalid_limit() -> None:
    """0 이하의 limit에 오류가 발생하는지 확인한다."""
    for invalid_limit in (0, -1):
        try:
            DatasetLoader(
                dataset_name="livecodebench_v6",
                limit=invalid_limit,
            )
        except ValueError as error:
            assert "limit must be greater than 0" in str(error)
        else:
            raise AssertionError(
                f"Invalid limit did not raise ValueError: "
                f"{invalid_limit}"
            )


def main() -> None:
    print("=" * 80)
    print("LiveCodeBench v6 DatasetLoader Test")
    print("=" * 80)

    examples = test_limit()
    print("[PASS] limit=3")

    test_unique_problem_ids(examples)
    print("[PASS] unique problem IDs")

    for index, example in enumerate(examples, start=1):
        validate_example(example)
        print_example(example, index)

    print()
    print("[PASS] ProblemExample fields")
    print("[PASS] public test decoding")
    print("[PASS] private test decoding")
    print("[PASS] metadata decoding")

    test_deterministic_order()
    print("[PASS] deterministic dataset order")

    test_invalid_dataset_name()
    print("[PASS] unsupported dataset validation")

    test_invalid_limit()
    print("[PASS] invalid limit validation")

    fingerprint = compute_fingerprint(examples)

    print()
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Number of examples : {len(examples)}")
    print(f"Problem IDs        : {[e.problem_id for e in examples]}")
    print(f"Fingerprint        : {fingerprint}")
    print()
    print("[PASS] All DatasetLoader tests passed.")


if __name__ == "__main__":
    main()