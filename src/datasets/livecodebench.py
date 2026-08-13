"""

phase0에서 데이터셋을 분석한 결과, 아래 9개 컬럼을 진단평가용으로 사용하기로 결정함.

| 역할    | 원본 컬럼                | 용도                                 |
| ----- | -------------------- | ---------------------------------- |
| 식별    | `question_id`        | 문제 고유 ID                           |
| 메타데이터 | `question_title`     | 문제명                                |
| 메타데이터 | `platform`           | AtCoder / LeetCode                 |
| 메타데이터 | `difficulty`         | Easy / Medium / Hard               |
| 메타데이터 | `contest_date`       | 출제 시점                              |
| 모델 입력 | `question_content`   | 모델에게 제공하는 문제                       |
| 모델 입력 | `starter_code`       | 함수 시그니처/초기 코드 제공, **functional 문제에만 활용** |
| 평가    | `public_test_cases`  | 공개 테스트                             |
| 평가    | `private_test_cases` | 비공개 테스트                            |

private_test_cases : 문자열로 압축/직렬화되어 저장있어서 evaluator가 테스트할 때 이 문자열을 실제 테스트케이스 목록으로 복원하는 작업이 필요함.

### Note.
- public_test_cases → JSON decode
- private_test_cases → JSON 또는 base64 + zlib + pickle decode

"""
# src/datasets/livecodebench.py

import base64
import json
import pickle
import zlib
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from src.schemas import ProblemExample


SUPPORTED_EVALUATION_TYPES = {
    "stdin",
    "functional",
}


def load_livecodebench(
    data_path: str | Path,
    evaluation_type: str,
) -> list[ProblemExample]:
    """
    Load the locally constructed LiveCodeBench-v6 diagnostic benchmark
    and normalize each row into ProblemExample.
    """

    if evaluation_type not in SUPPORTED_EVALUATION_TYPES:
        raise ValueError(
            f"Unsupported evaluation_type: {evaluation_type}. "
            f"Choose from {sorted(SUPPORTED_EVALUATION_TYPES)}."
        )

    dataset = load_from_disk(str(data_path))

    problems: list[ProblemExample] = []

    for row in dataset:
        problem_id = str(row["question_id"])

        public_tests = _decode_json_field(
            row["public_test_cases"],
            field_name="public_test_cases",
            problem_id=problem_id,
        )

        private_tests = _decode_private_tests(
            row["private_test_cases"],
            problem_id=problem_id,
        )

        inferred_evaluation_type = _infer_evaluation_type(
            public_tests=public_tests,
            private_tests=private_tests,
            problem_id=problem_id,
        )

        if inferred_evaluation_type != evaluation_type:
            raise ValueError(
                f"Evaluation type mismatch for {problem_id}: "
                f"expected={evaluation_type}, "
                f"inferred={inferred_evaluation_type}"
            )
        
        metadata = _decode_json_field(
            row["metadata"],
            field_name="metadata",
            problem_id=problem_id,
        )

        function_name = metadata.get("func_name")

        problem = ProblemExample(
            problem_id=problem_id,
            title=row["question_title"],
            problem=row["question_content"],
            starter_code=row["starter_code"] or "",

            dataset="livecodebench_v6",
            platform=row["platform"],

            difficulty=row["difficulty"],
            rating=None,
            contest_date=str(row["contest_date"]),

            evaluation_type=inferred_evaluation_type,

            public_tests=public_tests,
            private_tests=private_tests,

            time_limit=None,
            memory_limit=None,
            function_name= function_name,
        )

        _validate_problem(problem)

        problems.append(problem)

    _validate_unique_problem_ids(problems)

    return problems


def _infer_evaluation_type(
    *,
    public_tests: list[dict[str, Any]],
    private_tests: list[dict[str, Any]],
    problem_id: str,
) -> str:
    """
    Infer whether the problem uses stdin or functional evaluation.
    """

    all_tests = public_tests + private_tests

    if not all_tests:
        raise ValueError(
            f"No tests found: {problem_id}"
        )

    test_types = {
        test_case.get("testtype", "stdin")
        for test_case in all_tests
    }

    if len(test_types) != 1:
        raise ValueError(
            f"Mixed test types for {problem_id}: "
            f"{sorted(test_types)}"
        )

    evaluation_type = next(iter(test_types))

    if evaluation_type not in SUPPORTED_EVALUATION_TYPES:
        raise ValueError(
            f"Unknown evaluation type for "
            f"{problem_id}: {evaluation_type}"
        )

    return evaluation_type


def _decode_json_field(
    value: Any,
    field_name: str,
    problem_id: str,
) -> Any:
    """
    Decode a JSON-serialized dataset field.
    """

    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Failed to decode {field_name}: "
            f"{problem_id}"
        ) from error


def _decode_private_tests(
    value: Any,
    problem_id: str,
) -> list[dict[str, Any]]:
    """
    Decode LiveCodeBench private tests.

    The field can be either:
    1. JSON-serialized directly, or
    2. base64 -> zlib -> pickle encoded.
    """

    if not isinstance(value, str):
        return value

    # Case 1: directly serialized JSON
    try:
        decoded_json = json.loads(value)
        return decoded_json

    except json.JSONDecodeError:
        pass

    # Case 2: encoded private tests
    try:
        decoded = base64.b64decode(value)
        decompressed = zlib.decompress(decoded)
        unpickled = pickle.loads(decompressed)

    except Exception as error:
        raise ValueError(
            f"Failed to decode private tests: "
            f"{problem_id}"
        ) from error

    # Some versions contain JSON text after unpickling.
    if isinstance(unpickled, str):
        try:
            return json.loads(unpickled)

        except json.JSONDecodeError as error:
            raise ValueError(
                f"Failed to parse decoded "
                f"private tests: {problem_id}"
            ) from error

    if not isinstance(unpickled, list):
        raise TypeError(
            f"Unexpected private test type for "
            f"{problem_id}: "
            f"{type(unpickled).__name__}"
        )

    return unpickled


def _validate_problem(
    problem: ProblemExample,
) -> None:
    """
    Validate one normalized LiveCodeBench problem.
    """

    if not problem.problem_id:
        raise ValueError(
            "Empty problem_id detected."
        )

    if not problem.title.strip():
        raise ValueError(
            f"Empty title: {problem.problem_id}"
        )

    if not problem.problem.strip():
        raise ValueError(
            f"Empty problem: {problem.problem_id}"
        )

    if not problem.platform.strip():
        raise ValueError(
            f"Missing platform: {problem.problem_id}"
        )

    if problem.difficulty not in {
        "easy",
        "medium",
        "hard",
    }:
        raise ValueError(
            f"Unknown difficulty: "
            f"{problem.difficulty} "
            f"({problem.problem_id})"
        )

    if problem.evaluation_type not in (
        SUPPORTED_EVALUATION_TYPES
    ):
        raise ValueError(
            f"Unknown evaluation type: "
            f"{problem.evaluation_type} "
            f"({problem.problem_id})"
        )

    if (
        not problem.public_tests
        and not problem.private_tests
    ):
        raise ValueError(
            f"Missing all tests: "
            f"{problem.problem_id}"
        )

    if not isinstance(problem.starter_code, str):
        raise TypeError(
            f"starter_code must be str: "
            f"{problem.problem_id}"
        )


def _validate_unique_problem_ids(
    problems: list[ProblemExample],
) -> None:
    """
    Ensure that every diagnostic problem has a unique identifier.
    """

    seen: set[str] = set()

    for problem in problems:
        if problem.problem_id in seen:
            raise ValueError(
                f"Duplicated problem_id: "
                f"{problem.problem_id}"
            )

        seen.add(problem.problem_id)