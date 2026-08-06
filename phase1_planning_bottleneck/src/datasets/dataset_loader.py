"""
벤치마크 문제 로딩.

- 원본 데이터셋 로드
- 내부 ProblemExample 형식으로 변환
- subset 선택
- problem ID 기준 필터링
- 문제 순서 고정


초기에 확인할 것 :
examples = loader.load()

print(len(examples))
print(examples[0].problem_id)
print(examples[0].prompt)


Dataset Loader 완료 조건 :
- 문제 1개 이상 정상 로드
- problem_id가 고유함
- prompt와 test 정보가 누락되지 않음
- 같은 설정에서 항상 같은 문제 순서가 나옴

HumanEval+, MBPP+, APPS, CodeContests, Codeforces, LeetCode 등 다양한 문제를 로드할 수 있도록 구현 필요
우선 Livecodebench-v6 문제를 로드하는 DatasetLoader 구현
"""

import base64
import json
import pickle
import zlib
from typing import Any

from datasets import load_dataset

from src.schemas import ProblemExample


class DatasetLoader:
    def __init__(
        self,
        dataset_name: str = "livecodebench_v6",
        split: str = "test",
        limit: int | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.limit = limit

        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be greater than 0.")

    def load(self) -> list[ProblemExample]:
        if self.dataset_name != "livecodebench_v6":
            raise ValueError(
                f"Unsupported dataset: {self.dataset_name}"
            )

        examples = self._load_livecodebench_v6()

        if self.limit is not None:
            examples = examples[: self.limit]

        self._validate_examples(examples)

        return examples

    def _load_livecodebench_v6(self) -> list[ProblemExample]:
        dataset = load_dataset(
            "livecodebench/code_generation_lite",
            version_tag="release_v6",
            split=self.split,
            trust_remote_code=True,
        )

        examples: list[ProblemExample] = []

        for row in dataset:
            public_tests = self._decode_json_field(
                row["public_test_cases"]
            )

            private_tests = self._decode_private_tests(
                row["private_test_cases"]
            )

            metadata = self._decode_json_field(
                row["metadata"]
            )

            example = ProblemExample(
                problem_id=row["question_id"],
                title=row["question_title"],
                prompt=row["question_content"],
                platform=row["platform"],
                contest_id=row["contest_id"],
                contest_date=str(row["contest_date"]),
                difficulty=row["difficulty"],
                starter_code=row["starter_code"] or "",
                public_tests=public_tests,
                private_tests=private_tests,
                metadata=metadata,
            )

            examples.append(example)

        return examples

    @staticmethod
    def _decode_json_field(value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value)

        return value

    @staticmethod
    def _decode_private_tests(value: Any) -> list[dict]:
        if not isinstance(value, str):
            return value

        try:
            return json.loads(value)

        except json.JSONDecodeError:
            decoded = base64.b64decode(
                value.encode("utf-8")
            )
            decompressed = zlib.decompress(decoded)
            unpickled = pickle.loads(decompressed)

            if isinstance(unpickled, str):
                return json.loads(unpickled)

            return unpickled

    @staticmethod
    def _validate_examples(
        examples: list[ProblemExample],
    ) -> None:
        problem_ids: set[str] = set()

        for example in examples:
            if not example.problem_id:
                raise ValueError("Empty problem_id detected.")

            if example.problem_id in problem_ids:
                raise ValueError(
                    f"Duplicated problem_id: "
                    f"{example.problem_id}"
                )

            if not example.prompt.strip():
                raise ValueError(
                    f"Empty prompt: {example.problem_id}"
                )

            if example.platform not in {
                "leetcode",
                "codeforces",
                "atcoder",
            }:
                raise ValueError(
                    f"Unknown platform: {example.platform}"
                )

            if example.difficulty not in {
                "easy",
                "medium",
                "hard",
            }:
                raise ValueError(
                    f"Unknown difficulty: "
                    f"{example.difficulty}"
                )

            if not example.public_tests:
                raise ValueError(
                    f"Missing public tests: "
                    f"{example.problem_id}"
                )

            if not example.private_tests:
                raise ValueError(
                    f"Missing private tests: "
                    f"{example.problem_id}"
                )

            problem_ids.add(example.problem_id)