"""
벤치마크 문제 로딩.

기존 순서 : 
전체 데이터 → 첫 10개 선택 → stdin 필터

수정 버전 :
전체 데이터 → stdin 필터 → 필터된 문제 중 첫 10개 선택
"""
import base64
import json
import pickle
import zlib
from typing import Any

from datasets import load_dataset

from src.common.schemas import ProblemExample


class DatasetLoader:
    SUPPORTED_TEST_TYPES = {
        "stdin",
        "functional",
    }

    def __init__(
        self,
        dataset_name: str = "livecodebench_v6",
        split: str = "test",
        limit: int | None = None,
        test_type: str | None = "stdin",
        release_version: str = "release_v6",
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.limit = limit
        self.test_type = test_type
        self.release_version = release_version

        if self.limit is not None and self.limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        if (
            self.test_type is not None
            and self.test_type not in self.SUPPORTED_TEST_TYPES
        ):
            raise ValueError(
                f"Unsupported test_type: {self.test_type}. "
                f"Choose from "
                f"{sorted(self.SUPPORTED_TEST_TYPES)} "
                f"or None."
            )

    def load(self) -> list[ProblemExample]:
        if self.dataset_name != "livecodebench_v6":
            raise ValueError(
                f"Unsupported dataset: {self.dataset_name}"
            )

        examples = self._load_livecodebench_v6()
        self._validate_examples(examples)

        return examples

    def _load_livecodebench_v6(
        self,
    ) -> list[ProblemExample]:
        dataset = load_dataset(
            "livecodebench/code_generation_lite",
            version_tag=self.release_version,
            split=self.split,
            trust_remote_code=True,
        )

        examples: list[ProblemExample] = []

        for row in dataset:
            problem_id = row["question_id"]

            public_tests = self._decode_json_field(
                row["public_test_cases"],
                field_name="public_test_cases",
                problem_id=problem_id,
            )

            private_tests = self._decode_private_tests(
                row["private_test_cases"],
                problem_id=problem_id,
            )

            problem_test_type = self._infer_test_type(
                public_tests=public_tests,
                private_tests=private_tests,
                problem_id=problem_id,
            )

            # test_type=None이면 모든 유형을 로드한다.
            if (
                self.test_type is not None
                and problem_test_type != self.test_type
            ):
                continue

            metadata = self._decode_json_field(
                row["metadata"],
                field_name="metadata",
                problem_id=problem_id,
            )

            example = ProblemExample(
                problem_id=problem_id,
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
                test_type=problem_test_type,
            )

            examples.append(example)

            # 필터링 이후 limit을 적용한다.
            if (
                self.limit is not None
                and len(examples) >= self.limit
            ):
                break

        return examples

    @classmethod
    def _infer_test_type(
        cls,
        *,
        public_tests: list[dict[str, Any]],
        private_tests: list[dict[str, Any]],
        problem_id: str,
    ) -> str:
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

        test_type = next(iter(test_types))

        if test_type not in cls.SUPPORTED_TEST_TYPES:
            raise ValueError(
                f"Unknown test type for "
                f"{problem_id}: {test_type}"
            )

        return test_type

    @staticmethod
    def _decode_json_field(
        value: Any,
        field_name: str,
        problem_id: str,
    ) -> Any:
        if not isinstance(value, str):
            return value

        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Failed to decode {field_name}: "
                f"{problem_id}"
            ) from error

    @staticmethod
    def _decode_private_tests(
        value: Any,
        problem_id: str,
    ) -> list[dict[str, Any]]:
        if not isinstance(value, str):
            return value

        try:
            decoded_json = json.loads(value)
            return decoded_json
        except json.JSONDecodeError:
            pass

        try:
            decoded = base64.b64decode(value)
            decompressed = zlib.decompress(decoded)
            unpickled = pickle.loads(decompressed)
        except Exception as error:
            raise ValueError(
                f"Failed to decode private tests: "
                f"{problem_id}"
            ) from error

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

    @staticmethod
    def _validate_examples(
        examples: list[ProblemExample],
    ) -> None:
        problem_ids: set[str] = set()

        for example in examples:
            if not example.problem_id:
                raise ValueError(
                    "Empty problem_id detected."
                )

            if example.problem_id in problem_ids:
                raise ValueError(
                    f"Duplicated problem_id: "
                    f"{example.problem_id}"
                )

            if not example.title.strip():
                raise ValueError(
                    f"Empty title: {example.problem_id}"
                )

            if not example.prompt.strip():
                raise ValueError(
                    f"Empty prompt: {example.problem_id}"
                )

            if not example.platform.strip():
                raise ValueError(
                    f"Missing platform: "
                    f"{example.problem_id}"
                )

            if example.difficulty not in {
                "easy",
                "medium",
                "hard",
            }:
                raise ValueError(
                    f"Unknown difficulty: "
                    f"{example.difficulty} "
                    f"({example.problem_id})"
                )

            if example.test_type not in {
                "stdin",
                "functional",
            }:
                raise ValueError(
                    f"Unknown test type: "
                    f"{example.test_type} "
                    f"({example.problem_id})"
                )

            if (
                not example.public_tests
                and not example.private_tests
            ):
                raise ValueError(
                    f"Missing all tests: "
                    f"{example.problem_id}"
                )

            problem_ids.add(example.problem_id)