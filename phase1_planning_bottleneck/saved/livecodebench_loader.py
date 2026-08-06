from __future__ import annotations

import base64
import json
import pickle
import zlib
from typing import Any

from datasets import load_dataset

from src.schemas import ProblemExample


class LiveCodeBenchLoader:
    def __init__(
        self,
        release_version: str = "release_v6",
        split: str = "test",
        limit: int | None = None,
    ) -> None:
        self.release_version = release_version
        self.split = split
        self.limit = limit

        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be greater than 0.")

    def load(self) -> list[ProblemExample]:
        dataset = load_dataset(
            "livecodebench/code_generation_lite",
            version_tag=self.release_version,
            split=self.split,
            trust_remote_code=True,
        )

        examples = [
            self._convert_row(dict(row))
            for row in dataset
        ]

        if self.limit is not None:
            examples = examples[: self.limit]

        self._validate(examples)

        return examples

    def _convert_row(
        self,
        row: dict[str, Any],
    ) -> ProblemExample:
        return ProblemExample(
            problem_id=row["question_id"],
            title=row["question_title"],
            prompt=row["question_content"],
            platform=row["platform"],
            contest_id=row["contest_id"],
            contest_date=row["contest_date"],
            difficulty=row["difficulty"],
            starter_code=row["starter_code"],
            public_tests=self._decode_json(
                row["public_test_cases"]
            ),
            private_tests=self._decode_private_tests(
                row["private_test_cases"]
            ),
            metadata=self._decode_json(
                row["metadata"]
            ),
        )

    @staticmethod
    def _decode_json(value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value)

        return value

    @staticmethod
    def _decode_private_tests(
        value: Any,
    ) -> list[dict[str, Any]]:
        if not isinstance(value, str):
            return value

        try:
            return json.loads(value)

        except json.JSONDecodeError:
            decoded = base64.b64decode(value)
            decompressed = zlib.decompress(decoded)
            unpickled = pickle.loads(decompressed)

            if isinstance(unpickled, str):
                return json.loads(unpickled)

            return unpickled

    @staticmethod
    def _validate(
        examples: list[ProblemExample],
    ) -> None:
        seen_ids: set[str] = set()

        for example in examples:
            if not example.problem_id:
                raise ValueError("Empty problem_id detected.")

            if example.problem_id in seen_ids:
                raise ValueError(
                    f"Duplicated problem_id: "
                    f"{example.problem_id}"
                )

            if not example.prompt.strip():
                raise ValueError(
                    f"Empty prompt: {example.problem_id}"
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

            seen_ids.add(example.problem_id)