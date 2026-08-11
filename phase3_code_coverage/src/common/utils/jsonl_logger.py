"""
JSONL 기록/읽기 유틸.

실험이 중단되어도 이어서 실행할 수 있도록, 실험 결과를 JSONL 형식으로 안전하게 저장하고 읽는 기능을 제공한다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Iterable


class JSONLLogger:
    """실험 결과를 JSONL 형식으로 안전하게 저장한다."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        id_field: str = "problem_id",
    ) -> None:
        self.output_path = Path(output_path)
        self.id_field = id_field
        self._lock = Lock()

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def append(
        self,
        record: dict[str, Any],
    ) -> None:
        """레코드 하나를 JSONL 파일에 추가한다."""
        self._validate_record(record)

        serialized = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
        )

        with self._lock:
            with self.output_path.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(serialized)
                file.write("\n")
                file.flush()                            # 각 문제 처리 직후 결과를 실제 디스크 반영
                os.fsync(file.fileno())                 # note. 대규모 실험에서 fsync()는 약간의 I/O 비용을 추가하지만, 문제당 모델 생성 시간이 길기 때문에 현재 실험에서는 부담이 거의 없다고 함

    def append_many(
        self,
        records: Iterable[dict[str, Any]],
    ) -> None:
        """여러 레코드를 순차적으로 저장한다."""
        for record in records:
            self.append(record)

    def load_records(
        self,
        *,
        ignore_invalid_lines: bool = False,
    ) -> list[dict[str, Any]]:
        """저장된 모든 레코드를 읽는다."""
        if not self.output_path.exists():
            return []

        records: list[dict[str, Any]] = []

        with self.output_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                stripped = line.strip()

                if not stripped:
                    continue

                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as error:
                    if ignore_invalid_lines:
                        continue

                    raise ValueError(
                        f"Invalid JSONL line "
                        f"{line_number}: {self.output_path}"
                    ) from error

                if not isinstance(record, dict):
                    if ignore_invalid_lines:
                        continue

                    raise TypeError(
                        f"JSONL line {line_number} must "
                        f"contain an object."
                    )

                records.append(record)

        return records

    def completed_ids(self) -> set[str]:
        """기록된 레코드의 완료 ID 집합을 반환한다."""
        records = self.load_records()

        completed: set[str] = set()

        for index, record in enumerate(records):
            if self.id_field not in record:
                raise ValueError(
                    f"Missing id field '{self.id_field}' "
                    f"in record index {index}."
                )

            record_id = record[self.id_field]

            if not isinstance(record_id, str):
                raise TypeError(
                    f"{self.id_field} must be str, "
                    f"got {type(record_id).__name__}."
                )

            completed.add(record_id)

        return completed

    def contains(
        self,
        record_id: str,
    ) -> bool:
        """특정 ID가 이미 기록됐는지 확인한다."""
        return record_id in self.completed_ids()

    def count(self) -> int:
        """저장된 유효 레코드 수를 반환한다."""
        return len(self.load_records())

    def is_empty(self) -> bool:
        """파일이 없거나 유효 레코드가 없는지 확인한다."""
        return self.count() == 0

    def _validate_record(
        self,
        record: dict[str, Any],
    ) -> None:
        if not isinstance(record, dict):
            raise TypeError(
                "record must be a dict, "
                f"got {type(record).__name__}."
            )

        if self.id_field not in record:
            raise ValueError(
                f"Missing required ID field: "
                f"{self.id_field}"
            )

        record_id = record[self.id_field]

        if not isinstance(record_id, str):
            raise TypeError(
                f"{self.id_field} must be str, "
                f"got {type(record_id).__name__}."
            )

        if not record_id.strip():
            raise ValueError(
                f"{self.id_field} must not be empty."
            )