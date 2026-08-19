# src.plans.teacher_replan_store.py
"""Teacher replan storage and retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TeacherReplanRecord:
    """문제 하나의 Phase 1 failure에 대응하는 teacher replan."""

    problem_id: str
    teacher_replan: str
    teacher_model: str | None = None
    replan_version: str | None = None
    verified: bool | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "TeacherReplanRecord":
        problem_id = str(
            data.get("problem_id", "")
        ).strip()

        teacher_replan = str(
            data.get("teacher_replan", "")
        ).strip()

        if not problem_id:
            raise ValueError(
                "Teacher replan record has empty problem_id."
            )

        if not teacher_replan:
            raise ValueError(
                f"Teacher replan is empty: {problem_id}"
            )

        verified_value = data.get(
            "verified"
        )

        if (
            verified_value is not None
            and not isinstance(
                verified_value,
                bool,
            )
        ):
            raise TypeError(
                "verified must be bool or null: "
                f"{problem_id}"
            )

        teacher_model = data.get(
            "teacher_model"
        )

        if teacher_model is not None:
            teacher_model = str(
                teacher_model
            ).strip()

        replan_version = data.get(
            "replan_version"
        )

        if replan_version is not None:
            replan_version = str(
                replan_version
            ).strip()

        return cls(
            problem_id=problem_id,
            teacher_replan=teacher_replan,
            teacher_model=teacher_model,
            replan_version=replan_version,
            verified=verified_value,
        )


class TeacherReplanStore:
    """JSONL teacher replan 파일을 problem_id로 조회한다."""

    def __init__(
        self,
        replan_path: str | Path,
        *,
        require_verified: bool = False,
    ) -> None:
        self.replan_path = Path(
            replan_path
        )

        self.require_verified = (
            require_verified
        )

        if not self.replan_path.exists():
            raise FileNotFoundError(
                "Teacher replan file not found: "
                f"{self.replan_path}"
            )

        self._records = (
            self._load_records()
        )

    def _load_records(
        self,
    ) -> dict[
        str,
        TeacherReplanRecord,
    ]:
        records: dict[
            str,
            TeacherReplanRecord,
        ] = {}

        with self.replan_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                if not line.strip():
                    continue

                try:
                    raw_record = json.loads(
                        line
                    )

                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Invalid teacher replan JSON at "
                        f"line {line_number}: "
                        f"{self.replan_path}"
                    ) from error

                if not isinstance(
                    raw_record,
                    dict,
                ):
                    raise TypeError(
                        "Teacher replan record must be "
                        "an object at line "
                        f"{line_number}."
                    )

                record = (
                    TeacherReplanRecord.from_dict(
                        raw_record
                    )
                )

                if (
                    record.problem_id
                    in records
                ):
                    raise ValueError(
                        "Duplicated teacher replan: "
                        f"{record.problem_id}"
                    )

                if (
                    self.require_verified
                    and record.verified is not True
                ):
                    raise ValueError(
                        "Unverified teacher replan: "
                        f"{record.problem_id}"
                    )

                records[
                    record.problem_id
                ] = record

        if not records:
            raise ValueError(
                "No teacher replans loaded: "
                f"{self.replan_path}"
            )

        return records

    def get(
        self,
        problem_id: str,
    ) -> TeacherReplanRecord:
        try:
            return self._records[
                problem_id
            ]

        except KeyError as error:
            raise KeyError(
                "Teacher replan not found for "
                f"problem_id={problem_id}"
            ) from error

    def has(
        self,
        problem_id: str,
    ) -> bool:
        return (
            problem_id
            in self._records
        )

    def problem_ids(
        self,
    ) -> set[str]:
        return set(
            self._records
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self._records
        )