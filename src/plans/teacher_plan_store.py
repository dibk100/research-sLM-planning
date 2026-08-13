"""Teacher plan storage and retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TeacherPlanRecord:
    """문제 하나에 대응하는 teacher plan."""

    problem_id: str
    teacher_plan: str
    teacher_model: str | None = None
    plan_version: str | None = None
    verified: bool | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "TeacherPlanRecord":
        problem_id = str(
            data.get("problem_id", "")
        ).strip()

        teacher_plan = str(
            data.get("teacher_plan", "")
        ).strip()

        if not problem_id:
            raise ValueError(
                "Teacher plan record has empty problem_id."
            )

        if not teacher_plan:
            raise ValueError(
                f"Teacher plan is empty: {problem_id}"
            )

        verified_value = data.get("verified")

        if (
            verified_value is not None
            and not isinstance(verified_value, bool)
        ):
            raise TypeError(
                f"verified must be bool or null: "
                f"{problem_id}"
            )

        return cls(
            problem_id=problem_id,
            teacher_plan=teacher_plan,
            teacher_model=data.get("teacher_model"),
            plan_version=data.get("plan_version"),
            verified=verified_value,
        )


class TeacherPlanStore:
    """JSONL teacher plan 파일을 problem_id로 조회한다."""

    def __init__(
        self,
        plan_path: str | Path,
        *,
        require_verified: bool = False,
    ) -> None:
        self.plan_path = Path(plan_path)
        self.require_verified = require_verified

        if not self.plan_path.exists():
            raise FileNotFoundError(
                f"Teacher plan file not found: "
                f"{self.plan_path}"
            )

        self._records = self._load_records()

    def _load_records(
        self,
    ) -> dict[str, TeacherPlanRecord]:
        records: dict[str, TeacherPlanRecord] = {}

        with self.plan_path.open(
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
                    raw_record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Invalid teacher plan JSON at "
                        f"line {line_number}: "
                        f"{self.plan_path}"
                    ) from error

                if not isinstance(raw_record, dict):
                    raise TypeError(
                        "Teacher plan record must be "
                        f"an object at line {line_number}."
                    )

                record = TeacherPlanRecord.from_dict(
                    raw_record
                )

                if record.problem_id in records:
                    raise ValueError(
                        "Duplicated teacher plan: "
                        f"{record.problem_id}"
                    )

                if (
                    self.require_verified
                    and record.verified is not True
                ):
                    raise ValueError(
                        "Unverified teacher plan: "
                        f"{record.problem_id}"
                    )

                records[record.problem_id] = record

        if not records:
            raise ValueError(
                f"No teacher plans loaded: "
                f"{self.plan_path}"
            )

        return records

    def get(
        self,
        problem_id: str,
    ) -> TeacherPlanRecord:
        try:
            return self._records[problem_id]
        except KeyError as error:
            raise KeyError(
                "Teacher plan not found for "
                f"problem_id={problem_id}"
            ) from error

    def has(
        self,
        problem_id: str,
    ) -> bool:
        return problem_id in self._records

    def problem_ids(self) -> set[str]:
        return set(self._records)