"""Teacher re-plan storage and retrieval.

Phase 1의 TeacherPlanStore를 계승하되, 저장 대상이
"문제에 대한 최초 계획"이 아니라
"실패한 initial code에 대한 revised plan"이라는 점만 다르다.

JSONL 레코드 형식
-----------------
{
  "problem_id": "abc379_e",
  "teacher_replan": "- ...\\n- ...",
  "teacher_model": "claude-opus-5",
  "plan_version": "v1",
  "verified": true,
  "based_on": "direct_500_stdin"   # 어떤 initial trajectory를 보고 쓴 replan인지
}

`based_on` 은 어떤 Phase 1 실행 결과의 실패를 보고 작성한 replan인지 기록한다.
동일 문제라도 initial code가 달라지면 revised plan도 달라지므로,
Phase 1 실행을 다시 돌린 경우 replan을 재작성해야 한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TeacherReplanRecord:
    """문제 하나에 대응하는 teacher re-plan."""

    problem_id: str
    teacher_replan: str
    teacher_model: str | None = None
    plan_version: str | None = None
    verified: bool | None = None
    based_on: str | None = None

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
                "Teacher replan record has empty "
                "problem_id."
            )

        if not teacher_replan:
            raise ValueError(
                f"Teacher replan is empty: {problem_id}"
            )

        verified_value = data.get("verified")

        if (
            verified_value is not None
            and not isinstance(verified_value, bool)
        ):
            raise TypeError(
                "verified must be bool or null: "
                f"{problem_id}"
            )

        return cls(
            problem_id=problem_id,
            teacher_replan=teacher_replan,
            teacher_model=data.get("teacher_model"),
            plan_version=data.get("plan_version"),
            verified=verified_value,
            based_on=data.get("based_on"),
        )


class TeacherReplanStore:
    """JSONL teacher re-plan 파일을 problem_id로 조회한다."""

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
                "Teacher replan file not found: "
                f"{self.plan_path}"
            )

        self._records = self._load_records()

    def _load_records(
        self,
    ) -> dict[str, TeacherReplanRecord]:
        records: dict[str, TeacherReplanRecord] = {}

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
                        "Invalid teacher replan JSON at "
                        f"line {line_number}: "
                        f"{self.plan_path}"
                    ) from error

                if not isinstance(raw_record, dict):
                    raise TypeError(
                        "Teacher replan record must be "
                        f"an object at line {line_number}."
                    )

                record = TeacherReplanRecord.from_dict(
                    raw_record
                )

                if record.problem_id in records:
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

                records[record.problem_id] = record

        if not records:
            raise ValueError(
                f"No teacher replans loaded: "
                f"{self.plan_path}"
            )

        return records

    def get(
        self,
        problem_id: str,
    ) -> TeacherReplanRecord:
        try:
            return self._records[problem_id]
        except KeyError as error:
            raise KeyError(
                "Teacher replan not found for "
                f"problem_id={problem_id}"
            ) from error

    def has(
        self,
        problem_id: str,
    ) -> bool:
        return problem_id in self._records

    def problem_ids(self) -> set[str]:
        return set(self._records)
