"""
Teacher Re-plan JSONL store.

Phase 2-3 Teacher-Replanning Regeneration에서 사용한다.

Teacher re-plan 파일을 메모리에 로드한 뒤 problem_id로 조회하고,
현재 Phase 1 FailureCase와 teacher re-plan이 동일한 failure trajectory를
기준으로 생성되었는지 검증한다.

Expected JSONL schema:

{
    "problem_id": "1873_A",
    "teacher_replan": "- ...\\n- ...",
    "teacher_model": "claude-opus-...",
    "replan_version": "v1",
    "verified": true,
    "initial_status": "WRONG_ANSWER",
    "initial_passed_tests": 3,
    "initial_total_tests": 5
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.schemas import FailureCase


@dataclass(frozen=True)
class TeacherReplanEntry:
    """Teacher re-plan 한 건."""

    problem_id: str
    teacher_replan: str

    teacher_model: str
    replan_version: str
    verified: bool

    initial_status: str
    initial_passed_tests: int
    initial_total_tests: int


class TeacherReplanStore:
    """
    Teacher re-plan JSONL을 problem_id 기반으로 조회한다.

    기본적으로 verified=True인 re-plan만 허용한다.
    """

    def __init__(
        self,
        replan_path: str | Path,
        *,
        require_verified: bool = True,
    ) -> None:
        self.replan_path = Path(
            replan_path
        )

        self.require_verified = (
            require_verified
        )

        self._validate_path()

        self._entries = (
            self._load_entries()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        problem_id: str,
    ) -> TeacherReplanEntry:
        """
        problem_id에 해당하는 teacher re-plan을 반환한다.
        """

        if problem_id not in self._entries:
            raise KeyError(
                "Teacher re-plan not found for "
                f"problem_id={problem_id}"
            )

        return self._entries[
            problem_id
        ]

    def get_for_failure(
        self,
        case: FailureCase,
    ) -> TeacherReplanEntry:
        """
        FailureCase와 일치하는 teacher re-plan을 반환한다.

        problem_id뿐 아니라 initial execution state도 검증한다.
        """

        entry = self.get(
            case.example.problem_id
        )

        self._validate_failure_match(
            case=case,
            entry=entry,
        )

        return entry

    def has(
        self,
        problem_id: str,
    ) -> bool:
        return problem_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_entries(
        self,
    ) -> dict[str, TeacherReplanEntry]:
        entries: dict[
            str,
            TeacherReplanEntry,
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
                    record = json.loads(
                        line
                    )

                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Invalid JSON at line "
                        f"{line_number}: "
                        f"{self.replan_path}"
                    ) from error

                entry = self._build_entry(
                    record=record,
                    line_number=line_number,
                )

                if (
                    entry.problem_id
                    in entries
                ):
                    raise ValueError(
                        "Duplicate teacher re-plan "
                        f"for problem_id="
                        f"{entry.problem_id}"
                    )

                entries[
                    entry.problem_id
                ] = entry

        if not entries:
            raise ValueError(
                "Teacher re-plan file contains "
                f"no valid entries: "
                f"{self.replan_path}"
            )

        return entries

    def _build_entry(
        self,
        *,
        record: dict[str, Any],
        line_number: int,
    ) -> TeacherReplanEntry:
        """
        JSON record를 TeacherReplanEntry로 변환한다.
        """

        required_fields = (
            "problem_id",
            "teacher_replan",
            "teacher_model",
            "replan_version",
            "verified",
            "initial_status",
            "initial_passed_tests",
            "initial_total_tests",
        )

        missing = [
            field
            for field in required_fields
            if field not in record
        ]

        if missing:
            raise ValueError(
                "Missing teacher re-plan fields "
                f"at line {line_number}: "
                f"{missing}"
            )

        problem_id = str(
            record["problem_id"]
        )

        teacher_replan = str(
            record["teacher_replan"]
        ).strip()

        if not teacher_replan:
            raise ValueError(
                "teacher_replan is empty "
                f"at line {line_number}"
            )

        verified = bool(
            record["verified"]
        )

        if (
            self.require_verified
            and not verified
        ):
            raise ValueError(
                "Unverified teacher re-plan "
                f"for problem_id={problem_id}"
            )

        self._validate_plan_format(
            teacher_replan=teacher_replan,
            problem_id=problem_id,
        )

        return TeacherReplanEntry(
            problem_id=problem_id,

            teacher_replan=(
                teacher_replan
            ),

            teacher_model=str(
                record["teacher_model"]
            ),

            replan_version=str(
                record["replan_version"]
            ),

            verified=verified,

            initial_status=str(
                record["initial_status"]
            ),

            initial_passed_tests=int(
                record[
                    "initial_passed_tests"
                ]
            ),

            initial_total_tests=int(
                record[
                    "initial_total_tests"
                ]
            ),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_path(
        self,
    ) -> None:
        if not self.replan_path.exists():
            raise FileNotFoundError(
                "Teacher re-plan file "
                f"not found: "
                f"{self.replan_path}"
            )

        if not self.replan_path.is_file():
            raise ValueError(
                "Teacher re-plan path "
                f"is not a file: "
                f"{self.replan_path}"
            )

    @staticmethod
    def _validate_plan_format(
        *,
        teacher_replan: str,
        problem_id: str,
    ) -> None:
        """
        Teacher/Self plan format 통제를 위한 최소 검증.

        - 모든 non-empty line은 "- "로 시작
        - 최대 6 bullet
        """

        lines = [
            line.strip()
            for line
            in teacher_replan.splitlines()
            if line.strip()
        ]

        if not lines:
            raise ValueError(
                "Teacher re-plan has no "
                f"bullet lines: {problem_id}"
            )

        if len(lines) > 6:
            raise ValueError(
                "Teacher re-plan exceeds "
                f"6 bullets: {problem_id}, "
                f"count={len(lines)}"
            )

        invalid = [
            line
            for line in lines
            if not line.startswith("- ")
        ]

        if invalid:
            raise ValueError(
                "Teacher re-plan contains "
                "non-bullet lines for "
                f"problem_id={problem_id}: "
                f"{invalid[:3]}"
            )

    @staticmethod
    def _validate_failure_match(
        *,
        case: FailureCase,
        entry: TeacherReplanEntry,
    ) -> None:
        """
        Teacher re-plan 생성 당시 failure state와
        현재 FailureCase가 같은지 확인한다.
        """

        mismatches: list[str] = []

        if (
            entry.initial_status
            != case.initial_status
        ):
            mismatches.append(
                "initial_status: "
                f"teacher={entry.initial_status}, "
                f"current={case.initial_status}"
            )

        if (
            entry.initial_passed_tests
            != case.initial_passed_tests
        ):
            mismatches.append(
                "initial_passed_tests: "
                f"teacher="
                f"{entry.initial_passed_tests}, "
                f"current="
                f"{case.initial_passed_tests}"
            )

        if (
            entry.initial_total_tests
            != case.initial_total_tests
        ):
            mismatches.append(
                "initial_total_tests: "
                f"teacher="
                f"{entry.initial_total_tests}, "
                f"current="
                f"{case.initial_total_tests}"
            )

        if mismatches:
            raise ValueError(
                "Teacher re-plan failure state "
                "does not match current "
                f"FailureCase for "
                f"problem_id="
                f"{case.example.problem_id}: "
                + "; ".join(mismatches)
            )