"""★ Phase 1 Self-Plan 로드.

Phase 3-B는 plan을 생성하지 않는다.
Phase 1 self_plan run의 results.jsonl에서 문제당 plan 1개를 읽어 고정한다.

    /mnt/hdd/project_sLM_planning/output/self_plan_500_stdin/results.jsonl

problem_id 집합과 순서가 현재 데이터셋과 일치하는지 반드시 검증한다.
(scripts/freeze_problem_ids.py가 만든 manifest와 대조)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from src.common.schemas import ProblemExample


@dataclass(frozen=True)
class FixedPlan:
    """문제 하나에 고정된 Phase 1 self-plan."""

    problem_id: str
    plan_text: str
    source_path: str
    source_record_index: int


class FixedPlanLoader:
    """Phase 1 results.jsonl -> {problem_id: FixedPlan}."""

    def __init__(
        self,
        source_path: str | Path,
        plan_field: str = "plan_text",
        on_missing: str = "error",
    ) -> None:
        raise NotImplementedError

    def load(self) -> dict[str, FixedPlan]:
        """전체 plan을 problem_id 기준으로 읽어들인다."""
        raise NotImplementedError

    def for_examples(self, examples: Sequence[ProblemExample]) -> list[FixedPlan]:
        """데이터셋 순서에 맞춰 plan을 정렬해 반환한다."""
        raise NotImplementedError


def assert_plans_cover_examples(
    plans: dict[str, FixedPlan],
    examples: Iterable[ProblemExample],
) -> None:
    """모든 문제에 plan이 존재하는지 검증한다. 없으면 예외."""
    raise NotImplementedError
