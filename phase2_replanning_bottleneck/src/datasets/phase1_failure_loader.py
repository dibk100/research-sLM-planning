"""
Phase 1 실패 trajectory 로더.

Phase 2는 LiveCodeBench를 다시 읽고 initial code를 다시 생성하지 않는다.
Phase 1 Direct 실행 결과(results.jsonl)에서 passed == false 인 레코드만 골라
refinement 입력(FailureCase)으로 변환한다.

    /mnt/hdd/project_sLM_planning/output/direct_500_stdin/results.jsonl
        -> FailureCase (problem + initial code + execution feedback)

Phase 1 레코드는 test_results 에 모든 테스트의 input_text / expected_output /
actual_output 을 그대로 담고 있으므로(evaluator가 조기 종료하지 않음),
재평가에 필요한 테스트 케이스를 별도 다운로드 없이 복원할 수 있다.

주의
----
- Phase 1 레코드에는 public / private 구분이 남아 있지 않다.
  복원된 ProblemExample은 모든 테스트를 public_tests 에 넣고,
  Evaluator는 include_public_tests=true 로 실행하는 것을 전제로 한다.
- starter_code 는 남아 있지 않다. stdin 문제만 사용하므로 빈 문자열로 둔다.

TODO(구현)
----------
- [ ] _build_example : 레코드 -> ProblemExample 복원
- [ ] _build_feedback : test_results -> ExecutionFeedback 요약
- [ ] load : 필터링(passed/status/difficulty/limit) 후 FailureCase 목록 반환
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from src.schemas import (
    ExecutionFeedback,
    FailureCase,
    ProblemExample,
)

# refinement 대상에서 제외할 상태.
# 코드 추출 자체가 실패한 경우는 "실행 피드백"이 존재하지 않는다.
EXCLUDED_STATUSES = {
    "EXTRACTION_ERROR",
    "UNSUPPORTED_TEST_TYPE",
    "NO_TESTS",
}


class Phase1FailureLoader:
    """Phase 1 results.jsonl 에서 실패 케이스를 읽는다."""

    def __init__(
        self,
        results_path: str | Path,
        *,
        limit: int | None = None,
        difficulties: list[str] | None = None,
        max_feedback_chars: int = 2000,
        include_statuses: set[str] | None = None,
    ) -> None:
        self.results_path = Path(results_path)
        self.limit = limit
        self.difficulties = difficulties
        self.max_feedback_chars = max_feedback_chars
        self.include_statuses = include_statuses

        if not self.results_path.exists():
            raise FileNotFoundError(
                "Phase1 results file not found: "
                f"{self.results_path}"
            )

        if self.limit is not None and self.limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

    # -- public API ---------------------------------------------------------

    def load(self) -> list[FailureCase]:
        """실패 케이스를 FailureCase 목록으로 반환한다."""
        raise NotImplementedError(
            "TODO: iter_records -> 필터링 -> FailureCase 변환"
        )

    def iter_records(
        self,
    ) -> Iterator[dict[str, Any]]:
        """results.jsonl 을 한 줄씩 스트리밍한다.

        Phase 1 결과 파일은 500문항 기준 8~24GB 이므로
        전체를 메모리에 올리지 않는다.
        """
        with self.results_path.open(
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
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Invalid JSON at line "
                        f"{line_number}: "
                        f"{self.results_path}"
                    ) from error

                yield record

    # -- internal -----------------------------------------------------------

    def _is_failure(
        self,
        record: dict[str, Any],
    ) -> bool:
        """refinement 대상인지 판정한다."""
        if record.get("passed") is True:
            return False

        status = str(record.get("status", ""))

        if status in EXCLUDED_STATUSES:
            return False

        if (
            self.include_statuses is not None
            and status not in self.include_statuses
        ):
            return False

        if self.difficulties is not None:
            difficulty = record.get("difficulty")

            if difficulty not in self.difficulties:
                return False

        return True

    def _build_example(
        self,
        record: dict[str, Any],
    ) -> ProblemExample:
        """Phase 1 레코드에서 ProblemExample 을 복원한다."""
        raise NotImplementedError(
            "TODO: test_results -> public_tests 복원, "
            "메타데이터 매핑"
        )

    def _build_feedback(
        self,
        record: dict[str, Any],
    ) -> ExecutionFeedback:
        """실행 결과를 프롬프트용 피드백 텍스트로 요약한다.

        요약 규칙(초안)
        - 첫 번째 실패 테스트 1개만 사용한다
        - input / expected / actual 을 각각
          max_feedback_chars 로 잘라낸다
        - 런타임 에러는 stderr 마지막 줄을 함께 넣는다
        """
        raise NotImplementedError(
            "TODO: 첫 실패 테스트 추출 및 feedback_text 구성"
        )
