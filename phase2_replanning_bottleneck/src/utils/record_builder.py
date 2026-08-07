"""
Refinement record construction utilities.

Phase 1의 build_experiment_record()를 계승하되,
initial state(Phase 1에서 물려받은 값)와 refined state(Phase 2 생성 결과)를
한 레코드에 함께 담는다. recovery / regression 분석을 결과 파일 하나로
끝내기 위해서다.

    FailureCase + RefinementOutput + EvaluationResult -> RefinementRecord

파생 필드
---------
recovered
= initial FAIL -> refined PASS 인가

test_pass_delta
= refined_passed_tests - initial_passed_tests
  (전부 통과하지 못했더라도 개선 정도를 보기 위한 값)

TODO(구현)
----------
- [ ] build_refinement_record 본문
"""

from __future__ import annotations

from dataclasses import asdict

from src.schemas import (
    EvaluationResult,
    FailureCase,
    RefinementOutput,
    RefinementRecord,
)


def build_refinement_record(
    *,
    case: FailureCase,
    refinement_output: RefinementOutput,
    refined_code: str,
    evaluation: EvaluationResult,
    dataset_name: str,
    model_name: str,
    seed: int,
) -> RefinementRecord:
    """파이프라인 출력을 저장 가능한 단일 레코드로 결합한다."""
    if (
        case.example.problem_id
        != refinement_output.problem_id
    ):
        raise ValueError(
            "Problem ID mismatch: "
            f"case={case.example.problem_id}, "
            "refinement_output="
            f"{refinement_output.problem_id}"
        )

    raise NotImplementedError(
        "TODO: RefinementRecord 필드 매핑 "
        "(recovered / test_pass_delta 계산 포함)"
    )
