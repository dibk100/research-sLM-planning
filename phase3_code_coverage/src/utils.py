"""Phase 3-B 공통 유틸: candidate 레코드, coverage 지표.

Phase 3-A(src/utils.py)와 지표 정의를 동일하게 유지한다.
두 실험 결과를 같은 축에서 비교해야 하므로 계산식이 달라지면 안 된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.common.schemas import ProblemExample

_SEED_MODULUS = 2**31 - 1


def candidate_seed(base_seed: int, problem_id: str, candidate_index: int) -> int:
    """(문제, candidate index)에 대해 재현 가능한 시드를 만든다."""
    raise NotImplementedError


@dataclass
class CandidateRecord:
    """results.jsonl에 저장되는 candidate 단위 레코드."""

    problem_id: str
    candidate_index: int
    seed: int
    code_text: str
    passed: bool
    status: str
    num_tests: int
    num_passed: int
    test_pass_ratio: float
    generation_seconds: float = 0.0
    execution_seconds: float = 0.0
    prompt: str | None = None
    error_message: str | None = None
    test_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class ProblemRecord:
    """문제 하나 = 고정 plan 1개 + candidate N개."""

    problem_id: str
    difficulty: str | None
    plan_text: str
    plan_source: str
    num_samples: int
    candidates: list[CandidateRecord] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


def summarize_candidates(candidates: Sequence[CandidateRecord], num_samples: int) -> dict[str, Any]:
    """prefix별 oracle / best ratio 등 문제 단위 요약."""
    raise NotImplementedError


def prefix_ks(num_samples: int) -> list[int]:
    """1, 2, 4, ... num_samples 형태의 prefix 목록."""
    raise NotImplementedError


def oracle_at_k(passed_flags: Sequence[bool], k: int) -> float:
    """candidate prefix k개 중 하나라도 통과하면 1."""
    raise NotImplementedError


def best_ratio_at_k(ratios: Sequence[float], k: int) -> float:
    raise NotImplementedError


def unbiased_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Codex 방식 pass@k 불편 추정량."""
    raise NotImplementedError


def check_monotonicity(values: Sequence[float]) -> bool:
    """Oracle@k가 k에 대해 단조 증가하는지 확인 (sanity check)."""
    raise NotImplementedError


def load_problem_manifest(path: str | Path) -> list[dict[str, Any]]:
    raise NotImplementedError


def assert_examples_match_manifest(
    examples: Sequence[ProblemExample],
    manifest: Sequence[dict[str, Any]],
) -> None:
    """데이터셋 problem id/순서가 manifest와 동일한지 검증."""
    raise NotImplementedError


def format_duration(seconds: float) -> str:
    raise NotImplementedError


def mean(values: Iterable[float]) -> float:
    raise NotImplementedError
