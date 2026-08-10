"""
Phase 3 전용 유틸리티.

주요 책임:
- candidate 단위 seed 유도 (재현 가능한 고정 sampling sequence)
- candidate / problem 레코드 스키마
- Oracle@k prefix 계산 및 monotonicity 검증
- 문제 ID manifest 검증 (Phase 1과 동일한 500문제 보장)
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.common.schemas import ProblemExample


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

_SEED_MODULUS = 2**31 - 1


def candidate_seed(
    *,
    base_seed: int,
    problem_id: str,
    sample_id: int,
) -> int:
    """(문제, sample_id)마다 재현 가능한 고유 seed를 만든다.

    Python 내장 hash()는 프로세스마다 salt가 달라 재현되지 않으므로
    sha256 기반 안정 해시를 사용한다.

    같은 base_seed / problem_id / sample_id 조합은 항상 같은 seed를 주므로
    - 중단 후 resume 해도 동일한 candidate가 재현되고
    - candidate 순서(0..N-1)가 고정된 sampling sequence가 된다.
    """
    if sample_id < 0:
        raise ValueError("sample_id must be non-negative.")

    payload = (
        f"{base_seed}|{problem_id}|{sample_id}"
    ).encode("utf-8")

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


# ---------------------------------------------------------------------------
# 레코드 스키마
# ---------------------------------------------------------------------------


@dataclass
class CandidateRecord:
    """한 문제에 대한 하나의 (plan → code → execute) 후보."""

    sample_id: int
    sample_seed: int

    plan: str
    code: str

    passed: bool
    status: str
    passed_tests: int
    total_tests: int
    test_pass_ratio: float

    # 생성 비용
    plan_prompt_tokens: int = 0
    plan_completion_tokens: int = 0
    plan_generation_time: float = 0.0
    code_prompt_tokens: int = 0
    code_completion_tokens: int = 0
    code_generation_time: float = 0.0
    execution_time: float = 0.0

    # 무결성 확인 플래그
    plan_empty: bool = False
    plan_in_code_prompt: bool = False

    raw_output: str = ""
    error_message: str | None = None

    # store_prompts / store_test_results가 켜졌을 때만 채워진다.
    code_prompt: str | None = None
    test_results: list[dict[str, Any]] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProblemRecord:
    """한 문제의 전체 best-of-N 결과 (JSONL 한 줄)."""

    problem_id: str
    dataset: str
    strategy: str
    model_name: str
    seed: int
    num_samples: int

    title: str
    platform: str
    contest_id: str
    contest_date: str
    difficulty: str

    problem: str
    plan_prompt: str

    candidates: list[dict[str, Any]] = field(
        default_factory=list
    )

    # 요약 지표 (분석 단계에서 재계산 가능하지만 조회 편의를 위해 저장한다.)
    any_passed: bool = False
    num_passed: int = 0
    best_test_pass_ratio: float = 0.0
    total_generation_time: float = 0.0
    total_completion_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_candidates(
    candidates: Sequence[CandidateRecord],
) -> dict[str, Any]:
    """problem-level 요약 지표를 계산한다."""
    if not candidates:
        return {
            "any_passed": False,
            "num_passed": 0,
            "best_test_pass_ratio": 0.0,
            "total_generation_time": 0.0,
            "total_completion_tokens": 0,
        }

    return {
        "any_passed": any(
            candidate.passed for candidate in candidates
        ),
        "num_passed": sum(
            1
            for candidate in candidates
            if candidate.passed
        ),
        "best_test_pass_ratio": max(
            candidate.test_pass_ratio
            for candidate in candidates
        ),
        "total_generation_time": sum(
            candidate.plan_generation_time
            + candidate.code_generation_time
            for candidate in candidates
        ),
        "total_completion_tokens": sum(
            candidate.plan_completion_tokens
            + candidate.code_completion_tokens
            for candidate in candidates
        ),
    }


# ---------------------------------------------------------------------------
# Oracle@k
# ---------------------------------------------------------------------------


def prefix_ks(num_samples: int) -> list[int]:
    """1, 2, 4, 8, ... 형태의 prefix 크기 목록을 만든다.

    num_samples가 2의 거듭제곱이 아니면 마지막에 num_samples를 추가한다.
    """
    if num_samples <= 0:
        raise ValueError(
            "num_samples must be greater than 0."
        )

    ks: list[int] = []
    k = 1

    while k <= num_samples:
        ks.append(k)
        k *= 2

    if ks[-1] != num_samples:
        ks.append(num_samples)

    return ks


def oracle_at_k(
    candidates: Sequence[dict[str, Any]],
    k: int,
) -> bool:
    """앞에서 k개 candidate 중 하나라도 통과했는지 판정한다.

    candidate 순서는 고정된 sampling sequence이므로,
    prefix가 곧 'N=k로 실험했을 때의 결과'가 된다.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    if k > len(candidates):
        raise ValueError(
            f"k={k} exceeds candidate count "
            f"{len(candidates)}."
        )

    return any(
        bool(candidate["passed"])
        for candidate in candidates[:k]
    )


def best_ratio_at_k(
    candidates: Sequence[dict[str, Any]],
    k: int,
) -> float:
    """앞에서 k개 candidate 중 최고 test_pass_ratio를 반환한다."""
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    if k > len(candidates):
        raise ValueError(
            f"k={k} exceeds candidate count "
            f"{len(candidates)}."
        )

    return max(
        float(candidate["test_pass_ratio"])
        for candidate in candidates[:k]
    )


def unbiased_pass_at_k(
    *,
    num_samples: int,
    num_correct: int,
    k: int,
) -> float:
    """Codex(Chen et al., 2021) 방식의 unbiased pass@k 추정값.

    prefix 기반 Oracle@k는 특정 sampling sequence 하나에 대한 관측값이라
    표본 잡음이 있다. 같은 n개 표본을 모두 사용하는 이 추정량은
    정의상 k에 대해 단조 증가하므로 보조 지표로 함께 보고한다.
    """
    if k <= 0 or k > num_samples:
        raise ValueError(
            f"k must satisfy 0 < k <= num_samples "
            f"(k={k}, num_samples={num_samples})."
        )

    if num_samples - num_correct < k:
        return 1.0

    return 1.0 - math.prod(
        1.0 - k / (num_samples - i)
        for i in range(num_correct)
    )


def check_monotonicity(
    values: Sequence[float],
) -> list[tuple[int, float, float]]:
    """단조 증가가 깨진 위치를 (index, prev, curr)로 반환한다.

    Oracle@1 <= Oracle@2 <= Oracle@4 <= Oracle@8은 prefix 정의상
    반드시 성립한다. 깨졌다면 분석 구현에 버그가 있는 것이다.
    """
    violations: list[tuple[int, float, float]] = []

    for index in range(1, len(values)):
        previous = values[index - 1]
        current = values[index]

        if current + 1e-12 < previous:
            violations.append(
                (index, previous, current)
            )

    return violations


# ---------------------------------------------------------------------------
# 문제 ID manifest
# ---------------------------------------------------------------------------


def load_problem_manifest(
    manifest_path: str | Path,
) -> list[dict[str, Any]]:
    """data/livecodebench_500.jsonl 형태의 manifest를 읽는다."""
    path = Path(manifest_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Problem manifest not found: {path}. "
            f"Run scripts/freeze_problem_ids.py first."
        )

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
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
                raise ValueError(
                    f"Invalid manifest line "
                    f"{line_number}: {path}"
                ) from error

            if "problem_id" not in record:
                raise ValueError(
                    f"Missing problem_id in manifest line "
                    f"{line_number}: {path}"
                )

            records.append(record)

    if not records:
        raise ValueError(
            f"Problem manifest is empty: {path}"
        )

    return records


def assert_examples_match_manifest(
    examples: Sequence[ProblemExample],
    manifest: Sequence[dict[str, Any]],
) -> None:
    """로드된 문제 집합이 manifest와 순서까지 동일한지 검증한다.

    Phase 1과 동일한 500문제 위에서 실험하는지 확인하는 안전장치다.
    """
    loaded_ids = [
        example.problem_id for example in examples
    ]
    expected_ids = [
        record["problem_id"] for record in manifest
    ]

    if loaded_ids == expected_ids:
        return

    loaded_set = set(loaded_ids)
    expected_set = set(expected_ids)

    missing = expected_set - loaded_set
    unexpected = loaded_set - expected_set

    details = [
        f"loaded={len(loaded_ids)}, "
        f"expected={len(expected_ids)}",
    ]

    if missing:
        details.append(
            f"missing={sorted(missing)[:10]}"
        )

    if unexpected:
        details.append(
            f"unexpected={sorted(unexpected)[:10]}"
        )

    if not missing and not unexpected:
        first_diff = next(
            index
            for index, (left, right) in enumerate(
                zip(loaded_ids, expected_ids)
            )
            if left != right
        )
        details.append(
            f"order differs at index {first_diff}: "
            f"loaded={loaded_ids[first_diff]}, "
            f"expected={expected_ids[first_diff]}"
        )

    raise ValueError(
        "Loaded problems do not match the frozen "
        "manifest. " + " | ".join(details)
    )


# ---------------------------------------------------------------------------
# 기타
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """초 단위를 사람이 읽기 쉬운 형태로 바꾼다."""
    seconds = max(0.0, float(seconds))

    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {secs}s"

    if minutes:
        return f"{minutes}m {secs}s"

    return f"{secs}s"


def mean(values: Iterable[float]) -> float:
    """빈 시퀀스에서 0.0을 반환하는 안전한 평균."""
    materialized = list(values)

    if not materialized:
        return 0.0

    return sum(materialized) / len(materialized)
