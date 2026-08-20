"""
Phase 3-B Code Coverage에서 필요한
candidate-level / problem-level schema를 정의한다.

Phase 3-A:
    sampled plan -> greedy code

Phase 3-B:
    fixed Phase-1 self-plan -> sampled code

따라서 Phase 3-B의 candidate는
plan-generation cost를 포함하지 않고,
code-generation 및 execution 결과만 기록한다.
"""

# phase3_coverage_analysis/b_code_coverage/candidate.py

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


_SEED_MODULUS = 2**31 - 1


def candidate_seed(
    *,
    base_seed: int,
    problem_id: str,
    sample_id: int,
) -> int:
    """
    Return a deterministic seed for one code candidate.

    The same
        (base_seed, problem_id, sample_id)
    always produces the same seed.
    """

    if sample_id < 0:
        raise ValueError(
            "sample_id must be non-negative."
        )

    payload = (
        f"{base_seed}|{problem_id}|{sample_id}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        payload
    ).digest()

    return (
        int.from_bytes(
            digest[:8],
            "big",
        )
        % _SEED_MODULUS
    )


@dataclass
class CandidateRecord:
    """
    One stochastic code candidate generated
    from the fixed plan.
    """

    sample_id: int
    sample_seed: int

    code: str

    passed: bool
    status: str
    passed_tests: int
    total_tests: int
    test_pass_ratio: float

    code_prompt_tokens: int = 0
    code_completion_tokens: int = 0
    code_generation_time: float = 0.0

    execution_time: float = 0.0

    plan_in_code_prompt: bool = False

    raw_output: str = ""
    error_message: str | None = None

    code_prompt: str | None = None

    test_results: list[
        dict[str, Any]
    ] = field(
        default_factory=list
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(
            self
        )


@dataclass
class ProblemRecord:
    """
    Aggregated Phase 3-B result for one problem.

    All candidates share the same fixed Phase-1
    self-generated plan.
    """

    problem_id: str
    dataset: str
    strategy: str
    model_name: str
    seed: int
    num_samples: int

    title: str
    platform: str | None
    contest_date: str | None
    difficulty: str | None

    problem: str

    # Phase 1 Self-Plan에서 가져온 고정 plan.
    fixed_plan: str

    candidates: list[
        dict[str, Any]
    ] = field(
        default_factory=list
    )

    any_passed: bool = False
    num_passed: int = 0
    best_test_pass_ratio: float = 0.0

    total_generation_time: float = 0.0
    total_completion_tokens: int = 0

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(
            self
        )


def summarize_candidates(
    candidates: Sequence[
        CandidateRecord
    ],
) -> dict[str, Any]:
    """
    Summarize N stochastic code candidates
    generated from the same fixed plan.
    """

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
            candidate.passed
            for candidate in candidates
        ),
        "num_passed": sum(
            candidate.passed
            for candidate in candidates
        ),
        "best_test_pass_ratio": max(
            candidate.test_pass_ratio
            for candidate in candidates
        ),
        "total_generation_time": sum(
            candidate.code_generation_time
            for candidate in candidates
        ),
        "total_completion_tokens": sum(
            candidate.code_completion_tokens
            for candidate in candidates
        ),
    }