"""
Phase 3-B code sampling.

Phase 1 Self-Plan에서 생성된 plan을 문제별로 하나씩 고정하고,
동일한 plan으로부터 code만 N개 stochastic sampling한다.

Phase 3-A:
    stochastic plan × N
        -> greedy code

Phase 3-B:
    fixed Phase-1 plan
        -> stochastic code × N

각 code candidate는
(base_seed, problem_id, sample_id)에서 유도한 독립 seed를 사용한다.

이를 통해:
- candidate별 재현성 확보
- resume 후 동일 candidate 재생성 가능
- candidate[:k]를 Code Coverage@k로 해석 가능

Code sampling 설정은 Phase 3-A plan sampling과 맞춰
temperature=0.7, top_p=0.95를 기본 실험 설정으로 사용한다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from src.common.models.generator import ModelGenerator
from src.common.schemas import ProblemExample
from src.prompts import FixedPlanCodePromptBuilder


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def code_candidate_seed(
    *,
    base_seed: int,
    problem_id: str,
    sample_id: int,
) -> int:
    """
    Phase 3-B code candidate용 deterministic seed.

    Python built-in hash()는 프로세스마다 달라질 수 있으므로
    SHA-256 기반 stable hash를 사용한다.

    Phase 3-A의 plan candidate seed와 namespace를 분리하기 위해
    'code' prefix를 포함한다.
    """
    if sample_id < 0:
        raise ValueError(
            f"sample_id must be >= 0, got {sample_id}."
        )

    if not str(problem_id).strip():
        raise ValueError(
            "problem_id must not be empty."
        )

    payload = (
        f"code|{base_seed}|{problem_id}|{sample_id}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        payload
    ).digest()

    return (
        int.from_bytes(
            digest[:8],
            byteorder="big",
        )
        % (2**31)
    )


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


@dataclass
class CodeSample:
    """고정 plan에서 생성된 하나의 sampled code output."""

    sample_id: int
    sample_seed: int

    fixed_plan: str
    code_prompt: str
    raw_output: str

    prompt_tokens: int
    completion_tokens: int
    generation_time: float

    plan_in_code_prompt: bool

    @property
    def is_empty(self) -> bool:
        return not self.raw_output.strip()


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class FixedPlanCodeSampler:
    """
    동일 문제 + 동일 fixed plan에 대해
    서로 독립적인 code candidate를 N개 생성한다.
    """

    def __init__(
        self,
        generator: ModelGenerator,
        prompt_builder: FixedPlanCodePromptBuilder,
        *,
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        base_seed: int,
        system_prompt: str | None = None,
    ) -> None:
        if num_samples <= 0:
            raise ValueError(
                "num_samples must be greater than 0."
            )

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than 0."
            )

        if temperature <= 0:
            raise ValueError(
                "Phase 3-B code sampling requires "
                "temperature > 0."
            )

        if not 0.0 < top_p <= 1.0:
            raise ValueError(
                f"top_p must be in (0, 1], got {top_p}."
            )

        self.generator = generator
        self.prompt_builder = prompt_builder

        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.base_seed = base_seed
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Single candidate
    # ------------------------------------------------------------------

    def sample_one(
        self,
        example: ProblemExample,
        fixed_plan: str,
        sample_id: int,
        *,
        code_prompt: str | None = None,
    ) -> CodeSample:
        """
        sample_id번째 stochastic code candidate를 생성한다.
        """
        plan = fixed_plan.strip()

        if not plan:
            raise ValueError(
                "Fixed plan must not be empty."
            )

        if sample_id < 0:
            raise ValueError(
                f"sample_id must be >= 0, got {sample_id}."
            )

        if code_prompt is None:
            code_prompt = (
                self.prompt_builder.build_code_prompt(
                    example=example,
                    plan_text=plan,
                )
            )

        # --------------------------------------------------------------
        # Sanity check:
        # fixed plan이 실제 code prompt에 들어갔는지 확인
        # --------------------------------------------------------------

        plan_in_code_prompt = (
            plan in code_prompt
        )

        if not plan_in_code_prompt:
            raise RuntimeError(
                "Fixed plan is missing from "
                "the code prompt."
            )

        # --------------------------------------------------------------
        # Candidate-specific deterministic seed
        # --------------------------------------------------------------

        seed = code_candidate_seed(
            base_seed=self.base_seed,
            problem_id=example.problem_id,
            sample_id=sample_id,
        )

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # --------------------------------------------------------------
        # Stochastic code generation
        # --------------------------------------------------------------

        generation = self.generator.generate(
            prompt=code_prompt,
            system_prompt=self.system_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        return CodeSample(
            sample_id=sample_id,
            sample_seed=seed,
            fixed_plan=plan,
            code_prompt=code_prompt,
            raw_output=generation.text,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            generation_time=generation.generation_time,
            plan_in_code_prompt=plan_in_code_prompt,
        )

    # ------------------------------------------------------------------
    # N candidates
    # ------------------------------------------------------------------

    def sample(
        self,
        example: ProblemExample,
        fixed_plan: str,
    ) -> list[CodeSample]:
        """
        sample_id = 0..N-1 순서로 code candidate를 생성한다.

        code prompt는 모든 candidate에서 동일하므로 한 번만 구성한다.
        """
        plan = fixed_plan.strip()

        if not plan:
            raise ValueError(
                "Fixed plan must not be empty."
            )

        code_prompt = (
            self.prompt_builder.build_code_prompt(
                example=example,
                plan_text=plan,
            )
        )

        return [
            self.sample_one(
                example=example,
                fixed_plan=plan,
                sample_id=sample_id,
                code_prompt=code_prompt,
            )
            for sample_id in range(
                self.num_samples
            )
        ]