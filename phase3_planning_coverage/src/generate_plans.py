"""
Plan sampling (Phase 3-A의 핵심 변경 지점).

Phase 1:
    plan = generate_plan(problem)              # temperature=0.0, greedy

Phase 3-A:
    for sample_id in range(N):
        plan = generate_plan(problem, do_sample=True)

모델/프롬프트/max_new_tokens는 Phase 1과 동일하게 유지하고,
temperature/top_p만 sampling 설정으로 바꾼다.

candidate 순서는 고정된 sampling sequence여야 한다.
그래야 candidate[:k] prefix가 "N=k로 실험했을 때의 결과"로 해석되고,
Oracle@1 -> @2 -> @4 -> @8이 compute scaling curve가 된다.
이를 위해 (base_seed, problem_id, sample_id)에서 유도한 seed를
각 생성 직전에 torch RNG에 심는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.common.models.generator import ModelGenerator
from src.common.schemas import ProblemExample
from src.prompts import SelfPlanPromptBuilder
from src.utils import candidate_seed


@dataclass
class PlanSample:
    """하나의 sampled plan."""

    sample_id: int
    sample_seed: int

    plan: str
    plan_prompt: str

    prompt_tokens: int
    completion_tokens: int
    generation_time: float

    @property
    def is_empty(self) -> bool:
        return not self.plan.strip()


class PlanSampler:
    """동일 문제에 대해 서로 독립적인 plan을 N개 생성한다."""

    def __init__(
        self,
        generator: ModelGenerator,
        prompt_builder: SelfPlanPromptBuilder,
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
                "Plan sampling requires temperature > 0. "
                "temperature=0 은 8개 candidate가 모두 "
                "동일해지므로 best-of-N 실험이 성립하지 않는다."
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

    def sample_one(
        self,
        example: ProblemExample,
        sample_id: int,
        *,
        plan_prompt: str | None = None,
    ) -> PlanSample:
        """sample_id번째 plan 하나를 생성한다."""
        if plan_prompt is None:
            plan_prompt = (
                self.prompt_builder.build_plan_prompt(
                    example
                )
            )
            
        if sample_id < 0:
            raise ValueError(
                f"sample_id must be >= 0, got {sample_id}."
            )

        seed = candidate_seed(
            base_seed=self.base_seed,
            problem_id=example.problem_id,
            sample_id=sample_id,
        )

        # 각 candidate를 독립적으로 재현 가능하게 만든다.
        # (resume 후에도 같은 candidate가 같은 결과를 준다.)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        generation = self.generator.generate(
            prompt=plan_prompt,
            system_prompt=self.system_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        return PlanSample(
            sample_id=sample_id,
            sample_seed=seed,
            plan=generation.text.strip(),
            plan_prompt=plan_prompt,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            generation_time=generation.generation_time,
        )

    def sample(
        self,
        example: ProblemExample,
    ) -> list[PlanSample]:
        """sample_id = 0..N-1 순서로 plan을 생성한다."""
        plan_prompt = (
            self.prompt_builder.build_plan_prompt(example)
        )

        return [
            self.sample_one(
                example,
                sample_id,
                plan_prompt=plan_prompt,
            )
            for sample_id in range(self.num_samples)
        ]
