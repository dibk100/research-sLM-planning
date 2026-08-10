"""
Plan 기반 코드 생성.

Phase 3-A에서는 plan만 sampling하고 code 생성은 Phase 1과 동일하게
greedy(temperature=0.0)로 유지한다.
그래야 candidate 간 성능 차이를 'plan의 차이'로 귀속시킬 수 있다.

(code 쪽 sampling 효과는 Phase 3-B의 code best-of-N control 실험에서 본다.
 configs에서 code.temperature > 0으로 두면 그대로 재사용할 수 있다.)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.common.models.generator import ModelGenerator
from src.common.schemas import ProblemExample
from src.prompts import SelfPlanPromptBuilder


@dataclass
class CodeSample:
    """하나의 plan에서 생성된 코드 출력."""

    code_prompt: str
    raw_output: str

    prompt_tokens: int
    completion_tokens: int
    generation_time: float

    plan_in_code_prompt: bool


class PlanConditionedCodeGenerator:
    """주어진 plan을 코드 프롬프트에 넣어 코드를 생성한다."""

    def __init__(
        self,
        generator: ModelGenerator,
        prompt_builder: SelfPlanPromptBuilder,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        system_prompt: str | None = None,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than 0."
            )
            
        if temperature < 0:
            raise ValueError(
                "temperature must be >= 0."
            )

        if temperature > 0 and not 0.0 < top_p <= 1.0:
            raise ValueError(
                f"top_p must be in (0, 1], got {top_p}."
            )

        self.generator = generator
        self.prompt_builder = prompt_builder

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt

    def generate(
        self,
        example: ProblemExample,
        plan: str,
        *,
        seed: int | None = None,
    ) -> CodeSample:
        """plan을 조건으로 코드를 생성한다.

        For Phase 3-A (temperature=0), seed does not affect output.
        It is retained for reuse in stochastic code-sampling control experiments.
        
        """
        code_prompt = (
            self.prompt_builder.build_code_prompt(
                example=example,
                plan=plan,
            )
        )

        # sanity check: 이 candidate의 plan이 실제로 이 코드 프롬프트에
        # 들어갔는지 확인한다. (candidate 간 plan이 뒤섞이는 버그 방지)
        plan_in_code_prompt = (
            plan.strip() in code_prompt
        )

        if not plan_in_code_prompt:
            raise RuntimeError(
                "Generated plan is missing from the code prompt."
            )

        # greedy면 seed가 결과에 영향을 주지 않지만,
        # temperature > 0으로 바꿔 쓸 때를 위해 항상 심어둔다.
        if seed is not None:
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        generation = self.generator.generate(
            prompt=code_prompt,
            system_prompt=self.system_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        return CodeSample(
            code_prompt=code_prompt,
            raw_output=generation.text,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            generation_time=generation.generation_time,
            plan_in_code_prompt=plan_in_code_prompt,
        )
