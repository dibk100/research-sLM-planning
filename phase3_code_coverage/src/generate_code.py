"""★ Code sampling (Phase 3-B 핵심).

하나의 고정 plan에서 code를 N개 독립 sampling한다.
plan은 모든 candidate에서 동일하므로, candidate 간 차이는 전부 code sampling에서 온다.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.common.models.generator import ModelGenerator
from src.common.schemas import ProblemExample
from src.load_fixed_plans import FixedPlan
from src.prompts import FixedPlanCodePromptBuilder


@dataclass
class CodeSample:
    """candidate 하나의 code 생성 결과."""

    candidate_index: int
    code_text: str
    raw_output: str
    prompt: str | None
    seed: int
    generation_seconds: float


class FixedPlanCodeSampler:
    """고정 plan 조건부로 code를 N개 생성한다."""

    def __init__(
        self,
        generator: ModelGenerator,
        prompt_builder: FixedPlanCodePromptBuilder,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        store_prompts: bool = False,
    ) -> None:
        raise NotImplementedError

    @torch.inference_mode()
    def sample(
        self,
        example: ProblemExample,
        plan: FixedPlan,
        num_samples: int,
        base_seed: int,
    ) -> list[CodeSample]:
        raise NotImplementedError
