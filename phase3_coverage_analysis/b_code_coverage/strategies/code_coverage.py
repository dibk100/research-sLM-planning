"""
"""
# phase3_coverage_analysis/b_code_coverage/
# strategies/code_coverage.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.models.generator import ModelGenerator
from src.schemas import ProblemExample

from phase3_coverage_analysis.b_code_coverage.candidate import (
    candidate_seed,
)


@dataclass(frozen=True)
class CodeCoverageCandidateOutput:
    """
    One stochastic code-sampling candidate
    conditioned on a fixed Phase 1 self-plan.
    """

    sample_id: int
    sample_seed: int

    fixed_plan: str

    code_prompt: str
    code_raw_output: str

    code_prompt_tokens: int
    code_completion_tokens: int
    code_generation_time: float

    code_empty: bool
    plan_in_code_prompt: bool


class CodeCoverageStrategy:
    """
    Phase 3-B code-space exploration.

    For each problem:
    - use one fixed Phase 1 self-generated plan
    - sample N code solutions stochastically
      from the same plan-conditioned code prompt

    Unlike Phase 3-A, no plan generation occurs here.
    """

    name = "code_coverage"

    def __init__(
        self,
        generator: ModelGenerator,
        code_prompt_path: str | Path,
        *,
        base_seed: int,
        code_max_new_tokens: int = 1024,
        code_temperature: float = 0.7,
        code_top_p: float = 0.95,
        system_prompt: str | None = None,
    ) -> None:
        self.generator = generator

        self.code_prompt_path = Path(
            code_prompt_path
        )

        self.base_seed = base_seed

        self.code_max_new_tokens = (
            code_max_new_tokens
        )
        self.code_temperature = (
            code_temperature
        )
        self.code_top_p = (
            code_top_p
        )

        self.system_prompt = system_prompt

        self._validate_config()

        self.code_prompt_template = (
            self.code_prompt_path.read_text(
                encoding="utf-8"
            )
        )

        self._validate_template()

    def _validate_config(self) -> None:
        if not self.code_prompt_path.exists():
            raise FileNotFoundError(
                "Code prompt not found: "
                f"{self.code_prompt_path}"
            )

        if self.code_max_new_tokens <= 0:
            raise ValueError(
                "code_max_new_tokens must be "
                "greater than 0."
            )

        if self.code_temperature <= 0.0:
            raise ValueError(
                "Phase 3-B requires stochastic "
                "code generation "
                "(code_temperature > 0)."
            )

        if not 0 < self.code_top_p <= 1:
            raise ValueError(
                "code_top_p must be in (0, 1]."
            )

    def _validate_template(self) -> None:
        required_placeholders = {
            "{problem}",
            "{plan}",
            "{starter_code_section}",
        }

        missing = [
            placeholder
            for placeholder in required_placeholders
            if placeholder
            not in self.code_prompt_template
        ]

        if missing:
            raise ValueError(
                "Missing code prompt placeholders: "
                + ", ".join(missing)
            )

    @staticmethod
    def _set_generation_seed(
        seed: int,
    ) -> None:
        torch.manual_seed(
            seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )

    @staticmethod
    def _build_starter_code_section(
        starter_code: str,
    ) -> str:
        if not starter_code.strip():
            return ""

        return (
            "Starter Code:\n"
            f"{starter_code.strip()}"
        )

    def build_code_prompt(
        self,
        *,
        example: ProblemExample,
        fixed_plan: str,
    ) -> str:
        plan = fixed_plan.strip()

        if not plan:
            raise ValueError(
                "Fixed plan must not be empty."
            )

        starter_code_section = (
            self._build_starter_code_section(
                example.starter_code
            )
        )

        return self.code_prompt_template.format(
            problem=example.problem,
            plan=plan,
            starter_code_section=starter_code_section,
        ).strip()

    def run_candidate(
        self,
        *,
        example: ProblemExample,
        fixed_plan: str,
        sample_id: int,
        code_prompt: str | None = None,
    ) -> CodeCoverageCandidateOutput:
        if sample_id < 0:
            raise ValueError(
                "sample_id must be >= 0."
            )

        plan = fixed_plan.strip()

        if not plan:
            raise ValueError(
                "Fixed plan must not be empty."
            )

        if code_prompt is None:
            code_prompt = (
                self.build_code_prompt(
                    example=example,
                    fixed_plan=plan,
                )
            )

        plan_in_code_prompt = (
            plan in code_prompt
        )

        if not plan_in_code_prompt:
            raise RuntimeError(
                "Fixed plan is missing "
                "from the code prompt."
            )

        sample_seed = candidate_seed(
            base_seed=self.base_seed,
            problem_id=example.problem_id,
            sample_id=sample_id,
        )

        self._set_generation_seed(
            sample_seed
        )

        code_generation = (
            self.generator.generate(
                prompt=code_prompt,
                system_prompt=self.system_prompt,
                max_new_tokens=(
                    self.code_max_new_tokens
                ),
                temperature=(
                    self.code_temperature
                ),
                top_p=self.code_top_p,
            )
        )

        raw_output = (
            code_generation.text
        )

        return CodeCoverageCandidateOutput(
            sample_id=sample_id,
            sample_seed=sample_seed,

            fixed_plan=plan,

            code_prompt=code_prompt,
            code_raw_output=raw_output,

            code_prompt_tokens=(
                code_generation.prompt_tokens
            ),
            code_completion_tokens=(
                code_generation.completion_tokens
            ),
            code_generation_time=(
                code_generation.generation_time
            ),

            code_empty=(
                not raw_output.strip()
            ),
            plan_in_code_prompt=(
                plan_in_code_prompt
            ),
        )