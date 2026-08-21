"""
"""
# phase3_coverage_analysis/a_planning_coverage/
# strategies/planning_coverage.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.models.generator import ModelGenerator
from src.schemas import ProblemExample

from phase3_coverage_analysis.a_planning_coverage.candidate import (
    candidate_seed,
)


@dataclass(frozen=True)
class PlanningCandidateOutput:
    """
    One plan-sampling candidate before code evaluation.
    """

    sample_id: int
    sample_seed: int

    plan: str
    code_raw_output: str

    plan_prompt: str
    code_prompt: str

    plan_prompt_tokens: int
    plan_completion_tokens: int
    plan_generation_time: float

    code_prompt_tokens: int
    code_completion_tokens: int
    code_generation_time: float

    plan_empty: bool
    plan_in_code_prompt: bool


class PlanningCoverageStrategy:
    """
    Phase 3-A planning-space exploration.

    For each candidate:
    1. Sample one plan stochastically.
    2. Generate one code solution greedily from that plan.

    Only the plan generation is stochastic.
    """

    name = "planning_coverage"

    def __init__(
        self,
        generator: ModelGenerator,
        plan_prompt_path: str | Path,
        code_prompt_path: str | Path,
        *,
        base_seed: int,
        plan_max_new_tokens: int = 512,
        plan_temperature: float = 0.7,
        plan_top_p: float = 0.95,
        code_max_new_tokens: int = 1024,
        code_temperature: float = 0.0,
        code_top_p: float = 1.0,
        system_prompt: str | None = None,
    ) -> None:
        self.generator = generator

        self.plan_prompt_path = Path(
            plan_prompt_path
        )
        self.code_prompt_path = Path(
            code_prompt_path
        )

        self.base_seed = base_seed

        self.plan_max_new_tokens = (
            plan_max_new_tokens
        )
        self.plan_temperature = (
            plan_temperature
        )
        self.plan_top_p = (
            plan_top_p
        )

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

        self.plan_prompt_template = (
            self.plan_prompt_path.read_text(
                encoding="utf-8"
            )
        )

        self.code_prompt_template = (
            self.code_prompt_path.read_text(
                encoding="utf-8"
            )
        )

        self._validate_templates()

    def _validate_config(self) -> None:
        if not self.plan_prompt_path.exists():
            raise FileNotFoundError(
                "Plan prompt not found: "
                f"{self.plan_prompt_path}"
            )

        if not self.code_prompt_path.exists():
            raise FileNotFoundError(
                "Code prompt not found: "
                f"{self.code_prompt_path}"
            )

        if self.plan_max_new_tokens <= 0:
            raise ValueError(
                "plan_max_new_tokens must be "
                "greater than 0."
            )

        if self.code_max_new_tokens <= 0:
            raise ValueError(
                "code_max_new_tokens must be "
                "greater than 0."
            )

        # Phase 3-A core condition.
        if self.plan_temperature <= 0:
            raise ValueError(
                "Phase 3-A requires stochastic "
                "plan generation "
                "(plan_temperature > 0)."
            )

        if not 0 < self.plan_top_p <= 1:
            raise ValueError(
                "plan_top_p must be in (0, 1]."
            )

        # Phase 3-A control:
        # code generation must remain deterministic.
        if self.code_temperature != 0.0:
            raise ValueError(
                "Phase 3-A requires deterministic "
                "code generation "
                "(code_temperature = 0.0)."
            )

        if not 0 < self.code_top_p <= 1:
            raise ValueError(
                "code_top_p must be in (0, 1]."
            )

    def _validate_templates(self) -> None:
        plan_placeholders = {
            "{problem}",
        }

        missing_plan = [
            placeholder
            for placeholder in plan_placeholders
            if placeholder
            not in self.plan_prompt_template
        ]

        if missing_plan:
            raise ValueError(
                "Missing planning prompt placeholders: "
                + ", ".join(missing_plan)
            )

        code_placeholders = {
            "{problem}",
            "{plan}",
            "{starter_code_section}",
        }

        missing_code = [
            placeholder
            for placeholder in code_placeholders
            if placeholder
            not in self.code_prompt_template
        ]

        if missing_code:
            raise ValueError(
                "Missing code prompt placeholders: "
                + ", ".join(missing_code)
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

    def build_plan_prompt(
        self,
        example: ProblemExample,
    ) -> str:
        starter_code_section = (
            self._build_starter_code_section(
                example.starter_code
            )
        )

        return self.plan_prompt_template.format(
            problem=example.problem,
            starter_code_section=starter_code_section,
        ).strip()

    def build_code_prompt(
        self,
        *,
        example: ProblemExample,
        plan: str,
    ) -> str:
        plan = plan.strip()

        if not plan:
            raise ValueError(
                "Plan must not be empty."
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
        sample_id: int,
        plan_prompt: str | None = None,
    ) -> PlanningCandidateOutput:
        if sample_id < 0:
            raise ValueError(
                "sample_id must be >= 0."
            )

        if plan_prompt is None:
            plan_prompt = (
                self.build_plan_prompt(
                    example
                )
            )

        sample_seed = candidate_seed(
            base_seed=self.base_seed,
            problem_id=example.problem_id,
            sample_id=sample_id,
        )

        # ------------------------------------------------------
        # 1. Stochastic plan generation
        # ------------------------------------------------------

        self._set_generation_seed(
            sample_seed
        )

        plan_generation = (
            self.generator.generate(
                prompt=plan_prompt,
                system_prompt=self.system_prompt,
                max_new_tokens=(
                    self.plan_max_new_tokens
                ),
                temperature=(
                    self.plan_temperature
                ),
                top_p=self.plan_top_p,
            )
        )

        plan = (
            plan_generation.text.strip()
        )

        if not plan:
            return PlanningCandidateOutput(
                sample_id=sample_id,
                sample_seed=sample_seed,

                plan="",
                code_raw_output="",

                plan_prompt=plan_prompt,
                code_prompt="",

                plan_prompt_tokens=(
                    plan_generation.prompt_tokens
                ),
                plan_completion_tokens=(
                    plan_generation.completion_tokens
                ),
                plan_generation_time=(
                    plan_generation.generation_time
                ),

                code_prompt_tokens=0,
                code_completion_tokens=0,
                code_generation_time=0.0,

                plan_empty=True,
                plan_in_code_prompt=False,
            )

        # ------------------------------------------------------
        # 2. Greedy code generation
        # ------------------------------------------------------

        code_prompt = (
            self.build_code_prompt(
                example=example,
                plan=plan,
            )
        )

        plan_in_code_prompt = (
            plan in code_prompt
        )

        if not plan_in_code_prompt:
            raise RuntimeError(
                "Generated plan is missing "
                "from the code prompt."
            )

        # Not technically necessary for greedy generation,
        # but keeps candidate execution deterministic.
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

        return PlanningCandidateOutput(
            sample_id=sample_id,
            sample_seed=sample_seed,

            plan=plan,
            code_raw_output=(
                code_generation.text
            ),

            plan_prompt=plan_prompt,
            code_prompt=code_prompt,

            plan_prompt_tokens=(
                plan_generation.prompt_tokens
            ),
            plan_completion_tokens=(
                plan_generation.completion_tokens
            ),
            plan_generation_time=(
                plan_generation.generation_time
            ),

            code_prompt_tokens=(
                code_generation.prompt_tokens
            ),
            code_completion_tokens=(
                code_generation.completion_tokens
            ),
            code_generation_time=(
                code_generation.generation_time
            ),

            plan_empty=False,
            plan_in_code_prompt=(
                plan_in_code_prompt
            ),
        )