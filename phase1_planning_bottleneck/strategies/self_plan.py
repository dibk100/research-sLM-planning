# phase1_planning_bottleneck/strategies/self_plan.py

from __future__ import annotations

from pathlib import Path

from src.models.generator import ModelGenerator
from src.schemas import (
    GenerationStep,
    ProblemExample,
    StrategyOutput,
)


class SelfPlanningStrategy:
    """Generate a plan first, then generate code conditioned on that plan."""

    name = "self_plan"

    def __init__(
        self,
        generator: ModelGenerator,
        plan_prompt_path: str | Path,
        code_prompt_path: str | Path,
        *,
        system_prompt: str | None = None,
        plan_max_new_tokens: int = 512,
        code_max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.generator = generator

        self.plan_prompt_path = Path(plan_prompt_path)
        self.code_prompt_path = Path(code_prompt_path)

        self.system_prompt = system_prompt

        self.plan_max_new_tokens = plan_max_new_tokens
        self.code_max_new_tokens = code_max_new_tokens

        self.temperature = temperature
        self.top_p = top_p

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
                "Plan prompt template not found: "
                f"{self.plan_prompt_path}"
            )

        if not self.code_prompt_path.exists():
            raise FileNotFoundError(
                "Code prompt template not found: "
                f"{self.code_prompt_path}"
            )

        if self.plan_max_new_tokens <= 0:
            raise ValueError(
                "plan_max_new_tokens must be greater than 0."
            )

        if self.code_max_new_tokens <= 0:
            raise ValueError(
                "code_max_new_tokens must be greater than 0."
            )

        if self.temperature < 0:
            raise ValueError(
                "temperature must be greater than "
                "or equal to 0."
            )

        if not 0 < self.top_p <= 1:
            raise ValueError(
                "top_p must be in (0, 1]."
            )

    def _validate_templates(self) -> None:
        plan_placeholders = {
            "{title}",
            "{problem}",
            "{starter_code_section}",
        }

        code_placeholders = {
            "{title}",
            "{problem}",
            "{plan}",
            "{starter_code_section}",
        }

        missing_plan = [
            placeholder
            for placeholder in plan_placeholders
            if placeholder
            not in self.plan_prompt_template
        ]

        if missing_plan:
            raise ValueError(
                "Missing plan prompt placeholders: "
                + ", ".join(missing_plan)
            )

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
    def _build_starter_code_section(
        example: ProblemExample,
    ) -> str:
        if not example.starter_code.strip():
            return ""

        return (
            "Starter Code:\n"
            f"{example.starter_code.strip()}"
        )

    def build_plan_prompt(
        self,
        example: ProblemExample,
    ) -> str:
        starter_code_section = (
            self._build_starter_code_section(
                example
            )
        )

        return self.plan_prompt_template.format(
            title=example.title,
            problem=example.problem,
            starter_code_section=(
                starter_code_section
            ),
        ).strip()

    def build_code_prompt(
        self,
        example: ProblemExample,
        plan: str,
    ) -> str:
        if not isinstance(plan, str):
            raise TypeError(
                "plan must be str, "
                f"got {type(plan).__name__}"
            )

        plan = plan.strip()

        if not plan:
            raise ValueError(
                "Generated plan must not be empty."
            )

        starter_code_section = (
            self._build_starter_code_section(
                example
            )
        )

        return self.code_prompt_template.format(
            title=example.title,
            problem=example.problem,
            plan=plan,
            starter_code_section=(
                starter_code_section
            ),
        ).strip()

    def run(
        self,
        example: ProblemExample,
    ) -> StrategyOutput:
        # Step 1: plan generation
        plan_formatted_prompt = (
            self.build_plan_prompt(example)
        )

        plan_generation = (
            self.generator.generate(
                prompt=plan_formatted_prompt,
                system_prompt=self.system_prompt,
                max_new_tokens=(
                    self.plan_max_new_tokens
                ),
                temperature=self.temperature,
                top_p=self.top_p,
            )
        )

        plan = plan_generation.text.strip()

        if not plan:
            raise ValueError(
                "Empty plan generated: "
                f"{example.problem_id}"
            )

        # Step 2: code generation
        code_formatted_prompt = (
            self.build_code_prompt(
                example=example,
                plan=plan,
            )
        )

        code_generation = (
            self.generator.generate(
                prompt=code_formatted_prompt,
                system_prompt=self.system_prompt,
                max_new_tokens=(
                    self.code_max_new_tokens
                ),
                temperature=self.temperature,
                top_p=self.top_p,
            )
        )

        plan_step = GenerationStep(
            name="plan_generation",
            formatted_prompt=(
                plan_formatted_prompt
            ),
            raw_output=plan_generation.text,
            prompt_tokens=(
                plan_generation.prompt_tokens
            ),
            completion_tokens=(
                plan_generation.completion_tokens
            ),
            generation_time=(
                plan_generation.generation_time
            ),
        )

        code_step = GenerationStep(
            name="code_generation",
            formatted_prompt=(
                code_formatted_prompt
            ),
            raw_output=code_generation.text,
            prompt_tokens=(
                code_generation.prompt_tokens
            ),
            completion_tokens=(
                code_generation.completion_tokens
            ),
            generation_time=(
                code_generation.generation_time
            ),
        )

        return StrategyOutput(
            problem_id=example.problem_id,
            strategy=self.name,

            # Final code-generation stage
            formatted_prompt=(
                code_formatted_prompt
            ),
            raw_output=code_generation.text,

            # Total strategy cost
            prompt_tokens=(
                plan_generation.prompt_tokens
                + code_generation.prompt_tokens
            ),
            completion_tokens=(
                plan_generation.completion_tokens
                + code_generation.completion_tokens
            ),
            generation_time=(
                plan_generation.generation_time
                + code_generation.generation_time
            ),

            strategy_trace=[
                plan_step,
                code_step,
            ],
        )