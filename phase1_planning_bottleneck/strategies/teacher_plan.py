# phase1_planning_bottleneck/strategies/teacher_plan.py

from __future__ import annotations

from pathlib import Path

from src.models.generator import ModelGenerator
from src.plans.teacher_plan_store import TeacherPlanStore
from src.schemas import (
    GenerationStep,
    ProblemExample,
    StrategyOutput,
)


class TeacherPlanStrategy:
    """Generate code using an externally provided teacher plan."""

    name = "teacher_plan"

    def __init__(
        self,
        generator: ModelGenerator,
        plan_store: TeacherPlanStore,
        code_prompt_path: str | Path,
        *,
        system_prompt: str | None = None,
        code_max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.generator = generator
        self.plan_store = plan_store
        self.code_prompt_path = Path(
            code_prompt_path
        )

        self.system_prompt = system_prompt
        self.code_max_new_tokens = (
            code_max_new_tokens
        )
        self.temperature = temperature
        self.top_p = top_p

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
                "Teacher-plan code prompt not found: "
                f"{self.code_prompt_path}"
            )

        if self.code_max_new_tokens <= 0:
            raise ValueError(
                "code_max_new_tokens must be "
                "greater than 0."
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

    def _validate_template(self) -> None:
        required_placeholders = {
            "{title}",
            "{problem}",
            "{teacher_plan}",
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
                "Missing teacher-plan prompt "
                "placeholders: "
                + ", ".join(missing)
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

    def build_code_prompt(
        self,
        example: ProblemExample,
        teacher_plan: str,
    ) -> str:
        if not isinstance(teacher_plan, str):
            raise TypeError(
                "teacher_plan must be str, "
                f"got {type(teacher_plan).__name__}"
            )

        teacher_plan = teacher_plan.strip()

        if not teacher_plan:
            raise ValueError(
                "Teacher plan must not be empty."
            )

        starter_code_section = (
            self._build_starter_code_section(
                example
            )
        )

        return self.code_prompt_template.format(
            title=example.title,
            problem=example.problem,
            teacher_plan=teacher_plan,
            starter_code_section=(
                starter_code_section
            ),
        ).strip()

    def run(
        self,
        example: ProblemExample,
    ) -> StrategyOutput:
        # Step 1: retrieve external teacher plan
        plan_record = self.plan_store.get(
            example.problem_id
        )

        teacher_plan = (
            plan_record.teacher_plan.strip()
        )

        if not teacher_plan:
            raise ValueError(
                "Empty teacher plan: "
                f"{example.problem_id}"
            )

        # Step 2: generate code conditioned on teacher plan
        formatted_prompt = self.build_code_prompt(
            example=example,
            teacher_plan=teacher_plan,
        )

        generation = self.generator.generate(
            prompt=formatted_prompt,
            system_prompt=self.system_prompt,
            max_new_tokens=(
                self.code_max_new_tokens
            ),
            temperature=self.temperature,
            top_p=self.top_p,
        )

        code_step = GenerationStep(
            name="code_generation",
            formatted_prompt=formatted_prompt,
            raw_output=generation.text,
            prompt_tokens=(
                generation.prompt_tokens
            ),
            completion_tokens=(
                generation.completion_tokens
            ),
            generation_time=(
                generation.generation_time
            ),
        )

        return StrategyOutput(
            problem_id=example.problem_id,
            strategy=self.name,

            # Final code-generation stage
            formatted_prompt=formatted_prompt,
            raw_output=generation.text,

            # Only code generation is counted here.
            # Teacher-plan generation is external to this strategy.
            prompt_tokens=(
                generation.prompt_tokens
            ),
            completion_tokens=(
                generation.completion_tokens
            ),
            generation_time=(
                generation.generation_time
            ),

            strategy_trace=[
                code_step
            ],

            teacher_plan=teacher_plan,
            teacher_plan_source=(
                plan_record.teacher_model
            ),
            teacher_plan_version=(
                plan_record.plan_version
            ),
            teacher_plan_verified=(
                plan_record.verified
            ),
        )