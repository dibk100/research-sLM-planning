"""Teacher-plan code generation strategy."""

from __future__ import annotations

from pathlib import Path

from src.models.generator import ModelGenerator
from src.plans.teacher_plan_store import (
    TeacherPlanStore,
)
from src.schemas import (
    GenerationStep,
    ProblemExample,
    StrategyOutput,
)


class TeacherPlanStrategy:
    """외부 teacher plan을 이용해 코드를 생성한다."""

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

        self.code_prompt_template = (
            self.code_prompt_path.read_text(
                encoding="utf-8"
            )
        )

        self._validate_template()

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
        if not teacher_plan.strip():
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
            problem=example.prompt,
            teacher_plan=teacher_plan.strip(),
            starter_code_section=(
                starter_code_section
            ),
        ).strip()

    def run(
        self,
        example: ProblemExample,
    ) -> StrategyOutput:
        plan_record = self.plan_store.get(
            example.problem_id
        )

        formatted_prompt = self.build_code_prompt(
            example=example,
            teacher_plan=plan_record.teacher_plan,
        )

        generation = self.generator.generate(
            prompt=formatted_prompt,
            system_prompt=self.system_prompt,
            max_new_tokens=self.code_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        return StrategyOutput(
            problem_id=example.problem_id,
            strategy=self.name,
            formatted_prompt=formatted_prompt,
            raw_output=generation.text,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=(
                generation.completion_tokens
            ),
            generation_time=(
                generation.generation_time
            ),
            strategy_trace=[
                GenerationStep(
                    name="code_generation",
                    formatted_prompt=(
                        formatted_prompt
                    ),
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
            ],
            teacher_plan=plan_record.teacher_plan,
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