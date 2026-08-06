"""Self-planning code generation strategy."""

from __future__ import annotations

from pathlib import Path

from src.models.generator import ModelGenerator
from src.schemas import (
    GenerationStep,
    ProblemExample,
    StrategyOutput,
)


class SelfPlanningStrategy:
    """모델이 계획을 생성한 뒤 그 계획을 사용해 코드를 생성한다."""

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

        if not self.plan_prompt_path.exists():
            raise FileNotFoundError(
                f"Plan prompt template not found: "
                f"{self.plan_prompt_path}"
            )

        if not self.code_prompt_path.exists():
            raise FileNotFoundError(
                f"Code prompt template not found: "
                f"{self.code_prompt_path}"
            )

        if plan_max_new_tokens <= 0:
            raise ValueError(
                "plan_max_new_tokens must be greater than 0."
            )

        if code_max_new_tokens <= 0:
            raise ValueError(
                "code_max_new_tokens must be greater than 0."
            )

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
            if placeholder not in self.plan_prompt_template
        ]

        if missing_plan:
            raise ValueError(
                "Missing plan prompt placeholders: "
                + ", ".join(missing_plan)
            )

        missing_code = [
            placeholder
            for placeholder in code_placeholders
            if placeholder not in self.code_prompt_template
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
            self._build_starter_code_section(example)
        )

        return self.plan_prompt_template.format(
            title=example.title,
            problem=example.prompt,
            starter_code_section=starter_code_section,
        ).strip()

    def build_code_prompt(
        self,
        example: ProblemExample,
        plan: str,
    ) -> str:
        if not plan.strip():
            raise ValueError(
                "Generated plan must not be empty."
            )

        starter_code_section = (
            self._build_starter_code_section(example)
        )

        return self.code_prompt_template.format(
            title=example.title,
            problem=example.prompt,
            plan=plan.strip(),
            starter_code_section=starter_code_section,
        ).strip()

    def run(
        self,
        example: ProblemExample,
    ) -> StrategyOutput:
        """계획 생성 후 계획 기반 코드 생성을 수행한다."""

        # Step 1: 계획 생성
        plan_formatted_prompt = self.build_plan_prompt(
            example
        )

        plan_generation = self.generator.generate(
            prompt=plan_formatted_prompt,
            system_prompt=self.system_prompt,
            max_new_tokens=self.plan_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        plan = plan_generation.text.strip()

        if not plan:
            raise ValueError(
                f"Empty plan generated: "
                f"{example.problem_id}"
            )

        # Step 2: 계획 기반 코드 생성
        code_formatted_prompt = self.build_code_prompt(
            example=example,
            plan=plan,
        )

        code_generation = self.generator.generate(
            prompt=code_formatted_prompt,
            system_prompt=self.system_prompt,
            max_new_tokens=self.code_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        strategy_trace = [
            GenerationStep(
                name="plan_generation",
                formatted_prompt=plan_formatted_prompt,
                raw_output=plan_generation.text,
                prompt_tokens=plan_generation.prompt_tokens,
                completion_tokens=(
                    plan_generation.completion_tokens
                ),
                generation_time=(
                    plan_generation.generation_time
                ),
            ),
            GenerationStep(
                name="code_generation",
                formatted_prompt=code_formatted_prompt,
                raw_output=code_generation.text,
                prompt_tokens=code_generation.prompt_tokens,
                completion_tokens=(
                    code_generation.completion_tokens
                ),
                generation_time=(
                    code_generation.generation_time
                ),
            ),
        ]

        return StrategyOutput(
            problem_id=example.problem_id,
            strategy=self.name,

            # 최종 코드 생성 단계
            formatted_prompt=code_formatted_prompt,
            raw_output=code_generation.text,

            # 계획 + 코드 생성 전체 비용
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

            strategy_trace=strategy_trace,
        )