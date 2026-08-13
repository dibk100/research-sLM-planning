# phase1_planning_bottleneck/strategies/direct.py

from __future__ import annotations

from pathlib import Path

from src.models.generator import ModelGenerator
from src.schemas import (
    GenerationStep,
    ProblemExample,
    StrategyOutput,
)


class DirectStrategy:
    """Generate code directly from the problem without an explicit plan."""

    name = "direct"

    def __init__(
        self,
        generator: ModelGenerator,
        prompt_path: str | Path,
        *,
        system_prompt: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.generator = generator
        self.prompt_path = Path(prompt_path)

        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self._validate_config()

        self.prompt_template = self.prompt_path.read_text(
            encoding="utf-8"
        )

        self._validate_prompt_template()

    def _validate_config(self) -> None:
        if not self.prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: "
                f"{self.prompt_path}"
            )

        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than 0."
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

    def _validate_prompt_template(self) -> None:
        required_placeholders = {
            "{title}",
            "{problem}",
            "{starter_code_section}",
        }

        missing = [
            placeholder
            for placeholder in required_placeholders
            if placeholder not in self.prompt_template
        ]

        if missing:
            raise ValueError(
                "Missing prompt placeholders: "
                + ", ".join(missing)
            )

    def build_prompt(
        self,
        example: ProblemExample,
    ) -> str:
        starter_code_section = ""

        if example.starter_code.strip():
            starter_code_section = (
                "Starter Code:\n"
                f"{example.starter_code.strip()}"
            )

        return self.prompt_template.format(
            title=example.title,
            problem=example.problem,
            starter_code_section=starter_code_section,
        ).strip()

    def run(
        self,
        example: ProblemExample,
    ) -> StrategyOutput:
        formatted_prompt = self.build_prompt(
            example
        )

        generation = self.generator.generate(
            prompt=formatted_prompt,
            system_prompt=self.system_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        generation_step = GenerationStep(
            name="code_generation",
            formatted_prompt=formatted_prompt,
            raw_output=generation.text,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            generation_time=generation.generation_time,
        )

        return StrategyOutput(
            problem_id=example.problem_id,
            strategy=self.name,

            formatted_prompt=formatted_prompt,
            raw_output=generation.text,

            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            generation_time=generation.generation_time,

            strategy_trace=[
                generation_step
            ],
        )