"""
Direct code generation strategy

- 템플릿 로드
- 문제 필드 삽입
- Generator 호출
- 결과를 공통 스키마로 반환

Strategy 안에서 평가나 파일 저장까지 수행하지 않음. 
코드 추출, 실행 평가, JSONL 저장은 다른 파일에서 수행.
"""

from __future__ import annotations

from pathlib import Path

from src.models.generator import ModelGenerator
from src.schemas import (
    ProblemExample,
    StrategyOutput,
)


class DirectStrategy:
    """계획 없이 문제로부터 직접 코드를 생성한다."""

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

        if not self.prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: "
                f"{self.prompt_path}"
            )

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than 0."
            )

        self.prompt_template = self.prompt_path.read_text(
            encoding="utf-8"
        )

        self._validate_prompt_template()

    def _validate_prompt_template(self) -> None:
        """필수 placeholder가 있는지 확인한다."""
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
        """ProblemExample을 direct generation prompt로 변환한다."""
        starter_code_section = ""

        if example.starter_code.strip():
            starter_code_section = (
                "Starter Code:\n"
                f"{example.starter_code.strip()}"
            )

        return self.prompt_template.format(
            title=example.title,
            problem=example.prompt,
            starter_code_section=starter_code_section,
        ).strip()

    def run(
        self,
        example: ProblemExample,
    ) -> StrategyOutput:
        """문제 하나에 대해 direct generation을 수행한다."""
        formatted_prompt = self.build_prompt(example)

        generation = self.generator.generate(
            prompt=formatted_prompt,
            system_prompt=self.system_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        return StrategyOutput(
            problem_id=example.problem_id,
            strategy=self.name,
            formatted_prompt=formatted_prompt,
            raw_output=generation.text,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            generation_time=generation.generation_time,
        )