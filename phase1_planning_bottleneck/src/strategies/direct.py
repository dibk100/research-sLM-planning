"""
Direct Code Generation 전략: 계획 없이 문제 -> 코드.

1. 문제를 prompt에 삽입
2. 모델 호출
3. generation 결과 반환

Strategy 안에서 평가나 파일 저장까지 수행하지 않음. 평가나 파일 저장은 외부에서 수행.
"""

from pathlib import Path

from src.models.generator import ModelGenerator
from src.schemas import GenerationOutput, ProblemExample


class DirectStrategy:
    name = "direct"

    def __init__(
        self,
        generator: ModelGenerator,
        prompt_path: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        self.generator = generator
        self.prompt_template = Path(prompt_path).read_text(
            encoding="utf-8"
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def build_prompt(self, example: ProblemExample) -> str:
        return self.prompt_template.format(
            problem=example.prompt
        )

    def run(
        self,
        example: ProblemExample,
    ) -> tuple[str, GenerationOutput]:
        prompt = self.build_prompt(example)

        output = self.generator.generate(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        return prompt, output