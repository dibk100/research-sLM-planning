"""Feedback-only repair 전략 (baseline).

    Problem + Initial Code + Execution Feedback -> Code

계획을 다시 세우지 않고, 실패한 코드를 직접 수정하게 한다.
Phase 2의 baseline이며, 다른 두 전략과의 차이가
"re-planning의 기여분"을 의미하게 된다.

코드 생성 호출은 1회이므로 strategy_trace 는 단계 1개를 가진다.

TODO(구현)
----------
- [ ] build_prompt : 템플릿 채우기
- [ ] run : generator 호출 -> RefinementOutput 구성
"""

from __future__ import annotations

from pathlib import Path

from src.models.generator import ModelGenerator
from src.schemas import (
    FailureCase,
    GenerationStep,
    RefinementOutput,
)


class FeedbackRepairStrategy:
    """실행 피드백만 보고 코드를 수정한다."""

    name = "feedback_repair"

    # prompts/feedback_repair.txt 가 반드시 포함해야 하는 placeholder
    REQUIRED_PLACEHOLDERS = {
        "{title}",
        "{problem}",
        "{initial_code}",
        "{execution_feedback}",
    }

    def __init__(
        self,
        generator: ModelGenerator,
        prompt_path: str | Path,
        *,
        system_prompt: str | None = None,
        code_max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.generator = generator
        self.prompt_path = Path(prompt_path)

        self.system_prompt = system_prompt
        self.code_max_new_tokens = (
            code_max_new_tokens
        )
        self.temperature = temperature
        self.top_p = top_p

        if not self.prompt_path.exists():
            raise FileNotFoundError(
                "Feedback-repair prompt not found: "
                f"{self.prompt_path}"
            )

        if self.code_max_new_tokens <= 0:
            raise ValueError(
                "code_max_new_tokens must be "
                "greater than 0."
            )

        self.prompt_template = (
            self.prompt_path.read_text(
                encoding="utf-8"
            )
        )

        self._validate_template()

    def _validate_template(self) -> None:
        missing = [
            placeholder
            for placeholder in (
                self.REQUIRED_PLACEHOLDERS
            )
            if placeholder
            not in self.prompt_template
        ]

        if missing:
            raise ValueError(
                "Missing feedback-repair prompt "
                "placeholders: " + ", ".join(missing)
            )

    def build_prompt(
        self,
        case: FailureCase,
    ) -> str:
        raise NotImplementedError(
            "TODO: prompt_template.format(...)"
        )

    def run(
        self,
        case: FailureCase,
    ) -> RefinementOutput:
        raise NotImplementedError(
            "TODO: build_prompt -> generator.generate "
            "-> RefinementOutput"
        )
