"""
Feedback-based Regeneration strategy.

Phase 1 Direct generation에서 실패한 trajectory를 입력으로 받아,
문제 + 이전 코드 + execution feedback을 조건으로 solution code를
한 번 다시 생성한다.

Explicit planning / re-planning은 수행하지 않는다.

Flow:
    FailureCase
        -> Prompt
        -> ModelGenerator.generate()
        -> RefinementOutput

주의:
    - evaluation은 수행하지 않는다.
    - code extraction은 수행하지 않는다.
    - generation까지만 담당한다.
"""

from __future__ import annotations

from pathlib import Path

from src.models.generator import ModelGenerator
from src.schemas import (
    FailureCase,
    GenerationOutput,
    GenerationStep,
    RefinementOutput,
)


STRATEGY_NAME = "feedback_regeneration"


class FeedbackRegenerationStrategy:
    """
    Execution feedback을 이용하여 전체 solution code를 재생성하는 baseline.
    """

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

        self._validate_generation_config()

        self.prompt_template = (
            self._load_prompt_template()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        case: FailureCase,
    ) -> RefinementOutput:
        """
        FailureCase 하나에 대해 feedback-based regeneration을 수행한다.
        """

        formatted_prompt = self.build_prompt(
            case
        )

        generation = self._generate(
            formatted_prompt
        )

        generation_step = GenerationStep(
            name=STRATEGY_NAME,
            formatted_prompt=formatted_prompt,
            raw_output=generation.text,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            generation_time=generation.generation_time,
        )

        return RefinementOutput(
            problem_id=case.example.problem_id,
            strategy=STRATEGY_NAME,

            formatted_prompt=formatted_prompt,
            raw_output=generation.text,

            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            generation_time=generation.generation_time,

            strategy_trace=[
                generation_step
            ],

            self_replan=None,

            teacher_replan=None,
            teacher_replan_source=None,
            teacher_replan_version=None,
            teacher_replan_verified=None,
        )

    def build_prompt(
        self,
        case: FailureCase,
    ) -> str:
        """
        FailureCase를 feedback regeneration prompt로 변환한다.
        """

        values = {
            "problem": case.example.prompt,
            "previous_code": case.initial_code,
            "execution_feedback": (
                case.feedback.feedback_text
            ),
        }

        try:
            return self.prompt_template.format(
                **values
            )

        except KeyError as error:
            missing_key = error.args[0]

            raise ValueError(
                "Missing prompt placeholder value: "
                f"{missing_key}"
            ) from error

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(
        self,
        prompt: str,
    ) -> GenerationOutput:
        """
        ModelGenerator를 호출한다.
        """

        return self.generator.generate(
            prompt,
            system_prompt=self.system_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_generation_config(
        self,
    ) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than 0."
            )

        if self.temperature < 0:
            raise ValueError(
                "temperature must be greater than or equal to 0."
            )

        if not 0 < self.top_p <= 1:
            raise ValueError(
                "top_p must be in the range (0, 1]."
            )

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _load_prompt_template(
        self,
    ) -> str:
        """
        prompt template 파일을 읽고 필수 placeholder를 검사한다.
        """

        if not self.prompt_path.exists():
            raise FileNotFoundError(
                "Feedback regeneration prompt not found: "
                f"{self.prompt_path}"
            )

        if not self.prompt_path.is_file():
            raise ValueError(
                "Prompt path is not a file: "
                f"{self.prompt_path}"
            )

        template = self.prompt_path.read_text(
            encoding="utf-8"
        ).strip()

        if not template:
            raise ValueError(
                "Feedback regeneration prompt is empty: "
                f"{self.prompt_path}"
            )

        required_placeholders = (
            "{problem}",
            "{previous_code}",
            "{execution_feedback}",
        )

        missing = [
            placeholder
            for placeholder in required_placeholders
            if placeholder not in template
        ]

        if missing:
            raise ValueError(
                "Feedback regeneration prompt is missing "
                f"required placeholders: {missing}"
            )

        return template