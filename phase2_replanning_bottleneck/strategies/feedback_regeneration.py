# phase2_replanning_bottleneck/strategies/feedback_regeneration.py

from __future__ import annotations

from pathlib import Path

from src.datasets.phase1_failure_loader import (
    Phase1FailureRecord,
)
from src.models.generator import ModelGenerator
from src.schemas import (
    GenerationStep,
    StrategyOutput,
)

from src.utils.feedback import (
    truncate_input_text,
)

class FeedbackRegenerationStrategy:
    """
    Regenerate code directly from execution feedback.

    Input:
    - original problem
    - failed code
    - first failing test feedback

    Output:
    - regenerated code
    """

    name = "feedback_regeneration"

    def __init__(
        self,
        generator: ModelGenerator,
        prompt_path: str | Path,
        *,
        system_prompt: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_input_tokens: int | None = None,
    ) -> None:
        self.generator = generator

        self.prompt_path = Path(
            prompt_path
        )

        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_input_tokens = max_input_tokens

        self._validate_config()
        
        if (
            self.max_input_tokens is not None
            and self.max_input_tokens <= 0
        ):
            raise ValueError(
                "max_input_tokens must be "
                "greater than 0."
            )

        self.prompt_template = (
            self.prompt_path.read_text(
                encoding="utf-8"
            )
        )

        self._validate_template()

    def _validate_config(self) -> None:
        if not self.prompt_path.exists():
            raise FileNotFoundError(
                "Feedback-regeneration prompt "
                f"not found: {self.prompt_path}"
            )

        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be "
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
            "{problem}",
            "{extracted_code}",
            "{input_text}",
            # "{expected_output}",
            # "{actual_output}",
            "{stderr}",
        }

        missing = [
            placeholder
            for placeholder
            in required_placeholders
            if placeholder
            not in self.prompt_template
        ]

        if missing:
            raise ValueError(
                "Missing feedback-regeneration "
                "prompt placeholders: "
                + ", ".join(
                    missing
                )
            )

    def build_prompt(
        self,
        failure: Phase1FailureRecord,
    ) -> str:
        input_text = truncate_input_text(
            text=failure.input_text,
            tokenizer=self.generator.tokenizer,
            max_tokens=self.max_input_tokens,
        )

        return self.prompt_template.format(
            problem=failure.problem,
            extracted_code=(
                failure.extracted_code
            ),
            input_text=input_text,
            stderr=(
                failure.stderr
            ),
        ).strip()

    def run(
        self,
        failure: Phase1FailureRecord,
    ) -> StrategyOutput:
        formatted_prompt = self.build_prompt(
            failure
        )

        generation = self.generator.generate(
            prompt=formatted_prompt,
            system_prompt=self.system_prompt,
            max_new_tokens=(
                self.max_new_tokens
            ),
            temperature=self.temperature,
            top_p=self.top_p,
        )

        generation_step = GenerationStep(
            name="code_regeneration",
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
            problem_id=failure.problem_id,
            strategy=self.name,

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

            strategy_trace=[
                generation_step
            ],
        )