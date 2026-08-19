# phase2_replanning_bottleneck/strategies/self_replan.py

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


class SelfReplanStrategy:
    """
    Generate a revised plan from execution feedback,
    then regenerate code conditioned on that plan.
    """

    name = "self_replan"

    def __init__(
        self,
        generator: ModelGenerator,
        replan_prompt_path: str | Path,
        code_prompt_path: str | Path,
        *,
        system_prompt: str | None = None,
        plan_max_new_tokens: int = 384,
        code_max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_input_tokens: int | None = None,
    ) -> None:
        self.generator = generator

        self.replan_prompt_path = Path(
            replan_prompt_path
        )
        self.code_prompt_path = Path(
            code_prompt_path
        )

        self.system_prompt = system_prompt

        self.plan_max_new_tokens = (
            plan_max_new_tokens
        )
        self.code_max_new_tokens = (
            code_max_new_tokens
        )

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

        self.replan_prompt_template = (
            self.replan_prompt_path.read_text(
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
        if not self.replan_prompt_path.exists():
            raise FileNotFoundError(
                "Self-replan prompt not found: "
                f"{self.replan_prompt_path}"
            )

        if not self.code_prompt_path.exists():
            raise FileNotFoundError(
                "Self-replan code prompt not found: "
                f"{self.code_prompt_path}"
            )

        if self.plan_max_new_tokens <= 0:
            raise ValueError(
                "plan_max_new_tokens must be "
                "greater than 0."
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

    def _validate_templates(self) -> None:
        replan_placeholders = {
            "{problem}",
            "{extracted_code}",
            "{input_text}",
            "{stderr}",
        }

        missing_replan = [
            placeholder
            for placeholder in replan_placeholders
            if placeholder
            not in self.replan_prompt_template
        ]

        if missing_replan:
            raise ValueError(
                "Missing self-replan prompt placeholders: "
                + ", ".join(missing_replan)
            )

        code_placeholders = {
            "{problem}",
            "{plan}",
            "{starter_code_section}",
        }

        missing_code = [
            placeholder
            for placeholder in code_placeholders
            if placeholder
            not in self.code_prompt_template
        ]

        if missing_code:
            raise ValueError(
                "Missing self-replan code prompt placeholders: "
                + ", ".join(missing_code)
            )
            
    def build_replan_prompt(
        self,
        failure: Phase1FailureRecord,
    ) -> str:
        """
        Build the prompt used to generate a revised plan.
        """

        input_text = truncate_input_text(
            text=failure.input_text,
            tokenizer=self.generator.tokenizer,
            max_tokens=self.max_input_tokens,
        )

        return self.replan_prompt_template.format(
            problem=failure.problem,
            extracted_code=failure.extracted_code,
            input_text=input_text,
            stderr=failure.stderr,
        ).strip()

    @staticmethod
    def _build_starter_code_section(
        starter_code: str,
    ) -> str:
        if not starter_code.strip():
            return ""

        return (
            "Starter Code:\n"
            f"{starter_code.strip()}"
        )

    def build_code_prompt(
        self,
        *,
        failure: Phase1FailureRecord,
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
                "Generated replan must not be empty."
            )

        starter_code_section = (
            self._build_starter_code_section(
                failure.starter_code
            )
        )

        return self.code_prompt_template.format(
            problem=failure.problem,
            plan=plan,
            starter_code_section=starter_code_section,
        ).strip()

    def run(
        self,
        failure: Phase1FailureRecord,
    ) -> StrategyOutput:
        # ------------------------------------------------------
        # Step 1: generate revised plan from failure feedback
        # ------------------------------------------------------

        replan_prompt = (
            self.build_replan_prompt(
                failure
            )
        )

        plan_generation = (
            self.generator.generate(
                prompt=replan_prompt,
                system_prompt=self.system_prompt,
                max_new_tokens=(
                    self.plan_max_new_tokens
                ),
                temperature=self.temperature,
                top_p=self.top_p,
            )
        )

        revised_plan = (
            plan_generation.text.strip()
        )

        if not revised_plan:
            raise ValueError(
                "Self-replan generation returned "
                "an empty plan: "
                f"{failure.problem_id}"
            )

        replan_step = GenerationStep(
            name="replan_generation",
            formatted_prompt=replan_prompt,
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

        # ------------------------------------------------------
        # Step 2: regenerate code from revised plan
        # ------------------------------------------------------

        code_prompt = self.build_code_prompt(
            failure=failure,
            plan=revised_plan,
        )

        code_generation = (
            self.generator.generate(
                prompt=code_prompt,
                system_prompt=self.system_prompt,
                max_new_tokens=(
                    self.code_max_new_tokens
                ),
                temperature=self.temperature,
                top_p=self.top_p,
            )
        )

        code_step = GenerationStep(
            name="code_regeneration",
            formatted_prompt=code_prompt,
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

        # ------------------------------------------------------
        # Aggregate generation cost across both stages
        # ------------------------------------------------------

        total_prompt_tokens = (
            plan_generation.prompt_tokens
            + code_generation.prompt_tokens
        )

        total_completion_tokens = (
            plan_generation.completion_tokens
            + code_generation.completion_tokens
        )

        total_generation_time = (
            plan_generation.generation_time
            + code_generation.generation_time
        )

        # ------------------------------------------------------
        # Final StrategyOutput
        # ------------------------------------------------------

        return StrategyOutput(
            problem_id=failure.problem_id,
            strategy=self.name,

            # Final output consumed by CodeParser.
            formatted_prompt=code_prompt,
            raw_output=code_generation.text,

            # Total inference cost of the complete strategy.
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            generation_time=total_generation_time,

            strategy_trace=[
                replan_step,
                code_step,
            ],
        )