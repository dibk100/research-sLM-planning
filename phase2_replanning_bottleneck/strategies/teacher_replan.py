# phase2_replanning_bottleneck/strategies/teacher_replan.py

from __future__ import annotations

from pathlib import Path

from src.datasets.phase1_failure_loader import (
    Phase1FailureRecord,
)
from src.models.generator import ModelGenerator
from src.plans.teacher_replan_store import (
    TeacherReplanStore,
)
from src.schemas import (
    GenerationStep,
    StrategyOutput,
)


class TeacherReplanStrategy:
    """
    Regenerate code using an externally provided
    teacher-generated revised plan.
    """

    name = "teacher_replan"

    def __init__(
        self,
        generator: ModelGenerator,
        replan_store: TeacherReplanStore,
        code_prompt_path: str | Path,
        *,
        system_prompt: str | None = None,
        code_max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.generator = generator
        self.replan_store = replan_store

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
                "Teacher-replan code prompt "
                "not found: "
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
            "{problem}",
            "{plan}",
            "{starter_code_section}",
        }

        missing = [
            placeholder
            for placeholder
            in required_placeholders
            if placeholder
            not in self.code_prompt_template
        ]

        if missing:
            raise ValueError(
                "Missing teacher-replan code "
                "prompt placeholders: "
                + ", ".join(missing)
            )

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
        teacher_replan: str,
    ) -> str:
        if not isinstance(
            teacher_replan,
            str,
        ):
            raise TypeError(
                "teacher_replan must be str, "
                f"got "
                f"{type(teacher_replan).__name__}"
            )

        teacher_replan = (
            teacher_replan.strip()
        )

        if not teacher_replan:
            raise ValueError(
                "Teacher replan must not be empty."
            )

        starter_code_section = (
            self._build_starter_code_section(
                failure.starter_code
            )
        )

        return self.code_prompt_template.format(
            problem=failure.problem,
            plan=teacher_replan,
            starter_code_section=(
                starter_code_section
            ),
        ).strip()

    def run(
        self,
        failure: Phase1FailureRecord,
    ) -> StrategyOutput:
        # ------------------------------------------------------
        # Step 1: retrieve external teacher replan
        # ------------------------------------------------------

        replan_record = (
            self.replan_store.get(
                failure.problem_id
            )
        )

        teacher_replan = (
            replan_record.teacher_replan.strip()
        )

        if not teacher_replan:
            raise ValueError(
                "Empty teacher replan: "
                f"{failure.problem_id}"
            )

        # ------------------------------------------------------
        # Step 2: regenerate code conditioned on teacher replan
        # ------------------------------------------------------

        code_prompt = self.build_code_prompt(
            failure=failure,
            teacher_replan=teacher_replan,
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
        # Final StrategyOutput
        # ------------------------------------------------------

        return StrategyOutput(
            problem_id=failure.problem_id,
            strategy=self.name,

            formatted_prompt=code_prompt,
            raw_output=code_generation.text,

            # Teacher replan generation is external,
            # so only student code-generation cost is counted.
            prompt_tokens=(
                code_generation.prompt_tokens
            ),
            completion_tokens=(
                code_generation.completion_tokens
            ),
            generation_time=(
                code_generation.generation_time
            ),

            strategy_trace=[
                code_step
            ],

            teacher_plan=teacher_replan,
            teacher_plan_source=(
                replan_record.teacher_model
            ),
            teacher_plan_version=(
                replan_record.replan_version
            ),
            teacher_plan_verified=(
                replan_record.verified
            ),
        )