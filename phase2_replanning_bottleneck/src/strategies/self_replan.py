"""
Self-Replanning Regeneration strategy.

Phase 1 Direct generation에서 실패한 trajectory를 입력으로 받아:

1. Problem + Failed Code + Execution Feedback
   -> Self-generated Revised Plan

2. Problem + Failed Code + Execution Feedback + Revised Plan
   -> Code Regeneration

을 순차적으로 수행한다.

Flow:
    FailureCase
        -> self_replan_plan.txt
        -> ModelGenerator.generate()
        -> Revised Plan
        -> self_replan_code.txt
        -> ModelGenerator.generate()
        -> RefinementOutput

주의:
    - 동일한 student model을 re-plan 생성과 code 생성에 모두 사용한다.
    - evaluation은 수행하지 않는다.
    - code extraction은 수행하지 않는다.
    - prompt_tokens / completion_tokens / generation_time은
      re-plan 호출과 code 호출의 합계를 기록한다.
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


STRATEGY_NAME = "self_replan"


class SelfReplanStrategy:
    """
    Execution feedback을 기반으로 revised plan을 생성한 뒤,
    revised plan을 이용하여 solution code를 재생성한다.
    """

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

        self.plan_prompt_path = Path(
            plan_prompt_path
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

        self._validate_generation_config()

        self.plan_prompt_template = (
            self._load_prompt_template(
                path=self.plan_prompt_path,
                required_placeholders=(
                    "{problem}",
                    "{previous_code}",
                    "{execution_feedback}",
                ),
                prompt_name="self-replan plan",
            )
        )

        self.code_prompt_template = (
            self._load_prompt_template(
                path=self.code_prompt_path,
                required_placeholders=(
                    "{problem}",
                    "{previous_code}",
                    "{execution_feedback}",
                    "{self_replan}",
                ),
                prompt_name="self-replan code",
            )
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        case: FailureCase,
    ) -> RefinementOutput:
        """
        FailureCase 하나에 대해 self-replanning regeneration을 수행한다.

        Step 1:
            failure trajectory -> revised plan

        Step 2:
            failure trajectory + revised plan -> new code
        """

        # --------------------------------------------------------------
        # Step 1. Self Re-planning
        # --------------------------------------------------------------

        plan_prompt = self.build_plan_prompt(
            case
        )

        plan_generation = self._generate_plan(
            plan_prompt
        )

        self_replan = (
            plan_generation.text.strip()
        )

        if not self_replan:
            raise ValueError(
                "Self-replan generation returned empty output "
                f"for problem_id={case.example.problem_id}"
            )

        plan_step = GenerationStep(
            name="self_replan",

            formatted_prompt=plan_prompt,

            raw_output=self_replan,

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

        # --------------------------------------------------------------
        # Step 2. Code Regeneration
        # --------------------------------------------------------------

        code_prompt = self.build_code_prompt(
            case=case,
            self_replan=self_replan,
        )

        code_generation = self._generate_code(
            code_prompt
        )

        code_output = (
            code_generation.text.strip()
        )

        if not code_output:
            raise ValueError(
                "Code regeneration returned empty output "
                f"for problem_id={case.example.problem_id}"
            )

        code_step = GenerationStep(
            name="code_regeneration",

            formatted_prompt=code_prompt,

            raw_output=code_output,

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

        # --------------------------------------------------------------
        # Total inference cost
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Output
        # --------------------------------------------------------------

        return RefinementOutput(
            problem_id=(
                case.example.problem_id
            ),

            strategy=STRATEGY_NAME,

            # RefinementOutput의 main prompt/output은
            # 최종 code-generation step을 가리킨다.
            formatted_prompt=code_prompt,

            raw_output=code_output,

            # Strategy 전체 inference cost
            prompt_tokens=(
                total_prompt_tokens
            ),

            completion_tokens=(
                total_completion_tokens
            ),

            generation_time=(
                total_generation_time
            ),

            strategy_trace=[
                plan_step,
                code_step,
            ],

            # Phase 2-2 전용 field
            self_replan=self_replan,

            # Teacher strategy fields
            teacher_replan=None,
            teacher_replan_source=None,
            teacher_replan_version=None,
            teacher_replan_verified=None,
        )

    # ------------------------------------------------------------------
    # Prompt Construction
    # ------------------------------------------------------------------

    def build_plan_prompt(
        self,
        case: FailureCase,
    ) -> str:
        """
        FailureCase를 revised-plan generation prompt로 변환한다.
        """

        values = {
            "problem": (
                case.example.prompt
            ),

            "previous_code": (
                case.initial_code
            ),

            "execution_feedback": (
                case.feedback.feedback_text
            ),
        }

        try:
            return (
                self.plan_prompt_template.format(
                    **values
                )
            )

        except KeyError as error:
            missing_key = error.args[0]

            raise ValueError(
                "Missing self-replan plan prompt "
                f"placeholder value: {missing_key}"
            ) from error

    def build_code_prompt(
        self,
        *,
        case: FailureCase,
        self_replan: str,
    ) -> str:
        """
        Revised plan을 포함한 code-regeneration prompt를 생성한다.
        """

        if not isinstance(
            self_replan,
            str,
        ):
            raise TypeError(
                "self_replan must be a string, "
                f"got {type(self_replan).__name__}."
            )

        self_replan = (
            self_replan.strip()
        )

        if not self_replan:
            raise ValueError(
                "self_replan must not be empty."
            )

        values = {
            "problem": (
                case.example.prompt
            ),

            "previous_code": (
                case.initial_code
            ),

            "execution_feedback": (
                case.feedback.feedback_text
            ),

            "self_replan": (
                self_replan
            ),
        }

        try:
            return (
                self.code_prompt_template.format(
                    **values
                )
            )

        except KeyError as error:
            missing_key = error.args[0]

            raise ValueError(
                "Missing self-replan code prompt "
                f"placeholder value: {missing_key}"
            ) from error

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_plan(
        self,
        prompt: str,
    ) -> GenerationOutput:
        """
        Student model로 revised plan을 생성한다.
        """

        generation = (
            self.generator.generate(
                prompt,
                system_prompt=(
                    self.system_prompt
                ),
                max_new_tokens=(
                    self.plan_max_new_tokens
                ),
                temperature=(
                    self.temperature
                ),
                top_p=self.top_p,
            )
        )

        if not isinstance(
            generation,
            GenerationOutput,
        ):
            raise TypeError(
                "generator.generate() must return "
                "GenerationOutput during self-replanning, "
                f"got {type(generation).__name__}."
            )

        return generation

    def _generate_code(
        self,
        prompt: str,
    ) -> GenerationOutput:
        """
        Revised plan을 조건으로 student model이 새 코드를 생성한다.
        """

        generation = (
            self.generator.generate(
                prompt,
                system_prompt=(
                    self.system_prompt
                ),
                max_new_tokens=(
                    self.code_max_new_tokens
                ),
                temperature=(
                    self.temperature
                ),
                top_p=self.top_p,
            )
        )

        if not isinstance(
            generation,
            GenerationOutput,
        ):
            raise TypeError(
                "generator.generate() must return "
                "GenerationOutput during code regeneration, "
                f"got {type(generation).__name__}."
            )

        return generation

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_generation_config(
        self,
    ) -> None:
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
                "temperature must be greater "
                "than or equal to 0."
            )

        if not 0 < self.top_p <= 1:
            raise ValueError(
                "top_p must be in the range "
                "(0, 1]."
            )

    # ------------------------------------------------------------------
    # Prompt Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_prompt_template(
        *,
        path: Path,
        required_placeholders: tuple[str, ...],
        prompt_name: str,
    ) -> str:
        """
        Prompt template 파일을 읽고 필수 placeholder를 검증한다.
        """

        if not path.exists():
            raise FileNotFoundError(
                f"{prompt_name} prompt not found: "
                f"{path}"
            )

        if not path.is_file():
            raise ValueError(
                f"{prompt_name} prompt path "
                f"is not a file: {path}"
            )

        template = path.read_text(
            encoding="utf-8"
        ).strip()

        if not template:
            raise ValueError(
                f"{prompt_name} prompt is empty: "
                f"{path}"
            )

        missing = [
            placeholder
            for placeholder
            in required_placeholders
            if placeholder not in template
        ]

        if missing:
            raise ValueError(
                f"{prompt_name} prompt is missing "
                f"required placeholders: {missing}"
            )

        return template