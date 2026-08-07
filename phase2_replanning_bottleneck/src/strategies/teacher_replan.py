"""
Teacher-Replanning Regeneration strategy.

Phase 1 Direct generation에서 실패한 trajectory를 입력으로 받아:

1. TeacherReplanStore에서 해당 FailureCase의 teacher re-plan 조회
2. Problem + Failed Code + Execution Feedback + Teacher Re-plan
   -> Code Regeneration

을 수행한다.

Flow:
    FailureCase
        -> TeacherReplanStore.get_for_failure()
        -> Teacher Revised Plan
        -> teacher_replan_code.txt
        -> ModelGenerator.generate()
        -> RefinementOutput

주의:
    - teacher re-plan은 offline에서 이미 생성되어 있다고 가정한다.
    - student model(Qwen)은 code generation 1회만 수행한다.
    - evaluation은 수행하지 않는다.
    - code extraction은 수행하지 않는다.
    - prompt_tokens / completion_tokens / generation_time은
      student code-generation 호출의 비용만 기록한다.
"""

from __future__ import annotations

from pathlib import Path

from src.models.generator import ModelGenerator
from src.plans.teacher_replan_store import (
    TeacherReplanEntry,
    TeacherReplanStore,
)
from src.schemas import (
    FailureCase,
    GenerationOutput,
    GenerationStep,
    RefinementOutput,
)


STRATEGY_NAME = "teacher_replan"


class TeacherReplanStrategy:
    """
    외부 teacher가 생성한 revised plan을 이용하여
    student model이 solution code를 재생성한다.
    """

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

        self._validate_generation_config()

        self.code_prompt_template = (
            self._load_prompt_template(
                path=self.code_prompt_path,
                required_placeholders=(
                    "{problem}",
                    "{previous_code}",
                    "{execution_feedback}",
                    "{teacher_replan}",
                ),
                prompt_name="teacher-replan code",
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
        FailureCase 하나에 대해 teacher-replanning regeneration을 수행한다.
        """

        # --------------------------------------------------------------
        # Step 1. Teacher re-plan 조회
        # --------------------------------------------------------------

        entry = (
            self.replan_store.get_for_failure(
                case
            )
        )

        teacher_replan = (
            entry.teacher_replan.strip()
        )

        if not teacher_replan:
            raise ValueError(
                "Teacher re-plan is empty "
                f"for problem_id={case.example.problem_id}"
            )

        # --------------------------------------------------------------
        # Step 2. Code prompt 생성
        # --------------------------------------------------------------

        code_prompt = self.build_code_prompt(
            case=case,
            teacher_replan=teacher_replan,
        )

        # --------------------------------------------------------------
        # Step 3. Student code generation
        # --------------------------------------------------------------

        code_generation = (
            self._generate_code(
                code_prompt
            )
        )

        code_output = (
            code_generation.text.strip()
        )

        if not code_output:
            raise ValueError(
                "Teacher-replan code generation "
                "returned empty output "
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
        # Output
        # --------------------------------------------------------------

        return RefinementOutput(
            problem_id=(
                case.example.problem_id
            ),

            strategy=STRATEGY_NAME,

            # 최종 code generation prompt/output
            formatted_prompt=code_prompt,

            raw_output=code_output,

            # student model code-generation cost만 기록
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
                code_step,
            ],

            self_replan=None,

            teacher_replan=(
                teacher_replan
            ),

            teacher_replan_source=(
                entry.teacher_model
            ),

            teacher_replan_version=(
                entry.replan_version
            ),

            teacher_replan_verified=(
                entry.verified
            ),
        )

    # ------------------------------------------------------------------
    # Prompt Construction
    # ------------------------------------------------------------------

    def build_code_prompt(
        self,
        *,
        case: FailureCase,
        teacher_replan: str,
    ) -> str:
        """
        Teacher revised plan을 포함한 code-generation prompt를 만든다.
        """

        if not isinstance(
            teacher_replan,
            str,
        ):
            raise TypeError(
                "teacher_replan must be a string, "
                f"got {type(teacher_replan).__name__}."
            )

        teacher_replan = (
            teacher_replan.strip()
        )

        if not teacher_replan:
            raise ValueError(
                "teacher_replan must not be empty."
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

            "teacher_replan": (
                teacher_replan
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
                "Missing teacher-replan code prompt "
                f"placeholder value: {missing_key}"
            ) from error

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_code(
        self,
        prompt: str,
    ) -> GenerationOutput:
        """
        Teacher revised plan을 조건으로 student model이 코드를 생성한다.
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
                "GenerationOutput during teacher-replan "
                "code generation, "
                f"got {type(generation).__name__}."
            )

        return generation

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_generation_config(
        self,
    ) -> None:
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