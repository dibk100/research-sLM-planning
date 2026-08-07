"""Teacher re-plan + repair 전략.

    Problem + Initial Code + Execution Feedback
        + (외부 teacher가 작성한 Revised Plan) -> Code

teacher revised plan은 실행 시점에 생성하지 않고,
미리 만들어 둔 JSONL(TeacherReplanStore)에서 조회한다.
따라서 코드 생성 호출은 1회이며, Phase 1 TeacherPlanStrategy와 동일한 구조다.

Phase 1과의 차이는 프롬프트에 initial code와 execution feedback이 함께 들어가고,
plan이 "처음부터의 계획"이 아니라 "실패를 고친 계획"이라는 점이다.

TODO(구현)
----------
- [ ] build_code_prompt : 템플릿 채우기
- [ ] run : plan 조회 -> generator 호출 -> RefinementOutput
"""

from __future__ import annotations

from pathlib import Path

from src.models.generator import ModelGenerator
from src.plans.teacher_replan_store import (
    TeacherReplanStore,
)
from src.schemas import (
    FailureCase,
    GenerationStep,
    RefinementOutput,
)


class TeacherReplanStrategy:
    """외부 teacher의 revised plan으로 코드를 재생성한다."""

    name = "teacher_replan"

    # prompts/teacher_replan_code.txt
    REQUIRED_PLACEHOLDERS = {
        "{title}",
        "{problem}",
        "{initial_code}",
        "{execution_feedback}",
        "{teacher_replan}",
    }

    def __init__(
        self,
        generator: ModelGenerator,
        plan_store: TeacherReplanStore,
        code_prompt_path: str | Path,
        *,
        system_prompt: str | None = None,
        code_max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.generator = generator
        self.plan_store = plan_store
        self.code_prompt_path = Path(
            code_prompt_path
        )

        self.system_prompt = system_prompt
        self.code_max_new_tokens = (
            code_max_new_tokens
        )
        self.temperature = temperature
        self.top_p = top_p

        if not self.code_prompt_path.exists():
            raise FileNotFoundError(
                "Teacher-replan code prompt not found: "
                f"{self.code_prompt_path}"
            )

        if self.code_max_new_tokens <= 0:
            raise ValueError(
                "code_max_new_tokens must be "
                "greater than 0."
            )

        self.code_prompt_template = (
            self.code_prompt_path.read_text(
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
            not in self.code_prompt_template
        ]

        if missing:
            raise ValueError(
                "Missing teacher-replan prompt "
                "placeholders: " + ", ".join(missing)
            )

    def build_code_prompt(
        self,
        case: FailureCase,
        teacher_replan: str,
    ) -> str:
        raise NotImplementedError(
            "TODO: code_prompt_template.format(...)"
        )

    def run(
        self,
        case: FailureCase,
    ) -> RefinementOutput:
        raise NotImplementedError(
            "TODO: plan_store.get -> build_code_prompt "
            "-> generator.generate -> RefinementOutput"
        )
