"""Self re-plan + repair 전략.

    Problem + Initial Code + Execution Feedback -> Revised Plan -> Code

모델이 스스로 새로운 solution plan을 세운 뒤, 그 계획으로 코드를 다시 만든다.
Phase 1의 SelfPlanStrategy와 동일하게 2회 호출(plan -> code)이며,
비용(prompt_tokens / completion_tokens / generation_time)은 두 호출의 합으로 기록한다.

strategy_trace 는 ["replan_generation", "code_generation"] 두 단계를 가진다.

TODO(구현)
----------
- [ ] build_replan_prompt / build_code_prompt
- [ ] run : 2회 호출 및 비용 합산
- [ ] plan 추출 실패(빈 출력) 시 처리 정책 결정
"""

from __future__ import annotations

from pathlib import Path

from src.models.generator import ModelGenerator
from src.schemas import (
    FailureCase,
    GenerationStep,
    RefinementOutput,
)


class SelfReplanStrategy:
    """모델이 스스로 계획을 다시 세운 뒤 코드를 재생성한다."""

    name = "self_replan"

    # prompts/self_replan_plan.txt
    REQUIRED_PLAN_PLACEHOLDERS = {
        "{title}",
        "{problem}",
        "{initial_code}",
        "{execution_feedback}",
    }

    # prompts/self_replan_code.txt
    REQUIRED_CODE_PLACEHOLDERS = {
        "{title}",
        "{problem}",
        "{revised_plan}",
    }

    def __init__(
        self,
        generator: ModelGenerator,
        plan_prompt_path: str | Path,
        code_prompt_path: str | Path,
        *,
        system_prompt: str | None = None,
        plan_max_new_tokens: int = 384,
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

        for path in (
            self.plan_prompt_path,
            self.code_prompt_path,
        ):
            if not path.exists():
                raise FileNotFoundError(
                    "Self-replan prompt not found: "
                    f"{path}"
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

        self.plan_prompt_template = (
            self.plan_prompt_path.read_text(
                encoding="utf-8"
            )
        )
        self.code_prompt_template = (
            self.code_prompt_path.read_text(
                encoding="utf-8"
            )
        )

        self._validate_templates()

    def _validate_templates(self) -> None:
        checks = (
            (
                self.plan_prompt_template,
                self.REQUIRED_PLAN_PLACEHOLDERS,
                "self-replan plan",
            ),
            (
                self.code_prompt_template,
                self.REQUIRED_CODE_PLACEHOLDERS,
                "self-replan code",
            ),
        )

        for template, required, label in checks:
            missing = [
                placeholder
                for placeholder in required
                if placeholder not in template
            ]

            if missing:
                raise ValueError(
                    f"Missing {label} prompt "
                    "placeholders: "
                    + ", ".join(missing)
                )

    def build_replan_prompt(
        self,
        case: FailureCase,
    ) -> str:
        raise NotImplementedError(
            "TODO: plan_prompt_template.format(...)"
        )

    def build_code_prompt(
        self,
        case: FailureCase,
        revised_plan: str,
    ) -> str:
        raise NotImplementedError(
            "TODO: code_prompt_template.format(...)"
        )

    def run(
        self,
        case: FailureCase,
    ) -> RefinementOutput:
        raise NotImplementedError(
            "TODO: replan 호출 -> code 호출 -> 비용 합산"
        )
