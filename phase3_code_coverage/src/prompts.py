"""
Phase 3-B 프롬프트 구성.

plan은 새로 생성하지 않으므로 code 프롬프트만 사용한다.
프롬프트 원문은 Phase 1 디렉터리
(phase1_planning_bottleneck/prompts/)에서 직접 읽어
Phase 1 / Phase 3-A / Phase 3-B가 완전히 동일한
code prompt를 사용하도록 보장한다.
"""

from __future__ import annotations

from pathlib import Path

from src.common.schemas import ProblemExample


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE1_PROMPT_DIR = (
    PROJECT_ROOT
    / "phase1_planning_bottleneck"
    / "prompts"
)

DEFAULT_CODE_PROMPT_PATH = (
    PHASE1_PROMPT_DIR
    / "self_plan_code.txt"
)


def resolve_phase1_prompt_path(
    path_or_name: str | Path,
) -> Path:
    """
    파일명만 주어지면 Phase 1 prompts 디렉터리 기준으로 해석한다.

    예:
        self_plan_code.txt
        -> <project_root>/phase1_planning_bottleneck/prompts/self_plan_code.txt

    절대경로나 디렉터리가 포함된 상대경로라면
    전달된 경로를 그대로 사용한다.
    """
    path = Path(path_or_name)

    if path.is_absolute():
        resolved = path

    elif path.parent == Path("."):
        resolved = (
            PHASE1_PROMPT_DIR
            / path.name
        )

    else:
        resolved = path.resolve()

    if not resolved.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {resolved}"
        )

    if not resolved.is_file():
        raise ValueError(
            f"Prompt path is not a file: {resolved}"
        )

    return resolved


class FixedPlanCodePromptBuilder:
    """
    고정된 self-generated plan과 문제 설명으로
    code generation prompt를 생성한다.
    """

    REQUIRED_PLACEHOLDERS = (
        "{title}",
        "{problem}",
        "{plan}",
        "{starter_code_section}",
    )

    def __init__(
        self,
        code_prompt_path: str | Path = DEFAULT_CODE_PROMPT_PATH,
        system_prompt: str | None = None,
    ) -> None:
        self.code_prompt_path = (
            resolve_phase1_prompt_path(
                code_prompt_path
            )
        )

        self.system_prompt = system_prompt

        self.code_prompt_template = (
            self.code_prompt_path.read_text(
                encoding="utf-8"
            )
        )

        self._validate_template()

    def _validate_template(self) -> None:
        missing = [
            placeholder
            for placeholder
            in self.REQUIRED_PLACEHOLDERS
            if placeholder
            not in self.code_prompt_template
        ]

        if missing:
            raise ValueError(
                "Missing code prompt placeholders: "
                + ", ".join(missing)
            )

    @staticmethod
    def _build_starter_code_section(
        example: ProblemExample,
    ) -> str:
        """
        Phase 1과 동일한 starter_code_section 생성 방식.
        """
        if not example.starter_code.strip():
            return ""

        return (
            "Starter Code:\n"
            f"{example.starter_code.strip()}"
        )

    def build_code_prompt(
        self,
        example: ProblemExample,
        plan_text: str,
    ) -> str:
        """
        Phase 1에서 생성된 fixed plan을 그대로 넣어
        code generation prompt를 만든다.
        """
        plan = plan_text.strip()

        if not plan:
            raise ValueError(
                "Fixed plan must not be empty."
            )

        starter_code_section = (
            self._build_starter_code_section(
                example
            )
        )

        prompt = (
            self.code_prompt_template.format(
                title=example.title,
                problem=example.prompt,
                plan=plan,
                starter_code_section=(
                    starter_code_section
                ),
            )
        ).strip()

        # Phase 3-B에서 fixed plan이 실제 code prompt에
        # 포함되었는지 방어적으로 확인한다.
        if plan not in prompt:
            raise RuntimeError(
                "Fixed plan is missing from "
                "the generated code prompt."
            )

        return prompt