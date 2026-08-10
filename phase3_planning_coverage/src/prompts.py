"""
Self-Planning 프롬프트 구성.

Phase 1 `src/strategies/self_plan.py`의 프롬프트 구성 로직을 그대로 옮긴 것이다.
Phase 1과 Phase 3의 비교 가능성을 위해 다음은 절대 변경하지 않는다.

- placeholder 치환 방식과 .strip() 처리
- starter_code_section 구성 방식

프롬프트 템플릿은 Phase 3에 복사본을 두지 않고
`phase1_planning_bottleneck/prompts/`에서 직접 읽는다.
복사본을 두면 한쪽만 수정됐을 때 Phase 1 <-> Phase 3 비교가 조용히 깨지므로,
Phase 1 파일 하나를 single source of truth로 삼는다.

경로 해석 규칙:
- 절대 경로를 주면 그대로 사용한다.
- 상대 경로를 주면 파일명만 취해 Phase 1 prompts 디렉터리에서 찾는다.
  (config의 `prompts/self_plan_plan.txt` ->
   phase1_planning_bottleneck/prompts/self_plan_plan.txt)
- 실행 위치(cwd)와 무관하게 이 파일 위치를 기준으로 해석한다.
"""

from __future__ import annotations

from pathlib import Path

from src.common.schemas import ProblemExample


# .../project_sLM_planning/phase3_planning_coverage/src/prompts.py
#  parents[0] = src
#  parents[1] = phase3_planning_coverage
#  parents[2] = project_sLM_planning
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE1_ROOT = PROJECT_ROOT / "phase1_planning_bottleneck"

PHASE1_PROMPT_DIR = PHASE1_ROOT / "prompts"

DEFAULT_PLAN_PROMPT_PATH = (
    PHASE1_PROMPT_DIR / "self_plan_plan.txt"
)

DEFAULT_CODE_PROMPT_PATH = (
    PHASE1_PROMPT_DIR / "self_plan_code.txt"
)


def resolve_phase1_prompt_path(
    path: str | Path,
) -> Path:
    """프롬프트 경로를 Phase 1 prompts 디렉터리 기준으로 해석한다."""
    candidate = Path(path)

    if candidate.is_absolute():
        return candidate

    # 'prompts/self_plan_plan.txt', 'self_plan_plan.txt' 모두
    # Phase 1 prompts 디렉터리의 같은 파일을 가리키게 한다.
    return PHASE1_PROMPT_DIR / candidate.name


class SelfPlanPromptBuilder:
    """Self-Planning의 plan/code 프롬프트를 생성한다.

    템플릿은 Phase 1 폴더(`phase1_planning_bottleneck/prompts/`)에서 읽는다.
    """

    PLAN_PLACEHOLDERS = (
        "{title}",
        "{problem}",
        "{starter_code_section}",
    )

    CODE_PLACEHOLDERS = (
        "{title}",
        "{problem}",
        "{plan}",
        "{starter_code_section}",
    )

    def __init__(
        self,
        plan_prompt_path: str | Path = (
            DEFAULT_PLAN_PROMPT_PATH
        ),
        code_prompt_path: str | Path = (
            DEFAULT_CODE_PROMPT_PATH
        ),
    ) -> None:
        self.plan_prompt_path = (
            resolve_phase1_prompt_path(plan_prompt_path)
        )
        self.code_prompt_path = (
            resolve_phase1_prompt_path(code_prompt_path)
        )

        if not PHASE1_PROMPT_DIR.is_dir():
            raise FileNotFoundError(
                f"Phase 1 prompt directory not found: "
                f"{PHASE1_PROMPT_DIR}. "
                f"Phase 3는 Phase 1 폴더의 프롬프트를 직접 사용한다."
            )

        if not self.plan_prompt_path.exists():
            raise FileNotFoundError(
                f"Plan prompt template not found: "
                f"{self.plan_prompt_path}"
            )

        if not self.code_prompt_path.exists():
            raise FileNotFoundError(
                f"Code prompt template not found: "
                f"{self.code_prompt_path}"
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
        missing_plan = [
            placeholder
            for placeholder in self.PLAN_PLACEHOLDERS
            if placeholder not in self.plan_prompt_template
        ]

        if missing_plan:
            raise ValueError(
                "Missing plan prompt placeholders: "
                + ", ".join(missing_plan)
            )

        missing_code = [
            placeholder
            for placeholder in self.CODE_PLACEHOLDERS
            if placeholder not in self.code_prompt_template
        ]

        if missing_code:
            raise ValueError(
                "Missing code prompt placeholders: "
                + ", ".join(missing_code)
            )

    @staticmethod
    def _build_starter_code_section(
        example: ProblemExample,
    ) -> str:
        if not example.starter_code.strip():
            return ""

        return (
            "Starter Code:\n"
            f"{example.starter_code.strip()}"
        )

    def build_plan_prompt(
        self,
        example: ProblemExample,
    ) -> str:
        starter_code_section = (
            self._build_starter_code_section(example)
        )

        return self.plan_prompt_template.format(
            title=example.title,
            problem=example.prompt,
            starter_code_section=starter_code_section,
        ).strip()

    def build_code_prompt(
        self,
        example: ProblemExample,
        plan: str,
    ) -> str:
        if not plan.strip():
            raise ValueError(
                "Generated plan must not be empty."
            )

        starter_code_section = (
            self._build_starter_code_section(example)
        )

        return self.code_prompt_template.format(
            title=example.title,
            problem=example.prompt,
            plan=plan.strip(),
            starter_code_section=starter_code_section,
        ).strip()
