"""Phase 3-B 프롬프트 구성.

plan은 새로 생성하지 않으므로 code 프롬프트만 사용한다.
프롬프트 원문은 Phase 1 디렉터리(phase1_planning_bottleneck/prompts/)에서 직접 읽어
Phase 1 / Phase 3-A / Phase 3-B가 완전히 동일한 프롬프트를 쓰도록 보장한다.
"""
from __future__ import annotations

from pathlib import Path

from src.common.schemas import ProblemExample

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE1_PROMPT_DIR = PROJECT_ROOT / "phase1_planning_bottleneck" / "prompts"

DEFAULT_CODE_PROMPT_PATH = PHASE1_PROMPT_DIR / "self_plan_code.txt"


def resolve_phase1_prompt_path(path_or_name: str | Path) -> Path:
    """파일명만 주어지면 Phase 1 prompts 디렉터리 기준으로 해석한다."""
    raise NotImplementedError


class FixedPlanCodePromptBuilder:
    """고정 plan + 문제 설명 -> code 생성 프롬프트."""

    def __init__(
        self,
        code_prompt_path: str | Path = DEFAULT_CODE_PROMPT_PATH,
        system_prompt: str | None = None,
    ) -> None:
        raise NotImplementedError

    def build_code_prompt(self, example: ProblemExample, plan_text: str) -> str:
        raise NotImplementedError
