"""
SelfReplanStrategy sanity check.

확인 항목:

1. Phase1FailureLoader에서 FailureCase 1개를 정상 로드하는가?
2. self-replan plan prompt가 정상 구성되는가?
3. Mock generator가 plan -> code 순서로 2번 호출되는가?
4. Revised Plan이 code prompt에 정상 삽입되는가?
5. RefinementOutput의 비용이 두 호출의 합으로 계산되는가?
6. strategy_trace가 정확히 2-step으로 저장되는가?

Usage:
    PYTHONPATH=. python -m src.strategies.inspect_self_replan
"""

from __future__ import annotations

from src.datasets.phase1_failure_loader import (
    Phase1FailureLoader,
)
from src.schemas import (
    GenerationOutput,
)
from src.strategies.self_replan import (
    SelfReplanStrategy,
)


RESULTS_PATH = (
    "/mnt/hdd/project_sLM_planning/"
    "output/direct_500_stdin/results.jsonl"
)

PLAN_PROMPT_PATH = (
    "prompts/self_replan_plan.txt"
)

CODE_PROMPT_PATH = (
    "prompts/self_replan_code.txt"
)


class MockGenerator:
    """
    SelfReplanStrategy 테스트용 mock generator.

    첫 번째 호출:
        revised plan 반환

    두 번째 호출:
        regenerated code 반환
    """

    def __init__(self) -> None:
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> GenerationOutput:
        self.call_count += 1

        print()
        print("=" * 100)
        print(
            f"[MockGenerator.generate] "
            f"call #{self.call_count}"
        )
        print("=" * 100)

        print(
            "system_prompt  :",
            system_prompt,
        )
        print(
            "max_new_tokens :",
            max_new_tokens,
        )
        print(
            "temperature    :",
            temperature,
        )
        print(
            "top_p          :",
            top_p,
        )
        print(
            "prompt chars   :",
            len(prompt),
        )

        # -----------------------------------------------------------
        # Call 1: Revised Plan
        # -----------------------------------------------------------

        if self.call_count == 1:
            plan = (
                "- Identify the incorrect mismatch condition.\n"
                "- Compare each character with the target string abc.\n"
                "- Count the positions that differ from abc.\n"
                "- A string is valid if there are zero or exactly two mismatches.\n"
                "- Output YES for valid strings and NO otherwise."
            )

            return GenerationOutput(
                text=plan,
                prompt_tokens=200,
                completion_tokens=60,
                generation_time=0.10,
            )

        # -----------------------------------------------------------
        # Call 2: Code Regeneration
        # -----------------------------------------------------------

        if self.call_count == 2:
            code = """```python
def can_sort(cards):
    mismatches = sum(
        cards[i] != "abc"[i]
        for i in range(3)
    )

    if mismatches in (0, 2):
        return "YES"

    return "NO"


t = int(input())

for _ in range(t):
    cards = input().strip()
    print(can_sort(cards))
```"""

            return GenerationOutput(
                text=code,
                prompt_tokens=350,
                completion_tokens=100,
                generation_time=0.20,
            )

        raise RuntimeError(
            "MockGenerator received more than 2 calls."
        )


def preview(
    text: str,
    max_chars: int = 4000,
) -> str:
    """긴 prompt를 sanity-check용으로 일부만 출력한다."""

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + "\n"
        + f"...[truncated, total={len(text)} chars]"
    )


def main() -> None:
    # ------------------------------------------------------------------
    # 1. FailureCase 1개 로드
    # ------------------------------------------------------------------

    loader = Phase1FailureLoader(
        RESULTS_PATH,
        limit=1,
    )

    cases = list(
        loader.load()
    )

    assert len(cases) == 1

    case = cases[0]

    print("=" * 100)
    print("SelfReplanStrategy Sanity Check")
    print("=" * 100)

    print()
    print("[FailureCase]")

    print(
        "problem_id     :",
        case.example.problem_id,
    )

    print(
        "title          :",
        case.example.title,
    )

    print(
        "difficulty     :",
        case.example.difficulty,
    )

    print(
        "initial_status :",
        case.initial_status,
    )

    print(
        "initial_tests  :",
        (
            f"{case.initial_passed_tests}/"
            f"{case.initial_total_tests}"
        ),
    )

    # ------------------------------------------------------------------
    # 2. Strategy 생성
    # ------------------------------------------------------------------

    generator = MockGenerator()

    strategy = SelfReplanStrategy(
        generator=generator,
        plan_prompt_path=PLAN_PROMPT_PATH,
        code_prompt_path=CODE_PROMPT_PATH,
        plan_max_new_tokens=512,
        code_max_new_tokens=1024,
        temperature=0.0,
        top_p=1.0,
    )

    # ------------------------------------------------------------------
    # 3. Plan prompt 검사
    # ------------------------------------------------------------------

    plan_prompt = strategy.build_plan_prompt(
        case
    )

    print()
    print("=" * 100)
    print("[Self-Replan Prompt]")
    print("=" * 100)

    print(
        preview(plan_prompt)
    )

    assert (
        case.example.prompt
        in plan_prompt
    )

    assert (
        case.initial_code
        in plan_prompt
    )

    assert (
        case.feedback.feedback_text
        in plan_prompt
    )

    assert (
        "[Revised Solution Plan]"
        in plan_prompt
    )

    print()
    print(
        "[PASS] Self-replan prompt looks valid."
    )

    # ------------------------------------------------------------------
    # 4. 전체 strategy 실행
    # ------------------------------------------------------------------

    output = strategy.run(
        case
    )

    # ------------------------------------------------------------------
    # 5. Revised Plan 검사
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("[Generated Revised Plan]")
    print("=" * 100)

    print(
        output.self_replan
    )

    assert output.self_replan

    assert (
        output.self_replan.startswith("- ")
    )

    print()
    print(
        "[PASS] Revised plan generated."
    )

    # ------------------------------------------------------------------
    # 6. 최종 Code Prompt 검사
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("[Code Regeneration Prompt]")
    print("=" * 100)

    print(
        preview(
            output.formatted_prompt
        )
    )

    assert (
        output.self_replan
        in output.formatted_prompt
    )

    assert (
        case.example.prompt
        in output.formatted_prompt
    )

    assert (
        case.initial_code
        in output.formatted_prompt
    )

    assert (
        case.feedback.feedback_text
        in output.formatted_prompt
    )

    assert (
        "[Revised Solution Plan]"
        in output.formatted_prompt
    )

    print()
    print(
        "[PASS] Revised plan is injected "
        "into code prompt."
    )

    # ------------------------------------------------------------------
    # 7. RefinementOutput 검사
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("[RefinementOutput]")
    print("=" * 100)

    print(
        "problem_id        :",
        output.problem_id,
    )

    print(
        "strategy          :",
        output.strategy,
    )

    print(
        "prompt_tokens     :",
        output.prompt_tokens,
    )

    print(
        "completion_tokens :",
        output.completion_tokens,
    )

    print(
        "generation_time   :",
        output.generation_time,
    )

    print(
        "strategy_trace len:",
        len(output.strategy_trace),
    )

    print()
    print("[raw_output]")
    print(
        output.raw_output
    )

    # ------------------------------------------------------------------
    # 8. 비용 합산 검증
    # ------------------------------------------------------------------

    # Mock:
    # plan = 200 prompt + 60 completion + 0.10 sec
    # code = 350 prompt + 100 completion + 0.20 sec

    assert output.prompt_tokens == 550
    assert output.completion_tokens == 160

    assert abs(
        output.generation_time - 0.30
    ) < 1e-9

    print()
    print(
        "[PASS] Generation costs are "
        "correctly accumulated."
    )

    # ------------------------------------------------------------------
    # 9. strategy_trace 검사
    # ------------------------------------------------------------------

    assert len(
        output.strategy_trace
    ) == 2

    plan_step = (
        output.strategy_trace[0]
    )

    code_step = (
        output.strategy_trace[1]
    )

    assert (
        plan_step.name
        == "self_replan"
    )

    assert (
        code_step.name
        == "code_regeneration"
    )

    assert (
        plan_step.raw_output
        == output.self_replan
    )

    assert (
        code_step.raw_output
        == output.raw_output
    )

    print()
    print(
        "[PASS] strategy_trace contains "
        "self_replan -> code_regeneration."
    )

    # ------------------------------------------------------------------
    # 10. Strategy-specific field 검사
    # ------------------------------------------------------------------

    assert (
        output.strategy
        == "self_replan"
    )

    assert (
        output.teacher_replan is None
    )

    assert (
        output.teacher_replan_source is None
    )

    assert (
        output.teacher_replan_version is None
    )

    assert (
        output.teacher_replan_verified is None
    )

    # Generator는 정확히 2번 호출되어야 한다.
    assert generator.call_count == 2

    print()
    print("=" * 100)
    print(
        "[SUCCESS] SelfReplanStrategy "
        "sanity check passed."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()