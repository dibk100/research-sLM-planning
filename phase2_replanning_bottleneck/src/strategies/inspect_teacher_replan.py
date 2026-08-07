"""
TeacherReplanStrategy sanity check.

확인 항목:
1. Phase1FailureLoader에서 FailureCase 1개를 정상 로드하는가?
2. TeacherReplanStore에서 해당 failure의 teacher re-plan을 조회하는가?
3. teacher re-plan이 code prompt에 정상 삽입되는가?
4. MockGenerator가 정확히 1번 호출되는가?
5. RefinementOutput에 teacher provenance가 정상 저장되는가?
6. strategy_trace가 code_regeneration 1-step으로 저장되는가?

Usage:
    PYTHONPATH=. python -m src.strategies.inspect_teacher_replan
"""

from __future__ import annotations

from src.datasets.phase1_failure_loader import (
    Phase1FailureLoader,
)
from src.plans.teacher_replan_store import (
    TeacherReplanStore,
)
from src.schemas import (
    GenerationOutput,
)
from src.strategies.teacher_replan import (
    TeacherReplanStrategy,
)


PHASE1_RESULTS_PATH = (
    "/mnt/hdd/project_sLM_planning/"
    "output/direct_500_stdin/results.jsonl"
)

TEACHER_REPLAN_PATH = (
    "/mnt/hdd/project_sLM_planning/"
    "data/teacher_plans/"
    "livecodebench_v6_teacher_replans_opus5_v1_seed.jsonl"
)

CODE_PROMPT_PATH = (
    "prompts/teacher_replan_code.txt"
)


class MockGenerator:
    """
    TeacherReplanStrategy 테스트용 mock generator.

    Teacher re-plan은 store에서 이미 제공되므로
    student code generation 1회만 수행한다.
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

        if self.call_count != 1:
            raise RuntimeError(
                "TeacherReplanStrategy should call "
                "the student generator exactly once."
            )

        fake_code = """```python
def can_sort(cards):
    mismatches = [
        i
        for i in range(3)
        if cards[i] != "abc"[i]
    ]

    if len(mismatches) == 0:
        return "YES"

    if len(mismatches) == 2:
        i, j = mismatches

        chars = list(cards)
        chars[i], chars[j] = chars[j], chars[i]

        if "".join(chars) == "abc":
            return "YES"

    return "NO"


t = int(input())

for _ in range(t):
    cards = input().strip()
    print(can_sort(cards))
```"""

        return GenerationOutput(
            text=fake_code,
            prompt_tokens=420,
            completion_tokens=130,
            generation_time=0.25,
        )


def preview(
    text: str,
    max_chars: int = 4000,
) -> str:
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
        PHASE1_RESULTS_PATH,
        limit=1,
    )

    cases = list(
        loader.load()
    )

    assert len(cases) == 1

    case = cases[0]

    print("=" * 100)
    print("TeacherReplanStrategy Sanity Check")
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
    # 2. TeacherReplanStore
    # ------------------------------------------------------------------

    store = TeacherReplanStore(
        TEACHER_REPLAN_PATH,
        require_verified=True,
    )

    entry = store.get_for_failure(
        case
    )

    print()
    print("=" * 100)
    print("[Teacher Re-plan]")
    print("=" * 100)

    print(
        entry.teacher_replan
    )

    print()
    print(
        "teacher_model  :",
        entry.teacher_model,
    )

    print(
        "version        :",
        entry.replan_version,
    )

    print(
        "verified       :",
        entry.verified,
    )

    assert entry.teacher_replan.strip()
    assert entry.verified is True

    print()
    print(
        "[PASS] Teacher re-plan loaded."
    )

    # ------------------------------------------------------------------
    # 3. Strategy 생성
    # ------------------------------------------------------------------

    generator = MockGenerator()

    strategy = TeacherReplanStrategy(
        generator=generator,
        replan_store=store,
        code_prompt_path=CODE_PROMPT_PATH,

        code_max_new_tokens=1024,
        temperature=0.0,
        top_p=1.0,
    )

    # ------------------------------------------------------------------
    # 4. Code prompt 직접 검사
    # ------------------------------------------------------------------

    code_prompt = (
        strategy.build_code_prompt(
            case=case,
            teacher_replan=(
                entry.teacher_replan
            ),
        )
    )

    print()
    print("=" * 100)
    print("[Teacher-Replan Code Prompt]")
    print("=" * 100)

    print(
        preview(code_prompt)
    )

    assert (
        case.example.prompt
        in code_prompt
    )

    assert (
        case.initial_code
        in code_prompt
    )

    assert (
        case.feedback.feedback_text
        in code_prompt
    )

    assert (
        entry.teacher_replan
        in code_prompt
    )

    assert (
        "[Revised Solution Plan]"
        in code_prompt
    )

    print()
    print(
        "[PASS] Teacher re-plan is injected "
        "into code prompt."
    )

    # ------------------------------------------------------------------
    # 5. Strategy 실행
    # ------------------------------------------------------------------

    output = strategy.run(
        case
    )

    # ------------------------------------------------------------------
    # 6. RefinementOutput 확인
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
    # 7. Basic assertions
    # ------------------------------------------------------------------

    assert (
        output.problem_id
        == case.example.problem_id
    )

    assert (
        output.strategy
        == "teacher_replan"
    )

    assert output.raw_output.strip()

    # Teacher strategy는 student generation 1회
    assert output.prompt_tokens == 420
    assert output.completion_tokens == 130

    assert abs(
        output.generation_time - 0.25
    ) < 1e-9

    # ------------------------------------------------------------------
    # 8. Teacher provenance 검사
    # ------------------------------------------------------------------

    assert (
        output.self_replan is None
    )

    assert (
        output.teacher_replan
        == entry.teacher_replan
    )

    assert (
        output.teacher_replan_source
        == entry.teacher_model
    )

    assert (
        output.teacher_replan_version
        == entry.replan_version
    )

    assert (
        output.teacher_replan_verified
        == entry.verified
    )

    print()
    print(
        "[PASS] Teacher provenance fields "
        "are correctly stored."
    )

    # ------------------------------------------------------------------
    # 9. strategy_trace 검사
    # ------------------------------------------------------------------

    assert len(
        output.strategy_trace
    ) == 1

    code_step = (
        output.strategy_trace[0]
    )

    assert (
        code_step.name
        == "code_regeneration"
    )

    assert (
        code_step.formatted_prompt
        == output.formatted_prompt
    )

    assert (
        code_step.raw_output
        == output.raw_output
    )

    assert (
        code_step.prompt_tokens
        == output.prompt_tokens
    )

    assert (
        code_step.completion_tokens
        == output.completion_tokens
    )

    print()
    print(
        "[PASS] strategy_trace contains "
        "one code_regeneration step."
    )

    # ------------------------------------------------------------------
    # 10. Generator call count
    # ------------------------------------------------------------------

    assert generator.call_count == 1

    print()
    print(
        "[PASS] Student generator was "
        "called exactly once."
    )

    # ------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "[SUCCESS] TeacherReplanStrategy "
        "sanity check passed."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()