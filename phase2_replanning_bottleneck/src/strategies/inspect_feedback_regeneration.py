"""
FeedbackRegenerationStrategy sanity check.

확인 항목:
1. Phase1FailureLoader에서 FailureCase 1개를 정상 로드하는가?
2. feedback_regeneration prompt가 정상 구성되는가?
3. Mock generator 호출이 정상 동작하는가?
4. RefinementOutput / strategy_trace가 정상 생성되는가?

Usage:
    python -m src.strategies.inspect_feedback_regeneration
"""

from src.datasets.phase1_failure_loader import Phase1FailureLoader
from src.schemas import GenerationOutput
from src.strategies.feedback_regeneration import (
    FeedbackRegenerationStrategy,
)


RESULTS_PATH = (
    "/mnt/hdd/project_sLM_planning/"
    "output/direct_500_stdin/results.jsonl"
)

PROMPT_PATH = (
    "prompts/feedback_regeneration.txt"
)


class MockGenerator:
    """
    실제 모델을 로딩하지 않는 테스트용 generator.

    FeedbackRegenerationStrategy가 ModelGenerator와 동일한
    generate() interface를 사용할 수 있는지만 확인한다.
    """

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> GenerationOutput:

        print()
        print("[MockGenerator.generate]")
        print(f"system_prompt  : {system_prompt}")
        print(f"max_new_tokens : {max_new_tokens}")
        print(f"temperature    : {temperature}")
        print(f"top_p          : {top_p}")
        print(f"prompt chars   : {len(prompt)}")

        fake_code = """\
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    return 0;
}
"""

        return GenerationOutput(
            text=fake_code,
            prompt_tokens=123,
            completion_tokens=45,
            generation_time=0.01,
        )


def preview(
    text: str,
    max_chars: int = 3000,
) -> str:
    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + f"\n...[preview truncated, total={len(text)} chars]"
    )


def main() -> None:
    # ------------------------------------------------------------------
    # 1. FailureCase 1개 로드
    # ------------------------------------------------------------------

    loader = Phase1FailureLoader(
        RESULTS_PATH,
        limit=1,
    )

    cases = list(loader.load())

    assert len(cases) == 1, (
        f"Expected 1 FailureCase, got {len(cases)}"
    )

    case = cases[0]

    print("=" * 100)
    print("FeedbackRegenerationStrategy Sanity Check")
    print("=" * 100)

    print()
    print("[FailureCase]")
    print("problem_id     :", case.example.problem_id)
    print("title          :", case.example.title)
    print("difficulty     :", case.example.difficulty)
    print("initial_status :", case.initial_status)
    print(
        "initial_tests  :",
        f"{case.initial_passed_tests}/"
        f"{case.initial_total_tests}",
    )

    # ------------------------------------------------------------------
    # 2. Strategy 생성
    # ------------------------------------------------------------------

    mock_generator = MockGenerator()

    strategy = FeedbackRegenerationStrategy(
        generator=mock_generator,
        prompt_path=PROMPT_PATH,
        max_new_tokens=1024,
        temperature=0.0,
        top_p=1.0,
    )

    # ------------------------------------------------------------------
    # 3. build_prompt() 검사
    # ------------------------------------------------------------------

    formatted_prompt = strategy.build_prompt(
        case
    )

    print()
    print("=" * 100)
    print("[Formatted Prompt]")
    print("=" * 100)
    print(
        preview(
            formatted_prompt,
            max_chars=4000,
        )
    )

    # 필수 요소 검사
    assert case.example.prompt in formatted_prompt
    assert case.initial_code in formatted_prompt
    assert (
        case.feedback.feedback_text
        in formatted_prompt
    )

    assert "[Problem]" in formatted_prompt
    assert "[Previous Solution]" in formatted_prompt
    assert "[Execution Feedback]" in formatted_prompt
    assert "[Corrected Solution]" in formatted_prompt

    print()
    print(
        "[PASS] Prompt contains problem, previous code, and feedback."
    )

    # ------------------------------------------------------------------
    # 4. 실제 strategy.run() 실행
    # ------------------------------------------------------------------

    output = strategy.run(
        case
    )

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
    print(output.raw_output)

    # ------------------------------------------------------------------
    # 5. RefinementOutput validation
    # ------------------------------------------------------------------

    assert (
        output.problem_id
        == case.example.problem_id
    )

    assert (
        output.strategy
        == "feedback_regeneration"
    )

    assert (
        output.formatted_prompt
        == formatted_prompt
    )

    assert output.raw_output

    assert output.prompt_tokens == 123
    assert output.completion_tokens == 45

    assert len(output.strategy_trace) == 1

    step = output.strategy_trace[0]

    assert (
        step.name
        == "feedback_regeneration"
    )

    assert (
        step.formatted_prompt
        == formatted_prompt
    )

    assert (
        step.raw_output
        == output.raw_output
    )

    assert (
        output.self_replan is None
    )

    assert (
        output.teacher_replan is None
    )

    print()
    print(
        "[PASS] RefinementOutput looks valid."
    )

    print()
    print("=" * 100)
    print(
        "[SUCCESS] FeedbackRegenerationStrategy sanity check passed."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()