"""
Self-Replanning Regeneration end-to-end sanity check.

실제 Qwen2.5-Coder-3B-Instruct를 사용하여:

    Phase 1 FailureCase
        -> SelfReplanStrategy
        -> Revised Plan
        -> Code Regeneration
        -> CodeExtractor
        -> Evaluator

전체 흐름을 1문제에서 검증한다.

Usage:
    PYTHONPATH=. python -m src.strategies.inspect_self_replan_extract_one \
        --model Qwen/Qwen2.5-Coder-3B-Instruct
"""

from __future__ import annotations

import argparse

from src.datasets.phase1_failure_loader import (
    Phase1FailureLoader,
)
from src.execution.code_extractor import (
    CodeExtractor,
)
from src.execution.evaluator import (
    Evaluator,
)
from src.models.generator import (
    ModelGenerator,
)
from src.strategies.self_replan import (
    SelfReplanStrategy,
)


DEFAULT_RESULTS_PATH = (
    "/mnt/hdd/project_sLM_planning/"
    "output/direct_500_stdin/results.jsonl"
)

DEFAULT_PLAN_PROMPT_PATH = (
    "prompts/self_replan_plan.txt"
)

DEFAULT_CODE_PROMPT_PATH = (
    "prompts/self_replan_code.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--results-path",
        type=str,
        default=DEFAULT_RESULTS_PATH,
    )

    parser.add_argument(
        "--plan-prompt-path",
        type=str,
        default=DEFAULT_PLAN_PROMPT_PATH,
    )

    parser.add_argument(
        "--code-prompt-path",
        type=str,
        default=DEFAULT_CODE_PROMPT_PATH,
    )

    parser.add_argument(
        "--plan-max-new-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--code-max-new-tokens",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=[
            "float16",
            "bfloat16",
            "float32",
        ],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---------------------------------------------------------------
    # 1. Phase 1 실패 trajectory 1개 로드
    # ---------------------------------------------------------------

    loader = Phase1FailureLoader(
        args.results_path,
        limit=1,
    )

    case = next(
        loader.load()
    )

    print("=" * 100)
    print("Phase 1 Failure")
    print("=" * 100)

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

    # ---------------------------------------------------------------
    # 2. 실제 모델 로드
    # ---------------------------------------------------------------

    print()
    print("=" * 100)
    print("Loading Model")
    print("=" * 100)

    generator = ModelGenerator(
        model_name_or_path=args.model,
        dtype=args.dtype,
        device_map="auto",
    )

    # ---------------------------------------------------------------
    # 3. Self-Replan Strategy
    # ---------------------------------------------------------------

    strategy = SelfReplanStrategy(
        generator=generator,

        plan_prompt_path=(
            args.plan_prompt_path
        ),

        code_prompt_path=(
            args.code_prompt_path
        ),

        plan_max_new_tokens=(
            args.plan_max_new_tokens
        ),

        code_max_new_tokens=(
            args.code_max_new_tokens
        ),

        temperature=args.temperature,
        top_p=args.top_p,
    )

    # ---------------------------------------------------------------
    # 4. Revised Plan -> Code Generation
    # ---------------------------------------------------------------

    print()
    print("=" * 100)
    print("Running Self-Replanning Regeneration")
    print("=" * 100)

    output = strategy.run(
        case
    )

    # ---------------------------------------------------------------
    # 5. Revised Plan 확인
    # ---------------------------------------------------------------

    print()
    print("=" * 100)
    print("Generated Revised Plan")
    print("=" * 100)

    print(
        output.self_replan
    )

    assert output.self_replan
    assert output.self_replan.strip()

    # ---------------------------------------------------------------
    # 6. Raw Code Output
    # ---------------------------------------------------------------

    print()
    print("=" * 100)
    print("Raw Code Output")
    print("=" * 100)

    print(
        output.raw_output
    )

    assert output.raw_output.strip()

    # ---------------------------------------------------------------
    # 7. Code Extraction
    # ---------------------------------------------------------------

    extractor = CodeExtractor()

    refined_code = extractor.extract(
        output.raw_output
    )

    print()
    print("=" * 100)
    print("Extracted Refined Code")
    print("=" * 100)

    print(
        refined_code
    )

    assert refined_code.strip()
    
    print()
    print("=" * 100)
    print("Loaded Plan Prompt Template")
    print("=" * 100)

    # ---------------------------------------------------------------
    # 8. Evaluation
    # ---------------------------------------------------------------

    evaluator = Evaluator(
        timeout_seconds=5.0,

        # Phase 1 test_results에서 복원한 테스트는
        # 모두 public_tests에 들어 있다.
        include_public_tests=True,
        include_private_tests=False,
    )

    evaluation = evaluator.evaluate(
        example=case.example,
        code=refined_code,
    )

    # ---------------------------------------------------------------
    # 9. Evaluation Result
    # ---------------------------------------------------------------

    print()
    print("=" * 100)
    print("Refined Code Evaluation")
    print("=" * 100)

    print(
        "passed         :",
        evaluation.passed,
    )

    print(
        "status         :",
        evaluation.status,
    )

    print(
        "passed_tests   :",
        (
            f"{evaluation.passed_tests}/"
            f"{evaluation.total_tests}"
        ),
    )

    print(
        "execution_time :",
        f"{evaluation.execution_time:.3f}s",
    )

    print(
        "error_message  :",
        evaluation.error_message,
    )

    # ---------------------------------------------------------------
    # 10. Initial -> Refined 비교
    # ---------------------------------------------------------------

    recovered = (
        evaluation.passed
    )

    test_pass_delta = (
        evaluation.passed_tests
        - case.initial_passed_tests
    )

    print()
    print("=" * 100)
    print("Initial -> Self-Replanned")
    print("=" * 100)

    print(
        "initial status :",
        case.initial_status,
    )

    print(
        "refined status :",
        evaluation.status,
    )

    print(
        "initial tests  :",
        (
            f"{case.initial_passed_tests}/"
            f"{case.initial_total_tests}"
        ),
    )

    print(
        "refined tests  :",
        (
            f"{evaluation.passed_tests}/"
            f"{evaluation.total_tests}"
        ),
    )

    print(
        "test_pass_delta:",
        test_pass_delta,
    )

    print(
        "recovered      :",
        recovered,
    )

    # ---------------------------------------------------------------
    # 11. Strategy cost
    # ---------------------------------------------------------------

    print()
    print("=" * 100)
    print("Self-Replan Generation Cost")
    print("=" * 100)

    print(
        "total prompt tokens     :",
        output.prompt_tokens,
    )

    print(
        "total completion tokens :",
        output.completion_tokens,
    )

    print(
        "total generation time   :",
        f"{output.generation_time:.3f}s",
    )

    print(
        "strategy_trace length   :",
        len(output.strategy_trace),
    )

    # step별 비용도 출력
    for index, step in enumerate(
        output.strategy_trace,
        start=1,
    ):
        print()
        print(
            f"[Step {index}] {step.name}"
        )

        print(
            "prompt_tokens     :",
            step.prompt_tokens,
        )

        print(
            "completion_tokens :",
            step.completion_tokens,
        )

        print(
            "generation_time   :",
            f"{step.generation_time:.3f}s",
        )

    # ---------------------------------------------------------------
    # 12. Sanity assertions
    # ---------------------------------------------------------------

    assert (
        evaluation.total_tests
        == case.initial_total_tests
    ), (
        "Test count mismatch: "
        f"initial={case.initial_total_tests}, "
        f"refined={evaluation.total_tests}"
    )

    assert (
        len(output.strategy_trace)
        == 2
    )

    assert (
        output.strategy_trace[0].name
        == "self_replan"
    )

    assert (
        output.strategy_trace[1].name
        == "code_regeneration"
    )

    assert (
        output.self_replan
        == output.strategy_trace[0].raw_output
    )

    assert (
        output.raw_output
        == output.strategy_trace[1].raw_output
    )

    print()
    print("=" * 100)
    print(
        "[SUCCESS] Self-Replanning end-to-end "
        "sanity check completed."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()