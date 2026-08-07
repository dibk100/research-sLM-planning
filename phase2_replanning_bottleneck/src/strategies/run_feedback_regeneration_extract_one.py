"""
Feedback-based Regeneration + Code Extraction + Evaluation
실제 모델 1문제 end-to-end sanity check.

Flow:
    Phase 1 FailureCase
        -> FeedbackRegenerationStrategy
        -> raw_output
        -> CodeExtractor
        -> refined_code
        -> Evaluator
        -> EvaluationResult

Usage:
    python -m src.strategies.run_feedback_regeneration_extract_one \
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
from src.strategies.feedback_regeneration import (
    FeedbackRegenerationStrategy,
)


DEFAULT_RESULTS_PATH = (
    "/mnt/hdd/project_sLM_planning/"
    "output/direct_500_stdin/results.jsonl"
)

DEFAULT_PROMPT_PATH = (
    "prompts/feedback_regeneration.txt"
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
        "--prompt-path",
        type=str,
        default=DEFAULT_PROMPT_PATH,
    )

    parser.add_argument(
        "--max-new-tokens",
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
    # 1. Phase 1 failure 1개
    # ---------------------------------------------------------------

    loader = Phase1FailureLoader(
        args.results_path,
        limit=1,
    )

    case = next(loader.load())

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

    generator = ModelGenerator(
        model_name_or_path=args.model,
        dtype=args.dtype,
        device_map="auto",
    )

    # ---------------------------------------------------------------
    # 3. Feedback-based Regeneration
    # ---------------------------------------------------------------

    strategy = FeedbackRegenerationStrategy(
        generator=generator,
        prompt_path=args.prompt_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    output = strategy.run(case)

    # ---------------------------------------------------------------
    # 4. Raw output
    # ---------------------------------------------------------------

    print()
    print("=" * 100)
    print("Raw Model Output")
    print("=" * 100)

    print(output.raw_output)

    # ---------------------------------------------------------------
    # 5. Code extraction
    # ---------------------------------------------------------------

    extractor = CodeExtractor()

    refined_code = extractor.extract(
        output.raw_output
    )

    print()
    print("=" * 100)
    print("Extracted Refined Code")
    print("=" * 100)

    print(refined_code)

    # ---------------------------------------------------------------
    # 6. Refined code evaluation
    # ---------------------------------------------------------------

    evaluator = Evaluator(
        timeout_seconds=5.0,

        # Phase 1 test_results에서 복원한 테스트는
        # 모두 public_tests에 저장되어 있다.
        include_public_tests=True,
        include_private_tests=False,
    )

    evaluation = evaluator.evaluate(
        example=case.example,
        code=refined_code,
    )

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
    # 7. Initial -> Refined comparison
    # ---------------------------------------------------------------

    # loader가 failure만 반환하므로 initial은 항상 FAIL.
    recovered = evaluation.passed

    test_pass_delta = (
        evaluation.passed_tests
        - case.initial_passed_tests
    )

    print()
    print("=" * 100)
    print("Initial -> Refined Comparison")
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
    # 8. Sanity checks
    # ---------------------------------------------------------------

    assert (
        evaluation.total_tests
        == case.initial_total_tests
    ), (
        "Test count mismatch: "
        f"initial={case.initial_total_tests}, "
        f"refined={evaluation.total_tests}"
    )

    assert output.raw_output.strip(), (
        "Model output is empty."
    )

    assert refined_code.strip(), (
        "Extracted refined code is empty."
    )

    print()
    print("=" * 100)
    print(
        "[SUCCESS] Generation + extraction + evaluation completed."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()