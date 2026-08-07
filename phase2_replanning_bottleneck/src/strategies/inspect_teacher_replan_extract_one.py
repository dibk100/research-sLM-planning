"""
Teacher-Replanning Regeneration end-to-end sanity check.

실제 Qwen2.5-Coder-3B-Instruct를 사용하여:

    Phase 1 FailureCase
        -> TeacherReplanStore
        -> TeacherReplanStrategy
        -> Code Regeneration
        -> CodeExtractor
        -> Evaluator

전체 흐름을 1문제에서 검증한다.

Usage:
    PYTHONPATH=. python -m src.strategies.inspect_teacher_replan_extract_one \
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
from src.plans.teacher_replan_store import (
    TeacherReplanStore,
)
from src.strategies.teacher_replan import (
    TeacherReplanStrategy,
)


DEFAULT_RESULTS_PATH = (
    "/mnt/hdd/project_sLM_planning/"
    "output/direct_500_stdin/results.jsonl"
)

DEFAULT_TEACHER_REPLAN_PATH = (
    "/mnt/hdd/project_sLM_planning/"
    "data/teacher_plans/"
    "livecodebench_v6_teacher_replans_opus5_v1_seed.jsonl"
)

DEFAULT_CODE_PROMPT_PATH = (
    "prompts/teacher_replan_code.txt"
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
        "--teacher-replan-path",
        type=str,
        default=DEFAULT_TEACHER_REPLAN_PATH,
    )

    parser.add_argument(
        "--code-prompt-path",
        type=str,
        default=DEFAULT_CODE_PROMPT_PATH,
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

    # ------------------------------------------------------------------
    # 1. Phase 1 failure 1개 로드
    # ------------------------------------------------------------------

    loader = Phase1FailureLoader(
        args.results_path,
        limit=1,
    )

    cases = list(
        loader.load()
    )

    assert len(cases) == 1

    case = cases[0]

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

    # ------------------------------------------------------------------
    # 2. Teacher re-plan store
    # ------------------------------------------------------------------

    store = TeacherReplanStore(
        args.teacher_replan_path,
        require_verified=True,
    )

    entry = store.get_for_failure(
        case
    )

    print()
    print("=" * 100)
    print("Teacher Revised Plan")
    print("=" * 100)

    print(
        entry.teacher_replan
    )

    print()
    print(
        "teacher_model :",
        entry.teacher_model,
    )

    print(
        "version       :",
        entry.replan_version,
    )

    print(
        "verified      :",
        entry.verified,
    )

    # ------------------------------------------------------------------
    # 3. 실제 Student Model 로드
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("Loading Student Model")
    print("=" * 100)

    generator = ModelGenerator(
        model_name_or_path=args.model,
        dtype=args.dtype,
        device_map="auto",
    )

    # ------------------------------------------------------------------
    # 4. Teacher-Replan Strategy
    # ------------------------------------------------------------------

    strategy = TeacherReplanStrategy(
        generator=generator,
        replan_store=store,

        code_prompt_path=(
            args.code_prompt_path
        ),

        code_max_new_tokens=(
            args.code_max_new_tokens
        ),

        temperature=(
            args.temperature
        ),

        top_p=args.top_p,
    )

    # ------------------------------------------------------------------
    # 5. Teacher Re-plan -> Code Generation
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("Running Teacher-Replanning Regeneration")
    print("=" * 100)

    output = strategy.run(
        case
    )

    # ------------------------------------------------------------------
    # 6. Strategy metadata
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("Teacher-Replan Metadata")
    print("=" * 100)

    print(
        "problem_id       :",
        output.problem_id,
    )

    print(
        "strategy         :",
        output.strategy,
    )

    print(
        "teacher_source   :",
        output.teacher_replan_source,
    )

    print(
        "teacher_version  :",
        output.teacher_replan_version,
    )

    print(
        "teacher_verified :",
        output.teacher_replan_verified,
    )

    # ------------------------------------------------------------------
    # 7. Raw Code Output
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("Raw Code Output")
    print("=" * 100)

    print(
        output.raw_output
    )

    assert output.raw_output.strip()

    # ------------------------------------------------------------------
    # 8. Code Extraction
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 9. Evaluation
    # ------------------------------------------------------------------

    evaluator = Evaluator(
        timeout_seconds=5.0,

        # Phase 1 test_results에서 복원된 테스트는
        # 모두 public_tests에 저장되어 있다.
        include_public_tests=True,
        include_private_tests=False,
    )

    evaluation = evaluator.evaluate(
        example=case.example,
        code=refined_code,
    )

    # ------------------------------------------------------------------
    # 10. Evaluation Result
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 11. Initial -> Refined
    # ------------------------------------------------------------------

    recovered = (
        evaluation.passed
    )

    test_pass_delta = (
        evaluation.passed_tests
        - case.initial_passed_tests
    )

    print()
    print("=" * 100)
    print("Initial -> Teacher-Replanned")
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

    # ------------------------------------------------------------------
    # 12. Student inference cost
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("Teacher-Replan Student Generation Cost")
    print("=" * 100)

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
        f"{output.generation_time:.3f}s",
    )

    print(
        "strategy_trace len:",
        len(output.strategy_trace),
    )

    # Teacher re-plan은 external artifact이므로
    # student strategy trace는 code generation 1-step.
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

    # ------------------------------------------------------------------
    # 13. Sanity assertions
    # ------------------------------------------------------------------

    assert (
        evaluation.total_tests
        == case.initial_total_tests
    ), (
        "Test count mismatch: "
        f"initial={case.initial_total_tests}, "
        f"refined={evaluation.total_tests}"
    )

    assert (
        output.strategy
        == "teacher_replan"
    )

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

    assert (
        len(output.strategy_trace)
        == 1
    )

    assert (
        output.strategy_trace[0].name
        == "code_regeneration"
    )

    assert (
        output.strategy_trace[0].raw_output
        == output.raw_output
    )

    print()
    print("=" * 100)
    print(
        "[SUCCESS] Teacher-Replanning end-to-end "
        "sanity check completed."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()