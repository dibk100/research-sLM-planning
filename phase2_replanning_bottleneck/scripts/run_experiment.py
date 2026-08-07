"""
Phase 2 experiment runner.

현재 지원 strategy:
    - feedback_regeneration

Flow:
    config
        -> Phase1FailureLoader
        -> FailureCase
        -> Strategy
        -> ModelGenerator
        -> CodeExtractor
        -> Evaluator
        -> RefinementRecord
        -> results.jsonl

특징:
    - Phase 1 Direct 실패 trajectory만 refinement
    - JSONL streaming
    - 문제 단위 즉시 저장
    - resume 지원
    - code extraction failure도 결과로 기록

Usage:
    PYTHONPATH=. python scripts/run_experiment.py \
        --config configs/feedback_regeneration.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from src.datasets.phase1_failure_loader import (
    Phase1FailureLoader,
)
from src.execution.code_extractor import (
    CodeExtractionError,
    CodeExtractor,
)
from src.execution.evaluator import (
    Evaluator,
)
from src.models.generator import (
    ModelGenerator,
)
from src.schemas import (
    RefinementRecord,
)
from src.strategies.feedback_regeneration import (
    FeedbackRegenerationStrategy,
)

from src.strategies.self_replan import (
    SelfReplanStrategy,
)

from src.plans.teacher_replan_store import (
    TeacherReplanStore,
)

from src.strategies.teacher_replan import (
    TeacherReplanStrategy,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 refinement experiment."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Experiment YAML config path.",
    )

    return parser.parse_args()


def load_config(
    path: str | Path,
) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid config: {path}"
        )

    return config


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def load_completed_problem_ids(
    results_path: Path,
) -> set[str]:
    """
    기존 results.jsonl에서 완료된 problem_id를 읽는다.

    results.jsonl 전체를 메모리에 올리지 않고
    problem_id만 set으로 유지한다.
    """

    completed: set[str] = set()

    if not results_path.exists():
        return completed

    with results_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid existing results.jsonl at "
                    f"line {line_number}: {results_path}"
                ) from error

            problem_id = record.get(
                "problem_id"
            )

            if problem_id is not None:
                completed.add(
                    str(problem_id)
                )

    return completed


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------


def append_jsonl(
    path: Path,
    record: dict[str, Any],
) -> None:
    """
    한 record를 즉시 JSONL에 append한다.
    """

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        json.dump(
            record,
            file,
            ensure_ascii=False,
        )
        file.write("\n")
        file.flush()


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
def build_strategy(
    *,
    config: dict[str, Any],
    generator: ModelGenerator,
):
    strategy_config = config["strategy"]

    strategy_name = strategy_config["name"]

    generation_config = config.get(
        "generation",
        {},
    )

    # ---------------------------------------------------------------
    # Phase 2-1. Feedback-based Regeneration
    # ---------------------------------------------------------------

    if strategy_name == "feedback_regeneration":
        return FeedbackRegenerationStrategy(
            generator=generator,

            prompt_path=(
                strategy_config["prompt_path"]
            ),

            system_prompt=(
                strategy_config.get(
                    "system_prompt"
                )
            ),

            max_new_tokens=int(
                generation_config.get(
                    "max_new_tokens",
                    1024,
                )
            ),

            temperature=float(
                generation_config.get(
                    "temperature",
                    0.0,
                )
            ),

            top_p=float(
                generation_config.get(
                    "top_p",
                    1.0,
                )
            ),
        )

    # ---------------------------------------------------------------
    # Phase 2-2. Self-Replanning Regeneration
    # ---------------------------------------------------------------

    if strategy_name == "self_replan":
        return SelfReplanStrategy(
            generator=generator,

            plan_prompt_path=(
                strategy_config[
                    "plan_prompt_path"
                ]
            ),

            code_prompt_path=(
                strategy_config[
                    "code_prompt_path"
                ]
            ),

            system_prompt=(
                strategy_config.get(
                    "system_prompt"
                )
            ),

            plan_max_new_tokens=int(
                generation_config.get(
                    "plan_max_new_tokens",
                    512,
                )
            ),

            code_max_new_tokens=int(
                generation_config.get(
                    "code_max_new_tokens",
                    1024,
                )
            ),

            temperature=float(
                generation_config.get(
                    "temperature",
                    0.0,
                )
            ),

            top_p=float(
                generation_config.get(
                    "top_p",
                    1.0,
                )
            ),
        )

    # ---------------------------------------------------------------
    # Phase 2-3. Teacher-Replanning Regeneration
    # ---------------------------------------------------------------

    if strategy_name == "teacher_replan":
        replan_store = TeacherReplanStore(
            strategy_config[
                "replan_path"
            ],

            require_verified=bool(
                strategy_config.get(
                    "require_verified",
                    True,
                )
            ),
        )

        return TeacherReplanStrategy(
            generator=generator,

            replan_store=replan_store,

            code_prompt_path=(
                strategy_config[
                    "code_prompt_path"
                ]
            ),

            system_prompt=(
                strategy_config.get(
                    "system_prompt"
                )
            ),

            code_max_new_tokens=int(
                generation_config.get(
                    "code_max_new_tokens",
                    1024,
                )
            ),

            temperature=float(
                generation_config.get(
                    "temperature",
                    0.0,
                )
            ),

            top_p=float(
                generation_config.get(
                    "top_p",
                    1.0,
                )
            ),
        )

    raise ValueError(
        f"Unsupported strategy: {strategy_name}"
    )

# ---------------------------------------------------------------------------
# RefinementRecord
# ---------------------------------------------------------------------------


def build_refinement_record(
    *,
    case,
    refinement,
    refined_code: str,
    evaluation,
    config: dict[str, Any],
) -> RefinementRecord:
    """
    generation + evaluation 결과를 RefinementRecord로 변환한다.
    """

    experiment_config = config["experiment"]
    model_config = config["model"]

    recovered = bool(
        evaluation.passed
    )

    test_pass_delta = (
        evaluation.passed_tests
        - case.initial_passed_tests
    )

    return RefinementRecord(
        # -----------------------------------------------------------
        # Experiment identity
        # -----------------------------------------------------------

        problem_id=case.example.problem_id,

        dataset=case.example.source,

        strategy=refinement.strategy,

        model_name=model_config[
            "name_or_path"
        ],

        seed=int(
            experiment_config["seed"]
        ),

        # -----------------------------------------------------------
        # Problem metadata
        # -----------------------------------------------------------

        title=case.example.title,
        platform=case.example.platform,
        contest_id=case.example.contest_id,
        contest_date=case.example.contest_date,
        difficulty=case.example.difficulty,

        problem=case.example.prompt,

        # -----------------------------------------------------------
        # Initial trajectory
        # -----------------------------------------------------------

        initial_code=case.initial_code,
        initial_status=case.initial_status,

        # FailureCase loader는 실패만 반환한다.
        initial_passed=False,

        initial_passed_tests=(
            case.initial_passed_tests
        ),

        initial_total_tests=(
            case.initial_total_tests
        ),

        initial_error_message=(
            case.feedback.error_message
        ),

        # -----------------------------------------------------------
        # Refinement input
        # -----------------------------------------------------------

        feedback_text=(
            case.feedback.feedback_text
        ),

        # -----------------------------------------------------------
        # Refinement generation
        # -----------------------------------------------------------

        formatted_prompt=(
            refinement.formatted_prompt
        ),

        raw_output=(
            refinement.raw_output
        ),

        refined_code=refined_code,

        prompt_tokens=(
            refinement.prompt_tokens
        ),

        completion_tokens=(
            refinement.completion_tokens
        ),

        generation_time=(
            refinement.generation_time
        ),

        # -----------------------------------------------------------
        # Refined evaluation
        # -----------------------------------------------------------

        refined_passed=(
            evaluation.passed
        ),

        refined_status=(
            evaluation.status
        ),

        refined_passed_tests=(
            evaluation.passed_tests
        ),

        refined_total_tests=(
            evaluation.total_tests
        ),

        execution_time=(
            evaluation.execution_time
        ),

        # -----------------------------------------------------------
        # Analysis
        # -----------------------------------------------------------

        recovered=recovered,

        test_pass_delta=test_pass_delta,

        refined_error_message=(
            evaluation.error_message
        ),

        test_results=[],

        strategy_trace=[
            asdict(step)
            for step
            in refinement.strategy_trace
        ],

        self_replan=(
            refinement.self_replan
        ),

        teacher_replan=(
            refinement.teacher_replan
        ),

        teacher_replan_source=(
            refinement.teacher_replan_source
        ),

        teacher_replan_version=(
            refinement.teacher_replan_version
        ),

        teacher_replan_verified=(
            refinement.teacher_replan_verified
        ),
    )


# ---------------------------------------------------------------------------
# Extraction failure record
# ---------------------------------------------------------------------------


def build_extraction_failure_record(
    *,
    case,
    refinement,
    extraction_error: Exception,
    config: dict[str, Any],
) -> RefinementRecord:
    """
    모델 generation은 성공했지만 code extraction이 실패한 경우.

    이 경우 evaluator를 실행하지 않고 EXTRACTION_ERROR로 기록한다.
    """

    experiment_config = config["experiment"]
    model_config = config["model"]

    return RefinementRecord(
        problem_id=case.example.problem_id,
        dataset=case.example.source,
        strategy=refinement.strategy,
        model_name=model_config[
            "name_or_path"
        ],
        seed=int(
            experiment_config["seed"]
        ),

        title=case.example.title,
        platform=case.example.platform,
        contest_id=case.example.contest_id,
        contest_date=case.example.contest_date,
        difficulty=case.example.difficulty,
        problem=case.example.prompt,

        initial_code=case.initial_code,
        initial_status=case.initial_status,
        initial_passed=False,
        initial_passed_tests=(
            case.initial_passed_tests
        ),
        initial_total_tests=(
            case.initial_total_tests
        ),
        initial_error_message=(
            case.feedback.error_message
        ),

        feedback_text=(
            case.feedback.feedback_text
        ),

        formatted_prompt=(
            refinement.formatted_prompt
        ),

        raw_output=(
            refinement.raw_output
        ),

        refined_code="",

        prompt_tokens=(
            refinement.prompt_tokens
        ),

        completion_tokens=(
            refinement.completion_tokens
        ),

        generation_time=(
            refinement.generation_time
        ),

        refined_passed=False,
        refined_status="EXTRACTION_ERROR",
        refined_passed_tests=0,

        # 실제 evaluator가 돌지 않았으므로 0으로 기록.
        refined_total_tests=0,

        execution_time=0.0,

        recovered=False,

        test_pass_delta=(
            -case.initial_passed_tests
        ),

        refined_error_message=str(
            extraction_error
        ),

        test_results=[],

        strategy_trace=[
            asdict(step)
            for step
            in refinement.strategy_trace
        ],

        self_replan=(
            refinement.self_replan
        ),

        teacher_replan=(
            refinement.teacher_replan
        ),

        teacher_replan_source=(
            refinement.teacher_replan_source
        ),

        teacher_replan_version=(
            refinement.teacher_replan_version
        ),

        teacher_replan_verified=(
            refinement.teacher_replan_verified
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    config = load_config(
        config_path
    )

    # ---------------------------------------------------------------
    # Experiment config
    # ---------------------------------------------------------------

    experiment_config = config[
        "experiment"
    ]

    seed = int(
        experiment_config.get(
            "seed",
            42,
        )
    )

    set_seed(seed)

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------

    output_config = config["output"]

    results_path = Path(
        output_config["path"]
    )

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 사용 config snapshot 저장
    run_config_path = (
        results_path.parent
        / "run_config.yaml"
    )

    shutil.copy2(
        config_path,
        run_config_path,
    )

    # ---------------------------------------------------------------
    # Resume
    # ---------------------------------------------------------------

    resume = bool(
        output_config.get(
            "resume",
            True,
        )
    )

    if resume:
        completed_problem_ids = (
            load_completed_problem_ids(
                results_path
            )
        )
    else:
        completed_problem_ids = set()

        if results_path.exists():
            raise FileExistsError(
                "results.jsonl already exists and "
                f"resume=false: {results_path}"
            )

    print("=" * 100)
    print("Phase 2 Experiment")
    print("=" * 100)

    print(
        "experiment :",
        experiment_config["name"],
    )

    print(
        "strategy   :",
        config["strategy"]["name"],
    )

    print(
        "model      :",
        config["model"]["name_or_path"],
    )

    print(
        "input      :",
        config["input"]["phase1_results_path"],
    )

    print(
        "output     :",
        results_path,
    )

    print(
        "resume     :",
        resume,
    )

    print(
        "completed  :",
        len(completed_problem_ids),
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    model_config = config["model"]

    generator = ModelGenerator(
        model_name_or_path=(
            model_config["name_or_path"]
        ),

        dtype=model_config.get(
            "dtype",
            "bfloat16",
        ),

        device_map=model_config.get(
            "device_map",
            "auto",
        ),
    )

    # ---------------------------------------------------------------
    # Strategy
    # ---------------------------------------------------------------

    strategy = build_strategy(
        config=config,
        generator=generator,
    )

    extractor = CodeExtractor()

    # ---------------------------------------------------------------
    # Evaluator
    # ---------------------------------------------------------------

    evaluation_config = config.get(
        "evaluation",
        {},
    )

    evaluator = Evaluator(
        timeout_seconds=float(
            evaluation_config.get(
                "timeout_seconds",
                5.0,
            )
        ),

        include_public_tests=True,

        # Phase 1 test_results를 모두 public_tests로 복원했음.
        include_private_tests=False,
    )

    # ---------------------------------------------------------------
    # Failure loader
    # ---------------------------------------------------------------

    input_config = config["input"]

    loader = Phase1FailureLoader(
        input_config[
            "phase1_results_path"
        ],

        limit=input_config.get(
            "limit"
        ),

        difficulties=input_config.get(
            "difficulties"
        ),

        max_feedback_chars=int(
            input_config.get(
                "max_feedback_chars",
                2000,
            )
        ),

        include_statuses=(
            set(
                input_config[
                    "include_statuses"
                ]
            )
            if input_config.get(
                "include_statuses"
            )
            else None
        ),
    )

    # ---------------------------------------------------------------
    # Experiment loop
    # ---------------------------------------------------------------

    processed = 0
    skipped = 0
    recovered_count = 0
    extraction_errors = 0

    experiment_start = (
        time.perf_counter()
    )

    for case in loader.load():
        problem_id = (
            case.example.problem_id
        )

        # -----------------------------------------------------------
        # Resume skip
        # -----------------------------------------------------------

        if (
            problem_id
            in completed_problem_ids
        ):
            skipped += 1
            continue

        processed += 1

        print()
        print("-" * 100)

        print(
            f"[{processed}] "
            f"{problem_id} | "
            f"{case.example.difficulty} | "
            f"{case.initial_status} | "
            f"{case.initial_passed_tests}/"
            f"{case.initial_total_tests}"
        )

        # -----------------------------------------------------------
        # Generation
        # -----------------------------------------------------------

        refinement = strategy.run(
            case
        )

        # -----------------------------------------------------------
        # Code extraction
        # -----------------------------------------------------------

        try:
            refined_code = (
                extractor.extract(
                    refinement.raw_output
                )
            )

        except CodeExtractionError as error:
            extraction_errors += 1

            record = (
                build_extraction_failure_record(
                    case=case,
                    refinement=refinement,
                    extraction_error=error,
                    config=config,
                )
            )

            append_jsonl(
                results_path,
                record.to_dict(),
            )

            print(
                "result      : EXTRACTION_ERROR"
            )

            continue

        # -----------------------------------------------------------
        # Evaluation
        # -----------------------------------------------------------

        evaluation = evaluator.evaluate(
            example=case.example,
            code=refined_code,
        )

        # Phase 1에서 복원한 test 수와
        # Phase 2 evaluation test 수가 같아야 한다.
        if (
            evaluation.total_tests > 0
            and evaluation.total_tests
            != case.initial_total_tests
        ):
            raise RuntimeError(
                "Test count mismatch for "
                f"{problem_id}: "
                f"initial={case.initial_total_tests}, "
                f"refined={evaluation.total_tests}"
            )

        # -----------------------------------------------------------
        # Record
        # -----------------------------------------------------------

        record = build_refinement_record(
            case=case,
            refinement=refinement,
            refined_code=refined_code,
            evaluation=evaluation,
            config=config,
        )

        append_jsonl(
            results_path,
            record.to_dict(),
        )

        # -----------------------------------------------------------
        # Progress
        # -----------------------------------------------------------

        if record.recovered:
            recovered_count += 1

        print(
            "refined     :",
            record.refined_status,
        )

        print(
            "tests       :",
            (
                f"{record.initial_passed_tests}/"
                f"{record.initial_total_tests}"
                " -> "
                f"{record.refined_passed_tests}/"
                f"{record.refined_total_tests}"
            ),
        )

        print(
            "delta       :",
            record.test_pass_delta,
        )

        print(
            "recovered   :",
            record.recovered,
        )

        print(
            "generation  :",
            f"{record.generation_time:.3f}s",
        )

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - experiment_start
    )

    print()
    print("=" * 100)
    print("Experiment Finished")
    print("=" * 100)

    print(
        "processed         :",
        processed,
    )

    print(
        "skipped(resume)   :",
        skipped,
    )

    print(
        "recovered         :",
        recovered_count,
    )

    print(
        "extraction_errors :",
        extraction_errors,
    )

    if processed > 0:
        print(
            "recovery_rate     :",
            (
                f"{recovered_count / processed:.4f}"
            ),
        )

    print(
        "elapsed_seconds   :",
        f"{elapsed:.2f}",
    )

    print(
        "results           :",
        results_path,
    )


if __name__ == "__main__":
    main()