"""
Phase 3-B: Code Best-of-N Coverage Control.

PYTHONPATH=. python -m scripts.run_code_best_of_n \
  --config configs/qwen25_coder_3b.yaml

연구 질문:
    고정된 self-generated plan에서 code만 여러 번 sampling할 때
    solution coverage가 얼마나 증가하는가?

Phase 3-A:
    stochastic plan x N
        -> greedy code

Phase 3-B:
    fixed Phase-1 Self-Plan
        -> stochastic code x N

고정:
- Dataset / problem IDs
- Model
- Phase 1 self-plan
- Code prompt
- Evaluator
- Timeout
- N

변경:
- Code generation만 stochastic sampling
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.common.datasets.dataset_loader import DatasetLoader
from src.common.execution.code_extractor import CodeExtractor
from src.common.execution.evaluator import Evaluator
from src.common.models.generator import ModelGenerator
from src.common.utils.config import load_config
from src.execute import CandidateExecutor
from src.generate_code import (
    CodeSample,
    FixedPlanCodeSampler,
)
from src.load_fixed_plans import FixedPlanLoader
from src.prompts import FixedPlanCodePromptBuilder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 3-B fixed-plan Code Best-of-N experiment."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to Phase 3-B YAML config.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override dataset limit.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override number of code samples.",
    )

    parser.add_argument(
        "--output-path",
        default=None,
        help="Override results.jsonl path.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def get_nested(
    data: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    current: Any = data

    for key in keys:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL: "
                    f"{path}:{line_number}"
                ) from error

    return records


def completed_problem_ids(
    path: Path,
) -> set[str]:
    return {
        str(record["problem_id"])
        for record in read_jsonl(path)
    }


def append_jsonl(
    path: Path,
    record: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def save_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
        )


def prefix_ks(
    num_samples: int,
) -> list[int]:
    ks: list[int] = []

    k = 1

    while k <= num_samples:
        ks.append(k)
        k *= 2

    if ks[-1] != num_samples:
        ks.append(num_samples)

    return ks


# ---------------------------------------------------------------------------
# Candidate execution
# ---------------------------------------------------------------------------


def run_code_candidate(
    *,
    example: Any,
    fixed_plan: str,
    sample_id: int,
    sampler: FixedPlanCodeSampler,
    executor: CandidateExecutor,
    store_prompts: bool,
) -> dict[str, Any]:
    """
    fixed plan으로부터 code candidate 하나를 생성하고 평가한다.
    """

    try:
        code_sample: CodeSample = (
            sampler.sample_one(
                example=example,
                fixed_plan=fixed_plan,
                sample_id=sample_id,
            )
        )

    except Exception as error:
        outcome = executor.skipped(
            status="CODE_GENERATION_ERROR",
            error_message=str(error),
        )

        return {
            "sample_id": sample_id,
            "sample_seed": None,

            "raw_output": "",
            "code": "",

            "passed": outcome.passed,
            "status": outcome.status,

            "passed_tests": 0,
            "total_tests": 0,
            "test_pass_ratio": 0.0,

            "prompt_tokens": 0,
            "completion_tokens": 0,

            "generation_time": 0.0,
            "execution_time": 0.0,

            "plan_in_code_prompt": False,

            "error_message": (
                outcome.error_message
            ),
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    try:
        outcome = executor.run(
            example=example,
            raw_output=code_sample.raw_output,
        )

    except Exception as error:
        outcome = executor.skipped(
            status="EVALUATION_ERROR",
            error_message=str(error),
        )

    record: dict[str, Any] = {
        "sample_id": (
            code_sample.sample_id
        ),
        "sample_seed": (
            code_sample.sample_seed
        ),

        "raw_output": (
            code_sample.raw_output
        ),

        "extracted_code": outcome.extracted_code,

        "passed": (
            outcome.passed
        ),
        "status": (
            outcome.status
        ),

        # Phase 3-A execute.py schema에 따라 우선 지원
        "passed_tests": getattr(
            outcome,
            "passed_tests",
            getattr(
                outcome,
                "num_passed",
                0,
            ),
        ),
        "total_tests": getattr(
            outcome,
            "total_tests",
            getattr(
                outcome,
                "num_tests",
                0,
            ),
        ),

        "test_pass_ratio": (
            outcome.test_pass_ratio
        ),

        "prompt_tokens": (
            code_sample.prompt_tokens
        ),
        "completion_tokens": (
            code_sample.completion_tokens
        ),

        "generation_time": (
            code_sample.generation_time
        ),

        "execution_time": getattr(
            outcome,
            "execution_time",
            getattr(
                outcome,
                "execution_seconds",
                0.0,
            ),
        ),

        "plan_in_code_prompt": (
            code_sample.plan_in_code_prompt
        ),

        "error_message": (
            outcome.error_message
        ),
    }

    if store_prompts:
        record["code_prompt"] = (
            code_sample.code_prompt
        )

    if getattr(
        outcome,
        "test_results",
        None,
    ):
        record["test_results"] = (
            outcome.test_results
        )

    return record


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize_candidates(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    num_passed = sum(
        bool(candidate["passed"])
        for candidate in candidates
    )

    best_ratio = max(
        (
            float(
                candidate[
                    "test_pass_ratio"
                ]
            )
            for candidate in candidates
        ),
        default=0.0,
    )

    distinct_codes = len(
        {
            candidate["code"].strip()
            for candidate in candidates
            if candidate.get(
                "code",
                "",
            ).strip()
        }
    )

    generation_time = sum(
        float(
            candidate[
                "generation_time"
            ]
        )
        for candidate in candidates
    )

    return {
        "num_passed": num_passed,
        "oracle_passed": (
            num_passed > 0
        ),
        "best_test_pass_ratio": (
            best_ratio
        ),
        "distinct_codes": (
            distinct_codes
        ),
        "total_generation_time": (
            generation_time
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    config_path = Path(
        args.config
    )

    config = load_config(
        config_path
    )

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    experiment_config = config[
        "experiment"
    ]

    dataset_config = config[
        "dataset"
    ]

    model_config = config[
        "model"
    ]

    sampling_config = config[
        "sampling"
    ]

    generation_config = config[
        "generation"
    ]

    strategy_config = config[
        "strategy"
    ]

    evaluation_config = config[
        "evaluation"
    ]

    output_config = config[
        "output"
    ]

    experiment_name = str(
        experiment_config["name"]
    )

    base_seed = int(
        experiment_config["seed"]
    )

    limit = int(
        args.limit
        if args.limit is not None
        else dataset_config["limit"]
    )

    num_samples = int(
        args.num_samples
        if args.num_samples is not None
        else sampling_config[
            "num_samples"
        ]
    )

    code_config = (
        generation_config["code"]
        if "code" in generation_config
        else generation_config
    )

    code_max_new_tokens = int(
        code_config["max_new_tokens"]
    )

    code_temperature = float(
        code_config["temperature"]
    )

    code_top_p = float(
        code_config["top_p"]
    )

    # ------------------------------------------------------------------
    # Phase 3-B protocol checks
    # ------------------------------------------------------------------

    if num_samples <= 0:
        raise ValueError(
            "num_samples must be greater than 0."
        )

    if code_temperature <= 0.0:
        raise ValueError(
            "Phase 3-B requires stochastic "
            "code generation "
            "(temperature > 0)."
        )

    if not 0.0 < code_top_p <= 1.0:
        raise ValueError(
            f"top_p must be in (0, 1], "
            f"got {code_top_p}."
        )

    # ------------------------------------------------------------------
    # Source Phase 1 results
    # ------------------------------------------------------------------

    fixed_plan_results = Path(
        strategy_config[
            "fixed_plan_results_path"
        ]
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    output_path = Path(
        args.output_path
        if args.output_path
        else output_config["path"]
    )

    resume = (
        False
        if args.no_resume
        else bool(
            output_config.get(
                "resume",
                True,
            )
        )
    )

    store_prompts = bool(
        output_config.get(
            "store_prompts",
            False,
        )
    )

    store_test_results = bool(
        output_config.get(
            "store_test_results",
            False,
        )
    )

    # Existing output + no resume = dangerous.
    if output_path.exists() and not resume:
        raise FileExistsError(
            f"Output already exists: "
            f"{output_path}\n"
            "Use resume=True or choose "
            "another output path."
        )

    # ------------------------------------------------------------------
    # Print experiment setup
    # ------------------------------------------------------------------

    print("=" * 80)
    print(
        "Phase3-B Fixed-Plan Code Best-of-N Experiment"
    )
    print("=" * 80)

    print(
        f"Experiment    : "
        f"{experiment_name}"
    )
    print(
        f"Dataset       : "
        f"{dataset_config['name']}"
    )
    print(
        f"Model         : "
        f"{model_config['name_or_path']}"
    )
    print(
        f"Seed          : "
        f"{base_seed}"
    )
    print(
        f"Limit         : "
        f"{limit}"
    )
    print(
        f"N (samples)   : "
        f"{num_samples}"
    )
    print(
        f"Fixed plans   : "
        f"{fixed_plan_results}"
    )
    print(
        "Code sample   : "
        f"temperature={code_temperature}, "
        f"top_p={code_top_p}, "
        f"max_new_tokens="
        f"{code_max_new_tokens}"
    )
    print(
        f"Output        : "
        f"{output_path}"
    )
    print(
        f"Resume        : "
        f"{resume}"
    )
    print(
        f"Store prompts : "
        f"{store_prompts}"
    )
    print(
        "Store tests   : "
        f"{store_test_results}"
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    dataset_loader = DatasetLoader(
        dataset_name=(
            dataset_config["name"]
        ),
        split=dataset_config.get(
            "split",
            "test",
        ),
        limit=limit,
        test_type=dataset_config.get(
            "test_type",
            "stdin",
        ),
        release_version=(
            dataset_config.get(
                "release_version",
            )
        ),
    )

    examples = dataset_loader.load()

    if len(examples) != limit:
        raise ValueError(
            f"Expected {limit} problems, "
            f"loaded {len(examples)}."
        )

    # ------------------------------------------------------------------
    # Fixed plans
    # ------------------------------------------------------------------

    fixed_plan_loader = FixedPlanLoader(
        fixed_plan_results
    )

    validation = (
        fixed_plan_loader.validate_examples(
            examples,
            require_exact_match=(
                len(examples)
                == len(
                    fixed_plan_loader
                )
            ),
        )
    )

    fixed_plan_loader.validate_sequence(
        examples
    )

    print()
    print(
        "[Fixed Plans] "
        f"loaded={len(fixed_plan_loader)}, "
        f"matched={validation['matched']}, "
        f"missing={validation['missing_plans']}, "
        f"empty={validation['empty_plans']}"
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    generator = ModelGenerator(
        model_name_or_path=(
            model_config[
                "name_or_path"
            ]
        ),
        dtype=model_config.get(
            "dtype",
            "bfloat16",
        ),
        device_map=model_config.get(
            "device_map",
            "auto",
        ),
        trust_remote_code=bool(
            model_config.get(
                "trust_remote_code",
                True,
            )
        ),
    )

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    prompt_builder = (
        FixedPlanCodePromptBuilder(
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
        )
    )

    print(
        f"[Prompt] code : "
        f"{prompt_builder.code_prompt_path}"
    )

    # ------------------------------------------------------------------
    # Code sampler
    # ------------------------------------------------------------------

    code_sampler = (
        FixedPlanCodeSampler(
            generator=generator,
            prompt_builder=prompt_builder,
            num_samples=num_samples,
            max_new_tokens=(
                code_max_new_tokens
            ),
            temperature=(
                code_temperature
            ),
            top_p=code_top_p,
            base_seed=base_seed,
            system_prompt=(
                strategy_config.get(
                    "system_prompt"
                )
            ),
        )
    )

    # ------------------------------------------------------------------
    # Evaluator
    # ------------------------------------------------------------------

    extractor = CodeExtractor()

    evaluator = Evaluator(
        timeout_seconds=float(
            evaluation_config.get(
                "timeout_seconds",
                5.0,
            )
        ),
        include_public_tests=bool(
            evaluation_config.get(
                "include_public_tests",
                True,
            )
        ),
        include_private_tests=bool(
            evaluation_config.get(
                "include_private_tests",
                True,
            )
        ),
    )

    executor = CandidateExecutor(
        extractor=extractor,
        evaluator=evaluator,
        store_test_results=(
            store_test_results
        ),
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    completed_ids = (
        completed_problem_ids(
            output_path
        )
        if resume
        else set()
    )

    if completed_ids:
        print(
            f"[Resume] completed="
            f"{len(completed_ids)}"
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    start_time = time.perf_counter()

    processed_count = 0
    skipped_count = 0

    oracle_pass_count = 0
    candidate0_pass_count = 0

    for problem_index, example in enumerate(
        examples,
        start=1,
    ):
        problem_id = str(
            example.problem_id
        )

        if problem_id in completed_ids:
            skipped_count += 1
            continue

        fixed_plan_record = (
            fixed_plan_loader.get(
                problem_id
            )
        )

        fixed_plan = (
            fixed_plan_record.plan
        )

        print()
        print("-" * 80)

        print(
            f"[{problem_index}/{len(examples)}] "
            f"{problem_id} | "
            f"{getattr(example, 'difficulty', 'unknown')} | "
            f"{getattr(example, 'title', '')}"
        )

        print("-" * 80)

        candidates: list[
            dict[str, Any]
        ] = []

        # --------------------------------------------------------------
        # N sampled codes
        # --------------------------------------------------------------

        for sample_id in range(
            num_samples
        ):
            candidate = run_code_candidate(
                example=example,
                fixed_plan=fixed_plan,
                sample_id=sample_id,
                sampler=code_sampler,
                executor=executor,
                store_prompts=store_prompts,
            )

            candidates.append(
                candidate
            )

            print(
                f"  [sample {sample_id}] "
                f"{candidate['status']:<20} "
                f"tests="
                f"{candidate['passed_tests']}/"
                f"{candidate['total_tests']} "
                f"ratio="
                f"{candidate['test_pass_ratio']:.3f} "
                f"passed="
                f"{candidate['passed']} "
                f"code_tok="
                f"{candidate['completion_tokens']} "
                f"time="
                f"{candidate['generation_time']:.1f}s"
            )

        # --------------------------------------------------------------
        # Problem summary
        # --------------------------------------------------------------

        summary = summarize_candidates(
            candidates
        )

        if summary["oracle_passed"]:
            oracle_pass_count += 1

        if (
            candidates
            and candidates[0]["passed"]
        ):
            candidate0_pass_count += 1

        print(
            f"  => Oracle@{num_samples}="
            f"{summary['oracle_passed']} | "
            f"passed "
            f"{summary['num_passed']}/"
            f"{num_samples} | "
            f"best_ratio="
            f"{summary['best_test_pass_ratio']:.3f} | "
            f"distinct_codes="
            f"{summary['distinct_codes']}/"
            f"{num_samples} | "
            f"gen_time="
            f"{summary['total_generation_time']:.1f}s"
        )

        # --------------------------------------------------------------
        # Save immediately per problem
        # --------------------------------------------------------------

        record: dict[str, Any] = {
            "problem_id": problem_id,
            "title": getattr(
                example,
                "title",
                "",
            ),
            "difficulty": getattr(
                example,
                "difficulty",
                "unknown",
            ),

            "fixed_plan": fixed_plan,

            "fixed_plan_source": {
                "results_path": str(
                    fixed_plan_results
                ),
                "source_line": (
                    fixed_plan_record.source_line
                ),
            },

            "num_samples": (
                num_samples
            ),

            "candidates": (
                candidates
            ),

            "summary": (
                summary
            ),
        }

        append_jsonl(
            output_path,
            record,
        )

        processed_count += 1

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print("=" * 80)
    print(
        "Experiment Summary"
    )
    print("=" * 80)

    print(
        f"Selected problems : "
        f"{len(examples)}"
    )

    print(
        f"Processed         : "
        f"{processed_count}"
    )

    print(
        f"Skipped           : "
        f"{skipped_count}"
    )

    print(
        f"N (samples)       : "
        f"{num_samples}"
    )

    # Important:
    # These rates refer only to problems processed
    # during this invocation when resume is used.

    print(
        f"Oracle@{num_samples} passed "
        f"(this run) : "
        f"{oracle_pass_count}"
    )

    print(
        "Candidate-0 passed "
        "(this run): "
        f"{candidate0_pass_count}"
    )

    if processed_count > 0:
        print(
            f"Oracle@{num_samples} rate "
            f"(this run)  : "
            f"{oracle_pass_count / processed_count:.4f}"
        )

        print(
            "Candidate-0 rate "
            "(this run) : "
            f"{candidate0_pass_count / processed_count:.4f}"
        )

    hours = int(
        elapsed // 3600
    )

    minutes = int(
        (elapsed % 3600) // 60
    )

    seconds = int(
        elapsed % 60
    )

    print(
        f"Elapsed           : "
        f"{hours}h {minutes}m {seconds}s"
    )

    print(
        f"Output            : "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()