"""
Phase 3-A: Self-Planning Best-of-N 실행 스크립트.

한 문제당 plan을 N개 sampling하고, 각 plan으로 코드를 생성/실행한 뒤
candidate 단위로 결과를 저장한다.

N=1,2,4,8을 따로 돌리지 않는다.  N=8까지 한 번만 생성하고
분석 단계에서 candidate prefix를 사용한다.

    candidate 0        -> Oracle@1
    candidate 0..1     -> Oracle@2
    candidate 0..3     -> Oracle@4
    candidate 0..7     -> Oracle@8

Usage:

python -m scripts.run_best_of_n \
  --config configs/qwen25_coder_3b.yaml

# 부분 실행 / 출력 경로 변경
python -m scripts.run_best_of_n \
  --config configs/qwen25_coder_3b.yaml \
  --limit 20 \
  --num-samples 8 \
  --output-path /mnt/hdd/project_sLM_planning/output_phase3/tmp/results.jsonl

저장 결과 확인:

wc -l <output>/results.jsonl
head -n 1 <output>/results.jsonl | python -m json.tool | head -50
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

from src.common.datasets.dataset_loader import DatasetLoader
from src.common.execution.code_extractor import CodeExtractor
from src.common.execution.evaluator import Evaluator
from src.common.models.generator import ModelGenerator
from src.common.schemas import ProblemExample
from src.common.utils.config import load_config
from src.common.utils.jsonl_logger import JSONLLogger
from src.common.utils.run_metadata import (
    save_run_config,
    save_run_metadata,
)
from src.common.utils.seed import set_seed
from src.execute import CandidateExecutor
from src.generate_code import PlanConditionedCodeGenerator
from src.generate_plans import PlanSampler
from src.prompts import SelfPlanPromptBuilder
from src.utils import (
    CandidateRecord,
    ProblemRecord,
    assert_examples_match_manifest,
    format_duration,
    load_problem_manifest,
    summarize_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Self-Planning best-of-N generation "
            "on LiveCodeBench v6."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override dataset limit from config.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override number of plan samples (N).",
    )

    parser.add_argument(
        "--output-path",
        default=None,
        help="Override output JSONL path.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume.",
    )

    parser.add_argument(
        "--skip-manifest-check",
        action="store_true",
        help=(
            "Skip the frozen problem-ID manifest check. "
            "Use only for ad-hoc debugging runs."
        ),
    )

    return parser.parse_args()


def run_candidate(
    *,
    example: ProblemExample,
    sample_id: int,
    plan_prompt: str,
    plan_sampler: PlanSampler,
    code_generator: PlanConditionedCodeGenerator,
    executor: CandidateExecutor,
    store_prompts: bool,
) -> CandidateRecord:
    """하나의 candidate(plan -> code -> execute)를 수행한다."""
    plan_sample = plan_sampler.sample_one(
        example,
        sample_id,
        plan_prompt=plan_prompt,
    )

    if plan_sample.is_empty:
        # 생성이 비어도 실험 전체를 중단하지 않고 실패 candidate로 기록한다.
        outcome = executor.skipped(
            status="EMPTY_PLAN",
            error_message=(
                "Model produced an empty plan."
            ),
        )

        return CandidateRecord(
            sample_id=sample_id,
            sample_seed=plan_sample.sample_seed,
            plan="",
            code="",
            passed=False,
            status=outcome.status,
            passed_tests=0,
            total_tests=0,
            test_pass_ratio=0.0,
            plan_prompt_tokens=plan_sample.prompt_tokens,
            plan_completion_tokens=(
                plan_sample.completion_tokens
            ),
            plan_generation_time=(
                plan_sample.generation_time
            ),
            plan_empty=True,
            plan_in_code_prompt=False,
            error_message=outcome.error_message,
        )

    code_sample = code_generator.generate(
        example=example,
        plan=plan_sample.plan,
        seed=plan_sample.sample_seed,
    )

    outcome = executor.run(
        example=example,
        raw_output=code_sample.raw_output,
    )

    return CandidateRecord(
        sample_id=sample_id,
        sample_seed=plan_sample.sample_seed,
        plan=plan_sample.plan,
        code=outcome.code,
        passed=outcome.passed,
        status=outcome.status,
        passed_tests=outcome.passed_tests,
        total_tests=outcome.total_tests,
        test_pass_ratio=outcome.test_pass_ratio,
        plan_prompt_tokens=plan_sample.prompt_tokens,
        plan_completion_tokens=(
            plan_sample.completion_tokens
        ),
        plan_generation_time=plan_sample.generation_time,
        code_prompt_tokens=code_sample.prompt_tokens,
        code_completion_tokens=(
            code_sample.completion_tokens
        ),
        code_generation_time=code_sample.generation_time,
        execution_time=outcome.execution_time,
        plan_empty=False,
        plan_in_code_prompt=(
            code_sample.plan_in_code_prompt
        ),
        raw_output=code_sample.raw_output,
        error_message=outcome.error_message,
        code_prompt=(
            code_sample.code_prompt
            if store_prompts
            else None
        ),
        test_results=outcome.test_results,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiment_config = config["experiment"]
    dataset_config = config["dataset"]
    model_config = config["model"]
    sampling_config = config["sampling"]
    generation_config = config["generation"]
    strategy_config = config["strategy"]
    evaluation_config = config["evaluation"]
    output_config = config["output"]

    plan_generation_config = generation_config["plan"]
    code_generation_config = generation_config["code"]
    
    
    plan_temperature = float(
        plan_generation_config["temperature"]
    )
    code_temperature = float(
        code_generation_config["temperature"]
    )
    
    num_samples = int(
        args.num_samples
        if args.num_samples is not None
        else sampling_config["num_samples"]
    )

    if plan_temperature <= 0.0:
        raise ValueError(
            "Phase 3-A requires stochastic plan generation "
            "(plan temperature > 0)."
        )

    if code_temperature != 0.0:
        raise ValueError(
            "Phase 3-A requires deterministic code generation "
            "(code temperature = 0.0)."
        )

    if num_samples <= 0:
        raise ValueError(
            "num_samples must be greater than 0."
        )

    seed = int(experiment_config["seed"])
    set_seed(seed)

    dataset_limit = (
        args.limit
        if args.limit is not None
        else dataset_config.get("limit")
    )

    output_path = Path(
        args.output_path
        if args.output_path is not None
        else output_config["path"]
    )

    resume = (
        bool(output_config.get("resume", True))
        and not args.no_resume
    )

    store_prompts = bool(
        output_config.get("store_prompts", False)
    )

    store_test_results = bool(
        output_config.get("store_test_results", False)
    )

    # CLI override를 실제 실행 설정에 반영한다.
    dataset_config["limit"] = dataset_limit
    sampling_config["num_samples"] = num_samples
    output_config["path"] = str(output_path)
    output_config["resume"] = resume

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = save_run_config(
        config=config,
        output_dir=output_dir,
        overwrite=False,
    )

    run_metadata_path = save_run_metadata(
        config=config,
        output_dir=output_dir,
        overwrite=False,
    )

    print("=" * 80)
    print("Phase3-A Self-Planning Best-of-N Experiment")
    print("=" * 80)
    print(f"Experiment  : {experiment_config['name']}")
    print(f"Dataset     : {dataset_config['name']}")
    print(f"Model       : {model_config['name_or_path']}")
    print(f"Seed        : {seed}")
    print(f"Limit       : {dataset_limit}")
    print(f"N (samples) : {num_samples}")
    print(
        f"Plan sample : "
        f"temperature="
        f"{plan_generation_config['temperature']}, "
        f"top_p={plan_generation_config['top_p']}, "
        f"max_new_tokens="
        f"{plan_generation_config['max_new_tokens']}"
    )
    print(
        f"Code gen    : "
        f"temperature="
        f"{code_generation_config['temperature']}, "
        f"top_p={code_generation_config['top_p']}, "
        f"max_new_tokens="
        f"{code_generation_config['max_new_tokens']}"
    )
    print(f"Output      : {output_path}")
    print(f"Run config  : {run_config_path}")
    print(f"Metadata    : {run_metadata_path}")
    print(f"Resume      : {resume}")
    print(f"Store prompts      : {store_prompts}")
    print(f"Store test results : {store_test_results}")
    print(
        f"Test type   : "
        f"{dataset_config.get('test_type', 'stdin')}"
    )
    print()

    loader = DatasetLoader(
        dataset_name=dataset_config["name"],
        split=dataset_config["split"],
        limit=dataset_limit,
        test_type=dataset_config.get(
            "test_type",
            "stdin",
        ),
        release_version=dataset_config.get(
            "release_version",
            "release_v6",
        ),
    )

    examples = loader.load()

    # Phase 1과 동일한 문제 집합인지 확인한다.
    manifest_path = dataset_config.get(
        "problem_manifest_path"
    )

    if manifest_path and not args.skip_manifest_check:
        manifest = load_problem_manifest(manifest_path)

        # limit이 manifest보다 작으면 앞에서부터 prefix로 비교한다.
        assert_examples_match_manifest(
            examples,
            manifest[: len(examples)],
        )

        print(
            f"[Manifest] {len(examples)} problems match "
            f"{manifest_path} (Phase 1과 동일)."
        )
        print()

    generator = ModelGenerator(
        model_name_or_path=model_config["name_or_path"],
        dtype=model_config.get("dtype", "bfloat16"),
        device_map=model_config.get("device_map", "auto"),
        trust_remote_code=model_config.get(
            "trust_remote_code",
            True,
        ),
    )

    prompt_builder = SelfPlanPromptBuilder(
        plan_prompt_path=strategy_config[
            "plan_prompt_path"
        ],
        code_prompt_path=strategy_config[
            "code_prompt_path"
        ],
    )

    # 프롬프트는 Phase 1 폴더에서 직접 읽는다. 실제 사용된 경로를 남긴다.
    print(
        f"[Prompt] plan : "
        f"{prompt_builder.plan_prompt_path}"
    )
    print(
        f"[Prompt] code : "
        f"{prompt_builder.code_prompt_path}"
    )
    print()

    system_prompt = strategy_config.get("system_prompt")

    plan_sampler = PlanSampler(
        generator=generator,
        prompt_builder=prompt_builder,
        num_samples=num_samples,
        max_new_tokens=plan_generation_config[
            "max_new_tokens"
        ],
        temperature=plan_generation_config[
            "temperature"
        ],
        top_p=plan_generation_config["top_p"],
        base_seed=seed,
        system_prompt=system_prompt,
    )

    code_generator = PlanConditionedCodeGenerator(
        generator=generator,
        prompt_builder=prompt_builder,
        max_new_tokens=code_generation_config[
            "max_new_tokens"
        ],
        temperature=code_generation_config[
            "temperature"
        ],
        top_p=code_generation_config["top_p"],
        system_prompt=system_prompt,
    )

    executor = CandidateExecutor(
        extractor=CodeExtractor(),
        evaluator=Evaluator(
            timeout_seconds=evaluation_config[
                "timeout_seconds"
            ],
            include_public_tests=evaluation_config[
                "include_public_tests"
            ],
            include_private_tests=evaluation_config[
                "include_private_tests"
            ],
        ),
        store_test_results=store_test_results,
    )
    
    if output_path.exists() and not resume:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Use resume or choose a new output path."
        )

    logger = JSONLLogger(output_path)

    completed_ids = (
        logger.completed_ids() if resume else set()
    )

    if completed_ids:
        print(
            f"[Resume] Loaded {len(completed_ids)} "
            f"completed problems."
        )

    total_selected = len(examples)
    processed_count = 0
    skipped_count = 0
    oracle_pass_count = 0
    first_candidate_pass_count = 0

    run_start = time.perf_counter()

    for index, example in enumerate(
        examples,
        start=1,
    ):
        if example.problem_id in completed_ids:
            skipped_count += 1

            print(
                f"[{index}/{total_selected}] "
                f"[SKIP] {example.problem_id}"
            )
            continue

        print()
        print("-" * 80)
        print(
            f"[{index}/{total_selected}] "
            f"{example.problem_id} | "
            f"{example.difficulty} | "
            f"{example.title}"
        )
        print("-" * 80)

        try:
            plan_prompt = (
                prompt_builder.build_plan_prompt(example)
            )

            candidates: list[CandidateRecord] = []

            for sample_id in range(num_samples):
                candidate = run_candidate(
                    example=example,
                    sample_id=sample_id,
                    plan_prompt=plan_prompt,
                    plan_sampler=plan_sampler,
                    code_generator=code_generator,
                    executor=executor,
                    store_prompts=store_prompts,
                )

                candidates.append(candidate)

                print(
                    f"  [sample {sample_id}] "
                    f"{candidate.status:<20} "
                    f"tests="
                    f"{candidate.passed_tests}/"
                    f"{candidate.total_tests} "
                    f"ratio="
                    f"{candidate.test_pass_ratio:.3f} "
                    f"passed={candidate.passed} "
                    f"plan_tok="
                    f"{candidate.plan_completion_tokens} "
                    f"time="
                    f"{candidate.plan_generation_time + candidate.code_generation_time:.1f}s"
                )

            summary: dict[str, Any] = (
                summarize_candidates(candidates)
            )

            record = ProblemRecord(
                problem_id=example.problem_id,
                dataset=dataset_config["name"],
                strategy=strategy_config["name"],
                model_name=model_config["name_or_path"],
                seed=seed,
                num_samples=num_samples,
                title=example.title,
                platform=example.platform,
                contest_id=example.contest_id,
                contest_date=example.contest_date,
                difficulty=example.difficulty,
                problem=example.prompt,
                plan_prompt=(
                    plan_prompt if store_prompts else ""
                ),
                candidates=[
                    candidate.to_dict()
                    for candidate in candidates
                ],
                any_passed=summary["any_passed"],
                num_passed=summary["num_passed"],
                best_test_pass_ratio=summary[
                    "best_test_pass_ratio"
                ],
                total_generation_time=summary[
                    "total_generation_time"
                ],
                total_completion_tokens=summary[
                    "total_completion_tokens"
                ],
            )

            logger.append(record.to_dict())

            processed_count += 1

            if summary["any_passed"]:
                oracle_pass_count += 1

            if candidates and candidates[0].passed:
                first_candidate_pass_count += 1

            distinct_plans = len(
                {
                    candidate.plan.strip()
                    for candidate in candidates
                    if candidate.plan.strip()
                }
            )
            
            num_empty_plans = sum(
                candidate.plan_empty
                for candidate in candidates
            )

            print(
                f"  => Oracle@{num_samples}="
                f"{summary['any_passed']} | "
                f"passed {summary['num_passed']}"
                f"/{num_samples} | "
                f"best_ratio="
                f"{summary['best_test_pass_ratio']:.3f} | "
                f"distinct_plans={distinct_plans}"
                f"/{num_samples} | "
                f"gen_time="
                f"{summary['total_generation_time']:.1f}s"
            )

        except Exception as error:
            print(
                f"[ERROR] {example.problem_id}: {error}"
            )
            traceback.print_exc()
            raise

    elapsed = time.perf_counter() - run_start

    print()
    print("=" * 80)
    print("Experiment Summary")
    print("=" * 80)
    print(f"Selected problems : {total_selected}")
    print(f"Processed         : {processed_count}")
    print(f"Skipped           : {skipped_count}")
    print(f"N (samples)       : {num_samples}")
    print(
        f"Oracle@{num_samples} passed     : "
        f"{oracle_pass_count}"
    )
    print(
        f"Candidate-0 passed: "
        f"{first_candidate_pass_count}"
    )

    if processed_count > 0:
        print(
            f"Oracle@{num_samples} rate       : "
            f"{oracle_pass_count / processed_count:.4f}"
        )
        print(
            f"Candidate-0 rate  : "
            f"{first_candidate_pass_count / processed_count:.4f}"
        )

    print(f"Elapsed           : {format_duration(elapsed)}")
    print(f"Output            : {output_path}")
    print()
    print(
        "[DONE] Phase3-A best-of-N experiment completed."
    )
    print(
        "Next: python -m scripts.sanity_check "
        f"--results {output_path}"
    )
    print(
        "      python -m scripts.analyze_coverage "
        f"--results {output_path}"
    )


if __name__ == "__main__":
    main()
