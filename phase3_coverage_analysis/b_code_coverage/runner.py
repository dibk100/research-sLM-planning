"""
Phase 3-B Code Coverage runner.

Responsibilities:
- problem iteration
- resume handling
- fixed-plan lookup
- candidate iteration
- code parsing
- exhaustive evaluation
- candidate record construction
- problem-level aggregation
- JSONL logging
- progress / summary reporting

Code sampling remains inside CodeCoverageStrategy.
"""

# phase3_coverage_analysis/b_code_coverage/runner.py

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phase3_coverage_analysis.b_code_coverage.candidate import (
    CandidateRecord,
    ProblemRecord,
    summarize_candidates,
)
from phase3_coverage_analysis.b_code_coverage.fixed_plan_loader import (
    FixedPlanLoader,
)
from phase3_coverage_analysis.b_code_coverage.strategies.code_coverage import (
    CodeCoverageStrategy,
)

from src.execution.evaluator import Evaluator
from src.parsing.code_parser import CodeParser
from src.schemas import (
    EvaluationResult,
    ProblemExample,
)
from src.utils.jsonl_logger import JSONLLogger


@dataclass
class CodeCoverageRunnerSummary:
    selected: int
    processed: int
    skipped: int

    oracle_passed: int
    candidate0_passed: int

    @property
    def oracle_rate(self) -> float:
        if self.processed == 0:
            return 0.0

        return (
            self.oracle_passed
            / self.processed
        )

    @property
    def candidate0_rate(self) -> float:
        if self.processed == 0:
            return 0.0

        return (
            self.candidate0_passed
            / self.processed
        )


class CodeCoverageRunner:
    """
    Phase 3-B code coverage runner.

    Each problem uses one fixed Phase 1 Self-Plan.
    N code candidates are then sampled stochastically
    from the same plan-conditioned code prompt.
    """

    def __init__(
        self,
        *,
        strategy: CodeCoverageStrategy,
        fixed_plan_loader: FixedPlanLoader,
        evaluator: Evaluator,
        parser: CodeParser,
        output_path: str | Path,
        model_name: str,
        dataset_name: str,
        seed: int,
        num_samples: int,
        resume: bool = True,
        store_prompts: bool = False,
        store_test_results: bool = False,
    ) -> None:
        if num_samples <= 0:
            raise ValueError(
                "num_samples must be greater than 0."
            )

        self.strategy = strategy
        self.fixed_plan_loader = (
            fixed_plan_loader
        )

        self.evaluator = evaluator
        self.parser = parser

        self.output_path = Path(
            output_path
        )

        self.model_name = model_name
        self.dataset_name = dataset_name

        self.seed = seed
        self.num_samples = num_samples

        self.resume = resume
        self.store_prompts = (
            store_prompts
        )
        self.store_test_results = (
            store_test_results
        )

        self.logger = JSONLLogger(
            self.output_path
        )

    def run(
        self,
        examples: list[ProblemExample],
    ) -> CodeCoverageRunnerSummary:
        # ------------------------------------------------------
        # Validate fixed plans before running generation.
        # ------------------------------------------------------

        validation = (
            self.fixed_plan_loader.validate_examples(
                examples,
                require_exact_match=False,
            )
        )

        self.fixed_plan_loader.validate_sequence(
            examples
        )

        print(
            "[Fixed Plans] "
            f"loaded="
            f"{len(self.fixed_plan_loader)}, "
            f"matched="
            f"{validation['matched']}, "
            f"missing="
            f"{validation['missing_plans']}, "
            f"empty="
            f"{validation['empty_plans']}"
        )

        # ------------------------------------------------------
        # Resume
        # ------------------------------------------------------

        completed_ids = (
            self.logger.completed_ids()
            if self.resume
            else set()
        )

        if completed_ids:
            print(
                "[Resume] Loaded "
                f"{len(completed_ids)} "
                "completed problems."
            )

        total_selected = len(
            examples
        )

        processed_count = 0
        skipped_count = 0

        oracle_pass_count = 0
        candidate0_pass_count = 0

        # ------------------------------------------------------
        # Problem loop
        # ------------------------------------------------------

        for index, example in enumerate(
            examples,
            start=1,
        ):
            if (
                example.problem_id
                in completed_ids
            ):
                skipped_count += 1

                print(
                    f"[{index}/{total_selected}] "
                    f"[SKIP] "
                    f"{example.problem_id}"
                )

                continue

            self._print_problem_header(
                index=index,
                total=total_selected,
                example=example,
            )

            try:
                record = self.run_one(
                    example
                )

                self.logger.append(
                    record.to_dict()
                )

                processed_count += 1

                if record.any_passed:
                    oracle_pass_count += 1

                if (
                    record.candidates
                    and bool(
                        record.candidates[
                            0
                        ].get(
                            "passed",
                            False,
                        )
                    )
                ):
                    candidate0_pass_count += 1

                self._print_problem_result(
                    record
                )

            except Exception as error:
                print(
                    f"[ERROR] "
                    f"{example.problem_id}: "
                    f"{error}"
                )

                traceback.print_exc()

                raise

        # ------------------------------------------------------
        # Summary
        # ------------------------------------------------------

        summary = (
            CodeCoverageRunnerSummary(
                selected=total_selected,
                processed=processed_count,
                skipped=skipped_count,
                oracle_passed=(
                    oracle_pass_count
                ),
                candidate0_passed=(
                    candidate0_pass_count
                ),
            )
        )

        self._print_summary(
            summary
        )

        return summary

    def run_one(
        self,
        example: ProblemExample,
    ) -> ProblemRecord:
        """
        Run all N stochastic code candidates
        for one fixed Phase 1 Self-Plan.
        """

        # ------------------------------------------------------
        # 1. Retrieve fixed Phase 1 self-plan
        # ------------------------------------------------------

        fixed_plan_record = (
            self.fixed_plan_loader.get(
                example.problem_id
            )
        )

        fixed_plan = (
            fixed_plan_record.plan.strip()
        )

        if not fixed_plan:
            raise ValueError(
                "Fixed plan must not be empty: "
                f"{example.problem_id}"
            )

        # ------------------------------------------------------
        # 2. Build shared code prompt once
        #
        # All N candidates must use the exact same
        # problem + fixed-plan prompt.
        # ------------------------------------------------------

        code_prompt = (
            self.strategy.build_code_prompt(
                example=example,
                fixed_plan=fixed_plan,
            )
        )

        # ------------------------------------------------------
        # Optional integrity check:
        #
        # Phase 3-B should reconstruct the same
        # code prompt used by Phase 1 Self-Plan.
        # ------------------------------------------------------

        if (
            fixed_plan_record.phase1_code_prompt
            is not None
        ):
            phase1_prompt = (
                fixed_plan_record
                .phase1_code_prompt
                .strip()
            )

            if (
                code_prompt.strip()
                != phase1_prompt
            ):
                raise ValueError(
                    "Reconstructed Phase 3-B "
                    "code prompt differs from "
                    "Phase 1 Self-Plan prompt: "
                    f"{example.problem_id}"
                )

        candidates: list[
            CandidateRecord
        ] = []

        # ------------------------------------------------------
        # 3. Candidate sequence
        #
        # sample_id = 0 ... N-1
        #
        # Prefix candidates are later interpreted as
        # Code Coverage@k.
        # ------------------------------------------------------

        for sample_id in range(
            self.num_samples
        ):
            candidate = (
                self.run_candidate(
                    example=example,
                    fixed_plan=fixed_plan,
                    sample_id=sample_id,
                    code_prompt=code_prompt,
                )
            )

            candidates.append(
                candidate
            )

            self._print_candidate(
                candidate
            )

        # ------------------------------------------------------
        # 4. Problem-level summary
        # ------------------------------------------------------

        summary = summarize_candidates(
            candidates
        )

        return ProblemRecord(
            problem_id=example.problem_id,
            dataset=self.dataset_name,
            strategy=self.strategy.name,
            model_name=self.model_name,
            seed=self.seed,
            num_samples=self.num_samples,

            title=example.title,
            platform=example.platform,
            contest_date=example.contest_date,
            difficulty=example.difficulty,

            problem=example.problem,

            fixed_plan=fixed_plan,

            candidates=[
                candidate.to_dict()
                for candidate in candidates
            ],

            any_passed=bool(
                summary["any_passed"]
            ),
            num_passed=int(
                summary["num_passed"]
            ),
            best_test_pass_ratio=float(
                summary[
                    "best_test_pass_ratio"
                ]
            ),
            total_generation_time=float(
                summary[
                    "total_generation_time"
                ]
            ),
            total_completion_tokens=int(
                summary[
                    "total_completion_tokens"
                ]
            ),
        )

    def run_candidate(
        self,
        *,
        example: ProblemExample,
        fixed_plan: str,
        sample_id: int,
        code_prompt: str,
    ) -> CandidateRecord:
        """
        Generate one stochastic code candidate
        and evaluate it.
        """

        candidate_output = (
            self.strategy.run_candidate(
                example=example,
                fixed_plan=fixed_plan,
                sample_id=sample_id,
                code_prompt=code_prompt,
            )
        )

        # ------------------------------------------------------
        # Empty code generation
        # ------------------------------------------------------

        if candidate_output.code_empty:
            return CandidateRecord(
                sample_id=(
                    candidate_output.sample_id
                ),
                sample_seed=(
                    candidate_output.sample_seed
                ),

                code="",

                passed=False,
                status="EMPTY_CODE",

                passed_tests=0,
                total_tests=0,
                test_pass_ratio=0.0,

                code_prompt_tokens=(
                    candidate_output
                    .code_prompt_tokens
                ),
                code_completion_tokens=(
                    candidate_output
                    .code_completion_tokens
                ),
                code_generation_time=(
                    candidate_output
                    .code_generation_time
                ),

                execution_time=0.0,

                plan_in_code_prompt=(
                    candidate_output
                    .plan_in_code_prompt
                ),

                raw_output=(
                    candidate_output
                    .code_raw_output
                ),

                error_message=(
                    "Model produced an empty "
                    "code output."
                ),

                code_prompt=(
                    candidate_output.code_prompt
                    if self.store_prompts
                    else None
                ),

                test_results=[],
            )

        # ------------------------------------------------------
        # Parse generated code
        # ------------------------------------------------------

        parse_result = self.parser.parse(
            candidate_output.code_raw_output
        )

        if parse_result.status != "SUCCESS":
            evaluation = (
                self._build_pipeline_failure(
                    status="PARSING_ERROR",
                    error_message=(
                        "Code parsing failed: "
                        f"{parse_result.status}"
                    ),
                )
            )

        else:
            # --------------------------------------------------
            # Exhaustive evaluation
            # --------------------------------------------------

            try:
                evaluation = (
                    self.evaluator.evaluate(
                        problem=example,
                        code=parse_result.code,
                    )
                )

            except Exception as error:
                evaluation = (
                    self._build_pipeline_failure(
                        status=(
                            "EVALUATION_ERROR"
                        ),
                        error_message=str(
                            error
                        ),
                    )
                )

        # ------------------------------------------------------
        # Candidate record
        # ------------------------------------------------------

        test_pass_ratio = (
            self._compute_test_pass_ratio(
                passed_tests=(
                    evaluation.passed_tests
                ),
                total_tests=(
                    evaluation.total_tests
                ),
            )
        )

        return CandidateRecord(
            sample_id=(
                candidate_output.sample_id
            ),
            sample_seed=(
                candidate_output.sample_seed
            ),

            code=(
                parse_result.code
                if (
                    parse_result.status
                    == "SUCCESS"
                )
                else ""
            ),

            passed=(
                evaluation.passed
            ),
            status=(
                evaluation.status
            ),

            passed_tests=(
                evaluation.passed_tests
            ),
            total_tests=(
                evaluation.total_tests
            ),
            test_pass_ratio=(
                test_pass_ratio
            ),

            code_prompt_tokens=(
                candidate_output
                .code_prompt_tokens
            ),
            code_completion_tokens=(
                candidate_output
                .code_completion_tokens
            ),
            code_generation_time=(
                candidate_output
                .code_generation_time
            ),

            execution_time=(
                evaluation.execution_time
            ),

            plan_in_code_prompt=(
                candidate_output
                .plan_in_code_prompt
            ),

            raw_output=(
                candidate_output
                .code_raw_output
            ),

            error_message=(
                evaluation.error_message
            ),

            code_prompt=(
                candidate_output.code_prompt
                if self.store_prompts
                else None
            ),

            test_results=(
                self._serialize_test_results(
                    evaluation.test_results
                )
                if self.store_test_results
                else []
            ),
        )

    @staticmethod
    def _build_pipeline_failure(
        *,
        status: str,
        error_message: str,
    ) -> EvaluationResult:
        """
        Create a synthetic evaluation result for
        failures occurring outside the evaluator.
        """

        return EvaluationResult(
            passed=False,
            status=status,
            passed_tests=0,
            total_tests=0,
            execution_time=0.0,
            test_results=[],
            error_message=error_message,
        )

    @staticmethod
    def _compute_test_pass_ratio(
        *,
        passed_tests: int,
        total_tests: int,
    ) -> float:
        if total_tests <= 0:
            return 0.0

        return (
            passed_tests
            / total_tests
        )

    @staticmethod
    def _serialize_test_results(
        test_results: list[Any],
    ) -> list[
        dict[str, Any]
    ]:
        serialized: list[
            dict[str, Any]
        ] = []

        for test_result in test_results:
            if hasattr(
                test_result,
                "to_dict",
            ):
                serialized.append(
                    test_result.to_dict()
                )

            elif hasattr(
                test_result,
                "__dataclass_fields__",
            ):
                from dataclasses import (
                    asdict,
                )

                serialized.append(
                    asdict(
                        test_result
                    )
                )

            elif isinstance(
                test_result,
                dict,
            ):
                serialized.append(
                    test_result
                )

            else:
                serialized.append(
                    {
                        "value": str(
                            test_result
                        )
                    }
                )

        return serialized

    @staticmethod
    def _print_problem_header(
        *,
        index: int,
        total: int,
        example: ProblemExample,
    ) -> None:
        print()
        print("-" * 80)

        if example.difficulty is not None:
            difficulty_text = (
                example.difficulty
            )

        elif example.rating is not None:
            difficulty_text = (
                f"rating={example.rating}"
            )

        else:
            difficulty_text = (
                "unknown"
            )

        print(
            f"[{index}/{total}] "
            f"{example.problem_id} | "
            f"{difficulty_text} | "
            f"{example.title}"
        )

        print("-" * 80)

    @staticmethod
    def _print_candidate(
        candidate: CandidateRecord,
    ) -> None:
        print(
            f"  [sample "
            f"{candidate.sample_id}] "
            f"{candidate.status:<20} "
            f"tests="
            f"{candidate.passed_tests}/"
            f"{candidate.total_tests} "
            f"ratio="
            f"{candidate.test_pass_ratio:.3f} "
            f"passed="
            f"{candidate.passed} "
            f"code_tok="
            f"{candidate.code_completion_tokens} "
            f"time="
            f"{candidate.code_generation_time:.1f}s"
        )

    def _print_problem_result(
        self,
        record: ProblemRecord,
    ) -> None:
        candidates = (
            record.candidates
        )

        distinct_codes = len(
            {
                str(
                    candidate.get(
                        "code",
                        "",
                    )
                ).strip()
                for candidate in candidates
                if str(
                    candidate.get(
                        "code",
                        "",
                    )
                ).strip()
            }
        )

        empty_codes = sum(
            not str(
                candidate.get(
                    "code",
                    "",
                )
            ).strip()
            for candidate in candidates
        )

        print(
            f"  => Oracle@"
            f"{record.num_samples}="
            f"{record.any_passed} | "
            f"passed "
            f"{record.num_passed}/"
            f"{record.num_samples} | "
            f"best_ratio="
            f"{record.best_test_pass_ratio:.3f} | "
            f"distinct_codes="
            f"{distinct_codes}/"
            f"{record.num_samples} | "
            f"empty_codes="
            f"{empty_codes} | "
            f"gen_time="
            f"{record.total_generation_time:.1f}s"
        )

    def _print_summary(
        self,
        summary: CodeCoverageRunnerSummary,
    ) -> None:
        print()
        print("=" * 80)
        print(
            "Phase 3-B Experiment Summary"
        )
        print("=" * 80)

        print(
            f"Selected problems : "
            f"{summary.selected}"
        )

        print(
            f"Processed         : "
            f"{summary.processed}"
        )

        print(
            f"Skipped           : "
            f"{summary.skipped}"
        )

        print(
            f"N (samples)       : "
            f"{self.num_samples}"
        )

        print(
            f"Oracle@{self.num_samples} "
            f"passed : "
            f"{summary.oracle_passed}"
        )

        print(
            f"Candidate-0 passed: "
            f"{summary.candidate0_passed}"
        )

        if summary.processed > 0:
            print(
                f"Oracle@{self.num_samples} "
                f"rate   : "
                f"{summary.oracle_rate:.4f}"
            )

            print(
                f"Candidate-0 rate  : "
                f"{summary.candidate0_rate:.4f}"
            )

        print(
            f"Output            : "
            f"{self.output_path}"
        )