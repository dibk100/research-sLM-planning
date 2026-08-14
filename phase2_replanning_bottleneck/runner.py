# phase2_replanning_bottleneck/runner.py

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.datasets.phase1_failure_loader import (
    Phase1FailureRecord,
)
from src.execution.evaluator import Evaluator
from src.parsing.code_parser import CodeParser
from src.schemas import (
    EvaluationResult,
    ProblemExample,
    StrategyOutput,
)
from src.utils.jsonl_logger import JSONLLogger
from src.utils.record_builder import (
    build_experiment_record,
)


class StrategyProtocol(Protocol):
    name: str

    def run(
        self,
        failure: Phase1FailureRecord,
    ) -> StrategyOutput:
        ...


@dataclass
class RunnerSummary:
    selected: int
    processed: int
    skipped: int
    recovered: int

    @property
    def recovery_rate(self) -> float:
        if self.processed == 0:
            return 0.0

        return (
            self.recovered
            / self.processed
        )


class Phase2Runner:
    """
    Shared experiment runner for Phase 2 refinement strategies.

    Input:
    - Phase 1 Direct failure trajectories
    - canonical benchmark ProblemExample objects

    Responsibilities:
    - resume handling
    - Phase 1 failure / benchmark problem matching
    - refinement strategy execution
    - regenerated-code parsing
    - exhaustive evaluation
    - Phase 2 provenance logging
    - recovery reporting

    Strategy-specific prompt construction and generation logic
    remain outside this runner.
    """

    def __init__(
        self,
        *,
        strategy: StrategyProtocol,
        evaluator: Evaluator,
        parser: CodeParser,
        output_path: str | Path,
        model_name: str,
        seed: int,
        resume: bool = True,
    ) -> None:
        self.strategy = strategy
        self.evaluator = evaluator
        self.parser = parser

        self.output_path = Path(
            output_path
        )

        self.model_name = model_name
        self.seed = seed
        self.resume = resume

        self.logger = JSONLLogger(
            self.output_path
        )

    def run(
        self,
        *,
        failures: list[
            Phase1FailureRecord
        ],
        examples: list[
            ProblemExample
        ],
    ) -> RunnerSummary:
        """
        Run Phase 2 over the selected Phase 1 failures.

        The benchmark examples are matched to failures by
        problem_id and are used as the canonical source for
        evaluation.
        """

        example_map = self._build_example_map(
            examples
        )

        self._validate_failure_coverage(
            failures=failures,
            example_map=example_map,
        )

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
            failures
        )

        processed_count = 0
        skipped_count = 0
        recovered_count = 0

        for index, failure in enumerate(
            failures,
            start=1,
        ):
            if (
                failure.problem_id
                in completed_ids
            ):
                skipped_count += 1

                print(
                    f"[{index}/{total_selected}] "
                    f"[SKIP] "
                    f"{failure.problem_id}"
                )

                continue

            example = example_map[
                failure.problem_id
            ]

            self._print_problem_header(
                index=index,
                total=total_selected,
                example=example,
                failure=failure,
            )

            try:
                record = self.run_one(
                    failure=failure,
                    example=example,
                )

                self.logger.append(
                    record
                )

                processed_count += 1

                if record["passed"]:
                    recovered_count += 1

                self._print_result(
                    record
                )

            except Exception as error:
                print(
                    f"[ERROR] "
                    f"{failure.problem_id}: "
                    f"{error}"
                )

                traceback.print_exc()

                raise

        summary = RunnerSummary(
            selected=total_selected,
            processed=processed_count,
            skipped=skipped_count,
            recovered=recovered_count,
        )

        self._print_summary(
            summary
        )

        return summary

    def run_one(
        self,
        *,
        failure: Phase1FailureRecord,
        example: ProblemExample,
    ) -> dict:
        """
        Run the complete Phase 2 pipeline for one
        Phase 1 failure trajectory.
        """

        # ----------------------------------------------------------
        # 1. Refinement strategy / generation
        # ----------------------------------------------------------

        strategy_output = (
            self.strategy.run(
                failure
            )
        )

        if (
            strategy_output.problem_id
            != failure.problem_id
        ):
            raise ValueError(
                "Strategy output problem_id "
                "does not match Phase 1 failure: "
                f"{strategy_output.problem_id} "
                "!= "
                f"{failure.problem_id}"
            )

        # ----------------------------------------------------------
        # 2. Parse regenerated model output
        # ----------------------------------------------------------

        parse_result = self.parser.parse(
            strategy_output.raw_output
        )

        # ----------------------------------------------------------
        # 3. Exhaustive evaluation
        # ----------------------------------------------------------

        if (
            parse_result.status
            != "SUCCESS"
        ):
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

        # ----------------------------------------------------------
        # 4. Build common experiment record
        # ----------------------------------------------------------

        base_record = (
            build_experiment_record(
                example=example,
                strategy_output=(
                    strategy_output
                ),
                parse_result=parse_result,
                evaluation=evaluation,
                model_name=self.model_name,
                seed=self.seed,
            )
        )

        record = base_record.to_dict()

        # ----------------------------------------------------------
        # 5. Add Phase 1 failure provenance
        # ----------------------------------------------------------

        self._add_phase2_provenance(
            record=record,
            failure=failure,
        )

        return record

    @staticmethod
    def _add_phase2_provenance(
        *,
        record: dict,
        failure: Phase1FailureRecord,
    ) -> None:
        """
        Attach the original Phase 1 failure state.

        These fields are analysis metadata and are not
        necessarily included in the model prompt.
        """

        record[
            "initial_status"
        ] = failure.status

        record[
            "initial_passed_tests"
        ] = failure.passed_tests

        record[
            "initial_total_tests"
        ] = failure.total_tests

        record[
            "initial_test_pass_ratio"
        ] = failure.test_pass_ratio

        record[
            "initial_extracted_code"
        ] = failure.extracted_code

        record[
            "feedback_test_index"
        ] = failure.test_index

        record[
            "feedback_input_text"
        ] = failure.input_text

        record[
            "feedback_expected_output"
        ] = failure.expected_output

        record[
            "feedback_actual_output"
        ] = failure.actual_output

        record[
            "feedback_stderr"
        ] = failure.stderr

        # ------------------------------------------------------
        # Derived Phase 2 analysis fields
        # ------------------------------------------------------

        record[
            "recovered"
        ] = bool(
            record["passed"]
        )

        record[
            "test_pass_ratio_delta"
        ] = (
            float(
                record[
                    "test_pass_ratio"
                ]
            )
            - failure.test_pass_ratio
        )

    @staticmethod
    def _build_example_map(
        examples: list[
            ProblemExample
        ],
    ) -> dict[
        str,
        ProblemExample,
    ]:
        example_map: dict[
            str,
            ProblemExample,
        ] = {}

        for example in examples:
            if (
                example.problem_id
                in example_map
            ):
                raise ValueError(
                    "Duplicate benchmark "
                    "problem_id: "
                    f"{example.problem_id}"
                )

            example_map[
                example.problem_id
            ] = example

        return example_map

    @staticmethod
    def _validate_failure_coverage(
        *,
        failures: list[
            Phase1FailureRecord
        ],
        example_map: dict[
            str,
            ProblemExample,
        ],
    ) -> None:
        missing_ids = [
            failure.problem_id
            for failure in failures
            if (
                failure.problem_id
                not in example_map
            )
        ]

        if missing_ids:
            preview = ", ".join(
                missing_ids[:10]
            )

            raise ValueError(
                "Phase 1 failure problems "
                "are missing from the "
                "benchmark dataset. "
                f"Count={len(missing_ids)}, "
                f"examples={preview}"
            )

    @staticmethod
    def _build_pipeline_failure(
        *,
        status: str,
        error_message: str,
    ) -> EvaluationResult:
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
    def _print_problem_header(
        *,
        index: int,
        total: int,
        example: ProblemExample,
        failure: Phase1FailureRecord,
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
            difficulty_text = "unknown"

        print(
            f"[{index}/{total}] "
            f"{example.problem_id} | "
            f"{difficulty_text} | "
            f"{example.title}"
        )

        print(
            "Initial    : "
            f"{failure.status} | "
            f"{failure.passed_tests}/"
            f"{failure.total_tests} "
            f"({failure.test_pass_ratio:.4f})"
        )

        print(
            "Feedback   : "
            f"test {failure.test_index}"
        )

        print("-" * 80)

    @staticmethod
    def _print_result(
        record: dict,
    ) -> None:
        print(
            "Parse      : "
            f"{record['parse_status']} "
            f"({record['extraction_method']})"
        )

        print(
            "Status     : "
            f"{record['status']}"
        )

        print(
            "Tests      : "
            f"{record['passed_tests']}/"
            f"{record['total_tests']} "
            f"({record['test_pass_ratio']:.4f})"
        )

        print(
            "Delta      : "
            f"{record['test_pass_ratio_delta']:+.4f}"
        )

        print(
            "Recovered  : "
            f"{record['recovered']}"
        )

        print(
            "Gen tokens : "
            f"{record['completion_tokens']}"
        )

        print(
            "Gen time   : "
            f"{record['generation_time']:.2f}s"
        )

        print(
            "Exec time  : "
            f"{record['execution_time']:.4f}s"
        )

        if record.get(
            "error_message"
        ):
            print(
                "Error      : "
                f"{record['error_message'][:500]}"
            )

    def _print_summary(
        self,
        summary: RunnerSummary,
    ) -> None:
        print()
        print("=" * 80)
        print("Phase 2 Experiment Summary")
        print("=" * 80)

        print(
            "Selected failures : "
            f"{summary.selected}"
        )

        print(
            "Processed         : "
            f"{summary.processed}"
        )

        print(
            "Skipped           : "
            f"{summary.skipped}"
        )

        print(
            "Recovered         : "
            f"{summary.recovered}"
        )

        if summary.processed > 0:
            print(
                "Recovery rate     : "
                f"{summary.recovery_rate:.4f}"
            )

        print(
            "Output            : "
            f"{self.output_path}"
        )