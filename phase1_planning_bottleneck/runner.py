# phase1_planning_bottleneck/runner.py

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.execution.evaluator import Evaluator
from src.parsing.code_parser import CodeParser
from src.schemas import (
    EvaluationResult,
    ExperimentRecord,
    ProblemExample,
    StrategyOutput,
)
from src.utils.jsonl_logger import JSONLLogger
from src.utils.record_builder import build_experiment_record


class StrategyProtocol(Protocol):
    name: str

    def run(
        self,
        example: ProblemExample,
    ) -> StrategyOutput:
        ...


@dataclass
class RunnerSummary:
    selected: int
    processed: int
    skipped: int
    passed: int

    @property
    def pass_rate(self) -> float:
        if self.processed == 0:
            return 0.0

        return self.passed / self.processed


class Phase1Runner:
    """
    Shared experiment runner for Phase 1 strategies.

    Responsibilities:
    - resume handling
    - strategy execution
    - raw-output parsing
    - exhaustive evaluation
    - experiment record construction
    - JSONL logging
    - progress / summary reporting

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

        self.output_path = Path(output_path)

        self.model_name = model_name
        self.seed = seed
        self.resume = resume

        self.logger = JSONLLogger(
            self.output_path
        )

    def run(
        self,
        examples: list[ProblemExample],
    ) -> RunnerSummary:
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

        total_selected = len(examples)

        processed_count = 0
        skipped_count = 0
        pass_count = 0

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

                if record.passed:
                    pass_count += 1

                self._print_result(
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

        summary = RunnerSummary(
            selected=total_selected,
            processed=processed_count,
            skipped=skipped_count,
            passed=pass_count,
        )

        self._print_summary(
            summary
        )

        return summary

    def run_one(
        self,
        example: ProblemExample,
    ) -> ExperimentRecord:
        """
        Run the complete Phase 1 pipeline for one problem.
        """

        # --------------------------------------------------------------
        # 1. Strategy / generation
        # --------------------------------------------------------------

        strategy_output = (
            self.strategy.run(
                example
            )
        )

        # --------------------------------------------------------------
        # 2. Parse raw model output
        # --------------------------------------------------------------

        parse_result = self.parser.parse(
            strategy_output.raw_output
        )

        # --------------------------------------------------------------
        # 3. Exhaustive evaluation
        # --------------------------------------------------------------

        if parse_result.status != "SUCCESS":
            evaluation = self._build_pipeline_failure(
                status="PARSING_ERROR",
                error_message=(
                    f"Code parsing failed: "
                    f"{parse_result.status}"
                ),
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
                        status="EVALUATION_ERROR",
                        error_message=str(
                            error
                        ),
                    )
                )

        # --------------------------------------------------------------
        # 4. Build record
        # --------------------------------------------------------------

        return build_experiment_record(
            example=example,
            strategy_output=(
                strategy_output
            ),
            parse_result=parse_result,
            evaluation=evaluation,
            model_name=self.model_name,
            seed=self.seed,
        )

    @staticmethod
    def _build_pipeline_failure(
        *,
        status: str,
        error_message: str,
    ) -> EvaluationResult:
        """
        Create a synthetic evaluation failure for errors occurring
        outside the evaluator itself.
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
            difficulty_text = "unknown"

        print(
            f"[{index}/{total}] "
            f"{example.problem_id} | "
            f"{difficulty_text} | "
            f"{example.title}"
        )

        print("-" * 80)

    @staticmethod
    def _print_result(
        record: ExperimentRecord,
    ) -> None:
        print(
            f"Parse      : "
            f"{record.parse_status} "
            f"({record.extraction_method})"
        )

        print(
            f"Status     : "
            f"{record.status}"
        )

        print(
            f"Tests      : "
            f"{record.passed_tests}/"
            f"{record.total_tests} "
            f"({record.test_pass_ratio:.4f})"
        )

        print(
            f"Gen tokens : "
            f"{record.completion_tokens}"
        )

        print(
            f"Gen time   : "
            f"{record.generation_time:.2f}s"
        )

        print(
            f"Exec time  : "
            f"{record.execution_time:.4f}s"
        )

        if record.error_message:
            print(
                "Error      : "
                f"{record.error_message[:500]}"
            )

    def _print_summary(
        self,
        summary: RunnerSummary,
    ) -> None:
        print()
        print("=" * 80)
        print("Experiment Summary")
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
            f"Passed            : "
            f"{summary.passed}"
        )

        if summary.processed > 0:
            print(
                f"Current pass rate : "
                f"{summary.pass_rate:.4f}"
            )

        print(
            f"Output            : "
            f"{self.output_path}"
        )