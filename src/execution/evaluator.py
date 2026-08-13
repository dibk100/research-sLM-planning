"""

최종 passed/status의 기준은 official evaluator
diagnostic은 선택적으로 수행
둘의 결과를 한 객체로 묶어서 반환
diagnostic 결과가 official 판정을 덮어쓰지 않음

실제 최종 판정은 result.official
result.passed
result.status


"""
# src/execution/evaluator.py

from __future__ import annotations

from dataclasses import dataclass

from src.execution.diagnostic_evaluator import (
    DiagnosticEvaluator,
)
from src.execution.livecodebench_evaluator import (
    LiveCodeBenchEvaluator,
)
from src.schemas import (
    EvaluationResult,
    ProblemExample,
)


@dataclass
class CombinedEvaluationResult:
    """
    Combined evaluation result.

    official:
        Final correctness authority.

    diagnostic:
        Optional full-test diagnostic result used only for analysis.
    """

    official: EvaluationResult
    diagnostic: EvaluationResult | None = None

    @property
    def passed(self) -> bool:
        """
        Final correctness always follows the official evaluator.
        """
        return self.official.passed

    @property
    def status(self) -> str:
        """
        Final status always follows the official evaluator.
        """
        return self.official.status

    @property
    def diagnostic_test_pass_ratio(
        self,
    ) -> float | None:
        if self.diagnostic is None:
            return None

        if self.diagnostic.total_tests == 0:
            return 0.0

        return (
            self.diagnostic.passed_tests
            / self.diagnostic.total_tests
        )

    @property
    def evaluation_mismatch(self) -> bool | None:
        """
        Whether official and diagnostic PASS/FAIL disagree.

        This is diagnostic information only and must never override
        the official evaluation result.
        """

        if self.diagnostic is None:
            return None

        return (
            self.official.passed
            != self.diagnostic.passed
        )


class Evaluator:
    """
    Unified evaluation interface.

    For LiveCodeBench:
        - official evaluation:
          official LiveCodeBench code-generation evaluator
        - diagnostic evaluation:
          optional full-test per-case evaluation

    Final correctness is always determined by the official evaluator.
    """

    def __init__(
        self,
        *,
        official_timeout_seconds: int = 6,
        diagnostic_timeout_seconds: int | None = None,
        enable_diagnostic: bool = False,
        include_public_tests: bool = True,
        include_private_tests: bool = True,
        debug: bool = False,
    ) -> None:

        if official_timeout_seconds <= 0:
            raise ValueError(
                "official_timeout_seconds must be greater than 0."
            )

        if diagnostic_timeout_seconds is None:
            diagnostic_timeout_seconds = (
                official_timeout_seconds
            )

        if diagnostic_timeout_seconds <= 0:
            raise ValueError(
                "diagnostic_timeout_seconds must be greater than 0."
            )

        self.enable_diagnostic = enable_diagnostic

        self.official_evaluator = (
            LiveCodeBenchEvaluator(
                timeout_seconds=official_timeout_seconds,
                debug=debug,
                include_public_tests=include_public_tests,
                include_private_tests=include_private_tests,
            )
        )

        self.diagnostic_evaluator = (
            DiagnosticEvaluator(
                timeout_seconds=diagnostic_timeout_seconds,
                debug=debug,
                include_public_tests=include_public_tests,
                include_private_tests=include_private_tests,
            )
            if enable_diagnostic
            else None
        )

    def evaluate(
        self,
        problem: ProblemExample,
        code: str,
        *,
        run_diagnostic: bool | None = None,
    ) -> CombinedEvaluationResult:
        """
        Evaluate one generated program.

        Parameters
        ----------
        problem:
            Normalized benchmark problem.

        code:
            Parsed/generated Python code.

        run_diagnostic:
            Optional per-call override.

            None:
                use evaluator-wide enable_diagnostic setting

            True:
                run diagnostic evaluation

            False:
                skip diagnostic evaluation

        Returns
        -------
        CombinedEvaluationResult
            official result + optional diagnostic result
        """

        self._validate_problem(problem)

        official_result = (
            self.official_evaluator.evaluate(
                problem=problem,
                code=code,
            )
        )

        should_run_diagnostic = (
            self.enable_diagnostic
            if run_diagnostic is None
            else run_diagnostic
        )

        diagnostic_result = None

        if should_run_diagnostic:
            diagnostic_evaluator = (
                self._get_diagnostic_evaluator()
            )

            diagnostic_result = (
                diagnostic_evaluator.evaluate(
                    problem=problem,
                    code=code,
                )
            )

        return CombinedEvaluationResult(
            official=official_result,
            diagnostic=diagnostic_result,
        )

    def evaluate_official(
        self,
        problem: ProblemExample,
        code: str,
    ) -> EvaluationResult:
        """
        Run only the official evaluator.
        """

        self._validate_problem(problem)

        return self.official_evaluator.evaluate(
            problem=problem,
            code=code,
        )

    def evaluate_diagnostic(
        self,
        problem: ProblemExample,
        code: str,
    ) -> EvaluationResult:
        """
        Run only the diagnostic evaluator.
        """

        self._validate_problem(problem)

        evaluator = self._get_diagnostic_evaluator()

        return evaluator.evaluate(
            problem=problem,
            code=code,
        )

    def _get_diagnostic_evaluator(
        self,
    ) -> DiagnosticEvaluator:
        """
        Lazily create a diagnostic evaluator when diagnostic evaluation
        is requested through a per-call override.
        """

        if self.diagnostic_evaluator is None:
            self.diagnostic_evaluator = (
                DiagnosticEvaluator(
                    timeout_seconds=(
                        self.official_evaluator.timeout_seconds
                    ),
                    debug=(
                        self.official_evaluator.debug
                    ),
                    include_public_tests=(
                        self.official_evaluator.include_public_tests
                    ),
                    include_private_tests=(
                        self.official_evaluator.include_private_tests
                    ),
                )
            )

        return self.diagnostic_evaluator

    @staticmethod
    def _validate_problem(
        problem: ProblemExample,
    ) -> None:
        if not isinstance(
            problem,
            ProblemExample,
        ):
            raise TypeError(
                "problem must be ProblemExample, "
                f"got {type(problem).__name__}"
            )

        if problem.dataset != "livecodebench_v6":
            raise ValueError(
                "Evaluator currently supports only "
                "livecodebench_v6, got "
                f"{problem.dataset}"
            )