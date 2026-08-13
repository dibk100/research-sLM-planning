# src/execution/evaluator.py

from __future__ import annotations

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


class Evaluator:
    """
    Unified evaluation interface.

    Primary evaluation:
        Exhaustive diagnostic evaluation.

        - Executes all selected test cases.
        - A problem passes only when every test passes.
        - Preserves per-test results for analysis.

    Official LiveCodeBench evaluation:
        Available only as an optional reference evaluator.
    """

    def __init__(
        self,
        *,
        timeout_seconds: int = 6,
        include_public_tests: bool = True,
        include_private_tests: bool = True,
        debug: bool = False,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than 0."
            )

        self.timeout_seconds = timeout_seconds
        self.include_public_tests = (
            include_public_tests
        )
        self.include_private_tests = (
            include_private_tests
        )
        self.debug = debug

        # Main evaluator used by experiments.
        self.diagnostic_evaluator = (
            DiagnosticEvaluator(
                timeout_seconds=timeout_seconds,
                debug=debug,
                include_public_tests=(
                    include_public_tests
                ),
                include_private_tests=(
                    include_private_tests
                ),
            )
        )

        # Reference evaluator.
        # Not used in the main experiment path.
        self.official_evaluator = (
            LiveCodeBenchEvaluator(
                timeout_seconds=timeout_seconds,
                debug=debug,
                include_public_tests=(
                    include_public_tests
                ),
                include_private_tests=(
                    include_private_tests
                ),
            )
        )

    def evaluate(
        self,
        problem: ProblemExample,
        code: str,
    ) -> EvaluationResult:
        """
        Run the primary exhaustive evaluation.

        Final PASS:
            all selected tests pass.

        Final FAIL:
            at least one selected test fails.

        All test cases are evaluated so that per-test outcomes
        and partial correctness can be retained.
        """

        self._validate_problem(problem)

        return self.diagnostic_evaluator.evaluate(
            problem=problem,
            code=code,
        )

    def evaluate_official(
        self,
        problem: ProblemExample,
        code: str,
    ) -> EvaluationResult:
        """
        Run the official LiveCodeBench evaluator.

        This method is retained only for reference,
        validation, or comparison.
        """

        self._validate_problem(problem)

        return self.official_evaluator.evaluate(
            problem=problem,
            code=code,
        )

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