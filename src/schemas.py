"""
공통 데이터 구조

ProblemExample
    ↓
GenerationOutput
    ↓
GenerationStep / StrategyOutput
    ↓
CodeParseResult
    ↓
TestCaseResult / EvaluationResult
    ↓
ExperimentRecord

### Note.
actual_output: PASS cases may be unavailable depending on runner metadata
"""

from dataclasses import asdict, dataclass, field
from typing import Any


# ======================================================================
# Problem
# ======================================================================

@dataclass
class ProblemExample:
    # Identity
    problem_id: str
    title: str

    # Model input
    problem: str
    starter_code: str = ""

    # Dataset / source metadata
    dataset: str = ""
    platform: str = ""

    # Difficulty / temporal metadata
    difficulty: str | None = None
    rating: int | None = None
    contest_date: str = ""

    # Evaluation
    evaluation_type: str = "stdin"

    public_tests: list[dict[str, Any]] = field(
        default_factory=list
    )

    private_tests: list[dict[str, Any]] = field(
        default_factory=list
    )

    # Execution constraints
    time_limit: float | None = None
    memory_limit: float | None = None

    # Functional evaluation
    function_name: str | None = None


# ======================================================================
# Generation
# ======================================================================

@dataclass
class GenerationOutput:
    text: str
    prompt_tokens: int
    completion_tokens: int
    generation_time: float


@dataclass
class GenerationStep:
    name: str
    formatted_prompt: str
    raw_output: str

    prompt_tokens: int
    completion_tokens: int
    generation_time: float


@dataclass
class StrategyOutput:
    problem_id: str
    strategy: str

    # Final generation stage
    formatted_prompt: str
    raw_output: str

    # Total strategy generation cost
    prompt_tokens: int
    completion_tokens: int
    generation_time: float

    # Generation trajectory
    strategy_trace: list[GenerationStep] = field(
        default_factory=list
    )

    # Teacher-plan provenance
    teacher_plan: str | None = None
    teacher_plan_source: str | None = None
    teacher_plan_version: str | None = None
    teacher_plan_verified: bool | None = None


# ======================================================================
# Parsing
# ======================================================================

@dataclass
class CodeParseResult:
    code: str
    status: str
    extraction_method: str


# ======================================================================
# Evaluation
# ======================================================================

@dataclass
class TestCaseResult:
    test_index: int
    passed: bool                                # 해당 unit test 하나를 통과했는지
    status: str                                 # 해당 테스트의 실행/채점 상태 : "WRONG_ANSWER", "PASS", "TIMEOUT" 등

    input_text: str                             # 생성된 코드를 실행할 때 넣은 test input
    expected_output: str
    actual_output: str

    execution_time: float

    return_code: int | None = None
    stderr: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class EvaluationResult:
    """
    Exhaustive evaluation result.

    All selected test cases are evaluated.

    passed=True only when every selected test case passes.
    """

    passed: bool
    status: str

    passed_tests: int
    total_tests: int

    execution_time: float

    test_results: list[TestCaseResult] = field(
        default_factory=list
    )

    error_message: str | None = None

    @property
    def test_pass_ratio(self) -> float:
        if self.total_tests == 0:
            return 0.0

        return (
            self.passed_tests
            / self.total_tests
        )


# ======================================================================
# Experiment Record
# ======================================================================

@dataclass
class ExperimentRecord:
    # ------------------------------------------------------------------
    # Experiment identity
    # ------------------------------------------------------------------

    problem_id: str
    dataset: str
    strategy: str
    model_name: str
    seed: int

    # ------------------------------------------------------------------
    # Problem metadata
    # ------------------------------------------------------------------

    title: str
    platform: str
    contest_date: str

    difficulty: str | None
    rating: int | None

    # ------------------------------------------------------------------
    # Problem / model input
    # ------------------------------------------------------------------

    problem: str
    formatted_prompt: str

    # ------------------------------------------------------------------
    # Generation / parsing
    # ------------------------------------------------------------------

    raw_output: str
    extracted_code: str

    parse_status: str
    extraction_method: str

    prompt_tokens: int
    completion_tokens: int
    generation_time: float

    # ------------------------------------------------------------------
    # Exhaustive evaluation
    # ------------------------------------------------------------------

    passed: bool
    status: str

    passed_tests: int
    total_tests: int
    test_pass_ratio: float

    execution_time: float

    # ------------------------------------------------------------------
    # Optional evaluation details
    # ------------------------------------------------------------------

    error_message: str | None = None

    test_results: list[dict[str, Any]] = field(
        default_factory=list
    )

    # ------------------------------------------------------------------
    # Strategy trajectory
    # ------------------------------------------------------------------

    strategy_trace: list[dict[str, Any]] = field(
        default_factory=list
    )

    # ------------------------------------------------------------------
    # Teacher-plan provenance
    # ------------------------------------------------------------------

    teacher_plan: str | None = None
    teacher_plan_source: str | None = None
    teacher_plan_version: str | None = None
    teacher_plan_verified: bool | None = None

    # Evaluation/data schema changed from official+diagnostic
    # to exhaustive-main evaluation.
    schema_version: str = "3.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)