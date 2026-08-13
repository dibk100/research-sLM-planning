"""
공통 데이터 구조 : 실험 전반에서 공유하는 데이터 스키마 설정


ProblemExample
    ↓
GenerationOutput
    ↓
GenerationStep / StrategyOutput
    ↓
TestCaseResult / EvaluationResult
    ↓
ExperimentRecord

"""
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ProblemExample:
    # Identity
    problem_id: str
    title: str

    # Model input
    problem: str                            # livecodebench : question_content
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
    public_tests: list[dict[str, Any]] = field(default_factory=list)            # 모델 입력에 공개될 수 있는 sample/example tests
    private_tests: list[dict[str, Any]] = field(default_factory=list)           # 최종 correctness 평가에 사용하는 hidden/official tests

    # Execution constraints
    time_limit: float | None = None
    memory_limit: float | None = None
    
    function_name: str | None = None

@dataclass
class GenerationOutput:
    text: str
    prompt_tokens: int
    completion_tokens: int
    generation_time: float

@dataclass
class CodeParseResult:
    code: str
    status: str
    extraction_method: str
    
"""
공통 데이터 구조 : 실험 전반에서 공유하는 데이터 스키마 설정


ProblemExample
    ↓
GenerationOutput
    ↓
GenerationStep / StrategyOutput
    ↓
TestCaseResult / EvaluationResult
    ↓
ExperimentRecord

"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProblemExample:
    # Identity
    problem_id: str
    title: str

    # Model input
    problem: str                            # livecodebench : question_content
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
    public_tests: list[dict[str, Any]] = field(default_factory=list)            # 모델 입력에 공개될 수 있는 sample/example tests
    private_tests: list[dict[str, Any]] = field(default_factory=list)           # 최종 correctness 평가에 사용하는 hidden/official tests

    # Execution constraints
    time_limit: float | None = None
    memory_limit: float | None = None
    
    function_name: str | None = None

@dataclass
class GenerationOutput:
    text: str
    prompt_tokens: int
    completion_tokens: int
    generation_time: float

@dataclass
class CodeParseResult:
    code: str
    status: str
    extraction_method: str
    
@dataclass
class ExperimentRecord:
    # Experiment identity
    problem_id: str
    dataset: str
    strategy: str
    model_name: str
    seed: int

    # Problem metadata
    title: str
    platform: str
    contest_date: str
    difficulty: str | None
    rating: int | None

    # Problem and model input
    problem: str
    formatted_prompt: str

    # Generation / Parsing
    raw_output: str
    extracted_code: str
    parse_status: str
    extraction_method: str

    prompt_tokens: int
    completion_tokens: int
    generation_time: float

    # Official evaluation
    passed: bool
    status: str
    execution_time: float

    # Optional
    error_message: str | None = None

    # Diagnostic evaluation
    diagnostic_status: str | None = None
    diagnostic_passed_tests: int | None = None
    diagnostic_total_tests: int | None = None
    diagnostic_test_pass_ratio: float | None = None
    diagnostic_execution_time: float | None = None

    diagnostic_test_results: list[dict[str, Any]] = field(
        default_factory=list
    )

    strategy_trace: list[dict[str, Any]] = field(
        default_factory=list
    )

    teacher_plan: str | None = None
    teacher_plan_source: str | None = None
    teacher_plan_version: str | None = None
    teacher_plan_verified: bool | None = None

    schema_version: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

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

    formatted_prompt: str
    raw_output: str

    prompt_tokens: int
    completion_tokens: int
    generation_time: float

    strategy_trace: list[GenerationStep] = field(default_factory=list)

    teacher_plan: str | None = None
    teacher_plan_source: str | None = None
    teacher_plan_version: str | None = None
    teacher_plan_verified: bool | None = None


@dataclass
class TestCaseResult:
    test_index: int
    passed: bool
    status: str

    input_text: str
    expected_output: str
    actual_output: str

    execution_time: float
    return_code: int | None = None
    stderr: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    passed: bool
    status: str

    passed_tests: int
    total_tests: int
    execution_time: float

    test_results: list[TestCaseResult] = field(default_factory=list)
    error_message: str | None = None

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

    formatted_prompt: str
    raw_output: str

    prompt_tokens: int
    completion_tokens: int
    generation_time: float

    strategy_trace: list[GenerationStep] = field(default_factory=list)

    teacher_plan: str | None = None
    teacher_plan_source: str | None = None
    teacher_plan_version: str | None = None
    teacher_plan_verified: bool | None = None


@dataclass
class TestCaseResult:
    test_index: int
    passed: bool
    status: str

    input_text: str
    expected_output: str
    actual_output: str

    execution_time: float
    return_code: int | None = None
    stderr: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    passed: bool
    status: str

    passed_tests: int
    total_tests: int
    execution_time: float

    test_results: list[TestCaseResult] = field(default_factory=list)
    error_message: str | None = None
