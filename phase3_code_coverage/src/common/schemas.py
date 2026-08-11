
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ProblemExample:
    problem_id: str
    title: str
    prompt: str
    platform: str
    contest_id: str
    contest_date: str
    difficulty: str
    starter_code: str
    public_tests: list[dict[str, Any]]
    private_tests: list[dict[str, Any]]
    metadata: dict[str, Any]
    
    test_type: str
    source: str = "livecodebench_v6"


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

    # 최종 코드 생성 단계
    formatted_prompt: str
    raw_output: str

    # 전략 전체 비용
    prompt_tokens: int
    completion_tokens: int
    generation_time: float

    # 단계별 생성 기록
    strategy_trace: list[GenerationStep] = field(
        default_factory=list
    )

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


@dataclass
class EvaluationResult:
    passed: bool
    status: str

    passed_tests: int
    total_tests: int
    execution_time: float

    test_results: list[TestCaseResult] = field(
        default_factory=list
    )
    error_message: str | None = None


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
    contest_id: str
    contest_date: str
    difficulty: str

    # Problem and model input
    problem: str
    formatted_prompt: str

    # Generation
    raw_output: str
    extracted_code: str
    prompt_tokens: int
    completion_tokens: int
    generation_time: float

    # Evaluation
    passed: bool
    status: str
    passed_tests: int
    total_tests: int
    execution_time: float

    # Optional fields
    error_message: str | None = None
    test_results: list[dict[str, Any]] = field(
        default_factory=list
    )
    strategy_trace: list[dict[str, Any]] = field(
        default_factory=list
    )

    teacher_plan: str | None = None
    teacher_plan_source: str | None = None
    teacher_plan_version: str | None = None
    teacher_plan_verified: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
