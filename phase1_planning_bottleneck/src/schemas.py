"""
공통 데이터 구조 : 실험 전반에서 공유하는 데이터 스키마 설정

Direct 단계부터 strategy="direct"를 기록하고, 이후 동일한 스키마로 self_plan, teacher_plan을 저장할 수 있도록 구성함.

"""

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
class StrategyOutput:
    problem_id: str
    strategy: str
    formatted_prompt: str
    raw_output: str

    prompt_tokens: int
    completion_tokens: int
    generation_time: float


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

    error_message: str | None = None
    test_results: list[dict[str, Any]] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)