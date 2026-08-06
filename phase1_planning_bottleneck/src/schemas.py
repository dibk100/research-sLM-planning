"""
공통 데이터 구조 : 실험 전반에서 공유하는 데이터 스키마 설정

Direct 단계부터 strategy="direct"를 기록하고, 이후 동일한 스키마로 self_plan, teacher_plan을 저장할 수 있도록 구성함.

"""

from dataclasses import dataclass
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
    prompt: str
    raw_output: str

    prompt_tokens: int
    completion_tokens: int
    generation_time: float


@dataclass
class EvaluationResult:
    passed: bool
    status: str
    passed_tests: int
    total_tests: int
    execution_time: float
    stdout: str = ""
    stderr: str = ""


@dataclass
class GenerationRecord:
    problem_id: str
    dataset: str
    strategy: str
    model_name: str
    seed: int

    prompt: str
    raw_output: str
    code: str

    prompt_tokens: int
    completion_tokens: int
    generation_time: float

    passed: bool
    status: str
    passed_tests: int
    total_tests: int
    execution_time: float

    difficulty: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)