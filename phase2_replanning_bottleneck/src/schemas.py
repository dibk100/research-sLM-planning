"""
공통 데이터 구조 : Phase 2 refinement 실험에서 공유하는 데이터 스키마.

Phase 1의 스키마(ProblemExample / GenerationOutput / GenerationStep /
TestCaseResult / EvaluationResult)를 그대로 계승하고,
refinement 단계에 필요한 구조만 추가한다.

추가되는 개념
-------------
ExecutionFeedback
= initial code 실행 결과를 모델 입력용 텍스트로 정리한 것

FailureCase
= Phase 1 direct 결과 중 실패한 trajectory 하나
  (문제 + initial code + execution feedback)

RefinementOutput
= refinement 전략 1회 실행의 출력 (Phase 1 StrategyOutput의 refinement 판)

RefinementRecord
= results.jsonl 한 줄. initial_* (Phase 1에서 계승) 과
  refined_* (Phase 2에서 생성) 를 함께 보관하여
  recovery / regression 분석이 한 파일에서 가능하도록 한다.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Phase 1에서 계승한 구조
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase 2 전용 구조
# ---------------------------------------------------------------------------


@dataclass
class ExecutionFeedback:
    """initial code 실행 결과를 refinement 입력으로 정리한 것."""

    status: str
    passed_tests: int
    total_tests: int

    # 프롬프트에 실제로 주입되는 텍스트
    feedback_text: str

    # 첫 실패 테스트 (없으면 None)
    failed_test_index: int | None = None
    failed_input: str | None = None
    expected_output: str | None = None
    actual_output: str | None = None

    error_message: str | None = None
    stderr: str = ""


@dataclass
class FailureCase:
    """Phase 1 direct 결과 중 실패한 trajectory 하나."""

    example: ProblemExample

    # Phase 1에서 계승하는 initial state
    initial_code: str
    initial_raw_output: str
    initial_status: str
    initial_passed_tests: int
    initial_total_tests: int

    feedback: ExecutionFeedback

    # Phase 1 실행 비용 (누적 비용 계산용)
    initial_prompt_tokens: int = 0
    initial_completion_tokens: int = 0
    initial_generation_time: float = 0.0


@dataclass
class RefinementOutput:
    """refinement 전략 1회 실행의 출력."""

    problem_id: str
    strategy: str

    # 최종 코드 생성 단계
    formatted_prompt: str
    raw_output: str

    # 전략 전체 비용 (re-plan 호출 + code 호출 합계)
    prompt_tokens: int
    completion_tokens: int
    generation_time: float

    strategy_trace: list[GenerationStep] = field(
        default_factory=list
    )

    # self_replan 이 생성한 revised plan
    self_replan: str | None = None

    # teacher_replan 이 주입한 revised plan
    teacher_replan: str | None = None
    teacher_replan_source: str | None = None
    teacher_replan_version: str | None = None
    teacher_replan_verified: bool | None = None


@dataclass
class RefinementRecord:
    """results.jsonl 한 줄 (initial + refined 를 함께 보관)."""

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
    problem: str

    # Phase 1에서 계승한 initial state
    initial_code: str
    initial_status: str
    initial_passed: bool
    initial_passed_tests: int
    initial_total_tests: int
    initial_error_message: str | None

    # refinement 입력
    feedback_text: str

    # refinement 생성
    formatted_prompt: str
    raw_output: str
    refined_code: str
    prompt_tokens: int
    completion_tokens: int
    generation_time: float

    # refinement 평가
    refined_passed: bool
    refined_status: str
    refined_passed_tests: int
    refined_total_tests: int
    execution_time: float

    # 분석용 파생 필드
    # recovered : FAIL -> PASS
    # unchanged : FAIL -> FAIL
    recovered: bool
    test_pass_delta: int

    refined_error_message: str | None = None
    test_results: list[dict[str, Any]] = field(
        default_factory=list
    )
    strategy_trace: list[dict[str, Any]] = field(
        default_factory=list
    )

    self_replan: str | None = None
    teacher_replan: str | None = None
    teacher_replan_source: str | None = None
    teacher_replan_version: str | None = None
    teacher_replan_verified: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
