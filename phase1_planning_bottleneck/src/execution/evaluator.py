"""
테스트 실행 및 평가 결과 산출.

선택한 벤치마크의 공식 평가 코드를 감싸는 wrapper 형태로 구현할 것.
생성된 코드를 현재 Python 프로세스에서 직접 exec()하는 방식은 피하는 것이 좋다. 
다음 중 하나를 사용해야 한다.

- 벤치마크 공식 evaluator
- subprocess 기반 격리
- Docker sandbox
- EvalPlus 공식 evaluation pipeline

초기 MVP에서는 공식 evaluator wrapper를 사용하는 것이 가장 안정적이라고 함.

"""

import time
from typing import Protocol

from src.schemas import EvaluationResult, ProblemExample


class BenchmarkEvaluator(Protocol):
    def evaluate(
        self,
        example: ProblemExample,
        code: str,
    ) -> EvaluationResult:
        ...


class Evaluator:
    def __init__(self, benchmark_evaluator: BenchmarkEvaluator):
        self.benchmark_evaluator = benchmark_evaluator

    def evaluate(
        self,
        example: ProblemExample,
        code: str,
    ) -> EvaluationResult:
        start_time = time.perf_counter()

        try:
            result = self.benchmark_evaluator.evaluate(
                example=example,
                code=code,
            )
            return result

        except TimeoutError as error:
            return EvaluationResult(
                passed=False,
                status="TIMEOUT",
                passed_tests=0,
                total_tests=0,
                execution_time=time.perf_counter() - start_time,
                stderr=str(error),
            )

        except SyntaxError as error:
            return EvaluationResult(
                passed=False,
                status="SYNTAX_ERROR",
                passed_tests=0,
                total_tests=0,
                execution_time=time.perf_counter() - start_time,
                stderr=str(error),
            )

        except Exception as error:
            return EvaluationResult(
                passed=False,
                status="EVALUATION_ERROR",
                passed_tests=0,
                total_tests=0,
                execution_time=time.perf_counter() - start_time,
                stderr=repr(error),
            )