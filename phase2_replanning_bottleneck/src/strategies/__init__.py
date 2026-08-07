"""Phase 2 refinement 전략.

세 전략 모두 동일한 인터페이스를 따른다.

    strategy.name : str
    strategy.run(case: FailureCase) -> RefinementOutput

입력이 되는 FailureCase(initial code + execution feedback)는
Phase 1 direct 결과에서 그대로 로딩한 값이므로,
세 전략은 완전히 동일한 출발점을 공유한다.

    feedback_repair : feedback + initial code       -> code
    self_replan     : feedback + initial code       -> plan -> code
    teacher_replan  : feedback + teacher revised plan -> code
"""

from src.strategies.feedback_repair import (
    FeedbackRepairStrategy,
)
from src.strategies.self_replan import (
    SelfReplanStrategy,
)
from src.strategies.teacher_replan import (
    TeacherReplanStrategy,
)

__all__ = [
    "FeedbackRepairStrategy",
    "SelfReplanStrategy",
    "TeacherReplanStrategy",
]
