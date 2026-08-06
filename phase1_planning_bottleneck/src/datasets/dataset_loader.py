"""
벤치마크 문제 로딩.

- 원본 데이터셋 로드
- 내부 ProblemExample 형식으로 변환
- subset 선택
- problem ID 기준 필터링
- 문제 순서 고정


초기에 확인할 것 :
examples = loader.load()

print(len(examples))
print(examples[0].problem_id)
print(examples[0].prompt)


Dataset Loader 완료 조건 :
- 문제 1개 이상 정상 로드
- problem_id가 고유함
- prompt와 test 정보가 누락되지 않음
- 같은 설정에서 항상 같은 문제 순서가 나옴

HumanEval+, MBPP+, APPS, CodeContests, Codeforces, LeetCode 등 다양한 문제를 로드할 수 있도록 구현 필요
우선 Livecodebench-v6 문제를 로드하는 DatasetLoader 구현
"""

from collections.abc import Iterator

from src.schemas import ProblemExample


class DatasetLoader:
    def __init__(
        self,
        dataset_name: str,
        split: str = "test",
        limit: int | None = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.limit = limit

    def load(self) -> list[ProblemExample]:
        if self.dataset_name == "humaneval":
            examples = self._load_humaneval()
        else:
            raise ValueError(
                f"Unsupported dataset: {self.dataset_name}"
            )

        if self.limit is not None:
            examples = examples[: self.limit]

        return examples

    def _load_humaneval(self) -> list[ProblemExample]:
        raise NotImplementedError