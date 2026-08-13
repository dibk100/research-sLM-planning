"""

| 역할    | 원본 컬럼                                   | 용도                  |
| ----- | --------------------------------------- | ------------------- |
| 식별    | `id`                                    | 문제 고유 ID (`1796/F`) |
| 메타데이터 | `title`                                 | 문제명                 |
| 메타데이터 | `contest_start` | 출제 시점               |
| 메타데이터 | `rating`                                | Codeforces 난이도      |
| 모델 입력 | `description`                           | 문제 본문               |
| 모델 입력(보류) | `input_format`                          | 입력 형식               |
| 모델 입력(보류) | `output_format`                         | 출력 형식               |
| 모델 입력 | `examples`                              | Sample input/output |
| 모델 입력 | `note`                                  | 문제 설명에 포함되는 공식 note |
| 평가    | `official_tests`                        | 실제 채점 테스트           |
| 실행 설정 | `time_limit`                            | 문제별 실행 제한           |
| 실행 설정 | `memory_limit`                          | 문제별 메모리 제한          |

`examples`       → public_tests
`official_tests` → private_tests

# Note.
- problem 문자열을 만드는 작업을 함.

"""
# src/datasets/codeforces.py

from pathlib import Path

from datasets import load_from_disk

from src.schemas import ProblemExample


def _format_examples():
    # Todo
    return

def _build_problem_statement(row: dict) -> str:
    parts = []

    if row.get("description"):
        parts.append(row["description"].strip())

    if row.get("input_format"):
        parts.append(
            "Input\n\n"
            + row["input_format"].strip()
        )

    if row.get("output_format"):
        parts.append(
            "Output\n\n"
            + row["output_format"].strip()
        )

    if row.get("examples"):
        parts.append(
            _format_examples(row["examples"])
        )

    if row.get("note"):
        parts.append(
            "Note\n\n"
            + row["note"].strip()
        )

    return "\n\n".join(parts)

def load_codeforces(
    data_path: str | Path,
) -> list[ProblemExample]:

    dataset = load_from_disk(str(data_path))

    problems = []

    for row in dataset:
        problem_text = _build_problem_statement(row)

        problem = ProblemExample(
            problem_id=str(row["id"]),
            title=row["title"],

            problem=problem_text,
            starter_code="",

            dataset="codeforces",
            platform="codeforces",

            difficulty=None,
            rating=row["rating"],
            contest_date=_convert_timestamp(
                row["contest_start"]
            ),

            evaluation_type="stdin",

            public_tests=row["examples"] or [],
            private_tests=row["official_tests"] or [],

            time_limit=row["time_limit"],
            memory_limit=row["memory_limit"],
        )

        problems.append(problem)

    return problems
