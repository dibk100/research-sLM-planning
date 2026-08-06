"""
Teacher Plan 수동 구축 스크립트.

LiveCodeBench 문제를 불러와 Teacher Plan 생성 프롬프트를 출력하고,
외부 GPT/Claude에서 생성한 plan을 터미널로 입력받아 JSONL에 저장한다.

Usage:

1) 첫 10문제 구축

python -m scripts.build_teacher_plans \
  --limit 10

2) 특정 문제만 구축

python -m scripts.build_teacher_plans \
  --problem-id 1873_A

3) 기존 레코드도 다시 작성

python -m scripts.build_teacher_plans \
  --limit 10 \
  --overwrite-existing

4) 출력 파일 변경

python -m scripts.build_teacher_plans \
  --limit 10 \
  --output-path data/teacher_plans/manual_teacher_plans.jsonl

Teacher plan 입력 종료:
새로운 줄에 __END__ 입력
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.datasets.dataset_loader import DatasetLoader
from src.schemas import ProblemExample


DEFAULT_OUTPUT_PATH = Path(
    "data/teacher_plans/"
    "livecodebench_v6_teacher_plans.jsonl"
)

PLAN_END_MARKER = "__END__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively build teacher plans "
            "for LiveCodeBench problems."
        )
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Teacher-plan JSONL output path.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help=(
            "Number of stdin problems to load. "
            "Ignored when --problem-id is used "
            "except as the search range."
        ),
    )

    parser.add_argument(
        "--problem-id",
        default=None,
        help="Build a plan for one problem ID.",
    )

    parser.add_argument(
        "--teacher-model",
        default="manual_external_llm",
        help=(
            "Teacher model identifier stored "
            "in each record."
        ),
    )

    parser.add_argument(
        "--plan-version",
        default="v1",
        help="Plan version stored in each record.",
    )

    parser.add_argument(
        "--verified",
        action="store_true",
        help=(
            "Store newly entered plans as verified=true. "
            "Omit this flag for unverified plans."
        ),
    )

    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help=(
            "Allow rebuilding plans that already "
            "exist in the output file."
        ),
    )

    parser.add_argument(
        "--show-problem",
        action="store_true",
        help=(
            "Print the full problem separately before "
            "printing the teacher prompt."
        ),
    )

    return parser.parse_args()


def build_teacher_prompt(
    example: ProblemExample,
) -> str:
    starter_code_section = ""

    if example.starter_code.strip():
        starter_code_section = (
            "\n\nStarter Code:\n"
            f"{example.starter_code.strip()}"
        )

    return f"""You are an expert competitive programmer.

Read the following competitive programming problem.

Your task is NOT to write code. Write a concise implementation plan that another language model can follow to generate correct Python code.

Requirements:
- Return only the implementation plan.
- Do not write Python code or pseudocode.
- Use 5 to 8 bullet points.
- State the core algorithm precisely.
- State the exact decision condition or recurrence when applicable.
- Explain the key invariant or correctness reasoning.
- Mention important edge cases.
- Mention only the required data structures.
- State the time and space complexity.
- Do not repeat the problem statement.
- Ensure the plan is logically complete and sufficient for a correct implementation.

Problem Title:
{example.title}

Problem:
{example.prompt}{starter_code_section}
""".strip()


def read_existing_records(
    output_path: Path,
) -> dict[str, dict[str, Any]]:
    if not output_path.exists():
        return {}

    records: dict[str, dict[str, Any]] = {}

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: "
                    f"{output_path}"
                ) from error

            if not isinstance(record, dict):
                raise TypeError(
                    "Teacher plan record must be a JSON "
                    f"object at line {line_number}."
                )

            problem_id = str(
                record.get("problem_id", "")
            ).strip()

            if not problem_id:
                raise ValueError(
                    "Missing problem_id at line "
                    f"{line_number}: {output_path}"
                )

            if problem_id in records:
                raise ValueError(
                    "Duplicated existing teacher plan: "
                    f"{problem_id}"
                )

            records[problem_id] = record

    return records


def rewrite_records(
    output_path: Path,
    records: dict[str, dict[str, Any]],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records.values():
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            file.write("\n")

    temporary_path.replace(output_path)


def read_multiline_plan() -> str:
    print()
    print(
        "Paste the teacher plan below."
    )
    print(
        f"Enter {PLAN_END_MARKER} on a new line "
        "to finish."
    )
    print("-" * 80)

    lines: list[str] = []

    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip() == PLAN_END_MARKER:
            break

        lines.append(line)

    plan = "\n".join(lines).strip()

    if not plan:
        raise ValueError(
            "Teacher plan must not be empty."
        )

    return plan


def confirm_plan(
    plan: str,
) -> bool:
    print()
    print("=" * 80)
    print("Teacher Plan Preview")
    print("=" * 80)
    print(plan)
    print()

    while True:
        answer = input(
            "Save this plan? [y/n]: "
        ).strip().lower()

        if answer in {"y", "yes"}:
            return True

        if answer in {"n", "no"}:
            return False

        print("Please enter y or n.")


def select_examples(
    *,
    limit: int,
    problem_id: str | None,
) -> list[ProblemExample]:
    if limit <= 0:
        raise ValueError(
            "limit must be greater than 0."
        )

    # problem_id를 찾을 때는 전체 stdin subset을 검색한다.
    loader_limit = None if problem_id else limit

    loader = DatasetLoader(
        dataset_name="livecodebench_v6",
        split="test",
        limit=loader_limit,
        test_type="stdin",
        release_version="release_v6",
    )

    examples = loader.load()

    if problem_id is None:
        return examples

    matched = [
        example
        for example in examples
        if example.problem_id == problem_id
    ]

    if not matched:
        raise KeyError(
            f"Problem not found in stdin subset: "
            f"{problem_id}"
        )

    return matched


def build_record(
    *,
    example: ProblemExample,
    teacher_plan: str,
    teacher_model: str,
    plan_version: str,
    verified: bool,
) -> dict[str, Any]:
    return {
        "problem_id": example.problem_id,
        "teacher_plan": teacher_plan,
        "teacher_model": teacher_model,
        "teacher_interface": "manual_copy_paste",
        "plan_method": "problem_to_plan",
        "plan_version": plan_version,
        "verified": verified,
    }


def main() -> None:
    args = parse_args()

    examples = select_examples(
        limit=args.limit,
        problem_id=args.problem_id,
    )

    records = read_existing_records(
        args.output_path
    )

    print("=" * 80)
    print("Teacher Plan Builder")
    print("=" * 80)
    print(f"Selected problems : {len(examples)}")
    print(f"Output path       : {args.output_path}")
    print(f"Teacher model     : {args.teacher_model}")
    print(f"Plan version      : {args.plan_version}")
    print(f"Verified          : {args.verified}")
    print(
        f"Overwrite existing: "
        f"{args.overwrite_existing}"
    )
    print()

    saved_count = 0
    skipped_count = 0

    for index, example in enumerate(
        examples,
        start=1,
    ):
        if (
            example.problem_id in records
            and not args.overwrite_existing
        ):
            skipped_count += 1

            print(
                f"[{index}/{len(examples)}] "
                f"[SKIP] Existing plan: "
                f"{example.problem_id}"
            )
            continue

        print()
        print("=" * 100)
        print(
            f"[{index}/{len(examples)}] "
            f"{example.problem_id} | "
            f"{example.difficulty} | "
            f"{example.title}"
        )
        print("=" * 100)

        if args.show_problem:
            print()
            print("[Problem]")
            print(example.prompt)

        prompt = build_teacher_prompt(example)

        print()
        print("[Teacher Prompt]")
        print("-" * 100)
        print(prompt)
        print("-" * 100)

        print()
        print(
            "Copy the prompt above into the external "
            "teacher model."
        )

        while True:
            teacher_plan = read_multiline_plan()

            if confirm_plan(teacher_plan):
                break

            print()
            print(
                "The plan was not saved. "
                "Paste a replacement plan."
            )

        record = build_record(
            example=example,
            teacher_plan=teacher_plan,
            teacher_model=args.teacher_model,
            plan_version=args.plan_version,
            verified=args.verified,
        )

        records[example.problem_id] = record

        # 문제 하나를 완료할 때마다 저장하여
        # 중간 중단에도 결과를 보존한다.
        rewrite_records(
            output_path=args.output_path,
            records=records,
        )

        saved_count += 1

        print(
            f"[SAVED] {example.problem_id} "
            f"-> {args.output_path}"
        )

    print()
    print("=" * 80)
    print("Builder Summary")
    print("=" * 80)
    print(f"Selected : {len(examples)}")
    print(f"Saved    : {saved_count}")
    print(f"Skipped  : {skipped_count}")
    print(f"Total records in file: {len(records)}")
    print()
    print("[DONE] Teacher plan building completed.")


if __name__ == "__main__":
    main()