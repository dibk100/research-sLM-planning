"""
Teacher re-plan 작성용 배치 준비 / 빌드 스크립트.

Phase 1의 teacher plan은 "문제만 보고" 작성했지만,
Phase 2의 teacher re-plan은 "문제 + 실패한 initial code + execution feedback"을
보고 작성해야 한다. 따라서 teacher에게 넘길 입력 자체를 Phase 1 결과에서
추출해 두어야 한다.

두 가지 모드
------------
export : Phase 1 실패 케이스를 10건 단위 배치 JSON으로 떨어뜨린다
         (teacher가 읽을 입력. 문제 / initial code / feedback 포함)

build  : 작성된 배치별 replan JSON을 모아 최종 JSONL로 합치고 형식을 검증한다

    작업 디렉터리 구조 (Phase 1의 _v1_500_work 와 동일한 방식)

    /mnt/hdd/project_sLM_planning/data/teacher_replans/_v1_work/
        order.json            # 대상 problem_id 순서
        cases/b000.json ...   # teacher 입력 배치
        replans/b000.json ... # 작성된 replan (problem_id -> text)

Usage:

python -m scripts.build_teacher_replans export \
  --phase1-results /mnt/hdd/project_sLM_planning/output/direct_500_stdin/results.jsonl \
  --work-dir /mnt/hdd/project_sLM_planning/data/teacher_replans/_v1_work

python -m scripts.build_teacher_replans build \
  --work-dir /mnt/hdd/project_sLM_planning/data/teacher_replans/_v1_work \
  --output /mnt/hdd/project_sLM_planning/data/teacher_replans/replans_opus5_v1.jsonl

TODO(구현)
----------
- [ ] export : Phase1FailureLoader 로 케이스 로딩 후 배치 파일 기록
- [ ] build  : replans/*.json 병합 + bullet 형식 검증 + JSONL 기록
"""

from __future__ import annotations

import argparse

# Phase 1 teacher plan과 동일한 형식 규칙을 적용한다.
MAX_BULLETS = 6
BULLET_PREFIX = "- "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export teacher re-plan batches or build "
            "the final JSONL."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    export_parser = subparsers.add_parser(
        "export",
        help=(
            "Export phase1 failure cases as batch "
            "files for the teacher."
        ),
    )
    export_parser.add_argument(
        "--phase1-results",
        required=True,
    )
    export_parser.add_argument(
        "--work-dir",
        required=True,
    )
    export_parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
    )
    export_parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    build_parser = subparsers.add_parser(
        "build",
        help=(
            "Merge written replans into the final "
            "JSONL and validate the format."
        ),
    )
    build_parser.add_argument(
        "--work-dir",
        required=True,
    )
    build_parser.add_argument(
        "--output",
        required=True,
    )
    build_parser.add_argument(
        "--teacher-model",
        default="claude-opus-5",
    )
    build_parser.add_argument(
        "--plan-version",
        default="v1",
    )
    build_parser.add_argument(
        "--based-on",
        default="direct_500_stdin",
        help=(
            "어떤 phase1 실행 결과의 실패를 보고 "
            "작성한 replan인지 기록한다."
        ),
    )

    return parser.parse_args()


def run_export(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "TODO: FailureCase -> cases/bXXX.json 배치 기록"
    )


def run_build(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "TODO: replans/*.json 병합 및 형식 검증 후 JSONL 기록"
    )


def main() -> None:
    args = parse_args()

    if args.command == "export":
        run_export(args)
    else:
        run_build(args)


if __name__ == "__main__":
    main()
