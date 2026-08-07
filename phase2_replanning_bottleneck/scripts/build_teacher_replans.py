"""
Teacher Re-plan batch export / build script.

Phase 2-3 Teacher-Replanning Regeneration용 teacher re-plan 데이터를 구축한다.

Phase 1 Teacher Plan:
    Problem
        -> Teacher Plan

Phase 2 Teacher Re-plan:
    Problem
    + Failed Initial Code
    + Execution Feedback
        -> Teacher Revised Plan


Modes
-----

1. export

Phase 1 failure cases를 batch JSON 파일로 export한다.

Example:
    PYTHONPATH=. python scripts/build_teacher_replans.py export \
        --limit 50

Output:
    /mnt/hdd/project_sLM_planning/data/teacher_replans/_v1_work/
        order.json
        cases/
            b000.json
            b010.json
            ...


2. build

teacher가 작성한 replans/*.json을 읽어 최종 JSONL을 생성한다.

Example:
    PYTHONPATH=. python scripts/build_teacher_replans.py build \
        --teacher-model claude-opus-4.1 \
        --replan-version v1 \
        --verified

Input:
    replans/
        b000.json
        b010.json
        ...

Output:
    /mnt/hdd/project_sLM_planning/data/teacher_plans/
        livecodebench_v6_teacher_replans_opus5_v1_50.jsonl


Expected replan batch format
----------------------------

{
    "1873_A": "- ...\\n- ...",
    "1873_B": "- ...\\n- ..."
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.datasets.phase1_failure_loader import (
    Phase1FailureLoader,
)
from src.schemas import (
    FailureCase,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PHASE1_RESULTS_PATH = Path(
    "/mnt/hdd/project_sLM_planning/"
    "output/direct_500_stdin/results.jsonl"
)

DEFAULT_WORK_DIR = Path(
    "/mnt/hdd/project_sLM_planning/"
    "data/teacher_replans/_v1_work"
)

DEFAULT_OUTPUT_PATH = Path(
    "/mnt/hdd/project_sLM_planning/"
    "data/teacher_plans/"
    "livecodebench_v6_teacher_replans_opus5_v1_50.jsonl"
)

DEFAULT_BATCH_SIZE = 10


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Phase 1 failure trajectories "
            "and build Teacher Re-plan JSONL."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------

    export_parser = subparsers.add_parser(
        "export",
        help=(
            "Export Phase 1 failure cases "
            "into teacher-input batches."
        ),
    )

    export_parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_PHASE1_RESULTS_PATH,
    )

    export_parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
    )

    export_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help=(
            "Number of eligible Phase 1 failures "
            "to export."
        ),
    )

    export_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    export_parser.add_argument(
        "--max-feedback-chars",
        type=int,
        default=2000,
    )

    export_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing batch files.",
    )

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    build_parser = subparsers.add_parser(
        "build",
        help=(
            "Build final teacher re-plan JSONL "
            "from replans/*.json."
        ),
    )

    build_parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_PHASE1_RESULTS_PATH,
    )

    build_parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
    )

    build_parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    build_parser.add_argument(
        "--teacher-model",
        type=str,
        required=True,
    )

    build_parser.add_argument(
        "--replan-version",
        type=str,
        default="v1",
    )

    build_parser.add_argument(
        "--verified",
        action="store_true",
    )

    build_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Allow building a JSONL even when some "
            "problem_ids do not yet have re-plans."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# JSON utilities
# ---------------------------------------------------------------------------


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
        )

        file.write("\n")

    temporary_path.replace(path)


def read_json(
    path: Path,
) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON: {path}"
        ) from error


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )

            file.write("\n")

    temporary_path.replace(path)


# ---------------------------------------------------------------------------
# Teacher prompt
# ---------------------------------------------------------------------------


def build_teacher_prompt(
    case: FailureCase,
) -> str:
    """
    teacher가 실제 revised plan을 생성할 때 사용할 prompt.

    Self-Replan과 비교 가능하도록:
    - 최대 6 bullet
    - 각 bullet "- "
    - source code / pseudocode 금지

    형식으로 제한한다.
    """

    return f"""You are given a competitive programming problem, an incorrect previous solution, and execution feedback from that solution.

The previous solution failed.

Produce a concise revised solution plan that corrects the failure.

Requirements:
- Return at most 6 bullet points.
- Each bullet must start with "- ".
- Focus only on the corrected algorithm or logic.
- Use the problem statement, failed solution, and execution feedback to revise the failed approach.
- Identify the underlying cause of the failure and correct it in the revised plan.
- Do not include source code.
- Do not include pseudocode.
- Do not include explanations before or after the bullet list.

[Problem]
{case.example.prompt}

[Failed Previous Solution]
{case.initial_code}

[Execution Feedback]
{case.feedback.feedback_text}

[Revised Solution Plan]
""".strip()


# ---------------------------------------------------------------------------
# Failure case serialization
# ---------------------------------------------------------------------------


def build_case_record(
    case: FailureCase,
) -> dict[str, Any]:
    """
    Teacher에게 넘길 batch input record.

    최종 JSONL schema가 아니라 teacher 작성용 입력이다.
    """

    return {
        "problem_id": (
            case.example.problem_id
        ),

        "title": (
            case.example.title
        ),

        "difficulty": (
            case.example.difficulty
        ),

        "problem": (
            case.example.prompt
        ),

        "initial_code": (
            case.initial_code
        ),

        "execution_feedback": (
            case.feedback.feedback_text
        ),

        "initial_status": (
            case.initial_status
        ),

        "initial_passed_tests": (
            case.initial_passed_tests
        ),

        "initial_total_tests": (
            case.initial_total_tests
        ),

        # teacher에게 바로 복사할 수 있는 완성 prompt
        "teacher_prompt": (
            build_teacher_prompt(case)
        ),
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_batches(
    *,
    results_path: Path,
    work_dir: Path,
    limit: int,
    batch_size: int,
    max_feedback_chars: int,
    overwrite: bool,
) -> None:
    if limit <= 0:
        raise ValueError(
            "limit must be greater than 0."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than 0."
        )

    cases_dir = (
        work_dir / "cases"
    )

    replans_dir = (
        work_dir / "replans"
    )

    cases_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    replans_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    loader = Phase1FailureLoader(
        results_path,
        limit=limit,
        max_feedback_chars=max_feedback_chars,
    )

    cases = list(
        loader.load()
    )

    if not cases:
        raise ValueError(
            "No eligible Phase 1 failures found."
        )

    order = [
        case.example.problem_id
        for case in cases
    ]

    order_path = (
        work_dir / "order.json"
    )

    if (
        order_path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            "order.json already exists. "
            "Use --overwrite to replace it: "
            f"{order_path}"
        )

    write_json(
        order_path,
        order,
    )

    batch_count = 0

    for start in range(
        0,
        len(cases),
        batch_size,
    ):
        batch_cases = cases[
            start:start + batch_size
        ]

        batch_name = (
            f"b{start:03d}.json"
        )

        batch_path = (
            cases_dir / batch_name
        )

        if (
            batch_path.exists()
            and not overwrite
        ):
            raise FileExistsError(
                "Batch file already exists. "
                "Use --overwrite to replace it: "
                f"{batch_path}"
            )

        records = [
            build_case_record(case)
            for case in batch_cases
        ]

        write_json(
            batch_path,
            records,
        )

        batch_count += 1

        print(
            f"[EXPORTED] {batch_name} "
            f"({len(records)} cases)"
        )

    print()
    print("=" * 80)
    print("Teacher Re-plan Export Summary")
    print("=" * 80)

    print(
        "Phase 1 results :",
        results_path,
    )

    print(
        "Work dir        :",
        work_dir,
    )

    print(
        "Cases           :",
        len(cases),
    )

    print(
        "Batch size      :",
        batch_size,
    )

    print(
        "Batches         :",
        batch_count,
    )

    print(
        "Order           :",
        order_path,
    )

    print()
    print(
        "[DONE] Teacher re-plan input export completed."
    )


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------


def validate_teacher_replan(
    *,
    problem_id: str,
    text: Any,
) -> str:
    if not isinstance(
        text,
        str,
    ):
        raise TypeError(
            "Teacher re-plan must be a string "
            f"for problem_id={problem_id}, "
            f"got {type(text).__name__}."
        )

    replan = text.strip()

    if not replan:
        raise ValueError(
            "Teacher re-plan is empty for "
            f"problem_id={problem_id}"
        )

    lines = [
        line.strip()
        for line in replan.splitlines()
        if line.strip()
    ]

    if len(lines) > 6:
        raise ValueError(
            "Teacher re-plan exceeds 6 bullets "
            f"for problem_id={problem_id}: "
            f"{len(lines)}"
        )

    invalid_lines = [
        line
        for line in lines
        if not line.startswith("- ")
    ]

    if invalid_lines:
        raise ValueError(
            "Teacher re-plan contains non-bullet "
            f"lines for problem_id={problem_id}: "
            f"{invalid_lines[:3]}"
        )

    return replan


# ---------------------------------------------------------------------------
# Replan batch loading
# ---------------------------------------------------------------------------


def load_replan_batches(
    replans_dir: Path,
) -> dict[str, str]:
    """
    replans/*.json을 모두 합쳐
    problem_id -> replan mapping으로 반환한다.
    """

    if not replans_dir.exists():
        raise FileNotFoundError(
            "Teacher replan directory "
            f"not found: {replans_dir}"
        )

    batch_paths = sorted(
        replans_dir.glob("b*.json")
    )

    if not batch_paths:
        raise ValueError(
            "No teacher replan batch files found: "
            f"{replans_dir}"
        )

    replans: dict[str, str] = {}

    for path in batch_paths:
        data = read_json(path)

        if not isinstance(
            data,
            dict,
        ):
            raise TypeError(
                "Teacher replan batch must be "
                f"a JSON object: {path}"
            )

        for raw_problem_id, raw_text in data.items():
            problem_id = str(
                raw_problem_id
            ).strip()

            if not problem_id:
                raise ValueError(
                    "Empty problem_id in "
                    f"{path}"
                )

            if problem_id in replans:
                raise ValueError(
                    "Duplicate teacher re-plan for "
                    f"problem_id={problem_id}"
                )

            replans[
                problem_id
            ] = validate_teacher_replan(
                problem_id=problem_id,
                text=raw_text,
            )

        print(
            f"[LOADED] {path.name} "
            f"({len(data)} replans)"
        )

    return replans


# ---------------------------------------------------------------------------
# FailureCase mapping
# ---------------------------------------------------------------------------


def load_failure_case_map(
    *,
    results_path: Path,
    expected_problem_ids: list[str],
) -> dict[str, FailureCase]:
    """
    order.json 대상 문제의 FailureCase를 복원한다.

    loader의 ordering을 이용하되 problem_id 기반으로 다시 mapping한다.
    """

    expected = set(
        expected_problem_ids
    )

    cases: dict[
        str,
        FailureCase,
    ] = {}

    loader = Phase1FailureLoader(
        results_path,
        limit=None,
    )

    for case in loader.load():
        problem_id = (
            case.example.problem_id
        )

        if problem_id not in expected:
            continue

        if problem_id in cases:
            raise ValueError(
                "Duplicate Phase 1 failure "
                f"problem_id={problem_id}"
            )

        cases[
            problem_id
        ] = case

        if len(cases) == len(expected):
            break

    missing = [
        problem_id
        for problem_id in expected_problem_ids
        if problem_id not in cases
    ]

    if missing:
        raise ValueError(
            "Could not reconstruct Phase 1 "
            "FailureCases for problem_ids: "
            f"{missing[:10]}"
        )

    return cases


# ---------------------------------------------------------------------------
# Final record
# ---------------------------------------------------------------------------


def build_final_record(
    *,
    case: FailureCase,
    teacher_replan: str,
    teacher_model: str,
    replan_version: str,
    verified: bool,
) -> dict[str, Any]:
    """
    TeacherReplanStore가 읽을 최종 schema.
    """

    return {
        "problem_id": (
            case.example.problem_id
        ),

        "teacher_replan": (
            teacher_replan
        ),

        "teacher_model": (
            teacher_model
        ),

        "teacher_interface": (
            "external_batch"
        ),

        "replan_method": (
            "failure_to_replan"
        ),

        "replan_version": (
            replan_version
        ),

        "verified": (
            verified
        ),

        "initial_status": (
            case.initial_status
        ),

        "initial_passed_tests": (
            case.initial_passed_tests
        ),

        "initial_total_tests": (
            case.initial_total_tests
        ),
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_final_jsonl(
    *,
    results_path: Path,
    work_dir: Path,
    output_path: Path,
    teacher_model: str,
    replan_version: str,
    verified: bool,
    allow_partial: bool,
) -> None:
    order_path = (
        work_dir / "order.json"
    )

    replans_dir = (
        work_dir / "replans"
    )

    order = read_json(
        order_path
    )

    if not isinstance(
        order,
        list,
    ):
        raise TypeError(
            "order.json must contain "
            "a JSON list."
        )

    problem_ids = [
        str(problem_id).strip()
        for problem_id in order
    ]

    if not problem_ids:
        raise ValueError(
            "order.json is empty."
        )

    if len(problem_ids) != len(
        set(problem_ids)
    ):
        raise ValueError(
            "order.json contains duplicate "
            "problem_ids."
        )

    teacher_replans = (
        load_replan_batches(
            replans_dir
        )
    )

    unexpected = [
        problem_id
        for problem_id
        in teacher_replans
        if problem_id not in set(problem_ids)
    ]

    if unexpected:
        raise ValueError(
            "Teacher replans contain problem_ids "
            "not present in order.json: "
            f"{unexpected[:10]}"
        )

    missing_replans = [
        problem_id
        for problem_id in problem_ids
        if problem_id not in teacher_replans
    ]

    if (
        missing_replans
        and not allow_partial
    ):
        raise ValueError(
            "Teacher replans are incomplete. "
            f"Missing {len(missing_replans)} "
            "problem_ids. First missing: "
            f"{missing_replans[:10]}"
        )

    selected_ids = [
        problem_id
        for problem_id in problem_ids
        if problem_id in teacher_replans
    ]

    case_map = (
        load_failure_case_map(
            results_path=results_path,
            expected_problem_ids=(
                selected_ids
            ),
        )
    )

    records: list[
        dict[str, Any]
    ] = []

    for problem_id in selected_ids:
        case = case_map[
            problem_id
        ]

        teacher_replan = (
            teacher_replans[
                problem_id
            ]
        )

        record = build_final_record(
            case=case,
            teacher_replan=teacher_replan,
            teacher_model=teacher_model,
            replan_version=replan_version,
            verified=verified,
        )

        records.append(
            record
        )

    write_jsonl(
        output_path,
        records,
    )

    print()
    print("=" * 80)
    print("Teacher Re-plan Build Summary")
    print("=" * 80)

    print(
        "Order size        :",
        len(problem_ids),
    )

    print(
        "Replans available :",
        len(teacher_replans),
    )

    print(
        "Records built     :",
        len(records),
    )

    print(
        "Missing replans   :",
        len(missing_replans),
    )

    print(
        "Teacher model     :",
        teacher_model,
    )

    print(
        "Replan version    :",
        replan_version,
    )

    print(
        "Verified          :",
        verified,
    )

    print(
        "Output            :",
        output_path,
    )

    print()

    if missing_replans:
        print(
            "Missing IDs:"
        )

        for problem_id in (
            missing_replans[:20]
        ):
            print(
                "  -",
                problem_id,
            )

        print()

    print(
        "[DONE] Teacher re-plan JSONL build completed."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.command == "export":
        export_batches(
            results_path=args.results_path,
            work_dir=args.work_dir,
            limit=args.limit,
            batch_size=args.batch_size,
            max_feedback_chars=(
                args.max_feedback_chars
            ),
            overwrite=args.overwrite,
        )

        return

    if args.command == "build":
        build_final_jsonl(
            results_path=args.results_path,
            work_dir=args.work_dir,
            output_path=args.output_path,
            teacher_model=args.teacher_model,
            replan_version=(
                args.replan_version
            ),
            verified=args.verified,
            allow_partial=(
                args.allow_partial
            ),
        )

        return

    raise ValueError(
        f"Unsupported command: {args.command}"
    )


if __name__ == "__main__":
    main()