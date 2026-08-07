"""
TeacherReplanStore sanity check.

확인 항목:
1. Teacher re-plan JSONL이 정상 로드되는가?
2. problem_id로 entry를 조회할 수 있는가?
3. verified / bullet format validation이 정상 동작하는가?
4. Phase1 FailureCase와 failure state가 일치하는가?
5. get_for_failure()가 정상적으로 TeacherReplanEntry를 반환하는가?

Usage:
    PYTHONPATH=. python -m src.plans.inspect_teacher_replan_store
"""

from __future__ import annotations

from src.datasets.phase1_failure_loader import (
    Phase1FailureLoader,
)
from src.plans.teacher_replan_store import (
    TeacherReplanStore,
)


PHASE1_RESULTS_PATH = (
    "/mnt/hdd/project_sLM_planning/"
    "output/direct_500_stdin/results.jsonl"
)

TEACHER_REPLAN_PATH = (
    "/mnt/hdd/project_sLM_planning/"
    "data/teacher_plans/"
    "livecodebench_v6_teacher_replans_opus5_v1_seed.jsonl"
)

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Phase 1 failure 1개 로드
    # ------------------------------------------------------------------

    loader = Phase1FailureLoader(
        PHASE1_RESULTS_PATH,
        limit=1,
    )

    cases = list(loader.load())

    assert len(cases) == 1

    case = cases[0]

    print("=" * 100)
    print("TeacherReplanStore Sanity Check")
    print("=" * 100)

    print()
    print("[FailureCase]")

    print(
        "problem_id            :",
        case.example.problem_id,
    )

    print(
        "title                 :",
        case.example.title,
    )

    print(
        "difficulty            :",
        case.example.difficulty,
    )

    print(
        "initial_status        :",
        case.initial_status,
    )

    print(
        "initial_passed_tests  :",
        case.initial_passed_tests,
    )

    print(
        "initial_total_tests   :",
        case.initial_total_tests,
    )

    # ------------------------------------------------------------------
    # 2. TeacherReplanStore 로드
    # ------------------------------------------------------------------

    store = TeacherReplanStore(
        TEACHER_REPLAN_PATH,
        require_verified=True,
    )

    print()
    print("[Store]")

    print(
        "path          :",
        TEACHER_REPLAN_PATH,
    )

    print(
        "entry count   :",
        len(store),
    )

    assert len(store) > 0

    print(
        "[PASS] Teacher re-plan store loaded."
    )

    # ------------------------------------------------------------------
    # 3. has() 검사
    # ------------------------------------------------------------------

    problem_id = (
        case.example.problem_id
    )

    exists = store.has(
        problem_id
    )

    print()
    print("[Lookup]")

    print(
        "problem_id    :",
        problem_id,
    )

    print(
        "exists        :",
        exists,
    )

    assert exists, (
        "Teacher re-plan seed does not contain "
        f"problem_id={problem_id}"
    )

    print(
        "[PASS] problem_id exists in store."
    )

    # ------------------------------------------------------------------
    # 4. 단순 get() 검사
    # ------------------------------------------------------------------

    entry = store.get(
        problem_id
    )

    print()
    print("=" * 100)
    print("[TeacherReplanEntry]")
    print("=" * 100)

    print(
        "problem_id            :",
        entry.problem_id,
    )

    print(
        "teacher_model         :",
        entry.teacher_model,
    )

    print(
        "replan_version        :",
        entry.replan_version,
    )

    print(
        "verified              :",
        entry.verified,
    )

    print(
        "initial_status        :",
        entry.initial_status,
    )

    print(
        "initial_passed_tests  :",
        entry.initial_passed_tests,
    )

    print(
        "initial_total_tests   :",
        entry.initial_total_tests,
    )

    print()
    print("[teacher_replan]")
    print("-" * 100)

    print(
        entry.teacher_replan
    )

    print("-" * 100)

    assert (
        entry.problem_id
        == problem_id
    )

    assert entry.verified is True

    assert (
        entry.teacher_replan.strip()
    )

    # ------------------------------------------------------------------
    # 5. Bullet format 재확인
    # ------------------------------------------------------------------

    plan_lines = [
        line.strip()
        for line
        in entry.teacher_replan.splitlines()
        if line.strip()
    ]

    assert len(plan_lines) <= 6

    assert all(
        line.startswith("- ")
        for line in plan_lines
    )

    print()
    print(
        "[PASS] Teacher re-plan format is valid."
    )

    # ------------------------------------------------------------------
    # 6. Failure trajectory match 검사
    # ------------------------------------------------------------------

    matched_entry = (
        store.get_for_failure(
            case
        )
    )

    assert (
        matched_entry.problem_id
        == case.example.problem_id
    )

    assert (
        matched_entry.initial_status
        == case.initial_status
    )

    assert (
        matched_entry.initial_passed_tests
        == case.initial_passed_tests
    )

    assert (
        matched_entry.initial_total_tests
        == case.initial_total_tests
    )

    print()
    print(
        "[PASS] Teacher re-plan matches "
        "the Phase 1 failure trajectory."
    )

    # ------------------------------------------------------------------
    # 7. Provenance 확인
    # ------------------------------------------------------------------

    assert (
        matched_entry.teacher_model
    )

    assert (
        matched_entry.replan_version
    )

    print()
    print("[Provenance]")

    print(
        "teacher_model  :",
        matched_entry.teacher_model,
    )

    print(
        "version        :",
        matched_entry.replan_version,
    )

    print(
        "verified       :",
        matched_entry.verified,
    )

    # ------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "[SUCCESS] TeacherReplanStore "
        "sanity check passed."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()