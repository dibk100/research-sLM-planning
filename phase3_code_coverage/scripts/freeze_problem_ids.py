"""문제 ID와 순서가 Phase 1 / Phase 3-A와 동일한지 확인하고 manifest로 고정한다.

Phase 3-B는 Phase 1 self-plan을 그대로 재사용하므로,
problem id 정렬이 어긋나면 다른 문제의 plan이 붙는 치명적 오류가 난다.
실행 전에 반드시 통과해야 한다.

Usage:

PYTHONPATH=. python -m scripts.freeze_problem_ids

PYTHONPATH=. python -m scripts.freeze_problem_ids \
  --limit 10 \
  --output data/livecodebench_pilot_10.jsonl

phase1 결과 위치
/mnt/hdd/project_sLM_planning/output/direct_500_stdin/results.jsonl
/mnt/hdd/project_sLM_planning/output/self_plan_500_stdin/results.jsonl
/mnt/hdd/project_sLM_planning/output/teacher_plan_500_stdin/results.jsonl
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    raise NotImplementedError


def load_jsonl(path):
    raise NotImplementedError


def assert_unique_problem_ids(records) -> None:
    raise NotImplementedError


def assert_same_problem_sequence(left, right) -> None:
    raise NotImplementedError


def build_manifest_record(example, index: int):
    raise NotImplementedError


def write_manifest(path, records) -> None:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
