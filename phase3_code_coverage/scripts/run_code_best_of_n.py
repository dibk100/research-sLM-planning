"""★ Phase 3-B: Fixed-Plan Code Best-of-N 실행 스크립트.

Phase 1 self-plan을 문제당 1개 로드해 고정한 뒤,
그 plan으로 code를 N개 sampling하고 각각 실행/채점하여 candidate 단위로 저장한다.

N=1,2,4,8을 따로 돌리지 않는다. N=8까지 한 번만 생성하고
분석 단계에서 candidate prefix를 사용한다.

    candidate 0        -> Oracle@1
    candidate 0..1     -> Oracle@2
    candidate 0..3     -> Oracle@4
    candidate 0..7     -> Oracle@8

Usage:

PYTHONPATH=. python -m scripts.run_code_best_of_n \
  --config configs/qwen25_coder_3b.yaml

# 부분 실행 / 출력 경로 변경
PYTHONPATH=. python -m scripts.run_code_best_of_n \
  --config configs/qwen25_coder_3b.yaml \
  --limit 10 \
  --output /mnt/hdd/project_sLM_planning/output_phase3b/sanity/results.jsonl
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    raise NotImplementedError


def run_candidate(*args, **kwargs):
    """candidate 하나: code 생성 -> 실행 -> CandidateRecord."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
