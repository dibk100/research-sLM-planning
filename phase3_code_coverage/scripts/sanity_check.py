"""Phase 3-B pilot run 결과 점검.

PYTHONPATH=. python -m scripts.run_code_best_of_n \
  --config configs/qwen25_coder_3b.yaml --limit 10

PYTHONPATH=. python -m scripts.sanity_check \
  --results /mnt/hdd/project_sLM_planning/output_phase3b/sanity/results.jsonl

확인 항목:
- 문제당 candidate 수가 num_samples와 일치하는가
- 모든 candidate가 동일한 plan_text를 참조하는가 (Phase 3-B 핵심 전제)
- Oracle@k가 k에 대해 단조 증가하는가
- status 분포에 이상값이 없는가
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    raise NotImplementedError


def load_results(path):
    raise NotImplementedError


def assert_plan_is_fixed(records) -> None:
    """한 문제의 candidate들이 모두 같은 plan을 썼는지 검증."""
    raise NotImplementedError


def main() -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
