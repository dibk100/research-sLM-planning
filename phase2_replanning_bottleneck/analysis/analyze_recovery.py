"""
어떤 실패가 회복되었는지 분석한다.

Phase 2의 핵심 질문
-------------------
"feedback만으로 고칠 수 있는 실패"와 "계획을 다시 세워야 고쳐지는 실패"가
구분되는가.

산출
----
- recovered only by feedback_repair
- recovered only by self_replan
- recovered only by teacher_replan
- teacher_replan은 고쳤지만 self_replan은 못 고친 케이스
  -> re-planning 병목의 직접 증거

출력: archive/comparison_500/recovery_summary.csv,
      archive/comparison_500/recovery_detail.csv

TODO(구현)
----------
- [ ] 전략별 recovered 집합 계산
- [ ] 집합 연산 기반 분류 및 CSV 기록
"""
