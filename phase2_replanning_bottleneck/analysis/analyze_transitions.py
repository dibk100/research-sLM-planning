"""
initial -> refined 상태 전이를 분석한다.

전이 유형
---------
- FAIL -> PASS  : recovered
- FAIL -> FAIL  : unchanged (통과 테스트 수 변화는 별도 기록)
- 상태 코드 전이 (WRONG_ANSWER -> TIMEOUT 등)

또한 전략 쌍에 대한 McNemar 검정을 수행한다.
(동일 문제 집합에 대한 paired binary outcome이므로 적합)

출력: archive/comparison_500/transition_summary.csv,
      archive/comparison_500/transition_detail.csv,
      archive/comparison_500/mcnemar.csv

TODO(구현)
----------
- [ ] 상태 전이 집계
- [ ] 전략 쌍별 McNemar 검정
       (Phase 1 archive/compare_mcnemar.py 재사용 가능)
"""
