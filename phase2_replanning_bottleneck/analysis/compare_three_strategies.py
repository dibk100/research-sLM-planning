"""
세 refinement 전략의 성능을 한 표로 비교한다.

입력: 세 전략의 results.jsonl
출력: archive/comparison_500/overall_summary.csv

지표
----
- recovery_rate    : 실패 케이스 중 PASS로 전환된 비율 (핵심 지표)
- final_pass_rate  : Phase1 initial PASS + Phase2 recovered 를 합친 500문항 기준 pass rate
- mean_test_pass_ratio_delta : passed_tests/total_tests 의 평균 변화량
- cost             : prompt_tokens / completion_tokens / generation_time 평균

주의
----
세 전략은 동일한 실패 케이스 집합에서 출발하므로 문제 단위로 짝지어 비교할 수 있다.
(paired comparison; McNemar 검정은 analyze_transitions.py 참고)

TODO(구현)
----------
- [ ] 세 results.jsonl 로딩 및 problem_id 기준 정렬/검증
- [ ] 지표 계산 및 CSV 기록
"""
