# 확인 작업

- planner checkpoint별 차이를 확인하고자함.
- 문제, planner decoding, frozen coder, coder decoding, tests를 모두 동일하게 고정함.

```
Problem P_i
│
├── Base Planner
│   └── Plan
│       └── Frozen Coder
│           └── Code
│               └── execution result / TPR
│
├── Vanilla RLVR Planner
│   └── Plan
│       └── Frozen Coder
│           └── Code
│               └── execution result / TPR
│
└── TPR RLVR Planner
    └── Plan
        └── Frozen Coder
            └── Code
                └── execution result / TPR
```

### step1 : select_transition_cases
100문제 평가

| Transition      |  Count | 해석                          |
| --------------- | -----: | --------------------------- |
| PPP             |     10 | 세 모델 모두 성공                  |
| FPP             |      1 | Base 실패 → Vanilla/TPR 모두 성공 |
| FPF             |      0 | Vanilla만 recovery           |
| FFP             |      0 | TPR만 recovery               |
| PFP / PPF / PFF |      0 | regression 없음               |
| **FFF**         | **89** | **세 모델 모두 최종 실패**           |

### step2-1 : analyze_fff_cases
89개 문제 세부 분석

```
================================================================================
FFF Partial-Correctness Analysis
================================================================================
Total FFF cases: 89

tpr_only_improvement             6 ( 6.74%)
vanilla_only_improvement         7 ( 7.87%)
tpr_dominant_improvement         4 ( 4.49%)
vanilla_dominant_improvement     2 ( 2.25%)
both_equal_improvement           4 ( 4.49%)
crossed                          1 ( 1.12%)
both_degraded                    6 ( 6.74%)
tpr_degraded                     4 ( 4.49%)
vanilla_degraded                 4 ( 4.49%)
unchanged_nonzero               25 (28.09%)
all_zero                        26 (29.21%)
other_mixed                      0 ( 0.00%)
```

### step2-2 : case study
8개로 우선 고정 

| problem_id | Category                 |              B / V / T | 보는 이유                          |
| ---------- | ------------------------ | ---------------------: | ------------------------------ |
| `00417`    | TPR-only improvement     | .000 / .000 / **.918** | 가장 강한 TPR positive case        |
| `00379`    | Vanilla-only improvement | .000 / **.686** / .000 | 가장 강한 Vanilla positive case    |
| `00330`    | TPR-dominant             | .110 / .256 / **.500** | 점진적 improvement 여부             |
| `00022`    | TPR degraded             | .550 / .550 / **.000** | 강한 TPR negative case           |
| `00061`    | Vanilla degraded         |     .605 / .158 / .605 | Vanilla-specific negative case |
| `01069`    | Equal improvement        |     .000 / .513 / .513 | reward는 달라도 같은 improvement     |
| `00030`    | Both degraded            |     .222 / .024 / .102 | RL 자체가 악화시킨 case               |
| **FPP 1개** | Full recovery            |              F → P → P | 유일한 실제 Pass@1 recovery         |

```build_case_study.ipynb``` 참고
```inspect_case.py``` 문제 하나씩 확인용