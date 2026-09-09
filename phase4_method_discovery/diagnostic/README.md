# Diagnostic
- RQ-D1. 같은 문제에서도 self-generated plan에 따라 downstream code success가 크게 달라지는가?
- RQ-D2. 좋은 plan이 발견됐을 때 frozen coder는 이를 안정적으로 구현할 수 있는가?
- RQ-D3. 현재 RLVR reward가 실제로 “plan quality”를 얼마나 잘 식별하는가?

### Metric(설계)

| Metric                           | 정의                        | 해석                                |
| -------------------------------- | ------------------------- | --------------------------------- |
| **Plan Success Rate \(q_i\)**    | plan별 `#pass / M`         | plan의 empirical executability     |
| **Plan Oracle@N**                | \(\max_i q_i>0\)          | N개 중 최소 하나의 usable plan 존재        |
| **Stable Plan@τ**                | \(\max_i q_i \ge τ\)      | coder가 안정적으로 구현 가능한 plan 존재       |
| **Within-problem Plan Variance** | \(Var_i(q_i)\)            | plan 선택 효과                        |
| **Within-plan Code Variance**    | coder rollout 결과 variance | coder stochasticity               |
| **Best-plan Success**            | \(\max_i q_i\)            | 발견 가능한 최고 plan 품질                 |
| **Mean-plan Success**            | \(\frac1N\sum q_i\)       | 현재 planner 평균 품질                  |
| **Best–Mean Gap**                | \(\max q_i-\bar q\)       | plan discovery/selection headroom |

### 분석 결과(예측)

| 유형                              | N×M 결과                | 의미                                      |
| ------------------------------- | --------------------- | --------------------------------------- |
| **A. No-plan coverage**         | 모든 plan에서 거의 0/M      | planner discovery failure               |
| **B. Rare good plan**           | 일부 plan만 높은 M-success | planner selection / probability 문제      |
| **C. Good plan, unstable code** | 많은 plan에서 1/M~3/M     | coder stochasticity / implementation 문제 |
| **D. Robustly solvable**        | 여러 plan에서 높은 success  | 이미 충분히 해결 가능                            |

### Todo
실험 protocol과 output schema 확정
diagnostic config 작성
N×M generation runner 구현
1문제 sanity check
10문제 pilot
metric analyzer 구현

### 폴더구조
```
phase4_method_discovery/diagnostic/
├── configs/
│   └── nxm_qwen25coder3b.yaml
├── scripts/
│   └── run_nxm_diagnostic.py
├── analysis/
│   └── analyze_nxm_diagnostic.py      # sanity 후 작성
├── outputs/
│   ├── sanity/
│   └── pilot/
└── README.md
```

### 코드 재사용 범위
checkpoint_dataset.py
    → DeepCoder-TACO ProblemExample 복원

SelfPlanningStrategy
    → build_plan_prompt()만 재사용

planning_execution_reward.py
    → build_code_prompt()
    → select_reward_tests()

CodeParser
    → generated code extraction

TACOEvaluator
    → evaluate_non_fail_fast()


### 실험 정의 및 설계

목적 : 현재 base planner의 plan quality variation과 frozen coder realization noise를 분리하는 것
중요 원칙 : Training reward와 diagnostic execution semantics가 동일해야 한다.

Planner (= Phase 3-A) :
temperature: 0.7
top_p: 0.95
max_new_tokens: 512

Frozen coder (차이) :
temperature: 0.7
top_p: 0.95
max_new_tokens: 1024


- Binary downstream utility : {q_i}^pass​=(1/M)​∑_j ​I(c_ij​ passes all tests)
- Dense downstream utility : {q_i}^TPR​=(1/M)​∑_j TPR(c_ij)

- Seed 고정
```
problem
 ├─ plan 0 seed
 │   ├─ code 0 seed
 │   ├─ code 1 seed
 │   └─ ...
 └─ plan 1 seed
```