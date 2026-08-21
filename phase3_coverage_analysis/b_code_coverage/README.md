# Phase 3: Coverage Analysis

### 1. Research Objective :

Plan을 변경하지 않고 동일한 plan에서 여러 code implementation을 탐색하는 것만으로 solution coverage가 얼마나 증가하는지 측정한다.

이를 Phase 3-A와 비교하여 planning-space exploration의 효과가 단순한 code sampling 효과보다 큰지 분석한다.


#### Fixed Plan

Phase 3-B에서는 각 문제에 대해 Phase 1 Self-Plan 결과에서 생성된 plan을 사용한다.

```text
Phase 1 Self-Plan
        │
        ▼
    Fixed Plan
        │
        ├─ Sample Code C1
        ├─ Sample Code C2
        ├─ ...
        └─ Sample Code CN
```

`fixed_plan_loader.py`는 Phase 1 Self-Plan `results.jsonl`에서 각 `problem_id`에 대응하는 self-generated plan을 불러온다.


#### 실험 절차

각 문제에 대해 다음 과정을 수행한다.

1. Phase 1 Self-Plan에서 해당 문제의 plan을 불러온다.
2. 해당 plan을 fixed plan으로 설정한다.
3. 동일한 fixed plan으로 plan-conditioned code prompt를 구성한다.
4. 동일한 prompt에서 `N`개의 code candidate를 stochastic sampling한다.
5. 모든 code candidate를 benchmark evaluator로 실행한다.
6. candidate 순서를 기준으로 Oracle@k를 계산한다.


#### 주요 설정

```yaml
sampling:
  num_samples: 16

generation:
  code:
    max_new_tokens: 1024
    temperature: 0.7
    top_p: 0.95
```

### 2. 실험 실행

```bash
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase3_coverage_analysis/b_code_coverage/scripts/run_code_coverage.py \
  --config phase3_coverage_analysis/b_code_coverage/configs/qwen25Coder3b.yaml
```

### 3. 결과 분석

```bash
python phase3_coverage_analysis/b_code_coverage/analysis/analyze_code_coverage.py \
  --results <code_coverage_results.jsonl>
```

주요 분석 항목:

- Code Coverage@1
- Code Coverage@2
- Code Coverage@4
- Code Coverage@8
- Code Coverage@16
- 난이도별 coverage
- candidate-level success statistics
