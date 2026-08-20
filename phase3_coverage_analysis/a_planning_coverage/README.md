# Phase 3-A: Planning Coverage

### 1. Research Objective :

작은 코드 모델이 하나의 self-generated plan에 의존하는 대신 여러 plan을 탐색했을 때 성공 가능한 solution의 coverage가 얼마나 증가하는지 측정한다.

특히 다음 질문을 확인한다.

> 모델의 self-generated planning distribution 내부에 실제로 문제를 해결할 수 있는 plan/trajectory가 존재하는가?


#### 실험 절차

각 문제에 대해 다음 과정을 수행한다.

1. Phase 1과 동일한 Self-Plan prompt를 구성한다.
2. 동일한 문제에서 `N`개의 plan을 stochastic sampling한다.
3. 각 sampled plan에 대해 하나의 code를 greedy generation한다.
4. 생성된 모든 code candidate를 benchmark evaluator로 실행한다.
5. candidate 순서를 기준으로 Oracle@k를 계산한다.


#### 주요 설정

```yaml
sampling:
  num_samples: 16

generation:
  plan:
    max_new_tokens: 384
    temperature: 0.7
    top_p: 0.95

  code:
    max_new_tokens: 1024
    temperature: 0.0
    top_p: 1.0
```

### 2.실험 실행

```bash
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase3_coverage_analysis/a_planning_coverage/scripts/run_planning_coverage.py \
  --config phase3_coverage_analysis/a_planning_coverage/configs/planning_coverage_qwen25Coder3b.yaml
```

### 3. 결과 분석

```bash
python phase3_coverage_analysis/a_planning_coverage/analysis/analyze_planning_coverage.py \
  --results <planning_coverage_results.jsonl>
```

주요 분석 항목:

- Planning Coverage@1
- Planning Coverage@2
- Planning Coverage@4
- Planning Coverage@8
- Planning Coverage@16
- 난이도별 coverage
- candidate-level success statistics