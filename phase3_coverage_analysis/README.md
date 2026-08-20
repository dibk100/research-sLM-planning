# Phase 3: Coverage Analysis

## 1. Research Overview :

Phase 3에서는 작은 코드 모델의 제한된 성능이 **planning space를 충분히 탐색하지 못해서 발생하는지**, 또는 **동일한 plan에서도 적절한 code implementation을 생성하지 못해서 발생하는지** 분석한다.

핵심 연구 질문은 다음과 같다.

> Research Question :   
> 동일한 candidate budget에서 여러 plan을 탐색하는 것과 하나의 plan을 고정한 채 여러 code implementation을 탐색하는 것 중 어느 쪽이 문제 해결 coverage를 더 효과적으로 증가시키는가?

Phase 3는 두 개의 통제 실험으로 구성된다.

- **Phase 3-A: Planning Coverage**
  - 동일 문제에서 N개의 plan을 stochastic sampling한다.
  - 각 sampled plan으로부터 하나의 code를 greedy generation한다.
  - Planning space exploration이 성공 가능한 solution coverage를 얼마나 증가시키는지 측정한다.

- **Phase 3-B: Code Coverage**
  - Phase 1 Self-Plan에서 생성된 plan 하나를 고정한다.
  - 동일한 fixed plan으로부터 N 개의 code를 stochastic sampling한다.
  - 동일한 plan 아래에서 code implementation space exploration이 coverage를 얼마나 증가시키는지 측정한다.

두 실험의 Oracle@k를 동일한 candidate budget에서 비교하여 planning-space exploration과 code-space exploration의 효과를 분석한다.

## 2. 실험 구조

### 2.1 Phase 3-A: Planning Coverage

각 문제에 대해 N개의 plan을 stochastic sampling하고, 각 plan으로부터 하나의 code를 생성한다.

```text
Problem
  │
  ├─ Sample Plan P1 ──> Greedy Code C1 ──> Execute
  ├─ Sample Plan P2 ──> Greedy Code C2 ──> Execute
  ├─ ...
  └─ Sample Plan PN ──> Greedy Code CN ──> Execute
```

Plan generation은 stochastic sampling을 사용한다.

```text
temperature = 0.7
top_p      = 0.95
```

반면 code generation은 greedy decoding으로 고정한다.(Phase1과 동일)

```text
temperature = 0.0
top_p      = 1.0
```

따라서 candidate 간 주요 변화 요인은 **sampled plan의 차이**가 된다.


### 2.2 Phase 3-B: Code Coverage

Phase 1 Self-Plan에서 생성된 plan을 문제별로 하나씩 고정하고, 동일한 plan으로부터 N개의 code candidate를 stochastic sampling한다.

```text
Problem + Fixed Phase-1 Self-Plan
  │
  ├─ Sample Code C1 ──> Execute
  ├─ Sample Code C2 ──> Execute
  ├─ ...
  └─ Sample Code CN ──> Execute
```

Code generation은 다음 sampling 설정을 사용한다.

```text
temperature = 0.7
top_p      = 0.95
```

Phase 3-B에서는 새로운 plan을 생성하지 않는다.

따라서 candidate 간 주요 변화 요인은 **동일한 plan에서 생성되는 code implementation의 차이**가 된다.

## 3. Phase 3-A와 Phase 3-B의 통제 관계

두 실험의 차이는 다음과 같다.

| Experiment | Plan | Code |
|---|---|---|
| Phase 3-A | Stochastic Sampling × N | Plan당 Greedy Generation |
| Phase 3-B | Phase 1 Self-Plan으로 고정 | Stochastic Sampling × N |

Phase 3-A와 Phase 3-B 모두 최종적으로 문제당 `N`개의 code candidate를 생성한다.

따라서 동일한 candidate budget `k`에서 다음을 비교할 수 있다.

```text
Planning Coverage@k
vs.
Code Coverage@k
```

이를 통해 단순히 inference compute를 증가시켰을 때의 효과뿐만 아니라, **어느 generation space를 탐색하는 것이 더 효과적인지** 분석한다.

## 4. 평가 지표

주요 평가 지표는 **Oracle@k**이다.

문제 `x`에 대해 첫 `k`개의 candidate 중 하나라도 전체 unit test를 통과하면 해당 문제를 Oracle@k에서 해결한 것으로 정의한다.

```text
Oracle@k(x) = 1
if any candidate among the first k candidates passes all tests
```

전체 데이터셋의 Coverage@k는 다음과 같다.

```text
Coverage@k =
    Oracle@k에서 해결된 문제 수
    / 전체 문제 수
```

주요 candidate budget은 다음과 같다.

```text
@1, @2, @4, @8, @16
```

각 candidate는 deterministic candidate-specific seed를 사용하여 생성되므로 candidate 순서가 재현 가능하다.

## 5. 폴더 아키텍처

```text
phase3_coverage_analysis/
│
├── README.md
│
├── a_planning_coverage/
│   ├── configs/
│   │   └── planning_coverage_qwen25Coder3b.yaml
│   ├── scripts/
│   │   ├── run_planning_coverage.py
│   │   └── sanity_check.py
│   ├── strategies/
│   │   └── planning_coverage.py
│   ├── analysis/
│   │   └── analyze_planning_coverage.py
│   ├── candidate.py
│   └── runner.py
│
├── b_code_coverage/
│   ├── configs/
│   │   └── qwen25Coder3b.yaml
│   ├── scripts/
│   │   ├── run_code_coverage.py
│   │   └── sanity_check.py
│   ├── strategies/
│   │   └── code_coverage.py
│   ├── analysis/
│   │   └── analyze_code_coverage.py
│   ├── candidate.py
│   ├── fixed_plan_loader.py
│   └── runner.py
│
└── analysis/
    └── compare_planning_vs_code.py
```

## 6. Candidate Seed 및 재현성

각 stochastic candidate는 다음 세 값을 기반으로 deterministic seed를 생성한다.

```text
base_seed
problem_id
sample_id
```

개념적으로:

```text
candidate_seed =
    f(base_seed, problem_id, sample_id)
```

이를 통해 다음을 보장한다.

- 동일 candidate의 재현 가능성
- candidate 간 독립적인 sampling seed
- 중단된 실험의 resume 가능성
- Oracle@k 계산 시 candidate prefix의 일관성

예를 들어 `sample_id=0...15`로 생성된 결과에서 `candidate[:8]`은 항상 동일한 첫 8개 candidate를 의미한다.

## 7. 결과 분석

### Phase 3-A 분석



### Phase 3-B 분석


### Planning vs. Code 비교

```bash
python phase3_coverage_analysis/analysis/compare_planning_vs_code.py \
  --planning-results <planning_results.jsonl> \
  --code-results <code_results.jsonl>
```

동일 문제에 대한 paired comparison을 수행한다.

각 `k`에 대해 다음 네 가지 경우를 계산한다.

```text
Both PASS
Planning-only PASS
Code-only PASS
Both FAIL
```

또한 Planning Coverage와 Code Coverage의 차이가 통계적으로 유의한지 확인하기 위해 **McNemar exact test**를 수행한다.


## 8. 결과 해석 기준

### Case 1. Planning Coverage > Code Coverage

동일 candidate budget에서:

```text
Planning Coverage@k > Code Coverage@k
```

가 나타난다면, 하나의 plan 아래에서 여러 implementation을 생성하는 것보다 **서로 다른 plan을 탐색하는 것이 성공 가능한 solution을 찾는 데 더 효과적**이라는 것을 의미한다.

이는 작은 코드 모델에서 planning-space exploration이 중요한 recoverable capability임을 뒷받침한다.


### Case 2. Planning Coverage ≈ Code Coverage

두 coverage curve가 유사하다면 Phase 3-A의 성능 향상을 planning exploration에만 귀속시키기 어렵다.

이 경우 성능 향상은 planning 자체보다 **일반적인 best-of-N sampling 또는 inference compute 증가 효과**로 설명될 가능성이 있다.


### Case 3. Planning / Code Coverage 모두 낮은 수준에서 포화

특히 Medium/Hard 문제에서 두 방식 모두 낮은 coverage에 머문다면 sampling만으로는 해당 문제를 해결하기 어렵다는 것을 의미한다.

이는 남은 실패가 다음과 같은 broader capability bottleneck과 관련될 가능성을 시사한다.

- 고품질 plan 자체의 생성 능력
- plan을 정확한 implementation으로 변환하는 능력
- algorithmic reasoning capability
- 기본적인 model capacity

## 9. 전체 연구에서 Phase 3의 역할

Phase 1에서는 다음 비교를 통해 **planning quality bottleneck**을 분석한다.

```text
Direct
vs.
Self-Plan
vs.
Teacher-Plan
```

즉, 작은 모델이 스스로 생성한 plan과 외부의 고품질 plan을 사용했을 때의 차이를 비교한다.

Phase 2에서는 실패 이후의 refinement 상황에서:

```text
Feedback Regeneration
vs.
Self-Replanning
vs.
Teacher-Replanning
```

을 비교하여 **failure-time replanning bottleneck**을 분석한다.

Phase 3에서는 여기서 한 단계 더 나아가 모델 내부의 sampling distribution을 탐색한다.

```text
Planning-space Exploration
vs.
Code-space Exploration
```

이를 통해 다음을 구분하고자 한다.

1. 작은 모델이 고품질 plan을 **활용할 수 있는가**
2. 작은 모델이 고품질 plan을 **스스로 생성하거나 재생성할 수 있는가**
3. 모델의 self-generated distribution 내부에 **성공 가능한 planning trajectory가 존재하는가**
4. 동일한 inference budget에서 **planning exploration과 code exploration 중 어느 쪽이 더 효과적인가**