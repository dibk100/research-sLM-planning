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

Phase 3-A에서는 문제당 최대 16개의 plan을 stochastic sampling하고, 각 sampled plan에 대해 code는 greedy하게 생성하여 Planning Coverage@k를 측정하였다. 세 모델 모두 candidate budget이 증가할수록 coverage가 지속적으로 상승했다.

| Model            |     @1 |     @2 |     @4 |     @8 |        @16 |
| ---------------- | -----: | -----: | -----: | -----: | ---------: |
| Qwen2.5-Coder-3B | 16.67% | 23.33% | 27.00% | 29.33% | **35.67%** |
| Qwen2.5-3B       | 11.33% | 15.33% | 20.67% | 25.00% | **30.33%** |
| Phi-3-mini       | 17.00% | 24.00% | 25.33% | 30.00% | **32.67%** |


난이도별 @16 결과는 아래와 같다.

| Model            |    Easy |  Medium |    Hard |
| ---------------- | ------: | ------: | ------: |
| Qwen2.5-Coder-3B | **78%** | **22%** |      7% |
| Qwen2.5-3B       |     72% |     11% |      8% |
| Phi-3-mini       |     73% |     15% | **10%** |

**Findings :**
Planning sampling은 세 모델 모두에서 single-sample 대비 coverage를 크게 증가시켰다. Qwen2.5-Coder-3B는 @1 16.67%에서 @16 35.67%로 +19.00%p, Qwen2.5-3B는 +19.00%p, Phi-3-mini는 +15.67%p 증가했다. 이는 단일 decoding에서는 발견되지 않지만 모델의 self-generated planning distribution 내부에는 성공 가능한 trajectory가 추가로 존재함을 보여준다.

그러나 @16에서도 해결되지 않는 문제가 각각 193/300, 209/300, 202/300으로 여전히 많았다. 특히 Medium/Hard에서는 coverage가 낮게 유지되므로, 단순한 inference-time plan sampling만으로 planning bottleneck이 해소된다고 보기는 어렵다.

또한 세 모델 모두 문제당 16개의 sampled plan이 문자열 수준에서 모두 서로 달랐으며 empty plan은 없었다. 따라서 낮은 coverage가 단순한 sampling collapse 때문이라고 보기는 어렵다. 다만 이 결과는 문자열 다양성만을 의미하며 semantic planning diversity를 직접 보장하지는 않는다.

### Phase 3-B 분석

Phase 3-B에서는 Phase 1 Self-Plan에서 생성된 plan을 문제별로 하나씩 고정하고, 동일한 fixed plan으로부터 최대 16개의 code candidate를 stochastic sampling하여 Code Coverage@k를 측정하였다. 세 모델 모두 code sampling budget이 증가함에 따라 coverage가 상승했다.

| Model            |     @1 |     @2 |     @4 |     @8 |        @16 |
| ---------------- | -----: | -----: | -----: | -----: | ---------: |
| Qwen2.5-Coder-3B | 16.00% | 19.67% | 22.00% | 25.33% | **28.00%** |
| Qwen2.5-3B       | 10.67% | 14.67% | 17.00% | 20.33% | **24.33%** |
| Phi-3-mini       | 12.67% | 18.00% | 22.00% | 24.67% | **27.00%** |

난이도별 @16 결과는 아래와 같다.

| Model            |    Easy |  Medium |   Hard |
| ---------------- | ------: | ------: | -----: |
| Qwen2.5-Coder-3B | **68%** | **10%** |     6% |
| Qwen2.5-3B       |     64% |      7% |     2% |
| Phi-3-mini       |     64% | **10%** | **7%** |

**Findings :**
동일한 plan을 고정하더라도 code sampling은 coverage를 증가시켰다. 이는 일부 실패가 plan 자체뿐 아니라 plan을 구현하는 과정의 stochastic variation을 통해서도 회복될 수 있음을 보여준다.

그러나 @16에서도 Qwen2.5-Coder-3B 28.00%, Qwen2.5-3B 24.33%, Phi-3-mini 27.00%에 머물렀으며, Medium/Hard에서의 증가폭 역시 제한적이었다. 특히 Qwen2.5-3B의 Hard Code Coverage@16은 2%에 불과했다.

또한 Qwen2.5-Coder-3B의 전체 successful candidate 비율은 16.29%, Phi-3-mini는 15.96%로 Planning Coverage의 candidate-level pass 비율과 크게 차이나지 않았다. 그럼에도 problem-level coverage는 Planning 쪽이 더 높았기 때문에, Planning sampling이 성공을 더 넓은 문제 집합에 분산시키는지 여부를 paired comparison으로 추가 확인할 필요가 있었다.

### Planning vs. Code 비교

동일한 candidate budget과 동일한 sampling hyperparameter 조건에서 Planning Coverage와 Code Coverage를 직접 비교하였다. 세 모델 모두에서 Planning Coverage가 Code Coverage보다 높은 결과를 보였다.

| Model            | Planning@16 | Code@16 |           Δ |
| ---------------- | ----------: | ------: | ----------: |
| Qwen2.5-Coder-3B |  **35.67%** |  28.00% | **+7.67%p** |
| Qwen2.5-3B       |  **30.33%** |  24.33% | **+6.00%p** |
| Phi-3-mini       |  **32.67%** |  27.00% | **+5.67%p** |

문제별 paired comparison 결과는 다음과 같다.

| Model            | Both PASS | Planning-only | Code-only | Neither | McNemar exact p |
| ---------------- | --------: | ------------: | --------: | ------: | --------------: |
| Qwen2.5-Coder-3B |        77 |        **30** |         7 |     186 |     **0.00019** |
| Qwen2.5-3B       |        67 |        **24** |         6 |     203 |     **0.00143** |
| Phi-3-mini       |        74 |        **24** |         7 |     195 |     **0.00333** |

Best Test-Pass Ratio 역시 @16에서 세 모델 모두 Planning 쪽이 높았다.
| Model            | Planning Best TPR@16 | Code Best TPR@16 |       Δ |
| ---------------- | -------------------: | ---------------: | ------: |
| Qwen2.5-Coder-3B |           **0.6221** |           0.5248 | +0.0973 |
| Qwen2.5-3B       |           **0.5515** |           0.4810 | +0.0706 |
| Phi-3-mini       |           **0.6014** |           0.5216 | +0.0797 |

**Findings :**
세 모델 모두에서 Planning@16이 Code@16보다 높았고, 문제별 paired comparison에서도 Planning-only 문제가 Code-only 문제보다 약 3~4배 많았다. 또한 @16 기준 exact McNemar test가 세 모델 모두 통계적으로 유의했다. 따라서 동일한 candidate budget에서 서로 다른 plan을 탐색하는 것이 하나의 fixed plan 아래에서 code implementation만 반복 sampling하는 것보다 더 넓은 문제 집합에서 성공 trajectory를 발견하는 데 효과적이라는 결과가 일관되게 관찰되었다.

특히 Qwen2.5-Coder-3B에서는 Planning-only 30문제, Code-only 7문제로 차이가 가장 뚜렷했고, Medium 난이도에서도 Planning@16 22% 대 Code@16 10%로 +12%p의 차이를 보였다. 반면 Hard에서는 두 방식 모두 절대 coverage가 매우 낮아, sampling alone으로 고난도 문제의 capability bottleneck을 해결하기는 어려운 것으로 나타났다.

종합하면, 성공 가능한 trajectory의 일부는 3B 모델의 기존 planning distribution 내부에 이미 존재하지만 낮은 확률로 생성되고 있으며, planning-space exploration이 code-space exploration보다 이를 발견하는 데 더 효과적이다. 동시에 @16에서도 상당수 문제가 해결되지 않기 때문에, 다음 단계에서는 단순 sampling을 넘어 좋은 planning trajectory의 probability mass 자체를 증가시키는 학습 방법이 필요하다. 이 결과는 Phase 4에서 planning-targeted RLVR을 검토하는 직접적인 동기가 된다.
