# Phase1. Initial Analysis of the Planning Bottleneck

작업 기록 공간


## 1. Research Objective :

소형 언어 모델(Small Language Model)은 제한된 모델 용량으로 인해 대형 언어 모델(Large Language Model)에 비해 복잡한 문제를 분석하고 적절한 해결 전략을 구성하는 능력이 제한적일 수 있다. 본 연구에서는 이러한 관점에서 소형 언어 모델의 코드 문제 해결 과정을 계획 생성(Planning Generation)​과 코드 구현(Code Implementation)​의 두 단계로 구분하고, 성능 저하의 주요 병목이 어디에서 발생하는지를 분석한다.

본 단계(Phase 1)에서는 계획 생성 능력(Planning Capability)이 소형 언어 모델의 문제 해결 성능을 제한하는 주요 요인인지를 검증한다. 이를 위해 동일한 3B급 모델에 대해 Direct, Self-Plan, Teacher-Plan의 세 가지 추론 전략을 비교한다. Direct는 계획 없이 코드를 직접 생성하고, Self-Plan은 모델이 스스로 생성한 계획을 기반으로 코드를 작성하며, Teacher-Plan은 외부의 고품질 계획을 입력으로 제공받아 코드를 생성한다.

소형 언어 모델의 낮은 문제 해결 성능이 단순히 코드를 구현하지 못하기 때문인지, 또는 구현 가능한 해결 방법을 스스로 계획하지 못하는 것과 관련되는지를 분석한다.

핵심 질문은 다음과 같다.

> Research Question :   
> 3B급 소형 언어 모델의 코드 문제 해결 과정에서, 올바른 알고리즘 계획을 스스로 생성하는 능력의 부족이 주요 성능 병목으로 작용하는가?

## 2. Experimental Setup :
- Model: 
    - Code Model: ```Qwen/Qwen2.5-Coder-3B-Instruct```
        - maximum Code Generation Length: 1024 tokens
    - Plan Model: ```Qwen/Qwen2.5-Coder-3B-Instruct```
        - maximum Plan Generation Length: 384 tokens
    - Teacher Model: ```Claude Opus 5```

- Dataset:   
실험 파이프라인의 일관성을 위해 stdin 기반 실행 평가가 가능한 문제를 대상으로 실험하였다.
    - Dataset: LiveCodeBench v6
    - Split: Test
    - Evaluation Type: stdin
    - Number of Problems: 500
    - Difficulty Distribution
        - Easy: 162
        - Medium: 150
        - Hard: 188

- Comparison Strategies :
    1. Direct : 문제를 입력받아 계획 없이 바로 코드를 생성한다.
    2. Self-Plan : 모델이 스스로 해결 계획(Plan)을 생성한 뒤, 해당 계획을 기반으로 코드를 생성한다.
    3. Teacher-Plan : 강한 모델(claude-opus-5 사용)이 생성한 계획을 입력으로 제공하고, 소형 언어 모델은 해당 계획을 기반으로 코드만 구현한다.
    <details>
    <summary><b> Strategies 상세 내용</b></summary>

    동일한 모델을 대상으로 다음 세 가지 설정을 비교한다.

    1. Direct : ``` Problem → Code Generation ```    
    문제를 입력받아 계획 없이 바로 코드를 생성한다.   


    2. Self-Plan : ``` Problem → Plan Generation → Plan-based Code Generation ```    
    모델이 스스로 해결 계획(Plan)을 생성한 뒤, 해당 계획을 기반으로 코드를 생성한다.   
    총 두 번의 독립적인 모델 호출을 수행한다.   
    - call-1 계획 생성 : ```Problem → Plan Generation```
    - call-2 계획 기반 코드 생성 : ```Problem + Generated Plan → Code Generation```

    3. Teacher-Plan : ```Problem →  Teacher Plan (offline) → Code Generation```    
    강한 Teacher 모델(claude-opus-5 사용)이 생성한 계획을 입력으로 제공하고, 소형 언어 모델은 해당 계획을 기반으로 코드만 구현한다.

    - Teacher plan은 실험 전에 미리 생성하여 고정하며, 실험 중에는 Teacher 모델을 다시 호출하지 않고 저장된 계획만을 사용.

    </details>


## 3. Overall Results

| Strategy         | Solved Problems¹ | Pass Rate² | Mean Test Pass Ratio³ |
| ---------------- | ---------------: | ---------: | --------------------: |
| Direct           |         93 / 500 |      18.6% |                 0.334 |
| Self-Planning    |         84 / 500 |      16.8% |                 0.303 |
| Teacher-Planning |    **170 / 500** |  **34.0%** |             **0.443** |

- Solved Problems¹ : 모든 평가 test case를 통과하여 완전히 해결한 문제 수 / 전체 평가 문제 수.   
    - 하나 이상의 test case에서 실패한 경우 해당 문제는 해결되지 않은 것으로 간주한다. 
    
- Pass Rate² : 전체 평가 문제 중 모든 test case를 통과한 문제의 비율. 
    - 본 실험의 주 성능 지표로 사용한다. 

- Mean Test Pass Ratio³ : 각 문제에서 통과한 test case의 비율을 계산한 후 전체 문제에 대해 평균한 값. 
    - 완전 해결에는 실패했지만 일부 test case를 통과한 경우도 반영하는 보조 성능 지표이다.

### 3.1 Analysis

500개의 LiveCodeBench 문제를 대상으로 Direct, Self-Plan, Teacher-Plan 전략의 성능을 비교한 결과, Direct는 93개 문제를 해결하여 18.6%의 성공률을 기록한 반면, Self-Plan은 84개 문제를 해결하여 16.8%의 성공률을 보였다. 
즉, 3B 모델이 별도의 계획 없이 직접 코드를 생성하는 것보다 스스로 계획을 생성한 후 해당 계획을 기반으로 코드를 구현하도록 하는 것이 오히려 성능 저하로 이어졌다.

반면 동일한 3B 모델에 외부의 고품질 계획을 제공한 Teacher-Plan은 170개 문제를 해결하여 34.0%의 Pass Rate를 기록하였으며, Mean Test Pass Ratio 역시 0.443으로 가장 높은 값을 보였다. 
즉, Self-Planning에서는 성능이 감소했지만 Teacher-Planning에서는 Direct와 Self-Planning을 모두 크게 상회하는 성능 향상이 관찰되었다.

이러한 결과는 단순히 planning 단계를 추가하는 것 자체보다 모델이 사용하는 계획의 품질이 이후 코드 생성 성능에 중요한 영향을 미친다는 점을 시사한다. 
특히 Self-Plan과 Teacher-Plan에서 실제 코드를 생성하는 모델은 동일하므로, Teacher-Plan에서의 성능 향상은 일부 문제에 대해 3B 모델이 적절한 해결 계획이 주어질 경우 이를 코드로 구현할 수 있는 능력을 가지고 있음을 보여준다. 
반대로 Self-Plan이 Direct보다 낮은 성능을 보였다는 점은, 본 실험의 3B 모델이 그러한 계획을 스스로 안정적으로 생성하는 데에는 한계가 있을 가능성을 시사한다.

따라서 본 결과는 Planning Generation이 3B 모델의 문제 해결 성능을 제한하는 중요한 병목 중 하나일 수 있음을 보여주는 경험적 근거로 해석할 수 있다.


### 3.2 Performance by Problem Difficulty

| Difficulty | Problems | Direct | Self-Plan | Teacher-Plan |
| ---------- | -------: | -----: | ------------: | ---------------: |
| Easy       |      162 |  50.6% |         48.8% |        **74.7%** |
| Medium     |      150 |   4.7% |          2.7% |        **26.7%** |
| Hard       |      188 |   2.1% |          0.5% |         **4.8%** |

* **Problems** : 각 난이도에 포함된 평가 문제 수.
* **Direct / Self-Plan / Teacher-Plan** : 해당 난이도의 전체 문제 중 모든 test case를 통과하여 완전히 해결한 문제의 비율(Pass Rate).

난이도별 분석에서도 Teacher-Plan은 Easy와 Medium 문제에서 Direct 및 Self-Plan 대비 뚜렷한 성능 향상을 보였다. 
특히 Medium 문제에서는 Direct 4.7%, Self-Plan 2.7%에 비해 Teacher-Plan이 26.7%의 성공률을 기록하여 고품질 계획의 효과가 두드러졌다. 
반면 Hard 문제에서는 Teacher-Plan이 가장 높은 성공률을 기록했지만 4.8%에 그쳐, 고품질 계획을 제공하더라도 성능 향상이 제한적이었다.

이러한 결과는 고품질 계획의 제공만으로 소형 언어 모델의 문제 해결 한계를 항상 극복할 수 있는 것은 아님을 보여준다.
Easy와 Medium 문제에서는 planning quality를 개선함으로써 상당한 성능 향상이 나타났지만, Hard 문제에서는 고품질 계획을 제공한 이후에도 대부분의 문제가 해결되지 않았다. 
이는 문제 난이도가 증가할수록 planning generation뿐만 아니라 복잡한 알고리즘을 정확하게 코드로 구현하는 능력 등 추가적인 model capability가 병목으로 작용할 가능성을 시사한다.

따라서 난이도별 결과는 소형 코드 모델의 성능 병목이 단일한 요인으로 구성되어 있지 않음을 보여준다. 
Easy와 Medium 문제에서는 planning quality가 성능을 제한하는 중요한 요인으로 나타나는 반면, Hard 문제에서는 planning을 개선한 이후에도 추가적인 implementation 또는 model capability bottleneck이 강하게 남아 있는 것으로 해석할 수 있다. 
즉, 고품질 planning은 소형 모델의 문제 해결 성능을 향상시키는 중요한 요소이지만, 문제 난이도가 높아질수록 planning 개선만으로는 충분하지 않음을 시사한다.

## 4. Phase 1 Findings
500개의 LiveCodeBench 문제를 대상으로 수행한 실험에서 다음과 같은 주요 결과를 확인하였다.

1. Self-Plan은 Direct 대비 성능을 향상시키지 못했다.   
모델이 스스로 계획을 생성한 후 코드를 작성하도록 했을 때 더 많은 추론 비용이 발생했지만, 최종 성공률과 평균 test pass ratio는 오히려 감소하였다.

2. 고품질 계획을 제공하면 동일한 3B 모델의 코드 생성 성능이 크게 향상되었다.   
Teacher-Plan은 34.0%의 성공률을 기록하여 Direct의 18.6%와 Self-Plan의 16.8%를 크게 상회하였다.

3. Planning의 효과는 문제 난이도에 따라 다르게 나타났다.   
Teacher-Plan은 Easy와 Medium 문제에서 큰 성능 향상을 보였으나, Hard 문제에서는 고품질 계획을 제공한 이후에도 성공률이 4.8%에 머물렀다.


이러한 결과를 종합하면, 본 실험에서 사용한 3B 코드 모델은 적절한 해결 계획이 주어졌을 때 추가적인 문제를 해결할 수 있는 잠재력을 가지고 있지만, 그러한 계획을 스스로 생성하는 과정에서 성능 병목이 발생하는 것으로 관찰되었다. 따라서 Planning Generation은 소형 코드 모델의 중요한 성능 병목 중 하나로 볼 수 있다.

동시에 Hard 문제에서 Teacher-Planning의 효과가 제한적이었다는 결과는 planning의 개선만으로 모든 문제 해결 실패를 설명하거나 해결할 수 없음을 보여준다. 이는 높은 난이도의 문제에서는 Planning Generation뿐만 아니라 Code Implementation 및 모델 자체의 problem-solving capability와 관련된 추가적인 병목을 함께 고려해야 함을 시사한다.


## 5. Phase 1 Conclusion

단순히 소형 모델에게 더 많은 reasoning step을 부여하는 것보다 스스로 정확한 알고리즘 계획을 생성할 수 있는 Planning Capability를 향상시키는 것이 중요한 연구 방향이 될 수 있다.

Phase 1에서는 이러한 planning-generation bottleneck의 존재 가능성을 확인하였으며, 이후 단계에서는 이를 기반으로 소형 코드 모델의 planning 및 re-planning 능력을 보다 직접적으로 분석하고 개선하는 방향으로 연구를 확장한다.

<details>
<summary><b>Notes and Future Experiments.</b></summary>

#### Dataset Filtering :

- stdin 기반 문제만 사용하여 파이프라인 안정성 확보
  - Codeforces / AtCoder 스타일 문제만 사용
- 향후 functional evaluator를 별도 단계로 추가 예정

#### Teacher-Planning :

- Teacher model로 `claude-opus-5` 사용

#### Ablation Study: Cross-Dataset Generalization

추후 일반화 성능을 확인하기 위해 다음 데이터셋에서도 동일한 실험을 수행할 예정.

- APPS
- MBPP+
- HumanEval+
- LiveCodeBench v5
- etc.

#### Ablation Study: Models

- Qwen2.5-3B (Base)
- Phi-3.5-mini-instruct
- etc.

</details>
