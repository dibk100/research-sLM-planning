# project_sLM_planning

본 연구는 **3B–8B 규모의 작은 코드 언어 모델(Small Code Model)** 을 대상으로, Competitive Programming 환경에서 **Planning 능력의 한계와 병목을 분석하고 이를 개선하는 방법**을 연구한다.

연구는 다음 세 단계로 진행된다.
```
small-model-planning/
│
├── phase1_planning_bottleneck/
│
├── phase2_replanning_bottleneck/
│
└── phase3_planning_improvement/
```


## Phase1. Initial Analysis of the Planning Bottleneck ✅

"처음부터 좋은 계획을 만들 수 있는가?" 관점으로 실험을 진행함.

**실험 목표**   
소형 언어 모델의 코드 문제 해결 과정을 계획 생성(Planning Generation)​과 코드 구현(Code Implementation)​의 두 단계로 구분하고, 성능 저하의 주요 병목이 어디에서 발생하는지를 분석한다.

**실험 설정**
- Model: Qwen2.5-Coder-3B-Instruct
- Dataset: LiveCodeBench v6, stdin 기반 500문제
- 비교 실험 :
    - Direct: 바로 코드 생성
    - Self-Planning: 모델 스스로 Plan → Code
    - Teacher-Planning: 외부 고품질 Plan → 3B 모델이 Code 구현

**주요 결과**

| Method           |    Solved | Pass Rate | Mean Test Pass Ratio |
| ---------------- | --------: | --------: | -------------------: |
| Direct           |  93 / 500 | **18.6%** |            **0.334** |
| Self-Planning    |  84 / 500 | **16.8%** |            **0.303** |
| Teacher-Planning | 170 / 500 | **34.0%** |            **0.443** |

**Phase 1 Findings**   
1. Self-Plan은 Direct 대비 성능을 향상시키지 못했다.   
2. 고품질 계획을 제공하면 동일한 3B 모델의 코드 생성 성능이 크게 향상되었다.   
3. Planning의 효과는 문제 난이도에 따라 다르게 나타났다.   

본 실험에서 사용한 3B 코드 모델은 적절한 해결 계획이 주어졌을 때 추가적인 문제를 해결할 수 있는 잠재력을 가지고 있지만, 그러한 계획을 스스로 생성하는 과정에서 성능 병목이 발생하는 것으로 관찰되었다. 따라서 Planning Generation은 소형 코드 모델의 중요한 성능 병목 중 하나로 볼 수 있다.

## Phase2. Re-planning 병목 분석 ⏭️

"한번 잘못된 전략을 선택한 뒤 실행 피드백을 받았을 때, 그 전략 자체를 다시 계획할 수 있는가?" 관점으로 실험을 진행함.

**연구 질문**
- 작은 코드 모델은 잘못된 전략을 실행 피드백만으로 수정할 수 있는가?

**실험**
- Feedback-only Repair
- Re-plan + Repair

실패 이후의 복구 과정에서 단순 코드 수정(Repair)과 전략 재수립(Re-planning)의 역할을 비교·분석한다.


## Phase3. Planning 능력 향상

Phase1과 Phase2에서 확인된 병목을 기반으로 작은 모델의 Planning 능력을 향상시키는 방법을 연구한다.

후보 방법은 다음과 같다.

- Planning Policy Learning
- Planning Memory
- Compact Planning Representation

Phase3릐 구체적인 방법은 Phase1,2의 실증 결과를 바탕으로 결정한다.