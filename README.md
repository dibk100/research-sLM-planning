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


## Phase1. 초기 Planning 병목 분석

**연구 질문**
- 작은 코드 모델은 어려운 문제에서 올바른 알고리즘 계획을 스스로 생성할 수 있는가?

**실험**
- Direct Code Generation
- Self-Planning
- Teacher-Planning

초기 계획 생성 과정에서의 병목이 **Planning Generation**에 있는지, **Code Implementation**에 있는지를 분석한다.

---

## Phase2. Re-planning 병목 분석

**연구 질문**
- 작은 코드 모델은 잘못된 전략을 실행 피드백만으로 수정할 수 있는가?

**실험**
- Feedback-only Repair
- Re-plan + Repair

실패 이후의 복구 과정에서 단순 코드 수정(Repair)과 전략 재수립(Re-planning)의 역할을 비교·분석한다.

---

## Phase3. Planning 능력 향상

Phase1과 Phase2에서 확인된 병목을 기반으로 작은 모델의 Planning 능력을 향상시키는 방법을 연구한다.

후보 방법은 다음과 같다.

- Planning Policy Learning
- Planning Memory
- Compact Planning Representation

Phase3릐 구체적인 방법은 Phase1,2의 실증 결과를 바탕으로 결정한다.