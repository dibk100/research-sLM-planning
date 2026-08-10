# Phase2. Re-planning Bottleneck Analysis

phase1은 "처음부터 좋은 계획을 만들 수 있는가?" 관점으로 실험분석

## 1. Research Objective :

> Research Question :   
> 실패 후 실행 피드백을 받았을 때, 소형 언어 모델은 기존 전략을 수정하는 것보다 새로운 전략을 다시 계획하는 데 어려움을 겪는가?

(가설) 작은 모델이 실행 피드백을 활용하지 못하는 것이 아니라, 실패 원인을 알고도 더 나은 algorithmic strategy를 스스로 재구성하는 능력이 병목일 가능성이 높다.

## 2. Experimental Setup :
- Model: ```Qwen2.5-Coder-3B-Instruct```

- Dataset: Phase 1과 동일한 500문제

- Comparison Strategies :
  - **Feedback-based Regeneration**: 모델이 실패한 코드와 execution feedback을 바탕으로 코드를 재생성
  - **Self-Replanning Regeneration**: 모델이 실패한 코드와 execution feedback을 기반으로 새로운 revised plan을 생성한 뒤, 코드를 재생성
  - **Teacher-Replanning Regeneration**: 외부 Teacher Model이 제공한 고품질 revised plan을 바탕으로 모델이 코드를 재생성

- Budget :
  - Initial Code Generation: 1회
  - Refinement: 최대 1회
  - 따라서 각 문제당 최대 2회의 code generation을 허용

- Experimental Protocol :
  - 모든 전략은 동일한 Initial Code Generation 결과에서 시작
  - Initial execution이 PASS인 경우, refinement를 수행하지 않음
  - Initial execution이 FAIL인 경우에만 각 refinement strategy를 적용
  - 각 strategy는 동일한 initial code와 동일한 execution feedback을 입력으로 사용

- Workflow :

**Initial Success**
  `Problem → Initial Code Generation → [Execution: PASS] → Stop`

  **Feedback-only Repair**
  `Problem → Initial Code Generation → [Execution: FAIL] → Execution Feedback → Feedback-based Code Repair → [Execution]`

  **Self Re-plan + Repair**
  `Problem → Initial Code Generation → [Execution: FAIL] → Execution Feedback → Self Re-plan → Code Re-generation → [Execution]`

  **Teacher Re-plan + Repair**
  `Problem → Initial Code Generation → [Execution: FAIL] → Execution Feedback → Teacher Re-plan → Code Re-generation → [Execution]`

## Current
- qwen2.5-coder-3b-instruct [Done]
- qwen2.5-3b-instruct [ing]
- Phi-3.5-mini-instruct