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


### 폴더 구조
Phase 1의 실행 인프라를 그대로 계승하고, refinement 관련 모듈만 추가하는 방향으로 설계함.
첫째, Phase 1과 동일하게 코드/설정은 로컬 git 저장소, 대용량 results.jsonl과 teacher re-plan 데이터는 /mnt/hdd로 분리합니다. 둘째, Phase 2에서는 Phase 1의 direct_500_stdin/results.jsonl을 initial state의 source of truth로 재사용해서 Initial Generation을 다시 수행하지 않는 편이 좋습니다.
여기서 Phase 1과 비교해 가장 중요한 변경점은 dataset_loader.py가 아니라 phase1_failure_loader.py가 중심이 된다는 점

```
p~/workspace/project_sLM_planning/
└── phase2_replanning_bottleneck/
    ├── README.md
    ├── requirements.txt
    │
    ├── configs/
    │   ├── feedback_repair.yaml
    │   ├── self_replan.yaml
    │   └── teacher_replan.yaml
    │
    ├── prompts/
    │   ├── feedback_repair.txt
    │   ├── self_replan_plan.txt
    │   ├── self_replan_code.txt
    │   ├── teacher_replan_code.txt
    │   └── teacher_replan_generation.txt
    │
    ├── scripts/
    │   ├── run_experiment.py
    │   ├── run_all.sh
    │   └── build_teacher_replans.py
    │
    ├── src/
    │   ├── schemas.py
    │   │
    │   ├── datasets/
    │   │   └── phase1_failure_loader.py
    │   │
    │   ├── models/
    │   │   └── generator.py
    │   │
    │   ├── plans/
    │   │   └── teacher_replan_store.py
    │   │
    │   ├── strategies/
    │   │   ├── __init__.py
    │   │   ├── feedback_repair.py
    │   │   ├── self_replan.py
    │   │   └── teacher_replan.py
    │   │
    │   ├── execution/
    │   │   ├── code_extractor.py
    │   │   └── evaluator.py
    │   │
    │   └── utils/
    │       ├── config.py
    │       ├── seed.py
    │       ├── jsonl_logger.py
    │       ├── record_builder.py
    │       └── run_metadata.py
    │
    ├── outputs/
    │   └── pilot/
    │       ├── feedback_repair/
    │       ├── self_replan/
    │       └── teacher_replan/
    │
    ├── analysis/
    │   ├── compare_three_strategies.py
    │   ├── analyze_recovery.py
    │   ├── analyze_difficulty.py
    │   └── analyze_transitions.py
    │
    └── archive/
        └── comparison_500/
```

작업 흐름
```
Phase 1
direct_500_stdin/results.jsonl
            │
            │ initial passed == false
            ↓
 phase1_failure_loader.py
            │
            ├─────────────────────────────┐
            │                             │
            ↓                             ↓
   Initial Code                    Execution Feedback
            │                             │
            └──────────────┬──────────────┘
                           ↓
                    run_experiment.py
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
 Feedback Repair      Self Re-plan       Teacher Re-plan
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ↓
                       evaluator
                           ↓
                    Phase 2 results

```

Phase 2에서는 LiveCodeBench를 다시 읽고 initial code를 다시 생성하는 게 아니라, Phase 1 Direct 결과 중 실패 trajectory를 로딩하는 방식