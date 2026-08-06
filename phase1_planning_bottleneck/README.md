# Phase1. 초기 Planning 병목 분석

### 연구 목표 :

본 단계에서는 작은 코드 언어 모델이 어려운 프로그래밍 문제를 해결할 때 성능 저하의 원인이 **Planning Generation**에 있는지, **Code Implementation**에 있는지를 분석한다.

핵심 질문은 다음과 같다.

> **작은 코드 모델은 올바른 알고리즘 계획을 스스로 생성하지 못하는 것이 병목인가?**

### Note.

현재 진행 상황 기록하기.

- (Direct baseline Done) Phase1의 Direct baseline용 전체 실험 파이프라인과 로그,분석 인프라를 완성했고, stdin subset 10문제 smoke test까지 검증함.



### 폴더 아키텍처
```
phase1_planning_bottleneck/
├── README.md
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── model/
│   │   ├── qwen2.5_coder_3b.yaml
│   │   └── teacher_model.yaml
│   ├── dataset/
│   │   ├── humaneval_plus.yaml
│   │   ├── mbpp_plus.yaml
│   │   └── livecodebench.yaml
│   └── experiment/
│       ├── direct.yaml
│       ├── self_plan.yaml
│       └── teacher_plan.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── teacher_plans/
│   └── splits/
│
├── prompts/
│   ├── direct.txt
│   ├── self_plan.txt
│   ├── self_plan_code.txt
│   ├── teacher_plan.txt
│   └── teacher_plan_generation.txt
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   └── generator.py
│   │
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── base_dataset.py
│   │   ├── humaneval_plus.py
│   │   ├── mbpp_plus.py
│   │   └── livecodebench.py
│   │
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base_strategy.py
│   │   ├── direct.py
│   │   ├── self_plan.py
│   │   └── teacher_plan.py
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── code_extractor.py
│   │   ├── sandbox.py
│   │   ├── evaluator.py
│   │   └── result_parser.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── aggregate_results.py
│   │   ├── compare_conditions.py
│   │   ├── difficulty_analysis.py
│   │   ├── plan_analysis.py
│   │   └── failure_analysis.py
│   │
│   └── utils/
│       ├── config.py
│       ├── io.py
│       ├── logging.py
│       ├── seed.py
│       └── schema.py
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── generate_teacher_plans.py
│   ├── run_direct.py
│   ├── run_self_plan.py
│   ├── run_teacher_plan.py
│   ├── run_all.py
│   ├── evaluate.py
│   └── analyze.py
│
├── outputs/
│   ├── generations/
│   ├── executions/
│   ├── metrics/
│   ├── figures/
│   └── logs/
│
└── tests/
    ├── test_code_extractor.py
    ├── test_prompt_builder.py
    ├── test_result_schema.py
    └── test_evaluator.py

```


### 실험 구성 :

동일한 코드 모델을 대상으로 다음 세 가지 설정을 비교한다.

1. Direct Code Generation : 
문제를 입력받아 계획 없이 바로 코드를 생성한다.

2. Self-Planning :
모델이 스스로 해결 계획(Planning)을 생성한 뒤 해당 계획을 기반으로 코드를 생성한다.(두 번의 독립 호출)   
``` Problem → Student Plan → Student Code ```

- 호출 1: 계획 생성(Problem → Plan)
- 호출 2: 계획 기반 코드 생성(Problem + Generated Plan → Code)

3. Teacher-Planning :
강한 모델(Teacher,LLM)이 생성한 계획을 입력으로 제공하고, 작은 모델은 해당 계획만을 이용하여 코드를 구현한다.

- Teacher plan은 실험 전에 미리 생성하고 고정
- 실험 시에는 Teacher 모델을 다시 호출하지 않고 저장된 계획을 읽도록 함.


### 분석 목표

다음 항목을 중심으로 결과를 분석한다.

- Direct와 Self-Planning의 성능 비교
- Self-Planning과 Teacher-Planning의 성능 비교
- 문제 난이도에 따른 Planning 효과 변화
- Planning Generation과 Code Implementation 중 주요 병목 분석

### 기대 결과

만약 Teacher-Planning이 Self-Planning보다 높은 성능을 보인다면,

> 작은 코드 모델은 **좋은 계획을 구현할 능력은 보유하고 있지만,
스스로 고품질의 계획을 생성하는 능력이 부족하다**는 가설을 지지하는 근거가 될 수 있다.

반대로 Teacher-Planning 역시 성능이 낮다면,

> Planning Generation보다 **Code Implementation** 또는 **Plan Following** 능력이 주요 병목일 가능성을 고려한다.


### 다음 단계

Phase1에서 확인된 Planning 병목을 기반으로,
**Phase2. Re-planning 병목 분석**에서는 실행 피드백 이후 **Repair**와 **Re-planning**의 역할을 비교하여 실패 복구 과정의 병목을 분석한다.