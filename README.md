# project_sLM_planning

본 연구는 **3B–8B 규모의 작은 코드 언어 모델(Small Code Model)** 을 대상으로, Competitive Programming 환경에서 **Planning 능력의 한계와 병목을 분석하고 이를 개선하는 방법**을 연구한다.

### Note.

1. 평가 프로토콜 : Official evaluator vs Diagnostic evaluator 
- LiveCodeBench의 기본 평가 구현은 실패가 확인되면 나머지 테스트의 실행을 조기에 종료할 수 있다. 본 연구에서는 문제 단위 정답 판정 기준은 유지하면서, 실패한 생성물의 부분적 정확성과 refinement dynamics를 분석하기 위해 모든 테스트 케이스를 실행하였다. 모든 테스트를 통과한 경우에만 해당 문제를 성공으로 판정하였다.


연구는 다음 세 단계로 진행된다.
```
project-sLM-planning/
│
├── src/                              # 모든 Phase가 공유하는 코드
│   ├── schemas.py
│   │
│   ├── datasets/
│   │   ├── codeforces.py           # filtered official-test benchmark loading
│   │   ├── livecodebench.py        # stdin/functional benchmark loading 
│   │   └── dataset_loader.py               # config에 따라 dataset 선택
│   │
│   ├── models/
│   │   ├── generator.py
│   │   └── model_adapter.py
│   │
│   ├── parsing/
│   │   ├── code_parser.py
│   │   └── plan_parser.py
│   │
│   ├── execution/
│   │   ├── evaluator.py                 # 공통 dispatch
│   │   ├── livecodebench_evaluator.py   # 공식 LCB evaluator wrapper
│   │   ├── codeforces_evaluator.py      # Codeforces용 별도 구현
│   │   └── diagnostic_evaluator.py
│
│   └── utils/
│       ├── config.py
│       ├── seed.py
│       ├── jsonl_logger.py
│       ├── record_builder.py
│       └── run_metadata.py
│
├── prompt_templates/                 # 공통 Prompt Template
│   ├── direct.txt
│   ├── self_plan_plan.txt
│   ├── self_plan_code.txt
│   ├── teacher_plan_code.txt
│   └── teacher_plan_generation.txt
│
│
├── phase0_diagnostic_benchmark/
│
├── phase1_planning_bottleneck/
│   ├── configs/
│   ├── scripts/
│       ├── run_direct.py
│       ├── run_self_plan.py
│       └── run_teacher_plan.py
│   ├── strategies/
│   ├── runner.py
│   └── analysis/
│
├── phase2_replanning_bottleneck/
│   ├── configs/
│   ├── scripts/
│   ├── strategies/
│   └── analysis/
│
├── phase3_coverage_analysis/
│   │
│   ├── a_planning_coverage/
│   │   ├── configs/
│   │   ├── scripts/
│   │   └── analysis/
│   │
│   └── b_coder_coverage/
│       ├── configs/
│       ├── scripts/
│       └── analysis/
│
├── tests/
│   ├── test_code_parser.py
│   ├── test_executor.py
│   ├── test_evaluator.py
│   └── test_dataset.py
│
├── requirements.txt
└── README.md
```

나중에 아래와 같이 리드미 작성하기.

1. Research Overview
2. Environment
3. Dataset Preparation
4. Model Setup
5. Phase 1
6. Phase 2
7. Phase 3
8. Evaluation
9. Output Format
10. Reproduction

데이터셋 진단용 기록 

Primary diagnosis
- LCB-v6 stdin 300

Cross-evaluation
- LCB-v6 functional 300
- Codeforces 98



livecode는 modified APPS checker를 쓴다고 함