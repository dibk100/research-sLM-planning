# project_sLM_planning

본 연구는 **3B–8B 규모의 작은 코드 언어 모델(Small Code Model)** 을 대상으로, Competitive Programming 환경에서 **Planning 능력의 한계와 병목을 분석하고 이를 개선하는 방법**을 연구한다.

### Note.

1. 평가 프로토콜 : Official evaluator vs Diagnostic evaluator 
- LiveCodeBench의 기본 평가 구현은 실패가 확인되면 나머지 테스트의 실행을 조기에 종료할 수 있다. 본 연구에서는 문제 단위 정답 판정 기준은 유지하면서, 실패한 생성물의 부분적 정확성과 refinement dynamics를 분석하기 위해 모든 테스트 케이스를 실행하였다. 모든 테스트를 통과한 경우에만 해당 문제를 성공으로 판정하였다.


연구는 다음 세 단계로 진행된다.(Phase1-3은 inference experiment framework, Phase 4는 RL training framework)
```
project-sLM-planning/
│
├── src/                                    # 모든 Phase가 공유하는 코드 (※ __init__.py 없음)
│   ├── schemas.py
│   │
│   ├── datasets/
│   │   ├── codeforces.py
│   │   ├── livecodebench.py
│   │   ├── dataset_loader.py
│   │   └── phase1_failure_loader.py        
│   │
│   ├── models/
│   │   ├── generator.py
│   │   └── model_adapter.py
│   │
│   ├── parsing/
│   │   └── code_parser.py
│   │
│   ├── plans/
│   │   ├── teacher_plan_store.py
│   │   └── teacher_replan_store.py          
│   │
│   ├── execution/
│   │   ├── evaluator.py
│   │   ├── evaluator_off_dia.py             # [미사용] official/diagnostic 분기 버전
│   │   ├── livecodebench_evaluator.py
│   │   └── diagnostic_evaluator.py
│   │                                        # (codeforces_evaluator.py 는 아직 없음)
│   └── utils/
│       ├── config.py
│       ├── download_dataset.py
│       ├── feedback.py                      # Phase2용
│       ├── jsonl_logger.py
│       ├── record_builder.py
│       ├── run_metadata.py
│       └── seed.py
│
├── prompt_templates/
│   ├── direct.txt
│   ├── self_plan_code.txt
│   ├── self_plan_plan.txt
│   ├── self_replan_plan.txt
│   └── feedback_only.txt                    # (feedback_regeneration.txt)
│
├── phase0_diagnostic_benchmark/
│   ├── 01_EDA_livecodebench-v6.ipynb
│   ├── 02_preprocessing_livecodebench-v6.ipynb
│   ├── 02_preprocessing_codeforces.ipynb
│   └── README.md
│
├── phase1_planning_bottleneck/
│   ├── configs/                             # direct / self_plan / teacher_plan × {phi3, qwen253b, qwen25Coder3b}
│   │   └── teacher_plan_make.yaml
│   ├── scripts/
│   │   ├── run_direct.py
│   │   ├── run_self_plan.py
│   │   ├── run_teacher_plan.py
│   │   └── run_all.sh
│   ├── strategies/
│   │   ├── direct.py
│   │   ├── self_plan.py
│   │   └── teacher_plan.py
│   ├── teacher_plan_generation/
│   │   ├── export_teacher_inputs.py
│   │   ├── batch_teacher_plans.py
│   │   ├── build_teacher_plans.py
│   │   └── validate_teacher_plans.py
│   ├── analysis/
│   │   ├── compare_strategies.py
│   │   └── inspect_sample.py
│   ├── archive/                             # 모델별 비교 결과 CSV 보관
│   │   ├── comparison_phi3_300/
│   │   ├── comparison_qwen253b_300/
│   │   └── comparison_qwen25Coer3b_300/
│   ├── runner.py
│   └── README.md
│
├── phase2_replanning_bottleneck/
│   ├── configs/                             # feedback_regeneration / self_replan / teacher_replan × 3모델
│   │   └── teacher_replan_make.yaml
│   ├── scripts/
│   │   ├── run_experiment.py
│   │   ├── run_feedback_regeneration.py
│   │   ├── run_self_replan.py
│   │   ├── run_teacher_replan.py
│   │   └── run_all.sh
│   ├── strategies/
│   │   ├── feedback_regeneration.py
│   │   ├── self_replan.py
│   │   └── teacher_replan.py
│   ├── teacher_replan_generation/
│   │   ├── export_teacher_inputs.py
│   │   ├── batch_teacher_plans.py
│   │   ├── batch_teacher_replans.py
│   │   ├── build_teacher_plans.py
│   │   └── validate_teacher_plans.py
│   ├── analysis/
│   │   ├── analyze_phase2_results.py
│   │   ├── analyze_failure_feedback_structure.py
│   │   └── analyze_feedback_stderr.py
│   ├── runner.py
│   ├── .gitignore
│   └── README.md
│
├── phase3_coverage_analysis/                #
│   ├── a_planning_coverage/
│   │   ├── configs/qwen25Coder3b.yaml
│   │   ├── scripts/run_planning_coverage.py
│   │   ├── strategies/planning_coverage.py
│   │   ├── analysis/                        
│   │   ├── candidate.py
│   │   ├── runner.py
│   │   └── README.md
│   │
│   ├── b_code_coverage/
│   │   ├── configs/qwen25Coder3b.yaml
│   │   ├── scripts/run_code_coverage.py
│   │   ├── strategies/code_coverage.py
│   │   ├── candidate.py
│   │   ├── fixed_plan_loader.py
│   │   ├── runner.py
│   │   └── README.md
│   │
│   ├── sample_log/                          # 샘플 실행 로그
│   │   ├── planning_coverage_pilot/results.jsonl
│   │   └── code_coverage/results.jsonl
│   └── README.md
│
├── phase4_method_discovery/             
│
├── .gitignore
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