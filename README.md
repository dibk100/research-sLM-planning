# project_sLM_planning

본 연구는 **3B–8B 규모의 작은 코드 언어 모델(Small Code Model)** 을 대상으로, Competitive Programming 환경에서 **Planning 능력의 한계와 병목을 분석하고 이를 개선하는 방법**을 연구한다.

### Note.

1. 평가 프로토콜 : Official evaluator vs Diagnostic evaluator 
- LiveCodeBench의 기본 평가 구현은 실패가 확인되면 나머지 테스트의 실행을 조기에 종료할 수 있다. 본 연구에서는 문제 단위 정답 판정 기준은 유지하면서, 실패한 생성물의 부분적 정확성과 refinement dynamics를 분석하기 위해 모든 테스트 케이스를 실행하였다. 모든 테스트를 통과한 경우에만 해당 문제를 성공으로 판정하였다.

2. Phase1-3와 Phase4의 LiveCodeBench evaluator가 서로 다름.   
- Phase 4 training reward만 DeepCoder semantics에 고정함.


연구는 Phase 0–4로 진행된다. (Phase 0은 데이터셋 진단/전처리, Phase 1–3은 inference experiment framework, Phase 4는 RL training framework)
```
project_sLM_planning/
│
├── src/                                     # 전 Phase 공유 라이브러리 (※ __init__.py 없음 → PYTHONPATH=. 필요)
│   ├── schemas.py                           # 공통 레코드/스키마 정의
│   │
│   ├── datasets/                            # 벤치마크 로더
│   │   ├── livecodebench.py                 # LCB-v6 (Phase1–3 주 데이터셋)
│   │   ├── codeforces.py                    # cross-evaluation용
│   │   ├── deepcoder_taco.py                # Phase4 RL 학습셋
│   │   ├── dataset_loader.py                # 공통 진입점
│   │   └── phase1_failure_loader.py         # Phase1 실패 케이스 → Phase2 입력
│   │
│   ├── models/
│   │   ├── generator.py                     # vLLM 기반 배치 생성
│   │   └── model_adapter.py                 # 모델별 chat template / stop 처리
│   │
│   ├── parsing/
│   │   └── code_parser.py                   # 생성물에서 code block 추출
│   │
│   ├── plans/
│   │   ├── teacher_plan_store.py            # Phase1 teacher plan 조회
│   │   └── teacher_replan_store.py          # Phase2 teacher replan 조회
│   │
│   ├── execution/                           # 코드 실행 · 채점
│   │   ├── evaluator.py                     # 공통 evaluator 인터페이스
│   │   ├── livecodebench_evaluator.py       # Phase1–3 채점 backend
│   │   ├── diagnostic_evaluator.py          # 전 테스트 실행(early-stop 없음) 진단 채점
│   │   ├── taco_evaluator.py                # DeepCoder-TACO 채점
│   │   ├── evaluator_off_dia.py             # [미사용] official/diagnostic 분기 버전
│   │   └── deepcoder/                       # Phase4 reward 전용 (DeepCoder semantics 고정)
│   │       ├── livecodebench.py
│   │       └── utils.py
│   │                                        # (codeforces_evaluator.py 는 아직 없음)
│   └── utils/
│       ├── config.py                        # YAML config 로딩
│       ├── download_dataset.py
│       ├── feedback.py                      # Phase2 실행 피드백 포맷팅
│       ├── jsonl_logger.py                  # results.jsonl 스트리밍 기록
│       ├── record_builder.py
│       ├── run_metadata.py                  # run_metadata.json (실제 사용 체크포인트 기록)
│       └── seed.py
│
├── prompt_templates/                        # 전 Phase 공유 프롬프트
│   ├── direct.txt
│   ├── self_plan_plan.txt                   # (Phase4 planner prompt로도 재사용)
│   ├── self_plan_code.txt                   # (Phase4 frozen coder prompt로도 재사용)
│   ├── self_replan_plan.txt
│   └── feedback_only.txt                    # feedback_regeneration용
│
├── phase0_diagnostic_benchmark/             # 데이터셋 EDA · 전처리 (노트북)
│   ├── 01_EDA_livecodebench-v6.ipynb
│   ├── 02_preprocessing_livecodebench-v6.ipynb
│   ├── 02_preprocessing_codeforces.ipynb
│   └── README.md
│
├── phase1_planning_bottleneck/              # Q. planning이 병목인가? (direct vs self-plan vs teacher-plan)
│   ├── runner.py                            # 실험 오케스트레이션 (config → 생성 → 채점 → 로깅)
│   ├── configs/                             # {direct, self_plan, teacher_plan} × {phi3, qwen253b, qwen25Coder3b}
│   │   └── teacher_plan_make.yaml           # teacher plan 생성용 별도 config
│   ├── strategies/                          # 전략별 프롬프트 구성 · 다단계 호출
│   │   ├── direct.py
│   │   ├── self_plan.py
│   │   └── teacher_plan.py
│   ├── scripts/                             # CLI 엔트리포인트
│   │   ├── run_direct.py
│   │   ├── run_self_plan.py
│   │   ├── run_teacher_plan.py
│   │   └── run_all.sh
│   ├── teacher_plan_generation/             # teacher(Opus) plan 오프라인 생성 파이프라인
│   │   ├── export_teacher_inputs.py         #   1) 문제 → teacher 입력 export
│   │   ├── batch_teacher_plans.py           #   2) batch API 호출
│   │   ├── build_teacher_plans.py           #   3) 응답 → plan store 빌드
│   │   └── validate_teacher_plans.py        #   4) 검증
│   ├── analysis/
│   │   ├── compare_strategies.py            # 전략 간 비교표/전이 분석
│   │   └── inspect_sample.py
│   ├── archive/                             # 모델별 비교 결과 CSV 보관 (300문제)
│   │   ├── comparison_phi3_300/
│   │   ├── comparison_qwen253b_300/
│   │   └── comparison_qwen25Coer3b_300/
│   └── README.md
│
├── phase2_replanning_bottleneck/            # Q. 실패 후 re-planning이 병목인가? (Phase1 실패 케이스 입력)
│   ├── runner.py
│   ├── configs/                             # {feedback_regeneration, self_replan, teacher_replan} × 3모델
│   │   └── teacher_replan_make.yaml
│   ├── strategies/
│   │   ├── feedback_regeneration.py         # 피드백만 주고 코드 재생성
│   │   ├── self_replan.py                   # 스스로 plan 재작성
│   │   └── teacher_replan.py                # teacher replan 주입
│   ├── scripts/
│   │   ├── run_experiment.py                # 공통 러너 (전략 인자로 선택)
│   │   ├── run_feedback_regeneration.py
│   │   ├── run_self_replan.py
│   │   ├── run_teacher_replan.py
│   │   └── run_all.sh
│   ├── teacher_replan_generation/           # teacher replan 오프라인 생성 (Phase1과 동일 4단계)
│   │   ├── export_teacher_inputs.py
│   │   ├── batch_teacher_plans.py
│   │   ├── batch_teacher_replans.py
│   │   ├── build_teacher_plans.py
│   │   └── validate_teacher_plans.py
│   ├── analysis/
│   │   ├── analyze_phase2_results.py
│   │   ├── analyze_failure_feedback_structure.py
│   │   └── analyze_feedback_stderr.py
│   ├── .gitignore
│   └── README.md
│
├── phase3_coverage_analysis/                # Q. 병목이 sampling coverage로 설명되는가? (best-of-N)
│   ├── a_planning_coverage/                 #   plan을 N개 샘플링 → coverage 측정
│   │   ├── runner.py
│   │   ├── candidate.py                     #   후보 plan 표현/관리
│   │   ├── configs/                         #   phi3 / qwen253b / qwen25Coder3b
│   │   ├── strategies/planning_coverage.py
│   │   ├── scripts/run_planning_coverage.py
│   │   ├── analysis/analyze_planning_coverage.py
│   │   └── README.md
│   ├── b_code_coverage/                     #   plan 고정 후 code를 N개 샘플링 → coverage 측정
│   │   ├── runner.py
│   │   ├── candidate.py
│   │   ├── fixed_plan_loader.py             #   a_planning_coverage 결과에서 plan 고정
│   │   ├── configs/                         #   phi3 / qwen253b / qwen25Coder3b
│   │   ├── strategies/code_coverage.py
│   │   ├── scripts/run_code_coverage.py
│   │   ├── analysis/analyze_code_coverage.py
│   │   └── README.md
│   ├── analysis/compare_planning_vs_code.py # a vs b 교차 비교
│   ├── sample_log/                          # 샘플 실행 로그
│   │   ├── planning_coverage_pilot/results.jsonl
│   │   └── code_coverage/results.jsonl
│   └── README.md
│
├── phase4_method_discovery/                 # RL training framework (verl + vLLM, GRPO)
│   │                                        # 구조: x → [Trainable Planner] → P → [Frozen Coder] → C → R(C)
│   ├── vanilla_planning_rlvr/               # (A) binary execution reward R(C) ∈ {0,1}
│   │   ├── configs/                         #   verl GRPO config (smoke / pilot50 / 900step)
│   │   ├── data/
│   │   │   ├── download_deepcoder_taco.py
│   │   │   ├── eda_deepcoder_taco.py
│   │   │   └── build_verl_dataset.py        #   verl parquet 데이터셋 빌드
│   │   ├── reward/
│   │   │   ├── planning_execution_reward.py #   plan→coder→evaluator→reward
│   │   │   └── planning_reward_manager.py   #   verl reward manager 훅
│   │   ├── workers/frozen_coder_worker.py   #   고정 coder rollout worker
│   │   ├── logging/rollout_logger.py
│   │   ├── evaluation/
│   │   │   ├── rl_planner_strategy.py       #   학습된 planner를 Phase1 방식으로 평가
│   │   │   └── checkpoint_dataset.py
│   │   ├── scripts/                         #   run_training.sh / run_grpo_{smoke,pilot}.sh,
│   │   │                                    #   evaluate_checkpoint.py, evaluate_rl_planner.py,
│   │   │                                    #   export_verl_lora.py, diagnose_taco_reward_granularity.py
│   │   ├── analysis/analyze_full_training.py
│   │   ├── outputs/                         #   학습 로그 · checkpoint_eval jsonl · metrics csv
│   │   └── README.md
│   │
│   ├── tpr_planning_rlvr/                   # (B) TPR(test pass rate) dense reward 변형
│   │   ├── configs/verl_grpo_pilot_50step.yaml
│   │   ├── reward/
│   │   │   ├── planning_tpr_reward.py
│   │   │   ├── planning_tpr_reward_manager.py
│   │   │   └── test_non_fail_fast_equivalence.py   # early-stop 제거 동등성 테스트
│   │   ├── evaluation/rl_planner_strategy.py
│   │   ├── scripts/                         #   run_grpo_pilot.sh, evaluate_rl_planner.py, export_verl_lora.py
│   │   ├── analysis/                        #   analyze_rl_planner_eval / training_dynamics / training_trajectory
│   │   └── outputs/                         #   training log, trajectory_analysis(png+csv), analysis csv
│   │
│   ├── diagnostic/                          # (비어 있음)
│   ├── archive/                             # TACO/reward 초기 sanity·smoke 결과
│   └── README.md
│
├── .gitignore                               # *.jsonl, *.log, __pycache__ 등 실험 산출물 제외
└── README.md
```

### 실행 산출물 저장 위치

코드는 repo에 두고, 실험 산출물(`results.jsonl`, checkpoint, 데이터셋)은 전부 `/mnt/hdd/project_sLM_planning/` 아래에 둔다.
루트 파티션 여유가 적고 run 하나가 수 GB~수십 GB(전체 `test_results`를 저장하므로)이기 때문이며, `.gitignore`가 `*.jsonl` / `*.log`를 이미 제외한다.

```
/mnt/hdd/project_sLM_planning/
├── data/                                  # 데이터셋 및 teacher 산출물
│   ├── livecodebench_v6/  codeforces/  deepcoder_taco/
│   ├── teacher_plans/                     # Phase1 teacher plan (Opus 생성)
│   └── teacher_replans/                   # Phase2 teacher replan
│
├── phase{1,2,3}/                          # ← 현재 config들이 가리키는 정규 레이아웃
│   └── livecodebench_v6_stdin/
│       └── {phi3, qwen253b, qwen25Coder3b}/
│           └── <strategy>/                # phase1: direct | self_plan | teacher_plan
│               ├── results.jsonl          # phase2: feedback_regeneration | self_replan | teacher_replan
│               ├── run_config.yaml        # phase3: planning_coverage | code_coverage
│               └── run_metadata.json      #   ← 실제 사용된 체크포인트는 여기서 확인
│
├── checkpoints/                           # verl LoRA 체크포인트
│   └── {vanilla,tpr}_planning_rlvr_lora_pilot50/, vanilla_qwen25Coder_scale1000_900step/
│
└── (legacy) 초기 run 트리 — 모델별로 디렉터리가 갈려 있던 구버전
    ├── output/                            # Qwen2.5-Coder-3B-Instruct (README 기준 수치)
    │   ├── direct_500_stdin/  self_plan_500_stdin/  teacher_plan_500_stdin/
    │   ├── phase2_{feedback_regeneration,self_replan,teacher_replan}_500/
    │   └── phase4_rl_planner_eval/  phase4_tpr_rl_planner_eval/
    ├── output_qwen25_3b_inst/             # Qwen2.5-3B-Instruct
    ├── output_phi_3b_inst/                # Phi-3.5-mini-instruct
    └── output_phase3/qwen25_coder_3b/     # best_of_{8,16}, coder_best_of_{8,16}
```

- 모델 키 매핑: `phi3` = `microsoft/Phi-3.5-mini-instruct`, `qwen253b` = `Qwen/Qwen2.5-3B-Instruct`, `qwen25Coder3b` = `Qwen/Qwen2.5-Coder-3B-Instruct`.
- config의 `output.path`는 절대경로(HDD)로 둔다. repo 안에 `outputs/` 심볼릭 링크는 만들지 않는다.
- 어떤 체크포인트로 만든 run인지는 config가 아니라 각 run의 `run_metadata.json`으로 확인한다.
- `results.jsonl`은 8~24GB 수준이므로 통째로 로드하지 말고 스트리밍으로 파싱한다.
- Phase4는 예외적으로 학습 로그/분석 산출물이 repo 안 `phase4_method_discovery/*/outputs/`에 남아 있다(csv·png·json 위주, 용량 작음).

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