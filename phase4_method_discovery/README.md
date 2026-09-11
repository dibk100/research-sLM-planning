
### 실험 구조

Planner만 RL로 학습하고 Coder는 고정하는 구조.

```
Problem x
   │
   ▼
[Trainable Planner]  ← GRPO(verl)로 업데이트되는 유일한 대상
   │  plan P
   ▼
[Frozen Coder]       ← 동일 체크포인트, greedy 디코딩
   │  code C
   ▼
[TACO Evaluator]     ← DeepCoder semantics
   │
   ▼
Reward R  ──────────► update Planner only
```

reward 정의만 다른 두 갈래를 각각 독립 디렉터리로 둔다.

| 디렉터리                  | Reward                        |
| --------------------- | ----------------------------- |
| `vanilla_planning_rlvr/` | binary: 전체 테스트 통과 여부 R ∈ {0,1} |
| `tpr_planning_rlvr/`     | TPR(test pass rate): 통과 테스트 비율 (dense) |

두 갈래는 같은 학습 데이터(`vanilla_planning_rlvr_scale1000`)와 같은 verl 진입점(`verl.trainer.main_ppo_sync`)을 쓰고, reward 모듈과 reward manager만 갈아 끼운다.
그리고 별도로 `diagnostic/`이 학습과 무관하게 base planner의 plan quality를 N×M 샘플링으로 진단한다.

**`vanilla_planning_rlvr/`는 나머지 두 디렉터리의 의존성이기도 하다.**

- `tpr_planning_rlvr/reward/planning_tpr_reward.py` → `vanilla.../reward/planning_execution_reward.py` (프롬프트 구성·frozen coder RPC)
- `diagnostic/scripts/run_nxm_diagnostic.py` → `vanilla.../reward/planning_reward_utils.py` (`build_code_prompt`, `select_reward_tests`)

### 폴더 구조

```
phase4_method_discovery/
│
├── vanilla_planning_rlvr/               # (A) binary execution reward — 데이터/워커/로깅의 원본 구현
│   ├── configs/
│   │   ├── vanilla_planning_rlvr_qwen25coder3b.yaml   # 실험 config: planner/coder 생성 설정, grpo 옵션
│   │   │                                              #   (reward 모듈이 OmegaConf로 읽음)
│   │   └── verl_qwen25coder3b_900steps.yaml           # verl hydra config (ppo_trainer 상속), 900-step 본 학습
│   │
│   ├── data/                            # 학습 데이터 준비 (tpr·diagnostic도 여기서 만든 parquet을 씀)
│   │   ├── download_deepcoder_taco.py   #   1) HF DeepCoder-TACO 7,436문제 → raw jsonl
│   │   ├── eda_deepcoder_taco.py        #   2) EDA (stdin 6,387 / functional 1,049 확인)
│   │   └── build_verl_dataset.py        #   3) stdin-only 필터 → train/val split → verl parquet
│   │                                    #      ※ prompt에는 problem statement만, tests는
│   │                                    #        extra_info.problem.private_tests 에만 (leakage 방지)
│   │
│   ├── reward/                          # verl ↔ 우리 reward 환경 연결부
│   │   ├── planning_reward_utils.py     #   공용 유틸: build_code_prompt / select_reward_tests
│   │   │                                #   (tpr·diagnostic가 여기서 import — 갈래 간 프롬프트/테스트 선택 일치 보장)
│   │   ├── planning_execution_reward.py #   plan → code prompt → (frozen coder RPC) → CodeParser
│   │   │                                #   → TACOEvaluator → binary reward
│   │   └── planning_reward_manager.py   #   verl RewardManager 구현체 (reward_loop에 등록)
│   │
│   ├── workers/
│   │   ├── frozen_coder_worker.py       #   frozen coder GPU inference 서비스
│   │   │                                #   (init→CPU 대기 / wake_up→CUDA / sleep→GPU 반납)
│   │   └── __init__.py
│   │
│   ├── logging/
│   │   └── rollout_logger.py            # rollout 단위 기록 (plan/code/reward) → jsonl
│   │
│   ├── evaluation/
│   │   ├── rl_planner_strategy.py       # 학습된 planner + frozen coder를 Phase1 전략 인터페이스로 감쌈
│   │   └── checkpoint_dataset.py        # verl parquet val split → ProblemExample 변환
│   │
│   ├── scripts/
│   │   ├── run_training.sh              #   본 학습 진입점. conda env·PYTHONPATH·의존성 검증 후
│   │   │                                #   verl.trainer.main_ppo_sync 실행
│   │   │                                #   usage: bash run_training.sh <verl_config.yaml>
│   │   ├── run_grpo_smoke.sh            #   최소 스텝 smoke
│   │   ├── run_grpo_pilot.sh            #   50-step pilot
│   │   ├── export_verl_lora.py          #   verl global_step_N 체크포인트 → merge 가능한 LoRA adapter
│   │   ├── evaluate_checkpoint.py       #   TACO val split 평가 (base vs stepN)
│   │   ├── evaluate_rl_planner.py       #   Phase1 config/데이터(LCB-v6)로 평가 → Phase1과 직접 비교
│   │   └── diagnose_taco_reward_granularity.py  # all-zero GRPO group의 reward 해상도 진단
│   │
│   ├── analysis/
│   │   └── analyze_full_training.py     # reward sparsity, informative group 비율, entropy/안정성 분석
│   │
│   ├── outputs/                         # 경량 산출물만 repo에 유지 (대용량은 /mnt/hdd)
│   │   ├── full900_training.log         #   900-step 본 학습 로그
│   │   ├── full900_metrics.csv          #   analyze_full_training.py 산출물
│   │   └── checkpoint_eval/             #   TACO val100: base_val100.jsonl, step900_val100.jsonl
│   │
│   └── README.md                        # 평가기 버전, 학습 데이터 파이프라인 메모
│
├── tpr_planning_rlvr/                   # (B) TPR dense reward — reward/manager만 교체, 나머지는 vanilla 재사용
│   ├── configs/
│   │   ├── tpr_planning_rlvr_qwen25coder3b.yaml
│   │   └── verl_qwen25coder3b_900steps.yaml
│   ├── reward/
│   │   ├── planning_tpr_reward.py       #   통과 테스트 비율 reward (vanilla의 프롬프트/RPC 유틸 import)
│   │   ├── planning_tpr_reward_manager.py
│   │   └── test_non_fail_fast_equivalence.py  # early-stop 제거 backend가 기존 판정과 동일함을 검증
│   ├── evaluation/
│   │   └── rl_planner_strategy.py
│   ├── scripts/
│   │   ├── run_training.sh
│   │   ├── run_grpo_pilot.sh
│   │   ├── export_verl_lora.py
│   │   └── evaluate_rl_planner.py
│   ├── analysis/
│   │   ├── analyze_full_training.py         # 900-step 학습 로그 → metric csv (vanilla와 동일 기준)
│   │   ├── analyze_rl_planner_eval.py       # Phase1 self_plan(base) vs stepN 평가 결과 비교·전이 분석
│   │   ├── analyze_training_dynamics.py     # 학습 로그 → step별 metric csv/summary
│   │   ├── analyze_training_trajectory.py   # metric csv → trajectory plot(png)
│   │   └── compare_val100.py                # base vs vanilla-step900 vs tpr-step900 3자 비교 (val100)
│   └── outputs/
│       ├── full900_training.log
│       ├── full900_metrics.csv
│       └── checkpoint_eval/
│           ├── step900_val100.jsonl
│           └── base_vanilla_tpr_val100.csv  #   compare_val100.py 산출물 (세 갈래 통합 비교표)
│
├── diagnostic/                          # (C) 학습과 무관한 진단: 한 문제에 plan N개 × code M개 샘플링
│   │                                    #   RQ-D1 plan에 따른 성공률 변동 / RQ-D2 coder 구현 안정성
│   │                                    #   RQ-D3 현재 reward가 plan quality를 식별하는가
│   ├── configs/
│   │   └── nxm_qwen25coder3b.yaml       #   N/M, planner·coder 샘플링 설정 (둘 다 temp 0.7 / top_p 0.95)
│   │                                    #   데이터는 학습과 같은 val.parquet, evaluation 설정
│   │                                    #   (max_reward_tests·timeout)은 training config와 일치시킬 것
│   ├── scripts/
│   │   └── run_nxm_diagnostic.py        #   N×M generation + 채점 러너 (plan/code seed 고정)
│   ├── analysis/
│   │   ├── analyze_nxm_diagnostic.py    #   plan/problem 단위 기본 metric (Oracle@N, Best–Mean Gap 등)
│   │   ├── analyze_m1_reward_reliability.py  # M=1 reward가 plan 선택 기준으로 신뢰 가능한지 시뮬레이션
│   │   └── analyze_coverage_conditioned.py   # coverage 조건부 분석 (해결 가능 문제로 한정한 metric)
│   ├── outputs/
│   │   ├── sanity/                      #   base_n2_m2.jsonl
│   │   └── pilot/                       #   base_n8_m{2,4}_{10,100}problems.jsonl
│   │       └── *_analysis/              #   위 analysis 3종의 csv·json 산출물
│   └── README.md                        #   RQ 정의, metric 설계, 실험 프로토콜, 코드 재사용 범위
│
└── README.md
```

### 실행 환경 · 경로

Phase 1–3(`/mnt/hdd/conda_envs/slm`)과 **다른 환경**을 쓴다.

| 항목            | 값                                                                  |
| ------------- | ------------------------------------------------------------------ |
| conda env     | `/mnt/hdd/conda_envs/planning_rlvr`                                |
| verl          | `~/workspace/verl` (진입점 `verl.trainer.main_ppo_sync`)               |
| `PYTHONPATH`  | `<project_root>:~/workspace/verl` (+평가 시 `~/workspace/LiveCodeBench`) |
| 학습 데이터        | `/mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/vanilla_planning_rlvr_scale1000/{train,val}.parquet` |
| 체크포인트         | `/mnt/hdd/project_sLM_planning/checkpoints/<run>/global_step_N`, export본은 `<run>/exported/stepN` |
| Phase1 비교 평가 결과 | `/mnt/hdd/project_sLM_planning/output/phase4{,_tpr}_rl_planner_eval/stepN/results.jsonl` |

- `src/`는 패키지로 설치돼 있지 않으므로 항상 프로젝트 루트를 `PYTHONPATH`에 넣고 실행한다.
- Phase 1–3은 `src/execution/livecodebench_evaluator.py`, Phase 4 training reward는 `src/execution/taco_evaluator.py`(DeepCoder semantics)로 서로 다르다.
- 대부분의 스크립트는 파일 상단 docstring에 그대로 복사해 쓸 수 있는 실행 커맨드가 들어 있다.
