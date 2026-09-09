
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
`tpr_planning_rlvr/reward/planning_tpr_reward.py`는 `vanilla_planning_rlvr/reward/planning_execution_reward.py`의 프롬프트 구성·RPC 유틸을 그대로 import 하므로, **vanilla 쪽은 tpr의 의존성이기도 하다.**

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
│   ├── data/                            # 학습 데이터 준비 (두 갈래 공용)
│   │   ├── download_deepcoder_taco.py   #   1) HF DeepCoder-TACO 7,436문제 → raw jsonl
│   │   ├── eda_deepcoder_taco.py        #   2) EDA (stdin 6,387 / functional 1,049 확인)
│   │   └── build_verl_dataset.py        #   3) stdin-only 필터 → train/val split → verl parquet
│   │                                    #      ※ prompt에는 problem statement만, tests는
│   │                                    #        extra_info.problem.private_tests 에만 (leakage 방지)
│   │
│   ├── reward/                          # verl ↔ 우리 reward 환경 연결부
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
│   │   ├── full900_training.log
│   │   ├── full900_metrics.csv
│   │   └── checkpoint_eval/             #   base/step900 × val3/val100 jsonl
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
│   │   ├── analyze_rl_planner_eval.py       # Phase1 self_plan(base) vs stepN 평가 결과 비교·전이 분석
│   │   ├── analyze_training_dynamics.py     # 학습 로그 → step별 metric csv/summary
│   │   └── analyze_training_trajectory.py   # metric csv → trajectory plot(png)
│   └── outputs/
│       └── training_verl_qwen25coder3b_900steps_*.log
│
├── diagnostic/                          # RQ-D1~D3: plan별 downstream 성공률(N plans × M codes) 진단
│   └── README.md                        #   metric 정의와 예상 결과 유형만 정리된 설계 단계
│
├── archive/                             # 초기 sanity/smoke 결과 보관
│   ├── deepcoder_taco_eda_summary.json
│   ├── deepcoder_taco_multi_solution_sanity.jsonl
│   ├── deepcoder_taco_rllm_sanity.jsonl
│   └── reward_smoke_test.jsonl
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
