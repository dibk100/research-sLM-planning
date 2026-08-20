
### 실험 구조 
기본 학습 baseline :  x→P∼πθ(P∣x)→C∼πθ(C∣x,P)→R(C)

- Planner만 RL로 학습하고 Coder는 고정하는 구조 : P∼πθ(P∣x)만 RL action으로 봄. 그리고 R(P) = R(C)로 planner에 reward를 줌.
    - execution reward로 planning distribution 자체가 개선되는가?

초기 구현
```
Problem x
   │
   ▼
[Trainable Planner]
   │
   │  plan P
   ▼
[Frozen Coder]
   │
   │  code C
   ▼
[LiveCodeBench Evaluator]
   │
   ▼
Reward R(C) ∈ {0,1}
   │
   └──────────────► update Planner only
```

src/datasets/livecodebench.py
→ 그대로 사용

src/execution/livecodebench_evaluator.py
→ reward 계산 backend로 재사용

prompt_templates/self_plan_plan.txt
→ 초기 planner prompt의 기준으로 재사용 가능

prompt_templates/self_plan_code.txt
→ frozen coder prompt로 재사용 가능

src/models/model_adapter.py
→ evaluation 또는 frozen coder inference에서 재사용 가능

verl/vLLM rollout을 통한 RL training 부분은 기존 src/models/generator.py와 억지로 통합하지 않는 것

reward function내부
```
plan
  ↓
frozen coder
  ↓
code
  ↓
LiveCodeBench evaluator
  ↓
0 / 1 reward
```

```
Trainable:
    Planner

Frozen:
    Coder
    Evaluator

Reward:
    all_tests_pass ∈ {0, 1}
```

```
phase4_method_discovery/
├── configs/
│   ├── qwen25Coder3b_planning_rlvr.yaml
│   └── ...
│
├── prompts/
│   └── planning_rlvr.txt
│
├── data/
│   ├── build_train_data.py
│   ├── build_eval_data.py
│   └── README.md
│
├── reward/
│   ├── execution_reward.py
│   └── reward_adapter.py
│
├── rollout/
│   ├── planning_rollout.py
│   └── code_rollout.py
│
├── training/
│   ├── planning_rlvr_trainer.py
│   ├── verl_reward_fn.py
│   └── launch_train.py
│
├── evaluation/
│   ├── evaluate_checkpoint.py
│   ├── evaluate_planning.py
│   └── compare_checkpoints.py
│
├── scripts/
│   ├── build_data.sh
│   ├── run_train.sh
│   └── run_eval.sh
│
├── analysis/
│   ├── analyze_training_curve.py
│   ├── analyze_plan_distribution.py
│   └── analyze_pass_rate.py
│
├── sample_log/
│
└── README.md
```

기록 : 각 rollout마다
```
{
  "problem_id": "...",
  "global_step": 120,
  "group_id": "...",
  "sample_id": 3,

  "plan": "...",

  "plan_token_count": 132,
  "plan_logprob": -41.83,

  "generated_code": "...",

  "reward": 1,
  "passed_tests": 12,
  "total_tests": 12,

  "generation_seed": 42,

"plan_tokens": [...],
"token_logprobs": [...]

}

```