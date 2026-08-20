
### 평가기 버전
```
rLLM commit:
7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4

PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/rllm" \
python ...
```

### Training Dataset
- stdin-only DeepCoder TACO 6,387문제를 사용하기로함.
```
DeepCoder raw row
    problem
    tests
    solutions
        ↓
DeepCoderTACOLoader
        ↓
ProblemExample
    dataset="deepcoder_taco"
    evaluation_type="stdin"
    private_tests=[...]
        ↓
TACOEvaluator
        ↓
lcb_runner.check_correctness()
        ↓
EvaluationResult
```
테스트를 private_tests에만 넣는 이유는 학습 prompt로 test가 유출되는 것을 방지하기 위해서


verl GRPO+우리의 custom Planning reward

빼야 하는 DAPO-specific 요소는 clip-higher, dynamic sampling, overlong reward shaping 같은 추가 기법

초기 구현
```
phase4_method_discovery/
└── vanilla_planning_rlvr/
    ├── configs/
    │   └── qwen25Coder3b_grpo.yaml
    │
    ├── data/
    │   └── build_verl_dataset.py
    │
    ├── reward/
    │   └── planning_execution_reward.py
    │
    ├── scripts/
    │   ├── run_train.sh
    │   └── smoke_test_reward.py
    │
    ├── logging/
    │   └── rollout_logger.py
    │
    └── README.md
```