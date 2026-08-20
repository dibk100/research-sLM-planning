
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