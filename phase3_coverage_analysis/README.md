

Phase 1
Problem → Strategy → 1 output → Runner → Eval

Phase 2
Failure → Strategy → 1 output → Runner → Eval

Phase 3-A
Problem → PlanningCoverageStrategy → N candidates
       → PlanningCoverageRunner → N evaluations

candidate.py
├── CandidateRecord
├── ProblemRecord
├── candidate_seed()
└── summarize_candidates()

runner.py
├── CodeParser
├── Evaluator
├── candidate evaluation
├── problem/sample loop
├── resume
└── JSONL logging

analysis/analyze_planning_coverage.py
├── prefix_ks()
├── oracle_at_k()
├── best_ratio_at_k()
├── unbiased_pass_at_k()
└── check_monotonicity()

```
phase3_coverage_analysis/
├── README.md
│
├── a_planning_coverage/
    ├── configs/
    │   └── planning_coverage_qwen25Coder3b.yaml
    │
    ├── scripts/
    │   ├── run_planning_coverage.py
    │   └── sanity_check.py
    │
    ├── strategies/
    │   └── planning_coverage.py
    │
    ├── analysis/
    │   └── analyze_planning_coverage.py        # Oracle@k
    │
    ├── candidate.py             # PlanCandidate, CandidateEvaluation
    └── runner.py
│
├── b_code_coverage/
│   ├── configs/
│   │   └── code_coverage_qwen25Coder3b.yaml
│   ├── scripts/
│   │   └── run_code_coverage.py
│   ├── analysis/
│   │   └── analyze_code_coverage.py
│   ├── strategy.py
│   └── runner.py
│
└── analysis/
    └── compare_planning_vs_code.py
```