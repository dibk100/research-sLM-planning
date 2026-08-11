# Phase 3-B. Code Coverage (Sampling Control)

Phase 3-A(Planning Best-of-N)의 대조군.
**plan을 Phase 1 Self-Plan 하나로 고정**하고, 동일 plan에서 code만 N번 sampling하여
Best-of-N 상승분이 "plan 다양성" 때문인지 "code sampling 자체" 때문인지 분리한다.

```
Problem
 └── Fixed Self-Plan (Phase 1)
      ├── Code 1 → Test
      ├── Code 2 → Test
      ...
      └── Code N → Test
```

## 1. Research Objective

Phase 3-A에서 Oracle@N이 상승했을 때, 그 상승이 plan 탐색에서 오는 것인지
단순한 code sampling 효과인지 알 수 없다.
Phase 3-B는 plan을 고정하여 code sampling만의 coverage 상승분을 측정한다.

- Phase 3-A gain > Phase 3-B gain → plan 탐색이 실제로 기여 (selection 병목)
- Phase 3-A gain ≈ Phase 3-B gain → 상승분은 code sampling 노이즈 (generation 병목)

## 2. Experimental Setup

- Model: `Qwen2.5-Coder-3B-Instruct`
- Dataset: Phase 1과 동일한 LiveCodeBench v6 stdin 500문제 (동일 problem id / 순서)
- Plan: Phase 1 `self_plan_500_stdin/results.jsonl`에서 로드 (재생성 없음)
- Code: temperature > 0, 문제당 N=8 sampling

## 3. 구현 구조

```
phase3_code_coverage/
├── README.md
├── requirements.txt
├── configs/
│   └── qwen25_coder_3b.yaml
├── src/
│   ├── common/                 # Phase 1에서 검증된 인프라 (수정 금지)
│   │   ├── schemas.py
│   │   ├── datasets/dataset_loader.py
│   │   ├── models/generator.py
│   │   ├── execution/{code_extractor,evaluator}.py
│   │   └── utils/{config,seed,jsonl_logger,record_builder,run_metadata}.py
│   ├── prompts.py              # code 프롬프트 구성 (Phase 1 prompts/ 를 직접 읽음)
│   ├── load_fixed_plans.py     # ★ Phase 1 Self-Plan 로드 (문제당 plan 1개 고정)
│   ├── generate_code.py        # ★ 고정 plan 조건부 code N-sampling
│   ├── execute.py              # candidate 실행/채점
│   └── utils.py                # candidate/problem record, Oracle@k 등 지표
├── scripts/
│   ├── run_code_best_of_n.py   # ★ Phase 3-B runner
│   ├── analyze_code_coverage.py
│   ├── sanity_check.py
│   └── freeze_problem_ids.py
└── archive/
    └── analysis/
```


| 파일                        | 처리                                    |
| ------------------------- | ------------------------------------- |
| `src/common/*`            | **그대로 복사, 수정 금지**                     |
| `src/prompts.py`          | 거의 그대로 재사용                            |
| `src/generate_plans.py`   | **불필요**                               |
| `src/load_fixed_plans.py` | **새로 작성**                             |
| `src/generate_code.py`    | **B용 stochastic sampling으로 작성**       |
| `src/execute.py`          | 그대로 재사용                               |
| `src/utils.py`            | A 구조 참고해서 CodeSample schema 중심으로 수정   |
| `run_best_of_n.py`        | 사용하지 않고 `run_code_best_of_n.py` 새로 작성 |
| `analyze_coverage.py`     | `analyze_code_coverage.py`로 별도 작성     |



## 4. 실행

```bash
PYTHONPATH=. python -m scripts.run_code_best_of_n --config configs/qwen25_coder_3b.yaml
PYTHONPATH=. python -m scripts.analyze_code_coverage
```
