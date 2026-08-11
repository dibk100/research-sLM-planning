# Phase3. Planning Coverage and Selection Analysis

phase1,2에서 self-plan과 teacher-plan 사이에 gap을 발견했다. phase3에서 그 gap의 원인을 조금 더 세분화하여 무엇을 학습시켜야 하는지 찾아보고자 한다.

## 1. Research Objective

3B 모델의 낮은 self-planning 성능이 high-quality plan을 생성하지 못하기 때문인지, 아니면 생성 가능한 좋은 plan을 식별하지 못하기 때문인지 분석한다.

## 2. Experimental Setup

- Model: `Qwen2.5-Coder-3B-Instruct`
- Dataset: Phase 1과 동일한 LiveCodeBench v6 stdin 500문제

## 3. Planning Coverage (Phase 3-A)

- Best-of-N Planning

동일 문제에 대해 self-plan을 N개 독립 생성하도록 한다.
N = 1, 2, 4, 8로 증가시키면서 Oracle Pass@N을 측정하여 확인한다.

```
Problem
 ├── Plan 1 → Code 1 → Test
 ├── Plan 2 → Code 2 → Test
 ├── Plan 3 → Code 3 → Test
 ...
 └── Plan N → Code N → Test
```

### Case A : Best-of-N 상승
3B 모델이 좋은 plan을 생성할 가능성은 존재한다고 볼 수 있다.
병목은 plan selection / self-evaluation 문제로 봐야 한다.

### Case B : Best-of-N 변화 없음
3B 모델로 여러 번 생성시켜도 좋은 plan 자체가 거의 나오지 않는 상황으로 볼 수 있다.
planning generation capability 자체가 주요 병목이라는 Phase 1의 해석을 강하게 주장할 수 있다.

## 4. Sampling Control (Phase 3-B, Code Best-of-N)

Self-Plan을 하나만 생성한 뒤, 동일 plan에서 code를 N번 생성시켜서 비교하는 실험.
`configs`에서 `generation.code.temperature > 0`으로 두고 plan을 고정하면 같은 코드로 수행할 수 있다. (아직 별도 러너는 만들지 않았다.)

---

## 5. 구현 구조

```
phase3_planning_coverage/
├── README.md
├── requirements.txt
├── configs/
│   └── qwen25_coder_3b.yaml    
├── src/
│   ├── common/                        # Phase 1에서 검증된 인프라 (수정 금지)
│   │   ├── schemas.py
│   │   ├── datasets/dataset_loader.py
│   │   ├── models/generator.py
│   │   ├── execution/{code_extractor,evaluator}.py
│   │   └── utils/{config,seed,jsonl_logger,record_builder,run_metadata}.py
│   ├── prompts.py                     # Self-Planning 프롬프트 구성 (Phase 1 prompts/ 를 직접 읽음)
│   ├── generate_plans.py              # ★ plan sampling (Phase 3의 핵심 변경)
│   ├── generate_code.py               # plan 기반 코드 생성 (greedy)
│   ├── execute.py                     # 코드 추출 + 테스트 실행
│   └── utils.py                       # candidate 스키마 / Oracle@k / manifest 검증
├── scripts/
│   ├── run_best_of_n.py               # 실험 러너
│   ├── sanity_check.py                # full run 전 검증
│   └── freeze_problem_ids.py          # 500 problem ID manifest 생성/검증
└── .../
    
```

`src/common/`은 Phase 1 `src/`를 그대로 복사한 것이며, import 경로(`from src.` → `from src.common.`) 외에는 한 글자도 다르지 않다. 다음 명령으로 언제든 확인할 수 있다.

```bash
cd phase3_planning_coverage
for f in schemas.py datasets/dataset_loader.py models/generator.py \
         execution/code_extractor.py execution/evaluator.py \
         utils/config.py utils/jsonl_logger.py utils/record_builder.py \
         utils/run_metadata.py utils/seed.py; do
  diff <(sed 's/^from src\.common\./from src./' src/common/$f) \
       ../phase1_planning_bottleneck/src/$f && echo "same: $f"
done
```

프롬프트 템플릿은 Phase 3에 복사본을 두지 않는다. `src/prompts.py`가 `phase1_planning_bottleneck/prompts/`에서 직접 읽는다. 복사본을 두면 한쪽만 수정됐을 때 Phase 1 ↔ Phase 3 비교가 조용히 깨지기 때문이다.

- config의 `plan_prompt_path` / `code_prompt_path`에 상대 경로를 주면 **파일명만** 취해 Phase 1 prompts 디렉터리에서 찾는다.
- 절대 경로를 주면 그대로 사용한다. (Phase 3 전용 프롬프트를 실험할 때)
- 경로는 `src/prompts.py` 위치를 기준으로 해석하므로 실행 위치(cwd)와 무관하다.
- 실행 시 실제 사용된 프롬프트 경로가 `[Prompt] plan : ...` 로 로그에 남는다.

## 6. Phase 1과 동일하게 유지하는 것

| 항목 | 값 / 출처 |
| --- | --- |
| LiveCodeBench 500 problem IDs | `data/livecodebench_500.jsonl` (실행 시 대조) |
| Model checkpoint | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| chat template | `src/common/models/generator.py::build_chat_prompt` (Phase 1 코드 그대로) |
| system prompt | Phase 1 `configs/self_plan.yaml`과 동일 문자열 |
| Self-Planning plan prompt | `phase1_planning_bottleneck/prompts/self_plan_plan.txt` (같은 파일을 직접 읽음) |
| code generation prompt | `phase1_planning_bottleneck/prompts/self_plan_code.txt` (같은 파일을 직접 읽음) |
| plan max_new_tokens | 384 |
| code max_new_tokens | 1024 |
| execution environment | subprocess 격리, `sys.executable` |
| timeout | 5.0s |
| test evaluator | `src/common/execution/evaluator.py` (public + private) |
| code parsing | `src/common/execution/code_extractor.py` |

## 7. Phase 1과 달라지는 것 (핵심)

Phase 1:

```python
plan = generate_plan(problem)          # temperature=0.0, greedy
code = generate_code(problem, plan)
result = execute(code)
```

Phase 3-A:

```python
for sample_idx in range(8):
    plan = generate_plan(problem, do_sample=True)   # temperature=0.8, top_p=0.95
    code = generate_code(problem, plan)             # greedy (Phase 1과 동일)
    result = execute(code)
```

바뀌는 것은 **plan sampling 뿐**이다. code 생성은 greedy로 두어 candidate 간 성능 차이를 plan의 차이로 귀속시킨다.

### candidate 순서는 고정된 sampling sequence

`candidate[:k]` prefix가 곧 "N=k로 실험했을 때의 결과"가 되어야 compute scaling curve로 해석할 수 있다. 이를 위해 각 candidate의 seed를 `sha256(base_seed | problem_id | sample_id)`에서 유도하고 생성 직전에 torch RNG에 심는다.

- N=1, 2, 4, 8을 네 번 돌릴 필요가 없다. N=8까지 한 번만 생성한다.
- 중단 후 resume해도 같은 candidate가 재현된다.
- Python 내장 `hash()`는 프로세스마다 salt가 달라 쓰지 않는다.

```
candidate 0        → Oracle@1
candidate 0..1     → Oracle@2
candidate 0..3     → Oracle@4
candidate 0..7     → Oracle@8
```

## 8. 결과 저장 스키마

문제 하나가 JSONL 한 줄이고, candidate 단위로 저장한다.

```json
{
  "problem_id": "1873_A",
  "difficulty": "medium",
  "num_samples": 8,
  "candidates": [
    {
      "sample_id": 0,
      "sample_seed": 1226338148,
      "plan": "...",
      "code": "...",
      "passed": false,
      "test_pass_ratio": 0.42,
      "status": "WRONG_ANSWER",
      "passed_tests": 3,
      "total_tests": 7,
      "plan_in_code_prompt": true
    },
    {
      "sample_id": 1,
      "plan": "...",
      "code": "...",
      "passed": true,
      "test_pass_ratio": 1.0
    }
  ],
  "any_passed": true,
  "num_passed": 1,
  "best_test_pass_ratio": 1.0
}
```

용량 주의: Phase 1의 `self_plan_500_stdin/results.jsonl`은 `test_results`(테스트 입출력 원문)를 모두 저장해 8.5GB였다. candidate가 8배이므로 full run에서는 `output.store_test_results: false`, `output.store_prompts: false`가 기본값이다. `passed_tests / total_tests`만으로 `test_pass_ratio` 분석에는 충분하다.

결과 JSONL은 루트 파일시스템 용량 때문에 `/mnt/hdd/project_sLM_planning/output_phase3/` 아래에 쓴다. 저장소 안의 `outputs/`에는 분석 CSV만 둔다.

## 9. 실행 순서

환경: `/mnt/hdd/conda_envs/slm`, 프로젝트 루트에서 `PYTHONPATH=.`

### 0) 문제 ID manifest 검증 (선택)

`data/livecodebench_500.jsonl`은 Phase 1의 `self_plan_500_stdin` 실행 순서에서 추출해 이미 고정해 두었다. 현재 datasets 스냅샷이 그때와 같은지 확인하려면:

```bash
cd project_sLM_planning/phase3_planning_coverage
PYTHONPATH=. python -m scripts.freeze_problem_ids --verify
```

### 1) Sanity check (20문제)

```bash
PYTHONPATH=. python -m scripts.run_best_of_n \
  --config configs/qwen25_coder_3b_sanity.yaml

PYTHONPATH=. python -m scripts.sanity_check \
  --results /mnt/hdd/project_sLM_planning/output_phase3/qwen25_coder_3b/sanity_20/results.jsonl
```

`sanity_check.py`가 확인하는 것:

1. 8개의 plan이 실제로 서로 다르게 sampling 되는가
2. 각 plan이 해당 candidate의 code 프롬프트에 정확히 들어갔는가 (플래그 + 저장된 프롬프트 원문 재확인)
3. candidate별 결과가 누락 없이 저장되는가 (sample_id 0..N-1, seed 재현, `test_pass_ratio` 일관성)
4. Oracle@1 ≤ Oracle@2 ≤ Oracle@4 ≤ Oracle@8 이 전체·문제 단위 모두에서 성립하는가
5. 문제 ID가 Phase 1 manifest와 동일한가

**4번이 깨지면 분석 또는 저장 구현에 버그가 있는 것이다.** prefix 정의상 반드시 성립해야 한다.

### 2) Full experiment (500문제)

```bash
PYTHONPATH=. python -m scripts.run_best_of_n \
  --config configs/qwen25_coder_3b.yaml
```

문제당 생성 호출이 16회(plan 8 + code 8)라 Phase 1 self-plan 대비 약 8배 시간이 든다. `output.resume: true`이므로 중단해도 문제 단위로 이어서 실행된다.

### 3) 분석

```bash
PYTHONPATH=. python -m scripts.analyze_coverage \
  --results /mnt/hdd/project_sLM_planning/output_phase3/qwen25_coder_3b/best_of_8/results.jsonl \
  --output-dir outputs/qwen25_coder_3b/best_of_8 \
  --phase1-pass-rate 0.168 \
  --fail-on-violation
```

생성되는 CSV:

| 파일 | 내용 |
| --- | --- |
| `overall_coverage.csv` | k별 Oracle@k, unbiased pass@k, mean best test_pass_ratio |
| `difficulty_coverage.csv` | 난이도별 Oracle@k |
| `per_candidate.csv` | sample_id별 독립 pass rate (sampling 편향 확인) |
| `problem_coverage.csv` | 문제 단위 상세 (첫 성공 sample_id 포함) |
| `plan_diversity.csv` | distinct plan 수, pairwise Jaccard |
| `status_counts.csv` | candidate status 분포 |

### 지표 해석

- `oracle_at_k` — prefix 기반 관측값. 실제 compute scaling curve.
- `unbiased_pass_at_k` — n개 표본 전체를 쓰는 Codex(Chen et al., 2021) 추정량. 표본 잡음이 작고 정의상 단조 증가하므로 보조 지표로 함께 본다.
- `avg@1` — candidate 하나당 평균 pass rate. Phase 1 self-plan(16.8%, greedy)과 직접 비교하는 값이다. Phase 3-A의 candidate는 모두 temperature 0.8 sampling이므로 Oracle@1은 Phase 1의 greedy 결과와 같은 값이 아니다. 8개 candidate를 같은 분포에서 뽑아야 prefix가 i.i.d. scaling curve로 해석되기 때문에, 첫 candidate만 greedy로 두지 않았다.
