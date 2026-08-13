## Diagnostic Benchmark

본 연구에서는 소형 코드 모델의 Planning 능력을 진단하기 위해
LiveCodeBench-v6와 Codeforces 데이터를 사용하여 별도의 Diagnostic Benchmark를 구성하였다.

## 1. LiveCodeBench-v6

- **Source**: [LiveCodeBench-v6](https://huggingface.co/datasets/livecodebench/code_generation_lite)
- **수집 기간**: 2023-03 ~ 2025-04
- **전체 데이터 수**: 1,055
- **수집 플랫폼**: Codeforces, AtCoder, LeetCode
- **평가 방식**: `stdin`, `functional`

### Sampling 전략 및 기준

- LiveCodeBench-v6 전체 데이터에서 **Codeforces 플랫폼(9 problems)을 제외**하고,
AtCoder와 LeetCode 문제를 대상으로 Diagnostic Benchmark를 구성하였다.
- 문제의 `difficulty`를 기준으로 Easy / Medium / Hard 각 100개씩을
선정하였으며, 각 난이도에서는 가장 최근에 출제된 문제부터 우선적으로
선택하였다.
- 최종 Diagnostic Benchmark는 평가 방식에 따라 두 개의 데이터셋으로
분리하여 저장하였다.

| Evaluation Type | Platform | Samples |
|---|---|---:|
| `stdin` | AtCoder | 300 |
| `functional` | LeetCode | 300 |
| **Total** | | **600** |

> **Note:** LiveCodeBench-v6의 데이터 구성상 AtCoder 문제는 `stdin`,
> LeetCode 문제는 `functional` 평가 방식을 사용한다. 따라서 본 연구에서는
> `stdin`과 `functional`을 독립적인 실험 변수로 해석하지 않고,
> 플랫폼에 따른 평가 방식의 차이로 기록한다.

## 2. Codeforces

Codeforces는 LiveCodeBench와 다른 특성의 경쟁 프로그래밍 문제를 추가하여 Planning 진단 범위를 확장하기 위한 목적으로 사용하였다.

### ISSUE
- Codeforces 데이터셋에는 DeepSeek-R1 기반 생성 테스트(generated_tests) 및 평가기(generated_checker)가 존재한다.   
- 본 연구에서는 다른 LLM의 생성 결과에 의존하지 않는 평가를 위해 generated_checker에 의존하지 않고 공식 테스트 케이스(official_tests)를 중심으로 평가하고자 필터링을 진행한다.
- Codeforces test split에는 468개의 문제가 존재했으나,공식 테스트 케이스의 완전성이 존재하는 문제는 20개로 필터링되었고, train split의 문제도 추가 진행했다.

### Filtering
- Codeforces 데이터셋의 official_tests_complete 필드를 사용하여 공식 테스트 케이스가 완전한 문제만 우선적으로 선별하였다.
- 또한 최근 문제를 중심으로 구성하기 위해 2023–2024년에 출제된 문제를 대상으로 추가 필터링하였다.
- 최종적으로 train 78개, test 20개 문제를 필터링하여 Diagnostic Benchmark에 포함하였다.


## 3. Final Diagnostic Benchmark
| Dataset          | # Problems | Main Purpose           |
| ------------------------------ | --------------------- | ------: |
| `codeforces/`                  | Codeforces Diagnostic |      98 |
| `livecodebench_v6/stdin/`      | AtCoder / stdin       |     300 |
| `livecodebench_v6/functional/` | LeetCode / functional |     300 |
| **Total**                      |                       | **698** |

### 저장 구조

```text
/mnt/hdd/project_sLM_planning/data/
│
├── codeforces/
│   ├── data-00000-of-00001.arrow
│   ├── dataset_info.json
│   └── state.json
│
└── livecodebench_v6/
    │
    ├── stdin/
    │   ├── data-00000-of-00001.arrow
    │   ├── dataset_info.json
    │   └── state.json
    │
    └── functional/
        ├── data-00000-of-00001.arrow
        ├── dataset_info.json
        └── state.json
```