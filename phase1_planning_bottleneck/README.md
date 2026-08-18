# Phase1. Initial Analysis of the Planning Bottleneck

refactoring : 2026-08-14

## 1. Research Objective :

소형 언어 모델(Small Language Model)은 제한된 모델 용량으로 인해 대형 언어 모델(Large Language Model)에 비해 복잡한 문제를 분석하여 적절한 해결 전략을 구성하는 능력이 제한적일 수 있다. 
본 연구에서는 이러한 관점에서 소형 언어 모델의 코드 문제 해결 과정을 계획 생성(Planning Generation)​과 코드 구현(Code Implementation)​의 두 단계로 구분하고, 성능 저하의 주요 병목이 어디에서 발생하는지를 분석한다.

본 단계(Phase 1)에서는 계획 생성 능력(Planning Capability)이 소형 언어 모델의 문제 해결 성능을 제한하는 주요 요인인지를 검증한다. 
동일한 3B급 모델에 대해 Direct, Self-Plan, Teacher-Plan의 세 가지 추론 전략을 비교한다.
Direct는 계획 없이 코드를 직접 생성하고, Self-Plan은 모델이 스스로 생성한 계획을 기반으로 코드를 작성하며, Teacher-Plan은 외부의 고품질 계획을 입력으로 제공받아 코드를 생성한다.

소형 언어 모델의 낮은 문제 해결 성능이 단순히 코드를 구현하지 못하기 때문인지, 또는 구현 가능한 해결 방법을 스스로 계획하지 못하는 것과 관련되는지를 분석한다.

핵심 질문은 다음과 같다.

> Research Question :   
> 3B급 소형 언어 모델의 코드 문제 해결에서 알고리즘 계획 생성 능력의 부족이 주요 성능 병목으로 작용하는가?

## 2. Experimental Setup :

### 2.1. Base-Model: 
- Code Model: ```Qwen/Qwen2.5-Coder-3B-Instruct```
    - maximum Code Generation Length: 1024 tokens
- Plan Model: ```Qwen/Qwen2.5-Coder-3B-Instruct```
    - maximum Plan Generation Length: 384 tokens
- Teacher Model: ```Claude Opus 5```

### Additional Models(sLM) :
- ```Qwen/Qwen2.5-3B-Instruct```
- ```microsoft/Phi-3.5-mini-instruct```

### 2.2. Dataset: 
진단 실험을 위해, 기존 벤치마크 데이터셋을 실험 목적에 맞게 전처리하였다. 
자세한 내용은 `phase0_diagnostic_benchmark` 폴더의 README에 작성하였다.

- Source Dataset: LiveCodeBench v6
- Diagnositc Dataset : livecodebench-v6-stdin.jsonl
- Samples : 300
- Difficulty Distribution
    - Easy: 100
    - Medium: 100
    - Hard: 100

### 2.3. Comparison Strategies :
1. Direct : 문제를 입력받아 계획 없이 바로 코드를 생성한다.
2. Self-Plan : 모델이 스스로 해결 계획(Plan)을 생성한 뒤, 해당 계획을 기반으로 코드를 생성한다.
3. Teacher-Plan : 강한 모델(claude-opus-5 사용)이 생성한 계획을 입력으로 제공하고, 소형 언어 모델은 해당 계획을 기반으로 코드만 구현한다.

    <details>
    <summary><b> Strategies 상세 내용</b></summary>

    동일한 모델을 대상으로 다음 세 가지 설정을 비교한다.

    1. Direct : ``` Problem → Code Generation ```    
    문제를 입력받아 계획 없이 바로 코드를 생성한다.   


    2. Self-Plan : ``` Problem → Plan Generation → Plan-based Code Generation ```    
    모델이 스스로 해결 계획(Plan)을 생성한 뒤, 해당 계획을 기반으로 코드를 생성한다.   
    총 두 번의 독립적인 모델 호출을 수행한다.   
    - call-1 계획 생성 : ```Problem → Plan Generation```
    - call-2 계획 기반 코드 생성 : ```Problem + Generated Plan → Code Generation```

    3. Teacher-Plan : ```Problem →  Teacher Plan (offline) → Code Generation```    
    강한 Teacher 모델(claude-opus-5 사용)이 생성한 계획을 입력으로 제공하고, 소형 언어 모델은 해당 계획을 기반으로 코드만 구현한다.

    - Teacher plan은 실험 전에 미리 생성하여 고정하며, 실험 중에는 Teacher 모델을 다시 호출하지 않고 저장된 계획만을 사용.

    </details>

### 2.4. Evaluation Protocol :

본 연구에서는 모델(sLM)이 생성한 Python 코드를 LiveCodeBench의 unit test를 이용하여 실행 기반으로 평가한다.

LiveCodeBench의 기본 평가 방식은 하나의 test case에서 실패가 발생하면 이후 test case의 실행을 조기에 종료할 수 있다. 
그러나 본 연구에서는 최종 정답 여부뿐만 아니라 실패한 생성물이 전체 test case 중 어느 정도를 통과하는지 분석하기 위해 모든 test case를 독립적으로 실행하는 exhaustive evaluation 방식을 사용한다.

문제 단위의 최종 성공 판정 기준은 LiveCodeBench와 동일하게 엄격하게 유지한다. 
즉, 주어진 모든 test case를 통과한 경우에만 해당 문제를 PASS로 판정하며, 하나 이상의 test case에서 실패한 경우 해당 문제는 실패로 판정한다.

각 test case의 실행 결과는 LiveCodeBench evaluator가 반환하는 result code를 다음과 같이 local status로 변환하여 기록한다.

| LiveCodeBench Result | Local Status | Description |
|---|---|---|
| `True` | `PASS` | 해당 test case를 정상적으로 통과 |
| `False` 또는 `-2` | `WRONG_ANSWER` | 프로그램 실행은 완료되었으나 출력이 정답과 일치하지 않음 |
| `-3` | `TIME_LIMIT_EXCEEDED` | 해당 test case의 실행이 제한 시간 내에 완료되지 않음 |
| `-4` | `RUNTIME_ERROR` | 해당 test case 실행 중 예외 등의 runtime error가 발생하여 정상적으로 종료되지 않음 |

각 문제에 대해서는 다음 정보를 기록한다.

- `passed_tests`: 통과한 test case의 수
- `total_tests`: 전체 test case의 수
- `test_pass_ratio`: `passed_tests / total_tests`
- `passed`: 모든 test case를 통과했는지 여부
- `status`: 문제의 최종 평가 상태


### 2.5. Evaluation Metrics

- **Solved Problems**: 모든 unit test를 통과한 문제의 수.
- **Pass Rate**: 전체 문제 중 모든 unit test를 통과한 문제의 비율.
- **Mean Test Pass Ratio**: 각 문제에서 전체 unit test 중 통과한 test case의 비율을 계산한 후, 이를 전체 문제에 대해 평균한 값.

따라서 `Pass Rate`는 문제 단위의 완전한 해결 여부를 측정하며, `Mean Test Pass Ratio`는 완전히 해결하지 못한 문제에서의 부분적인 test 통과 정도까지 반영한다.


## 3. Overall Results

| Model                     | Strategy     | Solved Problems¹ | Pass Rate² | Mean Test Pass Ratio³ |
| ------------------------- | ------------ | ---------------: | ---------: | --------------------: |
| Qwen2.5-Coder-3B-Instruct | Direct       |         69 / 300 |     23.00% |                 0.387 |
|                           | Self-Plan    |         49 / 300 |     16.33% |                 0.328 |
|                           | Teacher-Plan |         85 / 300 |     28.33% |                 0.416 |
| Qwen2.5-3B-Instruct       | Direct       |         39 / 300 |     13.00% |                 0.305 |
|                           | Self-Plan    |         32 / 300 |     10.67% |                 0.274 |
|                           | Teacher-Plan |         67 / 300 |     22.33% |                 0.342 |
| Phi-3-mini-Instruct       | Direct       |         46 / 300 |     15.33% |                 0.327 |
|                           | Self-Plan    |         48 / 300 |     16.00% |                 0.331 |
|                           | Teacher-Plan |         85 / 300 |     28.33% |                 0.407 |



- Solved Problems¹ : 모든 unit test를 통과한 문제 수 / 전체 문제 수.
    
- Pass Rate² : 전체 문제 중 모든 unit test를 통과한 문제의 비율(Polved Problems의 비율 표기).

- Mean Test Pass Ratio³ : 각 문제별 unit tests를 통과한 test case의 비율을 계산한 후 전체 문제(300개)에 대해 평균한 값. 


### planning gap table

| Model                     | Direct | Self-Plan | Teacher-Plan | Self − Direct | Teacher − Direct | **Teacher − Self** |
| ------------------------- | -----: | --------: | -----------: | ------------: | ---------------: | -----------------: |
| Qwen2.5-Coder-3B-Instruct |  23.0% |     16.3% |    **28.3%** |        −6.7%p |           +5.3%p |        **+12.0%p** |
| Qwen2.5-3B-Instruct       |  13.0% |     10.7% |    **22.3%** |        −2.3%p |           +9.3%p |        **+11.7%p** |
| Phi-3-mini-Instruct       |  15.3% |     16.0% |    **28.3%** |        +0.7%p |          +13.0%p |        **+12.3%p** |


## 4. Performance by Problem Difficulty

(Qwen2.5-Coder-3B-Instruct)

| Difficulty  | Problems |   Direct | Self-Plan | Teacher-Plan |
| ----------- | -------: | -------: | --------: | -----------: |
| Easy        |      100 |    56.0% |     46.0% |    **70.0%** |
| Medium      |      100 |     6.0% |      6.0% |    **16.0%** |
| Hard        |      100 | **3.0%** |      2.0% |         1.0% |
| **Overall** |      300 |    21.7% |     18.0% |    **29.0%** |


<details>
<summary><b> Difficulty 상세 내용</b></summary>


| Model    | Strategy |    Easy |  Medium | Hard |
| -------- | -------- | ------: | ------: | ---: |
| Coder-3B | Direct   |     57% |      8% |   4% |
|          | Self     |     42% |      4% |   3% |
|          | Teacher  | **69%** | **14%** |   2% |
| Qwen-3B  | Direct   |     36% |      3% |   0% |
|          | Self     |     31% |      1% |   0% |
|          | Teacher  | **58%** |  **8%** |   1% |
| Phi-3    | Direct   |     38% |      6% |   2% |
|          | Self     |     42% |      3% |   3% |
|          | Teacher  | **70%** | **14%** |   1% |


</details>

## 5. Phase 1 Findings
300개의 Diagnositc Dataset(LiveCodeBench-v6-based) 문제를 대상으로 수행한 실험에서 다음과 같은 주요 결과를 확인하였다.

1. Self-Plan은 Direct 대비 성능을 향상시키지 못했다.   
모델이 스스로 계획을 생성한 후 코드를 작성하도록 했을 때 더 많은 추론 비용이 발생했지만, 최종 성공률과 평균 test pass ratio는 오히려 감소하였다.

2. 고품질 계획을 제공하면 동일한 3B 모델의 코드 생성 성능이 크게 향상되었다.   
Teacher-Plan은 29.0%의 성공률을 기록하여 Direct의 21.7%와 Self-Plan의 18.0%를 크게 상회하였다.

3. Planning의 효과는 문제 난이도에 따라 다르게 나타났다.   
Teacher-Plan은 Easy와 Medium 문제에서 큰 성능 향상을 보였으나, Hard 문제에서는 고품질 계획을 제공한 이후에도 성공률이 1.0%에 머물렀다.


이러한 결과를 종합하면, 본 실험에서 사용한 3B 코드 모델은 적절한 해결 계획이 주어졌을 때 추가적인 문제를 해결할 수 있는 잠재력을 가지고 있지만, 그러한 계획을 스스로 생성하는 과정에서 성능 병목이 발생하는 것으로 관찰되었다. 
따라서 Planning Generation은 소형 코드 모델의 중요한 성능 병목 중 하나로 볼 수 있다.

동시에 Hard 문제에서 Teacher-Planning의 효과가 제한적이었다는 결과는 planning의 개선만으로 모든 문제 해결 실패를 설명하거나 해결할 수 없음을 보여준다. 
이는 높은 난이도의 문제에서는 Planning Generation뿐만 아니라 Code Implementation 및 모델 자체의 problem-solving capability와 관련된 추가적인 병목을 함께 고려해야 함을 시사한다.


## 6. Phase 1 Conclusion

단순히 소형 모델에게 더 많은 reasoning step을 부여하는 것보다 스스로 정확한 알고리즘 계획을 생성할 수 있는 Planning Capability를 향상시키는 것이 중요한 연구 방향이 될 수 있다.

Phase 1에서는 이러한 planning-generation bottleneck의 존재 가능성을 확인하였으며, 이후 단계에서는 이를 기반으로 소형 코드 모델의 planning 및 re-planning 능력을 보다 직접적으로 분석하고 개선하는 방향으로 연구를 확장한다.


<details>
<summary><b>실험 환경 저장 경로 </b></summary>

코드/설정은 로컬 저장소에, 실험 산출물(대용량)과 teacher plan 데이터는 `/mnt/hdd`에 분리 저장한다.

### 1) 코드 저장소 (로컬)

`~/workspace/project_sLM_planning/phase1_planning_bottleneck/`

```
phase1_planning_bottleneck/

```

### 2) 데이터,실험 산출물 저장 위치 (HDD)

`/mnt/hdd/project_sLM_planning/`

```
/mnt/hdd/project_sLM_planning/

```

### 3) 산출물 스키마

</details>