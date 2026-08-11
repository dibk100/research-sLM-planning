"""
Phase 3-A: Planning Coverage 분석.

results.jsonl(문제당 candidate N개)을 읽어 candidate prefix로
Oracle@k (k = 1, 2, 4, 8)을 계산한다.  N=1,2,4,8을 따로 실행할 필요가 없다.

    candidate[:1] -> Oracle@1
    candidate[:2] -> Oracle@2
    candidate[:4] -> Oracle@4
    candidate[:8] -> Oracle@8

candidate 순서가 고정된 sampling sequence이므로 prefix가 곧 compute scaling이다.

보고 지표:
- oracle_at_k              : prefix 기반 관측값 (compute scaling curve)
- unbiased_pass_at_k       : n개 표본 전체를 쓰는 Codex 방식 추정량 (표본 잡음 보정)
- mean_best_test_pass_ratio: 부분 점수 관점의 coverage
- avg_at_1                 : candidate 하나당 평균 pass rate (Phase 1 self-plan과 직접 비교)

Usage:

PYTHONPATH=. python -m scripts.analyze_coverage \
  --results /mnt/hdd/project_sLM_planning/output_phase3/qwen25_coder_3b/best_of_8/results.jsonl \
  --phase1-pass-rate 0.168 \
  --teacher-pass-rate 0.340

"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.utils import (
    best_ratio_at_k,
    check_monotonicity,
    mean,
    oracle_at_k,
    prefix_ks,
    unbiased_pass_at_k,
)

DIFFICULTY_ORDER = ("easy", "medium", "hard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze planning coverage (Oracle@k) from "
            "best-of-N results."
        )
    )

    parser.add_argument(
        "--results",
        required=True,
        help="Path to best-of-N results.jsonl.",
    )

    parser.add_argument(
        "--output-dir",
        default='./archive/analysis',
        help=(
            "Directory for analysis CSV files. "
            "Defaults to <results dir>/analysis."
        ),
    )


    parser.add_argument(
        "--phase1-pass-rate",
        type=float,
        default=None,
        help=(
            "Phase 1 self-plan pass rate for reference "
            "(e.g. 0.168)."
        ),
    )

    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help=(
            "Exit with a non-zero code if Oracle@k "
            "monotonicity is violated."
        ),
    )
    
    parser.add_argument(
        "--teacher-pass-rate",
        type=float,
        default=None,
        help=(
            "Phase 1 Teacher-Plan pass rate for reference "
            "(e.g. 0.340)."
        ),
    )

    return parser.parse_args()


def load_results(
    results_path: Path,
) -> list[dict[str, Any]]:
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}"
        )

    records: list[dict[str, Any]] = []

    with results_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL line {line_number}: "
                    f"{results_path}"
                ) from error

            records.append(record)

    if not records:
        raise ValueError(
            f"No records found: {results_path}"
        )

    return records


def resolve_num_samples(
    records: list[dict[str, Any]],
) -> int:
    """모든 문제가 동일한 candidate 수를 가지는지 확인한다."""
    counts = {
        len(record["candidates"])
        for record in records
    }

    if len(counts) != 1:
        raise ValueError(
            f"Inconsistent candidate counts across "
            f"problems: {sorted(counts)}. "
            f"Run scripts/sanity_check.py to inspect."
        )

    num_samples = counts.pop()

    if num_samples <= 0:
        raise ValueError(
            "Records contain no candidates."
        )

    return num_samples

def validate_records(
    records: list[dict[str, Any]],
    num_samples: int,
) -> None:
    """Phase 3-A 결과의 기본 무결성을 검증한다."""
    expected_sample_ids = list(range(num_samples))
    seen_problem_ids: set[str] = set()

    for record in records:
        problem_id = record["problem_id"]

        if problem_id in seen_problem_ids:
            raise ValueError(
                f"Duplicate problem_id: {problem_id}"
            )

        seen_problem_ids.add(problem_id)

        candidates = record["candidates"]

        sample_ids = [
            int(candidate["sample_id"])
            for candidate in candidates
        ]

        if sample_ids != expected_sample_ids:
            raise ValueError(
                f"{problem_id}: invalid candidate sequence. "
                f"expected={expected_sample_ids}, "
                f"got={sample_ids}"
            )

        for candidate in candidates:
            ratio = float(candidate["test_pass_ratio"])

            if not 0.0 <= ratio <= 1.0:
                raise ValueError(
                    f"{problem_id}: invalid test_pass_ratio="
                    f"{ratio}"
                )


def coverage_rows(
    records: list[dict[str, Any]],
    ks: list[int],
    num_samples: int,
) -> list[dict[str, Any]]:
    """전체 문제에 대한 k별 coverage 지표를 계산한다."""
    rows: list[dict[str, Any]] = []

    for k in ks:
        oracle_flags = [
            oracle_at_k(record["candidates"], k)
            for record in records
        ]

        best_ratios = [
            best_ratio_at_k(record["candidates"], k)
            for record in records
        ]

        unbiased = [
            unbiased_pass_at_k(
                num_samples=num_samples,
                num_correct=sum(
                    1
                    for candidate in record["candidates"]
                    if candidate["passed"]
                ),
                k=k,
            )
            for record in records
        ]

        rows.append(
            {
                "k": k,
                "num_problems": len(records),
                "oracle_solved": sum(oracle_flags),
                "oracle_at_k": mean(
                    float(flag) for flag in oracle_flags
                ),
                "unbiased_pass_at_k": mean(unbiased),
                "mean_best_test_pass_ratio": mean(
                    best_ratios
                ),
            }
        )

    return rows


def per_candidate_rows(
    records: list[dict[str, Any]],
    num_samples: int,
) -> list[dict[str, Any]]:
    """sample_id별 독립 성능 (sampling 편향 확인용)."""
    rows: list[dict[str, Any]] = []

    for sample_id in range(num_samples):
        candidates = [
            record["candidates"][sample_id]
            for record in records
        ]

        rows.append(
            {
                "sample_id": sample_id,
                "num_problems": len(candidates),
                "solved": sum(
                    1
                    for candidate in candidates
                    if candidate["passed"]
                ),
                "pass_rate": mean(
                    float(candidate["passed"])
                    for candidate in candidates
                ),
                "mean_test_pass_ratio": mean(
                    float(candidate["test_pass_ratio"])
                    for candidate in candidates
                ),
            }
        )

    return rows


def difficulty_rows(
    records: list[dict[str, Any]],
    ks: list[int],
    num_samples: int,
) -> list[dict[str, Any]]:
    """난이도별 coverage 지표."""
    grouped: dict[str, list[dict[str, Any]]] = (
        defaultdict(list)
    )

    for record in records:
        grouped[record["difficulty"]].append(record)

    ordered = [
        difficulty
        for difficulty in DIFFICULTY_ORDER
        if difficulty in grouped
    ] + [
        difficulty
        for difficulty in sorted(grouped)
        if difficulty not in DIFFICULTY_ORDER
    ]

    rows: list[dict[str, Any]] = []

    for difficulty in ordered:
        for row in coverage_rows(
            grouped[difficulty],
            ks,
            num_samples,
        ):
            rows.append(
                {"difficulty": difficulty, **row}
            )

    return rows


def problem_rows(
    records: list[dict[str, Any]],
    ks: list[int],
) -> list[dict[str, Any]]:
    """문제 단위 상세 결과."""
    rows: list[dict[str, Any]] = []

    for record in records:
        candidates = record["candidates"]

        first_pass = next(
            (
                candidate["sample_id"]
                for candidate in candidates
                if candidate["passed"]
            ),
            None,
        )

        nonempty_plans = [
            candidate["plan"].strip()
            for candidate in candidates
            if candidate["plan"].strip()
        ]

        distinct_plans = len(set(nonempty_plans))
        
        empty_plans = sum(
            1
            for candidate in candidates
            if not candidate["plan"].strip()
        )

        row: dict[str, Any] = {
            "problem_id": record["problem_id"],
            "difficulty": record["difficulty"],
            "num_candidates": len(candidates),
            "num_passed": sum(
                1
                for candidate in candidates
                if candidate["passed"]
            ),
            "first_pass_sample_id": (
                "" if first_pass is None else first_pass
            ),
            "distinct_plans": distinct_plans,
        }

        for k in ks:
            row[f"oracle_at_{k}"] = int(
                oracle_at_k(candidates, k)
            )
            row[f"best_ratio_at_{k}"] = round(
                best_ratio_at_k(candidates, k),
                6,
            )

        rows.append(row)

    return rows


def plan_diversity_rows(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """plan이 실제로 서로 다르게 sampling 되는지 정량화한다."""
    rows: list[dict[str, Any]] = []

    for record in records:   
        plans = [
            candidate["plan"].strip()
            for candidate in record["candidates"]
        ]

        nonempty_plans = [
            plan for plan in plans if plan
        ]
        
        token_sets = [
            set(plan.lower().split())
            for plan in nonempty_plans
        ]

        similarities: list[float] = []

        for left in range(len(token_sets)):
            for right in range(
                left + 1,
                len(token_sets),
            ):
                union = (
                    token_sets[left] | token_sets[right]
                )

                if not union:
                    similarities.append(1.0)
                    continue

                intersection = (
                    token_sets[left] & token_sets[right]
                )

                similarities.append(
                    len(intersection) / len(union)
                )

        rows.append(
            {
                "problem_id": record["problem_id"],
                "difficulty": record["difficulty"],
                "num_candidates": len(plans),
                "distinct_plans": len(set(nonempty_plans)),
                "empty_plans": sum(
                    1 for plan in plans if not plan
                ),
                "mean_pairwise_jaccard": round(
                    mean(similarities),
                    6,
                ),
                "max_pairwise_jaccard": round(
                    max(similarities)
                    if similarities
                    else 0.0,
                    6,
                ),
            }
        )

    return rows


def status_rows(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """candidate status 분포."""
    counter: Counter[str] = Counter()

    for record in records:
        for candidate in record["candidates"]:
            counter[candidate["status"]] += 1

    total = sum(counter.values())

    return [
        {
            "status": status,
            "count": count,
            "ratio": count / total if total else 0.0,
        }
        for status, count in counter.most_common()
    ]


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    results_path = Path(args.results)
    records = load_results(results_path)

    num_samples = resolve_num_samples(records)
    validate_records(records, num_samples)
    ks = prefix_ks(num_samples)

    output_dir = Path(
        args.output_dir
        if args.output_dir
        else Path.cwd() / "archive" / "analysis"
    )

    overall = coverage_rows(records, ks, num_samples)
    by_difficulty = difficulty_rows(
        records,
        ks,
        num_samples,
    )
    by_candidate = per_candidate_rows(
        records,
        num_samples,
    )
    by_problem = problem_rows(records, ks)
    diversity = plan_diversity_rows(records)
    statuses = status_rows(records)

    write_csv(
        output_dir / "overall_coverage.csv",
        overall,
    )
    write_csv(
        output_dir / "difficulty_coverage.csv",
        by_difficulty,
    )
    write_csv(
        output_dir / "per_candidate.csv",
        by_candidate,
    )
    write_csv(
        output_dir / "problem_coverage.csv",
        by_problem,
    )
    write_csv(
        output_dir / "plan_diversity.csv",
        diversity,
    )
    write_csv(
        output_dir / "status_counts.csv",
        statuses,
    )

    print("=" * 80)
    print("Phase3-A Planning Coverage")
    print("=" * 80)
    print(f"Results      : {results_path}")
    print(f"Problems     : {len(records)}")
    print(f"N (samples)  : {num_samples}")
    print(f"Analysis dir : {output_dir}")
    print()

    print(
        f"{'k':>3}  {'solved':>7}  {'Oracle@k':>9}  "
        f"{'unbiased':>9}  {'mean best ratio':>15}"
    )
    print("-" * 52)

    for row in overall:
        print(
            f"{row['k']:>3}  "
            f"{row['oracle_solved']:>7}  "
            f"{row['oracle_at_k']:>9.4f}  "
            f"{row['unbiased_pass_at_k']:>9.4f}  "
            f"{row['mean_best_test_pass_ratio']:>15.4f}"
        )

    mean_single_sample_pass_rate = mean(
        row["pass_rate"] for row in by_candidate
    )
    
    sampling_oracle_at_1 = overall[0]["oracle_at_k"]

    print()

    print(
        f"Sampling Oracle@1               : "
        f"{sampling_oracle_at_1:.4f}"
    )
    print(
        f"Mean single-sample pass rate    : "
        f"{mean_single_sample_pass_rate:.4f}"
    )
    
    if args.phase1_pass_rate is not None:
        print(
            f"Phase 1 self-plan pass rate      : "
            f"{args.phase1_pass_rate:.4f} "
            f"(greedy, temperature=0.0)"
        )

    if args.teacher_pass_rate is not None:
        teacher = args.teacher_pass_rate
        baseline = sampling_oracle_at_1

        print(
            f"Phase 1 Teacher-Plan            : "
            f"{teacher:.4f}"
        )

        print()
        print("Teacher Gap Closed")
        print("-" * 52)

        denominator = teacher - baseline

        if denominator <= 0:
            print(
                "Teacher gap is non-positive; "
                "gap-closed metric is not defined."
            )
        else:
            for row in overall:
                k = row["k"]

                if k == 1:
                    continue

                gap_closed = (
                    row["oracle_at_k"] - baseline
                ) / denominator

                print(
                    f"@{k:<2} : {gap_closed:.4f} "
                    f"({gap_closed * 100:.1f}%)"
                )

    print()
    print("Difficulty breakdown")
    print("-" * 52)
    print(
        f"{'difficulty':>10}  {'k':>3}  {'n':>4}  "
        f"{'solved':>7}  {'Oracle@k':>9}"
    )

    for row in by_difficulty:
        print(
            f"{row['difficulty']:>10}  "
            f"{row['k']:>3}  "
            f"{row['num_problems']:>4}  "
            f"{row['oracle_solved']:>7}  "
            f"{row['oracle_at_k']:>9.4f}"
        )

    print()
    print("Plan diversity")
    print("-" * 52)
    print(
        f"mean distinct plans / {num_samples} : "
        f"{mean(row['distinct_plans'] for row in diversity):.2f}"
    )
    print(
        f"problems with all plans identical : "
        f"{sum(1 for row in diversity if row['distinct_plans'] == 1)}"
    )
    print(
        f"mean pairwise jaccard             : "
        f"{mean(row['mean_pairwise_jaccard'] for row in diversity):.4f}"
    )

    print()
    print("Candidate status distribution")
    print("-" * 52)

    for row in statuses:
        print(
            f"{row['status']:>22}  "
            f"{row['count']:>6}  "
            f"{row['ratio']:>7.4f}"
        )

    violations = check_monotonicity(
        [row["oracle_at_k"] for row in overall]
    )

    print()

    if violations:
        print("[FAIL] Oracle@k monotonicity violated.")
        print(
            "       prefix 정의상 Oracle@1 <= Oracle@2 "
            "<= Oracle@4 <= Oracle@8 은 항상 성립해야 한다."
        )
        print("       분석 또는 저장 구현에 버그가 있다.")

        for position, previous, current in violations:
            print(
                f"       k={ks[position]}: "
                f"{previous:.4f} -> {current:.4f}"
            )

        if args.fail_on_violation:
            return 1

    else:
        print(
            "[OK] Oracle@k monotonicity holds "
            "("
            + " <= ".join(
                f"@{row['k']}={row['oracle_at_k']:.4f}"
                for row in overall
            )
            + ")."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
