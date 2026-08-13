"""
Phase 3 Planning vs Code Paired Coverage Analysis.

동일한 LiveCodeBench 문제에 대해:

Phase 3-A:
    sampled plan x N -> greedy code

Phase 3-B:
    fixed Self-Plan -> sampled code x N

를 problem-level paired comparison한다.

각 k = 1, 2, 4, 8에서:

    Both PASS
    Planning-only PASS
    Code-only PASS
    Both FAIL

을 계산하고, discordant pair를 이용해
two-sided exact McNemar test를 수행한다.

또한 Easy / Medium / Hard별 paired analysis도 수행한다.

출력:
    ./archive/analysis/
        planning_vs_code_paired.csv
        planning_vs_code_paired_difficulty.csv
        planning_vs_code_problem_pairs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DIFFICULTY_ORDER = (
    "easy",
    "medium",
    "hard",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired Planning-vs-Code "
            "Best-of-N coverage analysis."
        )
    )

    parser.add_argument(
        "--planning-results",
        required=True,
        help=(
            "Phase 3-A Planning Best-of-N "
            "results.jsonl."
        ),
    )

    parser.add_argument(
        "--code-results",
        required=True,
        help=(
            "Phase 3-B Code Best-of-N "
            "results.jsonl."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. "
            "Default: ./archive/analysis"
        ),
    )

    parser.add_argument(
        "--expected-problems",
        type=int,
        default=500,
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open(
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
                record = json.loads(
                    stripped
                )

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL: "
                    f"{path}:{line_number}"
                ) from error

            if not isinstance(
                record,
                dict,
            ):
                raise ValueError(
                    "JSONL record must be "
                    f"an object: {path}:{line_number}"
                )

            records.append(
                record
            )

    if not records:
        raise ValueError(
            f"No records found: {path}"
        )

    return records


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


def records_by_id(
    records: list[dict[str, Any]],
    *,
    source_name: str,
) -> dict[str, dict[str, Any]]:
    mapping: dict[
        str,
        dict[str, Any],
    ] = {}

    for record in records:
        problem_id = str(
            record.get(
                "problem_id",
                "",
            )
        ).strip()

        if not problem_id:
            raise ValueError(
                f"{source_name}: "
                "empty problem_id."
            )

        if problem_id in mapping:
            raise ValueError(
                f"{source_name}: "
                "duplicate problem_id="
                f"{problem_id}"
            )

        mapping[
            problem_id
        ] = record

    return mapping


def validate_pairing(
    planning_records: list[dict[str, Any]],
    code_records: list[dict[str, Any]],
    expected_problems: int,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    planning = records_by_id(
        planning_records,
        source_name="Planning",
    )

    code = records_by_id(
        code_records,
        source_name="Code",
    )

    if (
        len(planning)
        != expected_problems
    ):
        raise ValueError(
            "Planning problem count mismatch: "
            f"{len(planning)} != "
            f"{expected_problems}"
        )

    if (
        len(code)
        != expected_problems
    ):
        raise ValueError(
            "Code problem count mismatch: "
            f"{len(code)} != "
            f"{expected_problems}"
        )

    planning_ids = set(
        planning
    )

    code_ids = set(
        code
    )

    missing_in_code = sorted(
        planning_ids
        - code_ids
    )

    missing_in_planning = sorted(
        code_ids
        - planning_ids
    )

    if missing_in_code:
        raise ValueError(
            "Problems missing in Code results: "
            f"{missing_in_code[:20]}"
        )

    if missing_in_planning:
        raise ValueError(
            "Problems missing in Planning results: "
            f"{missing_in_planning[:20]}"
        )

    return planning, code


def resolve_num_samples(
    records: list[dict[str, Any]],
    *,
    source_name: str,
) -> int:
    counts = {
        len(
            record.get(
                "candidates",
                [],
            )
        )
        for record in records
    }

    if len(counts) != 1:
        raise ValueError(
            f"{source_name}: inconsistent "
            f"candidate counts: {counts}"
        )

    n = counts.pop()

    if n <= 0:
        raise ValueError(
            f"{source_name}: "
            "no candidates."
        )

    return n


def prefix_ks(
    max_k: int,
) -> list[int]:
    ks: list[int] = []

    k = 1

    while k <= max_k:
        ks.append(k)
        k *= 2

    if ks[-1] != max_k:
        ks.append(
            max_k
        )

    return ks


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def oracle_at_k(
    record: dict[str, Any],
    k: int,
) -> bool:
    candidates = record[
        "candidates"
    ]

    if k > len(candidates):
        raise ValueError(
            f"k={k} exceeds candidate "
            f"count={len(candidates)} "
            f"for {record['problem_id']}."
        )

    return any(
        bool(
            candidate.get(
                "passed",
                False,
            )
        )
        for candidate
        in candidates[:k]
    )


# ---------------------------------------------------------------------------
# Exact McNemar test
# ---------------------------------------------------------------------------


def binomial_probability(
    n: int,
    k: int,
    p: float = 0.5,
) -> float:
    return (
        math.comb(n, k)
        * (p ** k)
        * ((1.0 - p) ** (n - k))
    )


def exact_mcnemar_pvalue(
    planning_only: int,
    code_only: int,
) -> float:
    """
    Two-sided exact McNemar test.

    Under H0:
        planning_only and code_only are
        equally likely among discordant pairs.

    Equivalent to exact Binomial(n=b+c, p=0.5).
    """

    b = int(
        planning_only
    )

    c = int(
        code_only
    )

    n = b + c

    if n == 0:
        return 1.0

    observed = min(
        b,
        c,
    )

    lower_tail = sum(
        binomial_probability(
            n,
            k,
            0.5,
        )
        for k in range(
            observed + 1
        )
    )

    p_value = min(
        1.0,
        2.0 * lower_tail,
    )

    return p_value


# ---------------------------------------------------------------------------
# Paired analysis
# ---------------------------------------------------------------------------


def paired_counts(
    problem_ids: list[str],
    planning: dict[
        str,
        dict[str, Any],
    ],
    code: dict[
        str,
        dict[str, Any],
    ],
    k: int,
) -> dict[str, Any]:
    both_pass = 0
    planning_only = 0
    code_only = 0
    both_fail = 0

    planning_solved = 0
    code_solved = 0

    for problem_id in problem_ids:
        planning_pass = (
            oracle_at_k(
                planning[
                    problem_id
                ],
                k,
            )
        )

        code_pass = (
            oracle_at_k(
                code[
                    problem_id
                ],
                k,
            )
        )

        planning_solved += int(
            planning_pass
        )

        code_solved += int(
            code_pass
        )

        if (
            planning_pass
            and code_pass
        ):
            both_pass += 1

        elif (
            planning_pass
            and not code_pass
        ):
            planning_only += 1

        elif (
            not planning_pass
            and code_pass
        ):
            code_only += 1

        else:
            both_fail += 1

    n = len(
        problem_ids
    )

    p_value = (
        exact_mcnemar_pvalue(
            planning_only,
            code_only,
        )
    )

    return {
        "k": k,
        "num_problems": n,

        "planning_solved": (
            planning_solved
        ),
        "planning_coverage": (
            planning_solved / n
            if n
            else 0.0
        ),

        "code_solved": (
            code_solved
        ),
        "code_coverage": (
            code_solved / n
            if n
            else 0.0
        ),

        "coverage_difference": (
            (
                planning_solved
                - code_solved
            )
            / n
            if n
            else 0.0
        ),

        "both_pass": (
            both_pass
        ),
        "planning_only": (
            planning_only
        ),
        "code_only": (
            code_only
        ),
        "both_fail": (
            both_fail
        ),

        "discordant_pairs": (
            planning_only
            + code_only
        ),

        "mcnemar_exact_p": (
            p_value
        ),
    }


# ---------------------------------------------------------------------------
# Problem-level table
# ---------------------------------------------------------------------------


def problem_pair_rows(
    planning: dict[
        str,
        dict[str, Any],
    ],
    code: dict[
        str,
        dict[str, Any],
    ],
    problem_ids: list[str],
    ks: list[int],
) -> list[dict[str, Any]]:
    rows: list[
        dict[str, Any]
    ] = []

    for problem_id in problem_ids:
        planning_record = (
            planning[
                problem_id
            ]
        )

        code_record = (
            code[
                problem_id
            ]
        )

        planning_difficulty = str(
            planning_record.get(
                "difficulty",
                "unknown",
            )
        ).lower()

        code_difficulty = str(
            code_record.get(
                "difficulty",
                "unknown",
            )
        ).lower()

        if (
            planning_difficulty
            != code_difficulty
        ):
            raise ValueError(
                f"{problem_id}: difficulty "
                "mismatch: "
                f"planning={planning_difficulty}, "
                f"code={code_difficulty}"
            )

        row: dict[
            str,
            Any,
        ] = {
            "problem_id": (
                problem_id
            ),
            "difficulty": (
                planning_difficulty
            ),
        }

        for k in ks:
            planning_pass = (
                oracle_at_k(
                    planning_record,
                    k,
                )
            )

            code_pass = (
                oracle_at_k(
                    code_record,
                    k,
                )
            )

            if (
                planning_pass
                and code_pass
            ):
                category = (
                    "both_pass"
                )

            elif planning_pass:
                category = (
                    "planning_only"
                )

            elif code_pass:
                category = (
                    "code_only"
                )

            else:
                category = (
                    "both_fail"
                )

            row[
                f"planning_at_{k}"
            ] = int(
                planning_pass
            )

            row[
                f"code_at_{k}"
            ] = int(
                code_pass
            )

            row[
                f"pair_category_at_{k}"
            ] = category

        rows.append(
            row
        )

    return rows


# ---------------------------------------------------------------------------
# Difficulty
# ---------------------------------------------------------------------------


def difficulty_rows(
    planning: dict[
        str,
        dict[str, Any],
    ],
    code: dict[
        str,
        dict[str, Any],
    ],
    problem_ids: list[str],
    ks: list[int],
) -> list[dict[str, Any]]:
    grouped: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for problem_id in problem_ids:
        difficulty = str(
            planning[
                problem_id
            ].get(
                "difficulty",
                "unknown",
            )
        ).lower()

        grouped[
            difficulty
        ].append(
            problem_id
        )

    ordered = [
        difficulty
        for difficulty
        in DIFFICULTY_ORDER
        if difficulty in grouped
    ]

    ordered += [
        difficulty
        for difficulty
        in sorted(grouped)
        if difficulty
        not in DIFFICULTY_ORDER
    ]

    rows: list[
        dict[str, Any]
    ] = []

    for difficulty in ordered:
        ids = grouped[
            difficulty
        ]

        for k in ks:
            analysis = (
                paired_counts(
                    ids,
                    planning,
                    code,
                    k,
                )
            )

            rows.append(
                {
                    "difficulty": (
                        difficulty
                    ),
                    **analysis,
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def significance_label(
    p_value: float,
) -> str:
    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return "ns"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    planning_path = Path(
        args.planning_results
    )

    code_path = Path(
        args.code_results
    )

    planning_records = (
        load_jsonl(
            planning_path
        )
    )

    code_records = (
        load_jsonl(
            code_path
        )
    )

    planning, code = (
        validate_pairing(
            planning_records,
            code_records,
            args.expected_problems,
        )
    )

    planning_n = (
        resolve_num_samples(
            planning_records,
            source_name="Planning",
        )
    )

    code_n = (
        resolve_num_samples(
            code_records,
            source_name="Code",
        )
    )

    common_n = min(
        planning_n,
        code_n,
    )

    ks = prefix_ks(
        common_n
    )

    # Keep planning file order
    problem_ids = [
        str(
            record["problem_id"]
        )
        for record
        in planning_records
    ]

    # --------------------------------------------------------------
    # Overall
    # --------------------------------------------------------------

    overall_rows = [
        paired_counts(
            problem_ids,
            planning,
            code,
            k,
        )
        for k in ks
    ]

    # --------------------------------------------------------------
    # Difficulty
    # --------------------------------------------------------------

    by_difficulty = (
        difficulty_rows(
            planning,
            code,
            problem_ids,
            ks,
        )
    )

    # --------------------------------------------------------------
    # Per-problem
    # --------------------------------------------------------------

    problem_rows = (
        problem_pair_rows(
            planning,
            code,
            problem_ids,
            ks,
        )
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    output_dir = Path(
        args.output_dir
        if args.output_dir
        else (
            Path.cwd()
            / "archive"
            / "analysis"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        output_dir
        / "planning_vs_code_paired.csv",
        overall_rows,
    )

    write_csv(
        output_dir
        / "planning_vs_code_paired_difficulty.csv",
        by_difficulty,
    )

    write_csv(
        output_dir
        / "planning_vs_code_problem_pairs.csv",
        problem_rows,
    )

    # --------------------------------------------------------------
    # Console output
    # --------------------------------------------------------------

    print("=" * 88)
    print(
        "Phase 3 Planning vs Code "
        "Paired Coverage Analysis"
    )
    print("=" * 88)

    print(
        f"Planning results : "
        f"{planning_path}"
    )

    print(
        f"Code results     : "
        f"{code_path}"
    )

    print(
        f"Problems         : "
        f"{len(problem_ids)}"
    )

    print(
        f"Planning N       : "
        f"{planning_n}"
    )

    print(
        f"Code N           : "
        f"{code_n}"
    )

    print(
        f"Compared k       : "
        f"{ks}"
    )

    print(
        f"Analysis dir     : "
        f"{output_dir}"
    )

    # --------------------------------------------------------------
    # Overall paired analysis
    # --------------------------------------------------------------

    print()
    print(
        "Overall Paired Coverage"
    )
    print("-" * 88)

    print(
        f"{'N':>3} "
        f"{'Plan':>8} "
        f"{'Code':>8} "
        f"{'Delta':>8} "
        f"{'Both+':>7} "
        f"{'PlanOnly':>9} "
        f"{'CodeOnly':>9} "
        f"{'Both-':>7} "
        f"{'p(exact)':>12}"
    )

    for row in overall_rows:
        p_value = float(
            row[
                "mcnemar_exact_p"
            ]
        )

        label = significance_label(
            p_value
        )

        print(
            f"{row['k']:>3} "
            f"{row['planning_coverage']:>8.4f} "
            f"{row['code_coverage']:>8.4f} "
            f"{row['coverage_difference']:>+8.4f} "
            f"{row['both_pass']:>7} "
            f"{row['planning_only']:>9} "
            f"{row['code_only']:>9} "
            f"{row['both_fail']:>7} "
            f"{p_value:>10.6g} "
            f"{label}"
        )

    # --------------------------------------------------------------
    # @max interpretation
    # --------------------------------------------------------------

    max_row = (
        overall_rows[-1]
    )

    print()
    print(
        f"Paired decomposition @"
        f"{max_row['k']}"
    )
    print("-" * 88)

    print(
        f"Both PASS      : "
        f"{max_row['both_pass']}"
    )

    print(
        f"Planning only  : "
        f"{max_row['planning_only']}"
    )

    print(
        f"Code only      : "
        f"{max_row['code_only']}"
    )

    print(
        f"Both FAIL      : "
        f"{max_row['both_fail']}"
    )

    print(
        f"Discordant     : "
        f"{max_row['discordant_pairs']}"
    )

    print(
        f"McNemar exact p: "
        f"{max_row['mcnemar_exact_p']:.8g}"
    )

    # --------------------------------------------------------------
    # Difficulty
    # --------------------------------------------------------------

    print()
    print(
        "Difficulty-wise Paired Analysis"
    )
    print("-" * 88)

    print(
        f"{'Diff':>8} "
        f"{'N':>3} "
        f"{'n':>5} "
        f"{'Plan':>8} "
        f"{'Code':>8} "
        f"{'P-only':>7} "
        f"{'C-only':>7} "
        f"{'p':>12}"
    )

    for row in by_difficulty:
        p_value = float(
            row[
                "mcnemar_exact_p"
            ]
        )

        print(
            f"{row['difficulty']:>8} "
            f"{row['k']:>3} "
            f"{row['num_problems']:>5} "
            f"{row['planning_coverage']:>8.4f} "
            f"{row['code_coverage']:>8.4f} "
            f"{row['planning_only']:>7} "
            f"{row['code_only']:>7} "
            f"{p_value:>12.6g}"
        )

    # --------------------------------------------------------------
    # Integrity summary
    # --------------------------------------------------------------

    category_counts = Counter(
        row[
            f"pair_category_at_{common_n}"
        ]
        for row
        in problem_rows
    )

    print()
    print(
        "Integrity"
    )
    print("-" * 88)

    print(
        "[OK] Planning / Code problem IDs "
        "match exactly."
    )

    print(
        f"[OK] Paired problems: "
        f"{len(problem_ids)}"
    )

    print(
        f"[OK] @"
        f"{common_n} categories sum: "
        f"{sum(category_counts.values())}"
    )

    print()
    print(
        "Saved:"
    )

    print(
        "  "
        + str(
            output_dir
            / "planning_vs_code_paired.csv"
        )
    )

    print(
        "  "
        + str(
            output_dir
            / "planning_vs_code_paired_difficulty.csv"
        )
    )

    print(
        "  "
        + str(
            output_dir
            / "planning_vs_code_problem_pairs.csv"
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )