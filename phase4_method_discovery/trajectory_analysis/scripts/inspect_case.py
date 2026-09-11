"""
PYTHONPATH=. python \
  phase4_method_discovery/trajectory_analysis/scripts/inspect_case.py \
  deepcoder_taco_00098
  
"""
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CANDIDATES_PATH = (
    PROJECT_ROOT
    / "phase4_method_discovery"
    / "trajectory_analysis"
    / "outputs"
    / "transition_candidates.jsonl"
)

BASE_PATH = (
    PROJECT_ROOT
    / "phase4_method_discovery"
    / "vanilla_planning_rlvr"
    / "outputs"
    / "checkpoint_eval"
    / "base_val100.jsonl"
)

VANILLA_PATH = (
    PROJECT_ROOT
    / "phase4_method_discovery"
    / "vanilla_planning_rlvr"
    / "outputs"
    / "checkpoint_eval"
    / "step900_val100.jsonl"
)

TPR_PATH = (
    PROJECT_ROOT
    / "phase4_method_discovery"
    / "tpr_planning_rlvr"
    / "outputs"
    / "checkpoint_eval"
    / "step900_val100.jsonl"
)


# ==============================================================================
# I/O
# ==============================================================================


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL: {path}:{line_no}"
                ) from exc

    return rows


def load_by_id(
    path: Path,
) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)

    result = {}

    for row in rows:
        problem_id = row.get(
            "problem_id"
        )

        if not problem_id:
            raise ValueError(
                f"Missing problem_id in {path}"
            )

        result[problem_id] = row

    return result


# ==============================================================================
# Trace extraction
# ==============================================================================


def extract_plan(
    row: dict[str, Any],
) -> str | None:
    trace = row.get(
        "strategy_trace"
    ) or []

    for item in trace:
        if not isinstance(
            item,
            dict,
        ):
            continue

        if (
            item.get("name")
            != "plan_generation"
        ):
            continue

        for key in (
            "raw_output",
            "output",
            "generated_text",
            "response",
            "plan",
            "text",
        ):
            value = item.get(key)

            if (
                isinstance(value, str)
                and value.strip()
            ):
                return value.strip()

    return None


# ==============================================================================
# Formatting
# ==============================================================================


def section(
    title: str,
    char: str = "=",
) -> None:
    print()
    print(char * 100)
    print(title)
    print(char * 100)


def subsection(
    title: str,
) -> None:
    print()
    print("-" * 100)
    print(title)
    print("-" * 100)


def format_bool(
    value: bool,
) -> str:
    return "PASS" if value else "FAIL"


def safe_tpr(
    row: dict[str, Any],
) -> float:
    value = row.get(
        "test_pass_ratio"
    )

    if value is None:
        return 0.0

    return float(value)


# ==============================================================================
# Result table
# ==============================================================================


def print_result_table(
    records: dict[
        str,
        dict[str, Any],
    ],
) -> None:

    headers = [
        "Method",
        "Passed",
        "Tests",
        "TPR",
        "Status",
    ]

    rows = []

    for method in (
        "Base",
        "Vanilla",
        "TPR",
    ):
        row = records[method]

        rows.append([
            method,
            format_bool(
                bool(
                    row.get("passed")
                )
            ),
            (
                f"{row.get('passed_tests', '-')}"
                f"/{row.get('total_tests', '-')}"
            ),
            f"{safe_tpr(row):.6f}",
            str(
                row.get(
                    "status",
                    "-",
                )
            ),
        ])

    widths = [
        max(
            len(headers[i]),
            max(
                len(str(row[i]))
                for row in rows
            ),
        )
        for i in range(
            len(headers)
        )
    ]

    header_line = " | ".join(
        headers[i].ljust(
            widths[i]
        )
        for i in range(
            len(headers)
        )
    )

    separator = "-+-".join(
        "-" * width
        for width in widths
    )

    print(header_line)
    print(separator)

    for row in rows:
        print(
            " | ".join(
                str(row[i]).ljust(
                    widths[i]
                )
                for i in range(
                    len(row)
                )
            )
        )


# ==============================================================================
# Plan / Code
# ==============================================================================


def print_plan(
    method: str,
    row: dict[str, Any],
) -> None:
    subsection(
        f"{method} Plan"
    )

    plan = extract_plan(row)

    if plan:
        print(plan)
    else:
        print(
            "[Plan not found]"
        )


def print_code(
    method: str,
    row: dict[str, Any],
) -> None:
    subsection(
        f"{method} Code"
    )

    code = row.get(
        "extracted_code"
    )

    if not code:
        code = row.get(
            "raw_output"
        )

    if code:
        print(code)
    else:
        print(
            "[Code not found]"
        )


# ==============================================================================
# Test-level comparison
# ==============================================================================


def validate_test_alignment(
    records: dict[
        str,
        dict[str, Any],
    ],
) -> int:

    totals = {
        records[method].get(
            "total_tests"
        )
        for method in records
    }

    if len(totals) != 1:
        raise ValueError(
            "total_tests mismatch "
            f"across methods: {totals}"
        )

    total = totals.pop()

    if total is None:
        return 0

    return int(total)


def get_test_result(
    row: dict[str, Any],
    index: int,
) -> dict[str, Any] | None:

    tests = row.get(
        "test_results"
    ) or []

    if index >= len(tests):
        return None

    item = tests[index]

    if not isinstance(
        item,
        dict,
    ):
        return None

    return item


def print_changed_tests(
    records: dict[
        str,
        dict[str, Any],
    ],
    *,
    max_tests: int = 30,
) -> None:

    total_tests = (
        validate_test_alignment(
            records
        )
    )

    changed = []

    for index in range(
        total_tests
    ):
        states = {}

        for method in (
            "Base",
            "Vanilla",
            "TPR",
        ):
            result = get_test_result(
                records[method],
                index,
            )

            if result is None:
                states[method] = (
                    None,
                    "MISSING",
                )
            else:
                states[method] = (
                    bool(
                        result.get(
                            "passed"
                        )
                    ),
                    result.get(
                        "status",
                        "-",
                    ),
                )

        passed_values = {
            states[method][0]
            for method in states
        }

        if len(
            passed_values
        ) > 1:
            changed.append(
                (
                    index,
                    states,
                )
            )

    section(
        "Discriminative Test Outcomes"
    )

    print(
        "Tests where Base / Vanilla / TPR "
        "do not share the same pass/fail outcome."
    )

    print(
        f"Changed tests: "
        f"{len(changed)}/{total_tests}"
    )

    if not changed:
        return

    print()

    print(
        f"{'Idx':>5} | "
        f"{'Base':<20} | "
        f"{'Vanilla':<20} | "
        f"{'TPR':<20}"
    )

    print(
        "-" * 74
    )

    for (
        index,
        states,
    ) in changed[:max_tests]:

        cells = []

        for method in (
            "Base",
            "Vanilla",
            "TPR",
        ):
            passed, status = (
                states[method]
            )

            if passed is None:
                label = "MISSING"
            else:
                label = (
                    "PASS"
                    if passed
                    else f"FAIL:{status}"
                )

            cells.append(label)

        print(
            f"{index:>5} | "
            f"{cells[0]:<20} | "
            f"{cells[1]:<20} | "
            f"{cells[2]:<20}"
        )

    if len(changed) > max_tests:
        print()
        print(
            f"... {len(changed) - max_tests} "
            "additional changed tests omitted."
        )


# ==============================================================================
# Failure details
# ==============================================================================


def print_failure_summary(
    records: dict[
        str,
        dict[str, Any],
    ],
) -> None:

    section(
        "Failure Summary"
    )

    for method in (
        "Base",
        "Vanilla",
        "TPR",
    ):
        row = records[method]

        print(
            f"[{method}]"
        )

        print(
            "status        :",
            row.get(
                "status"
            ),
        )

        print(
            "error_message :",
            row.get(
                "error_message"
            ),
        )

        failed_tests = [
            result
            for result in (
                row.get(
                    "test_results"
                )
                or []
            )
            if (
                isinstance(
                    result,
                    dict,
                )
                and not result.get(
                    "passed",
                    False,
                )
            )
        ]

        if failed_tests:
            first = failed_tests[0]

            print(
                "first failed test:",
                first.get(
                    "test_index"
                ),
            )

            print(
                "first status     :",
                first.get(
                    "status"
                ),
            )

            stderr = first.get(
                "stderr"
            )

            if stderr:
                stderr = str(
                    stderr
                ).strip()

                if len(stderr) > 500:
                    stderr = (
                        stderr[:500]
                        + "..."
                    )

                print(
                    "stderr           :",
                    stderr,
                )

        print()


# ==============================================================================
# Main inspector
# ==============================================================================


def inspect_case(
    problem_id: str,
    *,
    show_problem: bool = True,
    show_code: bool = True,
    show_tests: bool = True,
    max_changed_tests: int = 30,
) -> None:

    candidate_records = (
        load_by_id(
            CANDIDATES_PATH
        )
    )

    base_records = load_by_id(
        BASE_PATH
    )

    vanilla_records = load_by_id(
        VANILLA_PATH
    )

    tpr_records = load_by_id(
        TPR_PATH
    )

    if problem_id not in candidate_records:
        raise KeyError(
            f"Unknown problem_id: {problem_id}"
        )

    for name, records in [
        ("Base", base_records),
        ("Vanilla", vanilla_records),
        ("TPR", tpr_records),
    ]:
        if problem_id not in records:
            raise KeyError(
                f"{problem_id} missing "
                f"from {name} evaluation"
            )

    candidate = (
        candidate_records[
            problem_id
        ]
    )

    records = {
        "Base": base_records[
            problem_id
        ],
        "Vanilla": vanilla_records[
            problem_id
        ],
        "TPR": tpr_records[
            problem_id
        ],
    }

    section(
        problem_id
    )

    print(
        "Transition :",
        candidate.get(
            "transition"
        ),
    )

    print(
        "Category   :",
        candidate.get(
            "transition_description"
        ),
    )

    print()

    print_result_table(
        records
    )

    if show_problem:
        section(
            "Problem"
        )

        print(
            records[
                "Base"
            ].get(
                "problem",
                "[Problem not found]",
            )
        )

    section(
        "Plans"
    )

    for method in (
        "Base",
        "Vanilla",
        "TPR",
    ):
        print_plan(
            method,
            records[method],
        )

    if show_code:
        section(
            "Generated Codes"
        )

        for method in (
            "Base",
            "Vanilla",
            "TPR",
        ):
            print_code(
                method,
                records[method],
            )

    print_failure_summary(
        records
    )

    if show_tests:
        print_changed_tests(
            records,
            max_tests=(
                max_changed_tests
            ),
        )


# ==============================================================================
# CLI
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one Base / Vanilla / TPR "
            "planning trajectory."
        )
    )

    parser.add_argument(
        "problem_id",
        help=(
            "Example: deepcoder_taco_00417"
        ),
    )

    parser.add_argument(
        "--no-problem",
        action="store_true",
        help=(
            "Do not print the full problem statement."
        ),
    )

    parser.add_argument(
        "--no-code",
        action="store_true",
        help=(
            "Do not print generated code."
        ),
    )

    parser.add_argument(
        "--no-tests",
        action="store_true",
        help=(
            "Do not print discriminative tests."
        ),
    )

    parser.add_argument(
        "--max-changed-tests",
        type=int,
        default=30,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    inspect_case(
        args.problem_id,
        show_problem=(
            not args.no_problem
        ),
        show_code=(
            not args.no_code
        ),
        show_tests=(
            not args.no_tests
        ),
        max_changed_tests=(
            args.max_changed_tests
        ),
    )


if __name__ == "__main__":
    main()