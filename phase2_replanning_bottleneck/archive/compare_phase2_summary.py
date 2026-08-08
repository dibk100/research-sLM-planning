"""
PYTHONPATH=. python archive/compare_phase2_summary.py

"""

from __future__ import annotations

import json
from pathlib import Path


RESULT_PATHS = {
    "Feedback Regeneration": Path(
        "/mnt/hdd/project_sLM_planning/output/"
        "phase2_feedback_regeneration_500/results.jsonl"
    ),
    "Self-Replanning": Path(
        "/mnt/hdd/project_sLM_planning/output/"
        "phase2_self_replan_500/results.jsonl"
    ),
    "Teacher-Replanning": Path(
        "/mnt/hdd/project_sLM_planning/output/"
        "phase2_teacher_replan_500/results.jsonl"
    ),
}


def analyze_result_file(
    path: Path,
) -> dict[str, int | float]:
    if not path.exists():
        raise FileNotFoundError(
            f"Result file not found: {path}"
        )

    total = 0
    exact_same = 0
    changed = 0
    recovered = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {path}"
                ) from error

            initial_code = (
                record.get("initial_code")
                or ""
            ).strip()

            refined_code = (
                record.get("refined_code")
                or ""
            ).strip()

            is_same = (
                initial_code == refined_code
            )

            total += 1

            if is_same:
                exact_same += 1
            else:
                changed += 1

            if bool(
                record.get("recovered")
            ):
                recovered += 1

    if total == 0:
        raise ValueError(
            f"No result records found: {path}"
        )

    return {
        "total": total,
        "exact_same": exact_same,
        "changed": changed,
        "recovered": recovered,
        "exact_same_rate": (
            exact_same / total
        ),
        "changed_rate": (
            changed / total
        ),
        "recovery_rate": (
            recovered / total
        ),
    }


def format_metric(
    count: int,
    total: int,
) -> str:
    rate = count / total

    return (
        f"{count}/{total} "
        f"({rate:.1%})"
    )


def main() -> None:
    summaries: dict[
        str,
        dict[str, int | float],
    ] = {}

    for strategy, path in (
        RESULT_PATHS.items()
    ):
        summaries[strategy] = (
            analyze_result_file(
                path
            )
        )

    print()
    print(
        "| Strategy | Exact Same | Changed | Recovered |"
    )
    print(
        "|---|---:|---:|---:|"
    )

    for strategy, summary in (
        summaries.items()
    ):
        total = int(
            summary["total"]
        )

        exact_same = int(
            summary["exact_same"]
        )

        changed = int(
            summary["changed"]
        )

        recovered = int(
            summary["recovered"]
        )

        print(
            f"| {strategy} "
            f"| {format_metric(exact_same, total)} "
            f"| {format_metric(changed, total)} "
            f"| {format_metric(recovered, total)} |"
        )

    print()
    print("=" * 80)
    print("Raw Summary")
    print("=" * 80)

    for strategy, summary in (
        summaries.items()
    ):
        print()
        print(strategy)
        print("-" * 80)

        print(
            "Total      :",
            summary["total"],
        )

        print(
            "Exact Same :",
            format_metric(
                int(summary["exact_same"]),
                int(summary["total"]),
            ),
        )

        print(
            "Changed    :",
            format_metric(
                int(summary["changed"]),
                int(summary["total"]),
            ),
        )

        print(
            "Recovered  :",
            format_metric(
                int(summary["recovered"]),
                int(summary["total"]),
            ),
        )


if __name__ == "__main__":
    main()