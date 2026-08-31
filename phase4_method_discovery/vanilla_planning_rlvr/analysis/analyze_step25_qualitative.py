"""
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase4_method_discovery/vanilla_planning_rlvr/analysis/analyze_step25_qualitative.py \
  --base /mnt/hdd/project_sLM_planning/phase1/livecodebench_v6_stdin/qwen25Coder3b/self_plan/results.jsonl \
  --step25 /mnt/hdd/project_sLM_planning/output/phase4_rl_planner_eval/step25/results.jsonl \
  --output-dir /home/dibaeck/workspace/project_sLM_planning/phase4_method_discovery/vanilla_planning_rlvr/outputs/step25_qualitative
"""
# phase4_method_discovery/vanilla_planning_rlvr/analysis/analyze_step25_qualitative.py

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ============================================================================
# Data structures
# ============================================================================


@dataclass(frozen=True)
class GenerationTrace:
    plan: str
    code: str


@dataclass(frozen=True)
class QualitativeRecord:
    problem_id: str
    title: str
    difficulty: str

    base_passed: bool
    step25_passed: bool

    base_status: str
    step25_status: str

    base_test_pass_ratio: float
    step25_test_pass_ratio: float
    test_pass_ratio_delta: float

    base_plan: str
    step25_plan: str

    base_code: str
    step25_code: str

    transition: str


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qualitative analysis of Base Self-Plan -> "
            "RL step25 recovered/regressed cases."
        )
    )

    parser.add_argument(
        "--base",
        required=True,
        help="Phase 1 Base Self-Plan results.jsonl",
    )

    parser.add_argument(
        "--step25",
        required=True,
        help="RL step25 results.jsonl",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for qualitative analysis outputs.",
    )

    return parser.parse_args()


# ============================================================================
# Loading
# ============================================================================


def load_jsonl(
    path: str | Path,
) -> dict[str, dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    records: dict[str, dict[str, Any]] = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at "
                    f"{path}:{line_number}"
                ) from error

            problem_id = record.get(
                "problem_id"
            )

            if not problem_id:
                raise ValueError(
                    f"Missing problem_id at "
                    f"{path}:{line_number}"
                )

            if problem_id in records:
                raise ValueError(
                    f"Duplicate problem_id "
                    f"in {path}: {problem_id}"
                )

            records[problem_id] = record

    if not records:
        raise ValueError(
            f"No records found: {path}"
        )

    return records


# ============================================================================
# Helpers
# ============================================================================


def normalize_bool(
    value: Any,
) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value != 0

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
        }

    return bool(value)


def safe_float(
    value: Any,
) -> float:
    if value is None:
        return 0.0

    try:
        return float(value)

    except (TypeError, ValueError):
        return 0.0


def get_title(
    record: dict[str, Any],
) -> str:
    for key in (
        "title",
        "problem_title",
        "question_title",
    ):
        value = record.get(key)

        if value:
            return str(value)

    return ""


def get_difficulty(
    record: dict[str, Any],
) -> str:
    value = record.get(
        "difficulty"
    )

    if value is None:
        return "unknown"

    return str(value).strip().lower()


# ============================================================================
# Strategy trace extraction
# ============================================================================


def extract_trace(
    record: dict[str, Any],
) -> GenerationTrace:
    """
    Expected Phase 1 / Phase 4 structure:

    strategy_trace = [
        {
            "name": "plan_generation",
            "raw_output": ...
        },
        {
            "name": "code_generation",
            "raw_output": ...
        }
    ]

    Falls back to final raw_output for code when needed.
    """

    trace = record.get(
        "strategy_trace"
    )

    plan = ""
    code = ""

    if isinstance(trace, list):
        for step in trace:
            if not isinstance(
                step,
                dict,
            ):
                continue

            name = str(
                step.get(
                    "name",
                    "",
                )
            )

            raw_output = step.get(
                "raw_output",
                "",
            )

            if raw_output is None:
                raw_output = ""

            raw_output = str(
                raw_output
            )

            if name == "plan_generation":
                plan = raw_output

            elif name == "code_generation":
                code = raw_output

    if not code:
        raw_output = record.get(
            "raw_output",
            "",
        )

        if raw_output is not None:
            code = str(
                raw_output
            )

    return GenerationTrace(
        plan=plan.strip(),
        code=code.strip(),
    )


# ============================================================================
# Transition classification
# ============================================================================


def classify_transition(
    base_passed: bool,
    step25_passed: bool,
) -> str:
    if (
        not base_passed
        and step25_passed
    ):
        return "recovered"

    if (
        base_passed
        and not step25_passed
    ):
        return "regressed"

    if (
        base_passed
        and step25_passed
    ):
        return "both_pass"

    return "both_fail"


# ============================================================================
# Build qualitative records
# ============================================================================


def build_qualitative_records(
    base_records: dict[
        str,
        dict[str, Any],
    ],
    step25_records: dict[
        str,
        dict[str, Any],
    ],
) -> list[QualitativeRecord]:
    base_ids = set(
        base_records.keys()
    )

    step25_ids = set(
        step25_records.keys()
    )

    if base_ids != step25_ids:
        missing = sorted(
            base_ids
            - step25_ids
        )

        extra = sorted(
            step25_ids
            - base_ids
        )

        raise ValueError(
            "Problem sets do not match.\n"
            f"Missing in step25: "
            f"{missing[:10]}\n"
            f"Extra in step25: "
            f"{extra[:10]}"
        )

    output: list[
        QualitativeRecord
    ] = []

    for problem_id in sorted(
        base_ids
    ):
        base = base_records[
            problem_id
        ]

        step25 = step25_records[
            problem_id
        ]

        base_passed = normalize_bool(
            base.get(
                "passed",
                False,
            )
        )

        step25_passed = normalize_bool(
            step25.get(
                "passed",
                False,
            )
        )

        transition = classify_transition(
            base_passed,
            step25_passed,
        )

        # We only need the seven changed-success cases.
        if transition not in {
            "recovered",
            "regressed",
        }:
            continue

        base_trace = extract_trace(
            base
        )

        step25_trace = extract_trace(
            step25
        )

        base_ratio = safe_float(
            base.get(
                "test_pass_ratio",
                0.0,
            )
        )

        step25_ratio = safe_float(
            step25.get(
                "test_pass_ratio",
                0.0,
            )
        )

        output.append(
            QualitativeRecord(
                problem_id=problem_id,
                title=get_title(
                    base
                ),
                difficulty=get_difficulty(
                    base
                ),

                base_passed=base_passed,
                step25_passed=step25_passed,

                base_status=str(
                    base.get(
                        "status",
                        "UNKNOWN",
                    )
                ),
                step25_status=str(
                    step25.get(
                        "status",
                        "UNKNOWN",
                    )
                ),

                base_test_pass_ratio=(
                    base_ratio
                ),
                step25_test_pass_ratio=(
                    step25_ratio
                ),
                test_pass_ratio_delta=(
                    step25_ratio
                    - base_ratio
                ),

                base_plan=(
                    base_trace.plan
                ),
                step25_plan=(
                    step25_trace.plan
                ),

                base_code=(
                    base_trace.code
                ),
                step25_code=(
                    step25_trace.code
                ),

                transition=transition,
            )
        )

    return output


# ============================================================================
# Simple text statistics
# ============================================================================


def word_count(
    text: str,
) -> int:
    return len(
        text.split()
    )


def char_count(
    text: str,
) -> int:
    return len(text)


def line_count(
    text: str,
) -> int:
    if not text:
        return 0

    return len(
        text.splitlines()
    )


# ============================================================================
# Console output
# ============================================================================


def print_summary(
    records: list[
        QualitativeRecord
    ],
) -> None:
    recovered = [
        record
        for record in records
        if record.transition
        == "recovered"
    ]

    regressed = [
        record
        for record in records
        if record.transition
        == "regressed"
    ]

    print()
    print("=" * 100)
    print("Step25 Qualitative Transition Summary")
    print("=" * 100)

    print(
        f"Recovered : {len(recovered)}"
    )

    print(
        f"Regressed : {len(regressed)}"
    )

    print()

    print(
        f"{'Transition':<12}"
        f"{'Problem':<24}"
        f"{'Difficulty':<12}"
        f"{'Base TPR':>12}"
        f"{'Step25 TPR':>12}"
        f"{'Delta':>12}"
    )

    print("-" * 84)

    for record in records:
        print(
            f"{record.transition:<12}"
            f"{record.problem_id:<24}"
            f"{record.difficulty:<12}"
            f"{record.base_test_pass_ratio:>12.4f}"
            f"{record.step25_test_pass_ratio:>12.4f}"
            f"{record.test_pass_ratio_delta:>+12.4f}"
        )


# ============================================================================
# Markdown report
# ============================================================================


def write_markdown_report(
    records: list[
        QualitativeRecord
    ],
    output_path: Path,
) -> None:
    recovered = [
        record
        for record in records
        if record.transition
        == "recovered"
    ]

    regressed = [
        record
        for record in records
        if record.transition
        == "regressed"
    ]

    lines: list[str] = []

    lines.append(
        "# Step25 Qualitative Analysis"
    )

    lines.append("")

    lines.append(
        "Base Self-Plan vs RL Planner @ Step25"
    )

    lines.append("")

    lines.append(
        f"- Recovered: {len(recovered)}"
    )

    lines.append(
        f"- Regressed: {len(regressed)}"
    )

    lines.append("")

    lines.append(
        "## Overview"
    )

    lines.append("")

    lines.append(
        "| Transition | Problem | Difficulty | "
        "Base TPR | Step25 TPR | Delta |"
    )

    lines.append(
        "|---|---|---:|---:|---:|---:|"
    )

    for record in records:
        lines.append(
            f"| {record.transition} "
            f"| {record.problem_id} "
            f"| {record.difficulty} "
            f"| {record.base_test_pass_ratio:.4f} "
            f"| {record.step25_test_pass_ratio:.4f} "
            f"| {record.test_pass_ratio_delta:+.4f} |"
        )

    lines.append("")

    # ------------------------------------------------------------------
    # Detailed cases
    # ------------------------------------------------------------------

    ordered_groups = [
        (
            "Recovered Cases",
            recovered,
        ),
        (
            "Regressed Cases",
            regressed,
        ),
    ]

    for section_name, group in ordered_groups:
        lines.append(
            f"## {section_name}"
        )

        lines.append("")

        for index, record in enumerate(
            group,
            start=1,
        ):
            title_suffix = (
                f" — {record.title}"
                if record.title
                else ""
            )

            lines.append(
                f"### {index}. "
                f"{record.problem_id}"
                f"{title_suffix}"
            )

            lines.append("")

            lines.append(
                f"- Difficulty: "
                f"`{record.difficulty}`"
            )

            lines.append(
                f"- Base: "
                f"`{record.base_status}`, "
                f"TPR="
                f"`{record.base_test_pass_ratio:.4f}`"
            )

            lines.append(
                f"- Step25: "
                f"`{record.step25_status}`, "
                f"TPR="
                f"`{record.step25_test_pass_ratio:.4f}`"
            )

            lines.append(
                f"- TPR delta: "
                f"`{record.test_pass_ratio_delta:+.4f}`"
            )

            lines.append("")

            # ----------------------------------------------------------
            # Plan statistics
            # ----------------------------------------------------------

            lines.append(
                "#### Plan statistics"
            )

            lines.append("")

            lines.append(
                "| | Base | Step25 |"
            )

            lines.append(
                "|---|---:|---:|"
            )

            lines.append(
                "| Characters | "
                f"{char_count(record.base_plan)} | "
                f"{char_count(record.step25_plan)} |"
            )

            lines.append(
                "| Words | "
                f"{word_count(record.base_plan)} | "
                f"{word_count(record.step25_plan)} |"
            )

            lines.append(
                "| Lines | "
                f"{line_count(record.base_plan)} | "
                f"{line_count(record.step25_plan)} |"
            )

            lines.append("")

            # ----------------------------------------------------------
            # Base plan
            # ----------------------------------------------------------

            lines.append(
                "#### Base plan"
            )

            lines.append("")

            lines.append("```text")

            lines.append(
                record.base_plan
                if record.base_plan
                else "[NO PLAN FOUND]"
            )

            lines.append("```")

            lines.append("")

            # ----------------------------------------------------------
            # Step25 plan
            # ----------------------------------------------------------

            lines.append(
                "#### Step25 plan"
            )

            lines.append("")

            lines.append("```text")

            lines.append(
                record.step25_plan
                if record.step25_plan
                else "[NO PLAN FOUND]"
            )

            lines.append("```")

            lines.append("")

            # ----------------------------------------------------------
            # Base code
            # ----------------------------------------------------------

            lines.append(
                "#### Base generated code"
            )

            lines.append("")

            lines.append("```python")

            lines.append(
                record.base_code
                if record.base_code
                else "[NO CODE FOUND]"
            )

            lines.append("```")

            lines.append("")

            # ----------------------------------------------------------
            # Step25 code
            # ----------------------------------------------------------

            lines.append(
                "#### Step25 generated code"
            )

            lines.append("")

            lines.append("```python")

            lines.append(
                record.step25_code
                if record.step25_code
                else "[NO CODE FOUND]"
            )

            lines.append("```")

            lines.append("")

            # ----------------------------------------------------------
            # Manual annotation section
            # ----------------------------------------------------------

            lines.append(
                "#### Manual qualitative annotation"
            )

            lines.append("")

            lines.append(
                "- Plan-level change:"
            )

            lines.append(
                "- Algorithm / strategy change:"
            )

            lines.append(
                "- Missing or corrected reasoning step:"
            )

            lines.append(
                "- Edge-case handling change:"
            )

            lines.append(
                "- Code-level consequence:"
            )

            lines.append(
                "- Likely reason for transition:"
            )

            lines.append("")

            lines.append("---")

            lines.append("")

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================================
# CSV
# ============================================================================


def write_csv(
    records: list[
        QualitativeRecord
    ],
    output_path: Path,
) -> None:
    fieldnames = [
        "problem_id",
        "title",
        "difficulty",
        "transition",

        "base_passed",
        "step25_passed",

        "base_status",
        "step25_status",

        "base_test_pass_ratio",
        "step25_test_pass_ratio",
        "test_pass_ratio_delta",

        "base_plan_chars",
        "step25_plan_chars",

        "base_plan_words",
        "step25_plan_words",

        "base_plan_lines",
        "step25_plan_lines",

        "base_plan",
        "step25_plan",

        "base_code",
        "step25_code",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "problem_id": (
                        record.problem_id
                    ),
                    "title": (
                        record.title
                    ),
                    "difficulty": (
                        record.difficulty
                    ),
                    "transition": (
                        record.transition
                    ),

                    "base_passed": (
                        record.base_passed
                    ),
                    "step25_passed": (
                        record.step25_passed
                    ),

                    "base_status": (
                        record.base_status
                    ),
                    "step25_status": (
                        record.step25_status
                    ),

                    "base_test_pass_ratio": (
                        record.base_test_pass_ratio
                    ),
                    "step25_test_pass_ratio": (
                        record.step25_test_pass_ratio
                    ),
                    "test_pass_ratio_delta": (
                        record.test_pass_ratio_delta
                    ),

                    "base_plan_chars": (
                        char_count(
                            record.base_plan
                        )
                    ),
                    "step25_plan_chars": (
                        char_count(
                            record.step25_plan
                        )
                    ),

                    "base_plan_words": (
                        word_count(
                            record.base_plan
                        )
                    ),
                    "step25_plan_words": (
                        word_count(
                            record.step25_plan
                        )
                    ),

                    "base_plan_lines": (
                        line_count(
                            record.base_plan
                        )
                    ),
                    "step25_plan_lines": (
                        line_count(
                            record.step25_plan
                        )
                    ),

                    "base_plan": (
                        record.base_plan
                    ),
                    "step25_plan": (
                        record.step25_plan
                    ),

                    "base_code": (
                        record.base_code
                    ),
                    "step25_code": (
                        record.step25_code
                    ),
                }
            )


# ============================================================================
# JSON
# ============================================================================


def write_json(
    records: list[
        QualitativeRecord
    ],
    output_path: Path,
) -> None:
    output = []

    for record in records:
        output.append(
            {
                "problem_id": (
                    record.problem_id
                ),
                "title": (
                    record.title
                ),
                "difficulty": (
                    record.difficulty
                ),
                "transition": (
                    record.transition
                ),

                "base": {
                    "passed": (
                        record.base_passed
                    ),
                    "status": (
                        record.base_status
                    ),
                    "test_pass_ratio": (
                        record.base_test_pass_ratio
                    ),
                    "plan": (
                        record.base_plan
                    ),
                    "code": (
                        record.base_code
                    ),
                },

                "step25": {
                    "passed": (
                        record.step25_passed
                    ),
                    "status": (
                        record.step25_status
                    ),
                    "test_pass_ratio": (
                        record.step25_test_pass_ratio
                    ),
                    "plan": (
                        record.step25_plan
                    ),
                    "code": (
                        record.step25_code
                    ),
                },

                "test_pass_ratio_delta": (
                    record.test_pass_ratio_delta
                ),
            }
        )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()

    base_path = Path(
        args.base
    )

    step25_path = Path(
        args.step25
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("Step25 Qualitative Analysis")
    print("=" * 100)

    print(
        f"Base   : {base_path}"
    )

    print(
        f"Step25 : {step25_path}"
    )

    print(
        f"Output : {output_dir}"
    )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    base_records = load_jsonl(
        base_path
    )

    step25_records = load_jsonl(
        step25_path
    )

    print()

    print(
        f"[Loaded] base   = "
        f"{len(base_records)}"
    )

    print(
        f"[Loaded] step25 = "
        f"{len(step25_records)}"
    )

    # ------------------------------------------------------------------
    # Build changed-success cases
    # ------------------------------------------------------------------

    qualitative_records = (
        build_qualitative_records(
            base_records=base_records,
            step25_records=step25_records,
        )
    )

    recovered_count = sum(
        record.transition
        == "recovered"
        for record
        in qualitative_records
    )

    regressed_count = sum(
        record.transition
        == "regressed"
        for record
        in qualitative_records
    )

    # We already know the expected paired result from the aggregate
    # analysis. Warn rather than hard-fail so the script remains reusable.
    if (
        recovered_count != 6
        or regressed_count != 1
    ):
        print()
        print(
            "[WARNING] Expected 6 recovered "
            "and 1 regressed from the previous "
            "paired analysis, but found "
            f"{recovered_count} recovered / "
            f"{regressed_count} regressed."
        )

    print_summary(
        qualitative_records
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    markdown_path = (
        output_dir
        / "step25_qualitative_report.md"
    )

    csv_path = (
        output_dir
        / "step25_qualitative_cases.csv"
    )

    json_path = (
        output_dir
        / "step25_qualitative_cases.json"
    )

    write_markdown_report(
        qualitative_records,
        markdown_path,
    )

    write_csv(
        qualitative_records,
        csv_path,
    )

    write_json(
        qualitative_records,
        json_path,
    )

    print()
    print("=" * 100)
    print("Saved Outputs")
    print("=" * 100)

    print(
        f"Markdown : {markdown_path}"
    )

    print(
        f"CSV      : {csv_path}"
    )

    print(
        f"JSON     : {json_path}"
    )

    print()
    print(
        "[DONE] Qualitative analysis extraction completed."
    )


if __name__ == "__main__":
    main()