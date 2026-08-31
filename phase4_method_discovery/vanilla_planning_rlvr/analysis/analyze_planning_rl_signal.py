"""
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase4_method_discovery/vanilla_planning_rlvr/analysis/analyze_planning_rl_signal.py \
  --base /mnt/hdd/project_sLM_planning/phase1/livecodebench_v6_stdin/qwen25Coder3b/self_plan/results.jsonl \
  --step25 /mnt/hdd/project_sLM_planning/output/phase4_rl_planner_eval/step25/results.jsonl \
  --step50 /mnt/hdd/project_sLM_planning/output/phase4_rl_planner_eval/step50/results.jsonl \
  --output-dir /home/dibaeck/workspace/project_sLM_planning/phase4_method_discovery/vanilla_planning_rlvr/outputs/planning_rl_signal
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


# ============================================================================
# Data structures
# ============================================================================


@dataclass(frozen=True)
class Trace:
    plan: str
    code: str


@dataclass(frozen=True)
class EvalRecord:
    problem_id: str
    title: str
    difficulty: str
    passed: bool
    status: str
    test_pass_ratio: float
    plan: str
    code: str


@dataclass(frozen=True)
class PairRecord:
    problem_id: str
    title: str
    difficulty: str

    before_passed: bool
    after_passed: bool
    transition: str

    before_status: str
    after_status: str

    before_tpr: float
    after_tpr: float
    tpr_delta: float

    plan_similarity: float
    plan_change: float

    code_similarity: float
    code_change: float

    before_plan_words: int
    after_plan_words: int
    plan_word_delta: int

    before_code_chars: int
    after_code_chars: int

    before_plan: str
    after_plan: str
    before_code: str
    after_code: str


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether execution-reward changes in planning RL "
            "are attributable to meaningful plan changes."
        )
    )

    parser.add_argument(
        "--base",
        required=True,
        help="Base Self-Plan results.jsonl",
    )

    parser.add_argument(
        "--step25",
        required=True,
        help="RL step25 results.jsonl",
    )

    parser.add_argument(
        "--step50",
        default=None,
        help="Optional RL step50 results.jsonl",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--near-identical-plan-threshold",
        type=float,
        default=0.90,
        help=(
            "Plan similarity threshold used to flag reward changes "
            "despite nearly identical plans. Default: 0.90"
        ),
    )

    parser.add_argument(
        "--large-plan-change-threshold",
        type=float,
        default=0.50,
        help=(
            "Plan similarity <= this value is treated as a large plan "
            "change. Default: 0.50"
        ),
    )

    return parser.parse_args()


# ============================================================================
# Loading
# ============================================================================


def load_jsonl(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    records: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON: {path}:{line_number}"
                ) from exc

            problem_id = raw.get("problem_id")

            if not problem_id:
                raise ValueError(
                    f"Missing problem_id: {path}:{line_number}"
                )

            if problem_id in records:
                raise ValueError(
                    f"Duplicate problem_id: {problem_id}"
                )

            records[problem_id] = raw

    if not records:
        raise ValueError(f"No records found: {path}")

    return records


# ============================================================================
# Record extraction
# ============================================================================


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}

    return bool(value)


def safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def extract_trace(raw: dict[str, Any]) -> Trace:
    trace = raw.get("strategy_trace")

    plan = ""
    code = ""

    if isinstance(trace, list):
        for step in trace:
            if not isinstance(step, dict):
                continue

            name = str(step.get("name", ""))
            output = str(step.get("raw_output", "") or "")

            if name == "plan_generation":
                plan = output

            elif name == "code_generation":
                code = output

    if not code:
        code = str(raw.get("raw_output", "") or "")

    return Trace(
        plan=plan.strip(),
        code=code.strip(),
    )


def get_title(raw: dict[str, Any]) -> str:
    for key in (
        "title",
        "problem_title",
        "question_title",
    ):
        value = raw.get(key)

        if value:
            return str(value)

    return ""


def to_eval_record(raw: dict[str, Any]) -> EvalRecord:
    trace = extract_trace(raw)

    return EvalRecord(
        problem_id=str(raw["problem_id"]),
        title=get_title(raw),
        difficulty=str(
            raw.get("difficulty", "unknown")
        ).lower(),
        passed=normalize_bool(
            raw.get("passed", False)
        ),
        status=str(
            raw.get("status", "UNKNOWN")
        ),
        test_pass_ratio=safe_float(
            raw.get("test_pass_ratio", 0.0)
        ),
        plan=trace.plan,
        code=trace.code,
    )


# ============================================================================
# Text normalization / similarity
# ============================================================================


def normalize_text(text: str) -> str:
    """
    Normalize formatting differences while preserving semantic content
    reasonably well.

    This is deliberately simple: this script is a diagnostic, not a
    semantic-equivalence evaluator.
    """

    text = text.lower()

    # Remove markdown decoration.
    text = re.sub(r"[`*_#]+", " ", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def text_similarity(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b,
        autojunk=False,
    ).ratio()


def word_count(text: str) -> int:
    return len(text.split())


# ============================================================================
# Transition
# ============================================================================


def classify_transition(
    before_passed: bool,
    after_passed: bool,
) -> str:
    if before_passed and after_passed:
        return "both_pass"

    if not before_passed and not after_passed:
        return "both_fail"

    if not before_passed and after_passed:
        return "recovered"

    return "regressed"


# ============================================================================
# Pair construction
# ============================================================================


def build_pair_records(
    before_raw: dict[str, dict[str, Any]],
    after_raw: dict[str, dict[str, Any]],
) -> list[PairRecord]:

    before_ids = set(before_raw)
    after_ids = set(after_raw)

    if before_ids != after_ids:
        raise ValueError(
            "Problem ID sets do not match.\n"
            f"Missing after: "
            f"{sorted(before_ids - after_ids)[:10]}\n"
            f"Extra after: "
            f"{sorted(after_ids - before_ids)[:10]}"
        )

    output: list[PairRecord] = []

    for problem_id in sorted(before_ids):
        before = to_eval_record(
            before_raw[problem_id]
        )

        after = to_eval_record(
            after_raw[problem_id]
        )

        if before.difficulty != after.difficulty:
            raise ValueError(
                f"Difficulty mismatch for {problem_id}: "
                f"{before.difficulty} vs {after.difficulty}"
            )

        plan_similarity = text_similarity(
            before.plan,
            after.plan,
        )

        code_similarity = text_similarity(
            before.code,
            after.code,
        )

        output.append(
            PairRecord(
                problem_id=problem_id,
                title=before.title,
                difficulty=before.difficulty,

                before_passed=before.passed,
                after_passed=after.passed,

                transition=classify_transition(
                    before.passed,
                    after.passed,
                ),

                before_status=before.status,
                after_status=after.status,

                before_tpr=before.test_pass_ratio,
                after_tpr=after.test_pass_ratio,
                tpr_delta=(
                    after.test_pass_ratio
                    - before.test_pass_ratio
                ),

                plan_similarity=plan_similarity,
                plan_change=1.0 - plan_similarity,

                code_similarity=code_similarity,
                code_change=1.0 - code_similarity,

                before_plan_words=word_count(
                    before.plan
                ),
                after_plan_words=word_count(
                    after.plan
                ),
                plan_word_delta=(
                    word_count(after.plan)
                    - word_count(before.plan)
                ),

                before_code_chars=len(before.code),
                after_code_chars=len(after.code),

                before_plan=before.plan,
                after_plan=after.plan,

                before_code=before.code,
                after_code=after.code,
            )
        )

    return output


# ============================================================================
# Statistics
# ============================================================================


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")

    return sum(values) / len(values)


def median(values: list[float]) -> float:
    if not values:
        return float("nan")

    values = sorted(values)

    n = len(values)
    middle = n // 2

    if n % 2 == 1:
        return values[middle]

    return (
        values[middle - 1]
        + values[middle]
    ) / 2.0


def pearson(
    xs: list[float],
    ys: list[float],
) -> float:
    if len(xs) != len(ys):
        raise ValueError("Length mismatch")

    if len(xs) < 2:
        return float("nan")

    mx = mean(xs)
    my = mean(ys)

    numerator = sum(
        (x - mx) * (y - my)
        for x, y in zip(xs, ys)
    )

    dx = math.sqrt(
        sum(
            (x - mx) ** 2
            for x in xs
        )
    )

    dy = math.sqrt(
        sum(
            (y - my) ** 2
            for y in ys
        )
    )

    if dx == 0 or dy == 0:
        return float("nan")

    return numerator / (dx * dy)


def summarize_pair(
    records: list[PairRecord],
    near_identical_threshold: float,
    large_change_threshold: float,
) -> dict[str, Any]:

    transition_counts = Counter(
        r.transition
        for r in records
    )

    reward_changed = [
        r
        for r in records
        if r.transition
        in {"recovered", "regressed"}
    ]

    reward_unchanged = [
        r
        for r in records
        if r.transition
        in {"both_pass", "both_fail"}
    ]

    near_identical_reward_changed = [
        r
        for r in reward_changed
        if r.plan_similarity
        >= near_identical_threshold
    ]

    large_plan_change_reward_unchanged = [
        r
        for r in reward_unchanged
        if r.plan_similarity
        <= large_change_threshold
    ]

    recovered = [
        r
        for r in records
        if r.transition == "recovered"
    ]

    regressed = [
        r
        for r in records
        if r.transition == "regressed"
    ]

    # Does larger plan change correlate with partial reward change?
    plan_changes = [
        r.plan_change
        for r in records
    ]

    abs_tpr_changes = [
        abs(r.tpr_delta)
        for r in records
    ]

    # Does plan perturbation propagate to code perturbation?
    code_changes = [
        r.code_change
        for r in records
    ]

    return {
        "n": len(records),

        "transitions": dict(
            transition_counts
        ),

        "net_solved_gain": (
            transition_counts["recovered"]
            - transition_counts["regressed"]
        ),

        "plan_similarity": {
            "mean_all": mean(
                [r.plan_similarity for r in records]
            ),
            "median_all": median(
                [r.plan_similarity for r in records]
            ),

            "mean_recovered": mean(
                [r.plan_similarity for r in recovered]
            ),

            "mean_regressed": mean(
                [r.plan_similarity for r in regressed]
            ),

            "mean_reward_changed": mean(
                [
                    r.plan_similarity
                    for r in reward_changed
                ]
            ),

            "mean_reward_unchanged": mean(
                [
                    r.plan_similarity
                    for r in reward_unchanged
                ]
            ),
        },

        "code_similarity": {
            "mean_all": mean(
                [r.code_similarity for r in records]
            ),

            "mean_recovered": mean(
                [r.code_similarity for r in recovered]
            ),

            "mean_regressed": mean(
                [r.code_similarity for r in regressed]
            ),
        },

        "attribution_flags": {
            "near_identical_plan_threshold": (
                near_identical_threshold
            ),

            "reward_changed_count": len(
                reward_changed
            ),

            "reward_changed_with_near_identical_plan": len(
                near_identical_reward_changed
            ),

            "reward_changed_with_near_identical_plan_fraction": (
                len(near_identical_reward_changed)
                / len(reward_changed)
                if reward_changed
                else 0.0
            ),

            "large_plan_change_threshold": (
                large_change_threshold
            ),

            "reward_unchanged_count": len(
                reward_unchanged
            ),

            "reward_unchanged_despite_large_plan_change": len(
                large_plan_change_reward_unchanged
            ),

            "reward_unchanged_despite_large_plan_change_fraction": (
                len(large_plan_change_reward_unchanged)
                / len(reward_unchanged)
                if reward_unchanged
                else 0.0
            ),
        },

        "correlations": {
            "plan_change_vs_abs_tpr_change": pearson(
                plan_changes,
                abs_tpr_changes,
            ),

            "plan_change_vs_code_change": pearson(
                plan_changes,
                code_changes,
            ),
        },

        "problem_ids": {
            "recovered": [
                r.problem_id
                for r in recovered
            ],

            "regressed": [
                r.problem_id
                for r in regressed
            ],

            "reward_changed_with_near_identical_plan": [
                r.problem_id
                for r in near_identical_reward_changed
            ],

            "reward_unchanged_despite_large_plan_change": [
                r.problem_id
                for r in large_plan_change_reward_unchanged
            ],
        },
    }


# ============================================================================
# Console report
# ============================================================================


def fmt_float(value: float) -> str:
    if math.isnan(value):
        return "N/A"

    return f"{value:.4f}"


def print_pair_summary(
    name: str,
    summary: dict[str, Any],
) -> None:

    t = summary["transitions"]
    p = summary["plan_similarity"]
    c = summary["code_similarity"]
    a = summary["attribution_flags"]
    corr = summary["correlations"]

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    print()
    print("[Reward transitions]")

    print(
        f"Both pass : "
        f"{t.get('both_pass', 0)}"
    )

    print(
        f"Both fail : "
        f"{t.get('both_fail', 0)}"
    )

    print(
        f"Recovered : "
        f"{t.get('recovered', 0)}"
    )

    print(
        f"Regressed : "
        f"{t.get('regressed', 0)}"
    )

    print(
        f"Net gain  : "
        f"{summary['net_solved_gain']:+d}"
    )

    print()
    print("[Plan similarity]")

    print(
        "Mean all              : "
        f"{fmt_float(p['mean_all'])}"
    )

    print(
        "Median all            : "
        f"{fmt_float(p['median_all'])}"
    )

    print(
        "Mean recovered        : "
        f"{fmt_float(p['mean_recovered'])}"
    )

    print(
        "Mean regressed        : "
        f"{fmt_float(p['mean_regressed'])}"
    )

    print(
        "Mean reward changed   : "
        f"{fmt_float(p['mean_reward_changed'])}"
    )

    print(
        "Mean reward unchanged : "
        f"{fmt_float(p['mean_reward_unchanged'])}"
    )

    print()
    print("[Code similarity]")

    print(
        "Mean all       : "
        f"{fmt_float(c['mean_all'])}"
    )

    print(
        "Mean recovered : "
        f"{fmt_float(c['mean_recovered'])}"
    )

    print(
        "Mean regressed : "
        f"{fmt_float(c['mean_regressed'])}"
    )

    print()
    print("[Reward attribution diagnostics]")

    print(
        "Reward-changed cases                    : "
        f"{a['reward_changed_count']}"
    )

    print(
        f"Reward changed with plan similarity >= "
        f"{a['near_identical_plan_threshold']:.2f} : "
        f"{a['reward_changed_with_near_identical_plan']} "
        f"("
        f"{a['reward_changed_with_near_identical_plan_fraction']:.2%}"
        f")"
    )

    print(
        f"Reward unchanged with plan similarity <= "
        f"{a['large_plan_change_threshold']:.2f} : "
        f"{a['reward_unchanged_despite_large_plan_change']} "
        f"("
        f"{a['reward_unchanged_despite_large_plan_change_fraction']:.2%}"
        f")"
    )

    print()
    print("[Correlations]")

    print(
        "Plan change vs |TPR change| : "
        f"{fmt_float(corr['plan_change_vs_abs_tpr_change'])}"
    )

    print(
        "Plan change vs code change  : "
        f"{fmt_float(corr['plan_change_vs_code_change'])}"
    )


# ============================================================================
# Detailed console cases
# ============================================================================


def print_transition_cases(
    name: str,
    records: list[PairRecord],
) -> None:

    changed = [
        r
        for r in records
        if r.transition
        in {"recovered", "regressed"}
    ]

    print()
    print("-" * 100)
    print(f"{name}: Reward-changing cases")
    print("-" * 100)

    header = (
        f"{'Problem':<18}"
        f"{'Diff':<10}"
        f"{'Transition':<12}"
        f"{'PlanSim':>10}"
        f"{'CodeSim':>10}"
        f"{'BeforeTPR':>12}"
        f"{'AfterTPR':>11}"
        f"{'Delta':>10}"
    )

    print(header)
    print("-" * len(header))

    for r in changed:
        print(
            f"{r.problem_id:<18}"
            f"{r.difficulty:<10}"
            f"{r.transition:<12}"
            f"{r.plan_similarity:>10.4f}"
            f"{r.code_similarity:>10.4f}"
            f"{r.before_tpr:>12.4f}"
            f"{r.after_tpr:>11.4f}"
            f"{r.tpr_delta:>+10.4f}"
        )


# ============================================================================
# Saving
# ============================================================================


def save_pair_csv(
    records: list[PairRecord],
    path: Path,
) -> None:

    fields = list(
        asdict(records[0]).keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                asdict(record)
            )


def save_flagged_markdown(
    name: str,
    records: list[PairRecord],
    summary: dict[str, Any],
    path: Path,
) -> None:

    flagged_ids = set(
        summary[
            "problem_ids"
        ][
            "reward_changed_with_near_identical_plan"
        ]
    )

    changed = [
        r
        for r in records
        if r.transition
        in {"recovered", "regressed"}
    ]

    lines: list[str] = []

    lines.append(
        f"# Planning RL Signal Diagnostic: {name}"
    )
    lines.append("")

    lines.append(
        "This report focuses on cases where final execution reward "
        "changed between checkpoints."
    )
    lines.append("")

    lines.append(
        "| Problem | Difficulty | Transition | "
        "Plan Similarity | Code Similarity | "
        "TPR Delta | Near-identical-plan flag |"
    )

    lines.append(
        "|---|---|---|---:|---:|---:|---|"
    )

    for r in changed:
        flag = (
            "YES"
            if r.problem_id in flagged_ids
            else ""
        )

        lines.append(
            f"| {r.problem_id} "
            f"| {r.difficulty} "
            f"| {r.transition} "
            f"| {r.plan_similarity:.4f} "
            f"| {r.code_similarity:.4f} "
            f"| {r.tpr_delta:+.4f} "
            f"| {flag} |"
        )

    lines.append("")

    for r in changed:
        lines.append(
            f"## {r.problem_id} — {r.transition}"
        )

        lines.append("")

        lines.append(
            f"- Difficulty: `{r.difficulty}`"
        )

        lines.append(
            f"- Plan similarity: `{r.plan_similarity:.4f}`"
        )

        lines.append(
            f"- Code similarity: `{r.code_similarity:.4f}`"
        )

        lines.append(
            f"- TPR: `{r.before_tpr:.4f} -> "
            f"{r.after_tpr:.4f}`"
        )

        lines.append(
            f"- Status: `{r.before_status} -> "
            f"{r.after_status}`"
        )

        if r.problem_id in flagged_ids:
            lines.append(
                "- Attribution warning: **reward changed despite "
                "near-identical textual plans**"
            )

        lines.append("")
        lines.append("### Before plan")
        lines.append("")
        lines.append("```text")
        lines.append(r.before_plan)
        lines.append("```")
        lines.append("")

        lines.append("### After plan")
        lines.append("")
        lines.append("```text")
        lines.append(r.after_plan)
        lines.append("```")
        lines.append("")

        lines.append("### Before code")
        lines.append("")
        lines.append("```python")
        lines.append(r.before_code)
        lines.append("```")
        lines.append("")

        lines.append("### After code")
        lines.append("")
        lines.append("```python")
        lines.append(r.after_code)
        lines.append("```")
        lines.append("")

        lines.append("### Manual attribution")
        lines.append("")
        lines.append("- Is the algorithm different?")
        lines.append("- Is a reasoning condition corrected?")
        lines.append("- Is the change only implementation-level?")
        lines.append("- Did the plan change causally propagate to code?")
        lines.append("- Attribution: plan / code / ambiguous")
        lines.append("")
        lines.append("---")
        lines.append("")

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================================
# Interpretation
# ============================================================================


def build_interpretation(
    summary: dict[str, Any],
) -> list[str]:

    a = summary["attribution_flags"]
    corr = summary["correlations"]

    messages: list[str] = []

    fraction = (
        a[
            "reward_changed_with_near_identical_plan_fraction"
        ]
    )

    if a["reward_changed_count"] == 0:
        messages.append(
            "No binary reward transitions were observed."
        )

    elif fraction >= 0.50:
        messages.append(
            "Many reward transitions occur despite highly similar plans. "
            "This is evidence that downstream code generation can strongly "
            "affect execution reward attribution."
        )

    elif fraction > 0:
        messages.append(
            "Some reward transitions occur despite highly similar plans. "
            "Execution reward is therefore not a pure measure of planning "
            "quality."
        )

    else:
        messages.append(
            "Binary reward transitions are generally accompanied by "
            "non-trivial textual plan changes."
        )

    plan_code_corr = corr[
        "plan_change_vs_code_change"
    ]

    if not math.isnan(plan_code_corr):
        if plan_code_corr >= 0.5:
            messages.append(
                "Plan changes and code changes are strongly associated."
            )

        elif plan_code_corr >= 0.2:
            messages.append(
                "Plan changes and code changes show a moderate association."
            )

        else:
            messages.append(
                "Plan-change magnitude weakly predicts code-change magnitude."
            )

    messages.append(
        "This diagnostic cannot by itself distinguish reward sparsity, "
        "entropy collapse, or token-level credit assignment. "
        "Those require within-problem sampled groups and/or token logits."
    )

    return messages


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("Planning RL Signal Analysis")
    print("=" * 100)

    base_raw = load_jsonl(
        args.base
    )

    step25_raw = load_jsonl(
        args.step25
    )

    print(
        f"[Loaded] Base   : {len(base_raw)}"
    )

    print(
        f"[Loaded] Step25 : {len(step25_raw)}"
    )

    comparisons: dict[
        str,
        tuple[
            list[PairRecord],
            dict[str, Any],
        ],
    ] = {}

    # ------------------------------------------------------------------
    # Base -> Step25
    # ------------------------------------------------------------------

    base_to_25 = build_pair_records(
        base_raw,
        step25_raw,
    )

    summary_25 = summarize_pair(
        base_to_25,
        near_identical_threshold=(
            args.near_identical_plan_threshold
        ),
        large_change_threshold=(
            args.large_plan_change_threshold
        ),
    )

    comparisons["base_to_step25"] = (
        base_to_25,
        summary_25,
    )

    # ------------------------------------------------------------------
    # Optional Step50
    # ------------------------------------------------------------------

    if args.step50:
        step50_raw = load_jsonl(
            args.step50
        )

        print(
            f"[Loaded] Step50 : {len(step50_raw)}"
        )

        base_to_50 = build_pair_records(
            base_raw,
            step50_raw,
        )

        summary_50 = summarize_pair(
            base_to_50,
            near_identical_threshold=(
                args.near_identical_plan_threshold
            ),
            large_change_threshold=(
                args.large_plan_change_threshold
            ),
        )

        comparisons[
            "base_to_step50"
        ] = (
            base_to_50,
            summary_50,
        )

        step25_to_50 = build_pair_records(
            step25_raw,
            step50_raw,
        )

        summary_25_50 = summarize_pair(
            step25_to_50,
            near_identical_threshold=(
                args.near_identical_plan_threshold
            ),
            large_change_threshold=(
                args.large_plan_change_threshold
            ),
        )

        comparisons[
            "step25_to_step50"
        ] = (
            step25_to_50,
            summary_25_50,
        )

    # ------------------------------------------------------------------
    # Print + save
    # ------------------------------------------------------------------

    all_summaries: dict[str, Any] = {}

    for name, (
        records,
        summary,
    ) in comparisons.items():

        print_pair_summary(
            name,
            summary,
        )

        print_transition_cases(
            name,
            records,
        )

        interpretation = build_interpretation(
            summary
        )

        print()
        print("[Interpretation]")

        for message in interpretation:
            print(
                f"- {message}"
            )

        summary[
            "interpretation"
        ] = interpretation

        all_summaries[name] = summary

        csv_path = (
            output_dir
            / f"{name}_signal.csv"
        )

        report_path = (
            output_dir
            / f"{name}_signal_report.md"
        )

        save_pair_csv(
            records,
            csv_path,
        )

        save_flagged_markdown(
            name,
            records,
            summary,
            report_path,
        )

        print()
        print(
            f"[Saved] {csv_path}"
        )

        print(
            f"[Saved] {report_path}"
        )

    # ------------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------------

    summary_path = (
        output_dir
        / "planning_rl_signal_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_summaries,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )

    print()
    print("=" * 100)
    print("Analysis Complete")
    print("=" * 100)

    print(
        f"[Saved] {summary_path}"
    )


if __name__ == "__main__":
    main()