from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# Project root
# ============================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.schemas import ProblemExample


# ============================================================
# Constants
# ============================================================

DATA_SOURCE = "livecodebench_v6"
ABILITY = "code_planning"

DEFAULT_PLAN_INSTRUCTION = """\
Analyze the programming problem and produce a concise algorithmic plan.

Focus on:
- the core algorithm or strategy,
- necessary data structures,
- the main computational steps,
- important edge cases,
- time and space complexity.

Do not write code.
Output only the plan.
"""


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build verl-compatible Parquet datasets for "
            "Vanilla Planning-RLVR."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Input JSON or JSONL containing serialized "
            "ProblemExample records."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "phase4_method_discovery/"
            "vanilla_planning_rlvr/data/processed"
        ),
    )

    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help=(
            "Optional planning prompt template. "
            "If omitted, the built-in planning instruction is used."
        ),
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of problems.",
    )

    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args()


# ============================================================
# Input loading
# ============================================================

def load_problem_records(
    path: str | Path,
) -> list[ProblemExample]:

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_path}"
        )

    suffix = input_path.suffix.lower()

    if suffix == ".jsonl":
        payloads = _load_jsonl(input_path)

    elif suffix == ".json":
        payloads = _load_json(input_path)

    else:
        raise ValueError(
            "Input must be .json or .jsonl, "
            f"got: {input_path.suffix}"
        )

    problems: list[ProblemExample] = []

    seen_problem_ids: set[str] = set()

    for index, payload in enumerate(payloads):

        if not isinstance(payload, dict):
            raise TypeError(
                f"Input row {index} must be dict, "
                f"got {type(payload).__name__}"
            )

        try:
            problem = ProblemExample(**payload)

        except TypeError as exc:
            raise TypeError(
                f"Failed to construct ProblemExample "
                f"at input row {index}: {exc}"
            ) from exc

        validate_problem(problem)

        if problem.problem_id in seen_problem_ids:
            raise ValueError(
                "Duplicate problem_id detected: "
                f"{problem.problem_id}"
            )

        seen_problem_ids.add(
            problem.problem_id
        )

        problems.append(problem)

    if not problems:
        raise ValueError(
            "No problems were loaded."
        )

    return problems


def _load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                payload = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line "
                    f"{line_number}: {exc}"
                ) from exc

            rows.append(payload)

    return rows


def _load_json(
    path: Path,
) -> list[dict[str, Any]]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):

        # Allow:
        #
        # {
        #     "problems": [...]
        # }
        #
        if "problems" in payload:
            problems = payload["problems"]

            if not isinstance(problems, list):
                raise TypeError(
                    "'problems' must be a list."
                )

            return problems

        # Also allow one ProblemExample.
        return [payload]

    raise TypeError(
        "JSON input must contain either "
        "a problem object or a list of problems."
    )


# ============================================================
# Validation
# ============================================================

def validate_problem(
    problem: ProblemExample,
) -> None:

    if not problem.problem_id:
        raise ValueError(
            "problem_id must not be empty."
        )

    if not problem.problem.strip():
        raise ValueError(
            f"Empty problem statement: "
            f"{problem.problem_id}"
        )

    if problem.dataset != DATA_SOURCE:
        raise ValueError(
            f"Unsupported dataset for "
            f"{problem.problem_id}: "
            f"{problem.dataset!r}. "
            f"Expected {DATA_SOURCE!r}."
        )

    if problem.evaluation_type not in {
        "stdin",
        "functional",
    }:
        raise ValueError(
            f"Unsupported evaluation_type for "
            f"{problem.problem_id}: "
            f"{problem.evaluation_type}"
        )

    if (
        problem.evaluation_type == "functional"
        and not problem.function_name
    ):
        raise ValueError(
            f"Functional problem missing "
            f"function_name: {problem.problem_id}"
        )

    total_tests = (
        len(problem.public_tests)
        + len(problem.private_tests)
    )

    if total_tests == 0:
        raise ValueError(
            f"No tests available: "
            f"{problem.problem_id}"
        )


# ============================================================
# Prompt
# ============================================================

def load_prompt_template(
    path: str | Path | None,
) -> str | None:

    if path is None:
        return None

    prompt_path = Path(path)

    if not prompt_path.is_absolute():
        prompt_path = (
            PROJECT_ROOT / prompt_path
        )

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt template not found: "
            f"{prompt_path}"
        )

    template = prompt_path.read_text(
        encoding="utf-8",
    )

    if not template.strip():
        raise ValueError(
            f"Prompt template is empty: "
            f"{prompt_path}"
        )

    return template


def build_planning_prompt(
    problem: ProblemExample,
    *,
    template: str | None,
) -> str:
    """
    Build planner prompt.

    If Phase-1 self_plan_plan.txt is supplied, this function
    reuses it so that Planning-RLVR begins from the same
    prompt condition as the prior experiments.
    """

    if template is None:

        sections = [
            DEFAULT_PLAN_INSTRUCTION.strip(),
            "",
            "## Problem",
            problem.problem.strip(),
        ]

        if problem.starter_code.strip():
            sections.extend(
                [
                    "",
                    "## Starter Code",
                    problem.starter_code.strip(),
                ]
            )

        return "\n".join(sections).strip()

    # --------------------------------------------------------
    # Reuse Phase-1 template when possible.
    #
    # Supported placeholders:
    #   {problem}
    #   {title}
    #   {starter_code}
    # --------------------------------------------------------

    try:
        prompt = template.format(
            problem=problem.problem,
            title=problem.title,
            starter_code=problem.starter_code,
        )

    except KeyError as exc:
        raise KeyError(
            "Planning prompt template uses an "
            "unsupported placeholder. "
            "Supported placeholders are "
            "{problem}, {title}, {starter_code}. "
            f"Missing key: {exc}"
        ) from exc

    return prompt.strip()


# ============================================================
# verl row
# ============================================================

def build_verl_row(
    problem: ProblemExample,
    *,
    split: str,
    index: int,
    prompt_template: str | None,
) -> dict[str, Any]:
    """
    Convert one ProblemExample to verl's RLHF dataset schema.

    The policy response is expected to contain ONLY the plan.
    """

    planning_prompt = build_planning_prompt(
        problem,
        template=prompt_template,
    )

    problem_dict = asdict(problem)

    row = {
        # Used by reward manager / compute_score.
        "data_source": DATA_SOURCE,

        # verl applies the tokenizer chat template later.
        "prompt": [
            {
                "role": "user",
                "content": planning_prompt,
            }
        ],

        # Metadata/category field.
        "ability": ABILITY,

        # Required for compatibility with the normal
        # verl reward-manager interface.
        #
        # Vanilla Planning-RLVR does NOT use a textual
        # ground-truth answer; correctness comes from
        # LiveCodeBench execution.
        "reward_model": {
            "style": "rule",
            "ground_truth": "",
        },

        # Passed to our compute_score(...).
        "extra_info": {
            "split": split,
            "index": index,

            "problem_id": problem.problem_id,

            # Convenience duplicate used by coder prompt.
            "problem_text": problem.problem,

            # Serialized ProblemExample.
            #
            # IMPORTANT:
            # After Parquet loading this is a dict,
            # not a ProblemExample instance.
            "problem": problem_dict,
        },
    }

    return row


# ============================================================
# Split
# ============================================================

def split_problems(
    problems: list[ProblemExample],
    *,
    val_ratio: float,
    seed: int,
    shuffle: bool,
) -> tuple[
    list[ProblemExample],
    list[ProblemExample],
]:

    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(
            "val_ratio must satisfy "
            "0 <= val_ratio < 1."
        )

    items = list(problems)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)

    if val_ratio == 0.0:
        return items, []

    num_val = max(
        1,
        round(len(items) * val_ratio),
    )

    num_val = min(
        num_val,
        len(items) - 1,
    )

    val = items[:num_val]
    train = items[num_val:]

    return train, val


# ============================================================
# Save
# ============================================================

def save_parquet(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(rows)

    df.to_parquet(
        output_path,
        engine="pyarrow",
        index=False,
    )


def save_manifest(
    *,
    output_dir: Path,
    train_problems: list[ProblemExample],
    val_problems: list[ProblemExample],
    seed: int,
    val_ratio: float,
) -> None:

    manifest = {
        "schema_version": "1.0",
        "data_source": DATA_SOURCE,
        "ability": ABILITY,
        "seed": seed,
        "val_ratio": val_ratio,
        "num_train": len(train_problems),
        "num_val": len(val_problems),
        "train_problem_ids": [
            problem.problem_id
            for problem in train_problems
        ],
        "val_problem_ids": [
            problem.problem_id
            for problem in val_problems
        ],
    }

    manifest_path = (
        output_dir
        / "dataset_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("Build Vanilla Planning-RLVR Dataset")
    print("=" * 80)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    problems = load_problem_records(
        args.input
    )

    print(
        f"[Data] loaded={len(problems)}"
    )

    # --------------------------------------------------------
    # Optional limit
    # --------------------------------------------------------

    if args.max_samples is not None:

        if args.max_samples <= 0:
            raise ValueError(
                "--max-samples must be > 0."
            )

        problems = problems[
            : args.max_samples
        ]

        print(
            f"[Data] limited={len(problems)}"
        )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt_template = (
        load_prompt_template(
            args.prompt_template
        )
    )

    if args.prompt_template is None:
        print(
            "[Prompt] using built-in planning prompt"
        )
    else:
        print(
            f"[Prompt] template="
            f"{args.prompt_template}"
        )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_problems, val_problems = (
        split_problems(
            problems,
            val_ratio=args.val_ratio,
            seed=args.seed,
            shuffle=args.shuffle,
        )
    )

    print(
        f"[Split] train={len(train_problems)}"
    )

    print(
        f"[Split] val={len(val_problems)}"
    )

    # --------------------------------------------------------
    # Build rows
    # --------------------------------------------------------

    train_rows = [
        build_verl_row(
            problem,
            split="train",
            index=index,
            prompt_template=prompt_template,
        )
        for index, problem in enumerate(
            train_problems
        )
    ]

    val_rows = [
        build_verl_row(
            problem,
            split="val",
            index=index,
            prompt_template=prompt_template,
        )
        for index, problem in enumerate(
            val_problems
        )
    ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    if not output_dir.is_absolute():
        output_dir = (
            PROJECT_ROOT / output_dir
        )

    train_path = (
        output_dir / "train.parquet"
    )

    val_path = (
        output_dir / "val.parquet"
    )

    save_parquet(
        train_rows,
        train_path,
    )

    if val_rows:
        save_parquet(
            val_rows,
            val_path,
        )

    save_manifest(
        output_dir=output_dir,
        train_problems=train_problems,
        val_problems=val_problems,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Dataset Summary")
    print("=" * 80)

    print(
        f"train             : "
        f"{train_path}"
    )

    if val_rows:
        print(
            f"val               : "
            f"{val_path}"
        )

    print(
        f"train samples     : "
        f"{len(train_rows)}"
    )

    print(
        f"val samples       : "
        f"{len(val_rows)}"
    )

    print(
        "policy response   : PLAN ONLY"
    )

    print(
        "reward            : execution-based 0/1"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()