"""
흐름 : DeepCoder TACO raw JSONL → stdin-only 6,387문제 → train/val split → verl parquet으로 고정

학습데이터 원칙
1. 학습 prompt에는 problem statement만 들어간다.
2. unit tests는 extra_info.problem.private_tests 안에만 저장해서 reward evaluator에서만 사용한다.
3. solutions는 학습 데이터에 넣지 않는다. sanity check에서 evaluator 검증용으로만 쓴 것이므로 training leakage를 막기 위해 제외할 것!


PYTHONPATH="$HOME/workspace/project_sLM_planning" \
python \
  phase4_method_discovery/vanilla_planning_rlvr/data/build_verl_dataset.py \
  --max-samples 100 \
  --val-ratio 0.1 \
  --seed 42

데이터 저장 위치
/mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/
└── vanilla_planning_rlvr/
    ├── train.parquet
    ├── val.parquet
    └── dataset_manifest.json
    
    
(/mnt/hdd/conda_envs/slm) dibaeck@diserver:~/workspace/project_sLM_planning$ PYTHONPATH="$HOME/workspace/project_sLM_planning" \
python \
  phase4_method_discovery/vanilla_planning_rlvr/data/build_verl_dataset.py \
  --max-samples 100 \
  --val-ratio 0.1 \
  --seed 42
==========================================================================================
Build DeepCoder TACO Vanilla Planning-RLVR Dataset
==========================================================================================
input            : /mnt/hdd/project_sLM_planning/data/deepcoder_taco/raw/deepcoder_taco_train.jsonl
output dir       : /mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/vanilla_planning_rlvr
prompt template  : /home/dibaeck/workspace/project_sLM_planning/prompt_templates/self_plan_plan.txt
val ratio        : 0.1
seed             : 42
[DeepCoderTACO] stdin=6387, functional_skipped=1049, invalid_skipped=0

[Load] stdin problems=6387
[Validate] problem pool OK
[Limit] using 100 problems

[Split] train=90
[Split] val=10
[Validate] no evaluator-test schema leakage in prompts

==========================================================================================
Dataset Build Complete
==========================================================================================
stdin pool       : 100
train            : 90
val              : 10
train parquet    : /mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/vanilla_planning_rlvr/train.parquet
val parquet      : /mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/vanilla_planning_rlvr/val.parquet
manifest         : /mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/vanilla_planning_rlvr/dataset_manifest.json
policy output    : plan only
reward           : frozen-coder execution 0/1
reference code   : NOT included
evaluation tests : extra_info only
==========================================================================================
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd


# ======================================================================
# Project root
# ======================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.datasets.deepcoder_taco import (
    load_deepcoder_taco_stdin,
)
from src.schemas import ProblemExample


# ======================================================================
# Constants
# ======================================================================

DEFAULT_INPUT = Path(
    "/mnt/hdd/project_sLM_planning/data/"
    "deepcoder_taco/raw/"
    "deepcoder_taco_train.jsonl"
)

DEFAULT_OUTPUT_DIR = Path(
    "/mnt/hdd/project_sLM_planning/data/"
    "deepcoder_taco/processed/"
    "vanilla_planning_rlvr"
)

DEFAULT_PROMPT_TEMPLATE = (
    PROJECT_ROOT
    / "prompt_templates"
    / "self_plan_plan.txt"
)

DATA_SOURCE = "deepcoder_taco"
ABILITY = "code_planning"

SCHEMA_VERSION = "1.0"


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build verl-compatible train/val parquet files "
            "from the DeepCoder TACO stdin subset."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="DeepCoder TACO raw JSONL.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--prompt-template",
        type=str,
        default=str(DEFAULT_PROMPT_TEMPLATE),
        help=(
            "Planning prompt template. "
            "Recommended: Phase 1 self_plan_plan.txt"
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
        help=(
            "Optional limit applied after stdin filtering. "
            "Useful for pilot dataset construction."
        ),
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable deterministic shuffling before split.",
    )

    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fail immediately on malformed TACO stdin rows. "
            "Default: true."
        ),
    )

    return parser.parse_args()


# ======================================================================
# Prompt template
# ======================================================================

def load_prompt_template(
    path: str | Path,
) -> str:
    template_path = Path(path)

    if not template_path.is_absolute():
        template_path = (
            PROJECT_ROOT
            / template_path
        )

    if not template_path.exists():
        raise FileNotFoundError(
            f"Prompt template not found: "
            f"{template_path}"
        )

    template = template_path.read_text(
        encoding="utf-8",
    )

    if not template.strip():
        raise ValueError(
            f"Prompt template is empty: "
            f"{template_path}"
        )

    return template

def build_planning_prompt(
    problem: ProblemExample,
    *,
    template: str,
) -> str:
    """
    Reuse the Phase 1 Self-Plan prompt condition.

    Supported placeholders:
        {problem}
        {title}
        {starter_code}
        {starter_code_section}
    """

    starter_code_section = ""

    if problem.starter_code.strip():
        starter_code_section = (
            "\n\nStarter Code:\n"
            f"{problem.starter_code.strip()}"
        )

    try:
        prompt = template.format(
            problem=problem.problem,
            title=problem.title,
            starter_code=problem.starter_code,
            starter_code_section=starter_code_section,
        )

    except KeyError as exc:
        raise KeyError(
            "Unsupported placeholder in planning prompt. "
            "Supported placeholders: "
            "{problem}, {title}, {starter_code}, "
            "{starter_code_section}. "
            f"Missing key: {exc}"
        ) from exc

    prompt = prompt.strip()

    if not prompt:
        raise ValueError(
            f"Built prompt is empty: "
            f"{problem.problem_id}"
        )

    return prompt

# ======================================================================
# Split
# ======================================================================

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

    items = list(
        problems
    )

    if shuffle:
        rng = random.Random(
            seed
        )

        rng.shuffle(
            items
        )

    if val_ratio == 0.0:
        return items, []

    if len(items) < 2:
        raise ValueError(
            "At least two problems are required "
            "for train/val split."
        )

    num_val = round(
        len(items)
        * val_ratio
    )

    num_val = max(
        1,
        num_val,
    )

    num_val = min(
        num_val,
        len(items) - 1,
    )

    val = items[
        :num_val
    ]

    train = items[
        num_val:
    ]

    return train, val


# ======================================================================
# verl row construction
# ======================================================================

def build_verl_row(
    problem: ProblemExample,
    *,
    split: str,
    index: int,
    prompt_template: str,
) -> dict[str, Any]:
    """
    Convert one TACO ProblemExample into a verl-compatible row.

    Important:
    - prompt contains NO unit tests
    - reference solutions are NOT stored
    - execution tests live only in extra_info.problem.private_tests
    """

    planning_prompt = (
        build_planning_prompt(
            problem,
            template=prompt_template,
        )
    )

    problem_payload = asdict(
        problem
    )

    return {
        # --------------------------------------------------------------
        # verl dataset identity
        # --------------------------------------------------------------

        "data_source": DATA_SOURCE,

        # --------------------------------------------------------------
        # Policy input
        #
        # verl will apply the tokenizer's chat template later.
        # --------------------------------------------------------------

        "prompt": [
            {
                "role": "user",
                "content": planning_prompt,
            }
        ],

        # --------------------------------------------------------------
        # Task category
        # --------------------------------------------------------------

        "ability": ABILITY,

        # --------------------------------------------------------------
        # Reward manager compatibility
        #
        # There is no textual ground-truth answer.
        # Correctness is determined through code execution.
        # --------------------------------------------------------------

        "reward_model": {
            "style": "rule",
            "ground_truth": "",
        },

        # --------------------------------------------------------------
        # Custom Planning-RLVR reward payload
        # --------------------------------------------------------------

        "extra_info": {
            "schema_version": (
                SCHEMA_VERSION
            ),

            "split": split,

            "index": index,

            "problem_id": (
                problem.problem_id
            ),

            "problem_text": (
                problem.problem
            ),

            # Store ProblemExample as JSON instead of a nested
            # Arrow struct. TACO stdin inputs can be either str
            # or list[str], which PyArrow cannot represent in one
            # homogeneous nested field.
            "problem_json": json.dumps(
                problem_payload,
                ensure_ascii=False,
            ),
        },
    }


# ======================================================================
# Dataset checks
# ======================================================================

def validate_problem_pool(
    problems: list[
        ProblemExample
    ],
) -> None:
    seen_ids: set[str] = set()

    for problem in problems:
        if (
            problem.dataset
            != DATA_SOURCE
        ):
            raise ValueError(
                f"Unexpected dataset: "
                f"{problem.problem_id} -> "
                f"{problem.dataset}"
            )

        if (
            problem.evaluation_type
            != "stdin"
        ):
            raise ValueError(
                f"Non-stdin problem found: "
                f"{problem.problem_id}"
            )

        if not problem.problem.strip():
            raise ValueError(
                f"Empty problem text: "
                f"{problem.problem_id}"
            )

        if not problem.private_tests:
            raise ValueError(
                f"No private tests: "
                f"{problem.problem_id}"
            )

        if problem.public_tests:
            raise ValueError(
                f"TACO training problem unexpectedly "
                f"contains public_tests: "
                f"{problem.problem_id}"
            )

        if (
            problem.problem_id
            in seen_ids
        ):
            raise ValueError(
                f"Duplicate problem_id: "
                f"{problem.problem_id}"
            )

        seen_ids.add(
            problem.problem_id
        )


def check_prompt_leakage(
    rows: list[
        dict[str, Any]
    ],
) -> None:
    """
    Lightweight sanity check.

    We intentionally do NOT search test strings inside the natural-language
    statement because sample cases can legitimately appear in problem text.

    Instead, verify structurally that tests only exist under extra_info.
    """

    for index, row in enumerate(
        rows
    ):
        prompt = row.get(
            "prompt"
        )

        if not isinstance(
            prompt,
            list,
        ):
            raise TypeError(
                f"row={index}: prompt must be list."
            )

        prompt_serialized = (
            json.dumps(
                prompt,
                ensure_ascii=False,
            )
        )

        if (
            '"private_tests"'
            in prompt_serialized
            or '"public_tests"'
            in prompt_serialized
        ):
            raise ValueError(
                f"row={index}: evaluator tests "
                f"leaked into prompt."
            )


# ======================================================================
# Parquet save
# ======================================================================

def save_parquet(
    rows: list[
        dict[str, Any]
    ],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(
        rows
    )

    df.to_parquet(
        path,
        engine="pyarrow",
        index=False,
    )


# ======================================================================
# Manifest
# ======================================================================

def save_manifest(
    *,
    output_dir: Path,
    input_path: Path,
    prompt_template_path: Path,
    train_problems: list[
        ProblemExample
    ],
    val_problems: list[
        ProblemExample
    ],
    seed: int,
    val_ratio: float,
    shuffled: bool,
) -> None:
    train_test_counts = [
        len(
            problem.private_tests
        )
        for problem
        in train_problems
    ]

    val_test_counts = [
        len(
            problem.private_tests
        )
        for problem
        in val_problems
    ]

    manifest = {
        "schema_version": (
            SCHEMA_VERSION
        ),

        "source": {
            "dataset": (
                "agentica-org/"
                "DeepCoder-Preview-Dataset"
            ),
            "config": "taco",
            "raw_input": str(
                input_path
            ),
        },

        "filtering": {
            "evaluation_type": (
                "stdin"
            ),
            "functional_excluded": (
                True
            ),
            "solutions_excluded": (
                True
            ),
            "tests_excluded_from_prompt": (
                True
            ),
        },

        "verl": {
            "data_source": (
                DATA_SOURCE
            ),
            "ability": (
                ABILITY
            ),
            "reward_type": (
                "execution_binary"
            ),
        },

        "prompt_template": str(
            prompt_template_path
        ),

        "split": {
            "seed": seed,
            "val_ratio": val_ratio,
            "shuffled": shuffled,

            "num_train": len(
                train_problems
            ),

            "num_val": len(
                val_problems
            ),
        },

        "test_statistics": {
            "train_min_tests": (
                min(train_test_counts)
                if train_test_counts
                else None
            ),

            "train_max_tests": (
                max(train_test_counts)
                if train_test_counts
                else None
            ),

            "val_min_tests": (
                min(val_test_counts)
                if val_test_counts
                else None
            ),

            "val_max_tests": (
                max(val_test_counts)
                if val_test_counts
                else None
            ),
        },

        "train_problem_ids": [
            problem.problem_id
            for problem
            in train_problems
        ],

        "val_problem_ids": [
            problem.problem_id
            for problem
            in val_problems
        ],
    }

    path = (
        output_dir
        / "dataset_manifest.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output_dir
    )

    prompt_template_path = Path(
        args.prompt_template
    )

    if not prompt_template_path.is_absolute():
        prompt_template_path = (
            PROJECT_ROOT
            / prompt_template_path
        )

    print("=" * 90)
    print(
        "Build DeepCoder TACO "
        "Vanilla Planning-RLVR Dataset"
    )
    print("=" * 90)

    print(
        f"input            : "
        f"{input_path}"
    )

    print(
        f"output dir       : "
        f"{output_dir}"
    )

    print(
        f"prompt template  : "
        f"{prompt_template_path}"
    )

    print(
        f"val ratio        : "
        f"{args.val_ratio}"
    )

    print(
        f"seed             : "
        f"{args.seed}"
    )

    # ------------------------------------------------------------------
    # 1. Load stdin-only DeepCoder TACO
    # ------------------------------------------------------------------

    problems = (
        load_deepcoder_taco_stdin(
            input_path,
            strict=args.strict,
        )
    )

    print()
    print(
        f"[Load] stdin problems="
        f"{len(problems)}"
    )

    # Expected with the current DeepCoder release:
    #
    #     6387 stdin problems
    #
    # Do not hard-fail here because future dataset versions may differ.

    # ------------------------------------------------------------------
    # 2. Validate pool
    # ------------------------------------------------------------------

    validate_problem_pool(
        problems
    )

    print(
        "[Validate] problem pool OK"
    )

    # ------------------------------------------------------------------
    # 3. Optional pilot limit
    # ------------------------------------------------------------------

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError(
                "--max-samples must be > 0."
            )

        problems = problems[
            : args.max_samples
        ]

        print(
            f"[Limit] using "
            f"{len(problems)} problems"
        )

    # ------------------------------------------------------------------
    # 4. Prompt
    # ------------------------------------------------------------------

    prompt_template = (
        load_prompt_template(
            prompt_template_path
        )
    )

    # ------------------------------------------------------------------
    # 5. Train/val split
    # ------------------------------------------------------------------

    train_problems, val_problems = (
        split_problems(
            problems,
            val_ratio=args.val_ratio,
            seed=args.seed,
            shuffle=(
                not args.no_shuffle
            ),
        )
    )

    print()
    print(
        f"[Split] train="
        f"{len(train_problems)}"
    )

    print(
        f"[Split] val="
        f"{len(val_problems)}"
    )

    # ------------------------------------------------------------------
    # 6. Build verl rows
    # ------------------------------------------------------------------

    train_rows = [
        build_verl_row(
            problem,
            split="train",
            index=index,
            prompt_template=(
                prompt_template
            ),
        )
        for index, problem
        in enumerate(
            train_problems
        )
    ]

    val_rows = [
        build_verl_row(
            problem,
            split="val",
            index=index,
            prompt_template=(
                prompt_template
            ),
        )
        for index, problem
        in enumerate(
            val_problems
        )
    ]

    # ------------------------------------------------------------------
    # 7. Leakage sanity check
    # ------------------------------------------------------------------

    check_prompt_leakage(
        train_rows
    )

    check_prompt_leakage(
        val_rows
    )

    print(
        "[Validate] no evaluator-test "
        "schema leakage in prompts"
    )

    # ------------------------------------------------------------------
    # 8. Save parquet
    # ------------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        output_dir
        / "train.parquet"
    )

    val_path = (
        output_dir
        / "val.parquet"
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

    # ------------------------------------------------------------------
    # 9. Manifest
    # ------------------------------------------------------------------

    save_manifest(
        output_dir=output_dir,
        input_path=input_path,
        prompt_template_path=(
            prompt_template_path
        ),
        train_problems=(
            train_problems
        ),
        val_problems=(
            val_problems
        ),
        seed=args.seed,
        val_ratio=args.val_ratio,
        shuffled=(
            not args.no_shuffle
        ),
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 90)
    print("Dataset Build Complete")
    print("=" * 90)

    print(
        f"stdin pool       : "
        f"{len(problems)}"
    )

    print(
        f"train            : "
        f"{len(train_rows)}"
    )

    print(
        f"val              : "
        f"{len(val_rows)}"
    )

    print(
        f"train parquet    : "
        f"{train_path}"
    )

    if val_rows:
        print(
            f"val parquet      : "
            f"{val_path}"
        )

    print(
        f"manifest         : "
        f"{output_dir / 'dataset_manifest.json'}"
    )

    print(
        "policy output    : plan only"
    )

    print(
        "reward           : "
        "frozen-coder execution 0/1"
    )

    print(
        "reference code   : NOT included"
    )

    print(
        "evaluation tests : "
        "extra_info only"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()