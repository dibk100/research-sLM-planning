"""
작업 순서 1번
teacher용 prompt 생성하는 작업

output :
/mnt/hdd/project_sLM_planning/data/teacher_plans/
└── livecodebench_v6/
    └── claude-opus-5_v1/
        └── teacher_inputs_300.jsonl

PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase1_planning_bottleneck/teacher_plan_generation/export_teacher_inputs.py \
  --config phase1_planning_bottleneck/configs/teacher_plan_make.yaml
"""
# phase1_planning_bottleneck/teacher_plan_generation/export_teacher_inputs.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.datasets.dataset_loader import load_dataset
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export teacher-plan input prompts "
            "from a benchmark dataset."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to teacher-plan generation "
            "YAML config."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiment_config = config["experiment"]
    dataset_config = config["dataset"]
    teacher_config = config["teacher"]
    prompt_config = config["prompt"]
    output_config = config["output"]

    # --------------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------------

    problems = load_dataset(
        dataset_name=dataset_config["name"],
        data_path=dataset_config["path"],
        limit=dataset_config.get("limit"),
    )

    if not problems:
        raise ValueError(
            "No problems were loaded."
        )

    # --------------------------------------------------------------
    # Load teacher prompt template
    # --------------------------------------------------------------

    prompt_path = Path(
        prompt_config["path"]
    )

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Teacher prompt not found: "
            f"{prompt_path}"
        )

    prompt_template = prompt_path.read_text(
        encoding="utf-8"
    )

    required_placeholders = {
        # "{title}",
        "{problem}",
        "{starter_code_section}",
    }

    missing_placeholders = [
        placeholder
        for placeholder in required_placeholders
        if placeholder not in prompt_template
    ]

    if missing_placeholders:
        raise ValueError(
            "Missing teacher prompt placeholders: "
            + ", ".join(
                missing_placeholders
            )
        )

    # --------------------------------------------------------------
    # Output directory
    # --------------------------------------------------------------

    output_dir = Path(
        output_config["dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / output_config.get(
            "input_file",
            "teacher_inputs.jsonl",
        )
    )

    # --------------------------------------------------------------
    # Export
    # --------------------------------------------------------------

    num_written = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for problem in problems:
            starter_code_section = ""

            if problem.starter_code.strip():
                starter_code_section = (
                    "Starter Code:\n"
                    f"{problem.starter_code.strip()}"
                )

            teacher_prompt = (
                prompt_template.format(
                    title=problem.title,
                    problem=problem.problem,
                    starter_code_section=starter_code_section,
                ).strip()
            )

            record = {
                "problem_id": problem.problem_id,
                "dataset": problem.dataset,
                "title": problem.title,
                "difficulty": problem.difficulty,
                "rating": problem.rating,
                "platform": problem.platform,
                "contest_date": problem.contest_date,
                "teacher_model": "",                    # teacher_config["model"] 라벨링하는 모델이 입력하게 하기
                "plan_version": teacher_config.get(
                    "plan_version",
                    "v1",
                ),
                "teacher_prompt": teacher_prompt,
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            num_written += 1

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print("=" * 80)
    print("Teacher Input Export")
    print("=" * 80)

    print(
        f"Experiment : "
        f"{experiment_config['name']}"
    )
    print(
        f"Dataset    : "
        f"{dataset_config['name']}"
    )
    print(
        f"Data path  : "
        f"{dataset_config['path']}"
    )
    print(
        f"Teacher    : "
        f"{teacher_config['model']}"
    )
    print(
        f"Plan ver.  : "
        f"{teacher_config.get('plan_version', 'v1')}"
    )
    print(
        f"Problems   : "
        f"{len(problems)}"
    )
    print(
        f"Written    : "
        f"{num_written}"
    )
    print(
        f"Output     : "
        f"{output_path}"
    )

    print()
    print(
        "[DONE] Teacher inputs exported."
    )


if __name__ == "__main__":
    main()