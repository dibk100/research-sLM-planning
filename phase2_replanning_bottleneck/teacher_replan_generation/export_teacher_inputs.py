"""
PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase2_replanning_bottleneck/teacher_replan_generation/export_teacher_inputs.py \
  --config phase2_replanning_bottleneck/configs/teacher_replan_make.yaml
  
"""
# phase2_replanning_bottleneck/teacher_replan_generation/
# export_teacher_replan_inputs.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from src.datasets.phase1_failure_loader import (
    load_phase1_failures,
)
from src.utils.config import load_config
from src.utils.feedback import (
    truncate_input_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export teacher-replan input prompts "
            "from Phase 1 Direct failures."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to teacher-replan generation "
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
    
    
    feedback_config = config.get(
        "feedback",
        {},
    )

    model_config = config["model"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=model_config.get(
            "trust_remote_code",
            False,
        ),
    )


    # --------------------------------------------------------------
    # Load Phase 1 failures
    # --------------------------------------------------------------

    phase1_result_path = Path(
        dataset_config["path"]
    )

    failures = load_phase1_failures(
        result_path=phase1_result_path,
        limit=dataset_config.get("limit"),
    )

    if not failures:
        raise ValueError(
            "No refinable Phase 1 failures were loaded."
        )

    # --------------------------------------------------------------
    # Load shared Self-Replan prompt template
    # --------------------------------------------------------------

    prompt_path = Path(
        prompt_config["path"]
    )

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Teacher-replan prompt not found: "
            f"{prompt_path}"
        )

    prompt_template = prompt_path.read_text(
        encoding="utf-8"
    )

    required_placeholders = {
        "{problem}",
        "{extracted_code}",
        "{input_text}",
        "{stderr}",
    }

    missing_placeholders = [
        placeholder
        for placeholder in required_placeholders
        if placeholder not in prompt_template
    ]

    if missing_placeholders:
        raise ValueError(
            "Missing teacher-replan prompt placeholders: "
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
            "teacher_replan_inputs.jsonl",
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
        for failure in failures:
            input_text = truncate_input_text(
                text=failure.input_text,
                tokenizer=tokenizer,
                max_tokens=feedback_config.get(
                    "max_input_tokens"
                ),
            )

            teacher_prompt = (
                prompt_template.format(
                    problem=failure.problem,
                    extracted_code=(
                        failure.extracted_code
                    ),
                    input_text=input_text,
                    stderr=(
                        failure.stderr
                    ),
                ).strip()
            )

            record = {
                "problem_id": failure.problem_id,
                "dataset": dataset_config["name"],
                "difficulty": failure.difficulty,

                "initial_status": failure.status,
                "initial_passed_tests": failure.passed_tests,
                "initial_total_tests": failure.total_tests,
                "initial_test_pass_ratio": failure.test_pass_ratio,

                "feedback_test_index": failure.test_index,

                # 실제 labeling 과정에서 채움
                "teacher_model": "",

                "replan_version": teacher_config.get(
                    "replan_version",
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
    print("Teacher-Replan Input Export")
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
        f"Phase1 src : "
        f"{phase1_result_path}"
    )

    print(
        f"Prompt     : "
        f"{prompt_path}"
    )

    print(
        f"Replan ver.: "
        f"{teacher_config.get('replan_version', 'v1')}"
    )

    print(
        f"Failures   : "
        f"{len(failures)}"
    )

    print(
        f"Written    : "
        f"{num_written}"
    )

    print(
        f"Output     : "
        f"{output_path}"
    )
    
    print(
        f"Max input  : "
        f"{feedback_config.get('max_input_tokens')}"
    )

    print()
    print(
        "[DONE] Teacher-Replan inputs exported."
    )


if __name__ == "__main__":
    main()