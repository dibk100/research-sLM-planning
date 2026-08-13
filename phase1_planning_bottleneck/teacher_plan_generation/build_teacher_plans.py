"""
작업 순서 2번

teacher_inputs_300.jsonl을 읽고 Teacher-Moder을 API로 불러서 plan을 생성한 뒤 teacher_plans_300.jsonl에 append 저장함.

output : 
/mnt/hdd/project_sLM_planning/data/teacher_plans/
└── livecodebench_v6/
    └── claude-opus-5_v1/
        ├── teacher_inputs_300.jsonl
        └── teacher_plans_300.jsonl             # 저장

### Note.
- export ANTHROPIC_API_KEY="..." (불가)
- 바이브코딩으로 실행 : VS Code Claude를 batch teacher labeling 도구로 사용.
    - 



PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/LiveCodeBench" \
python phase1_planning_bottleneck/teacher_plan_generation/build_teacher_plans.py \
  --config phase1_planning_bottleneck/configs/teacher_plan_generation/livecodebench_v6_opus5_v1.yaml \
  --limit 1
"""
# phase1_planning_bottleneck/teacher_plan_generation/build_teacher_plans.py

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate teacher plans using the Anthropic API."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to teacher-plan generation YAML config.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of teacher inputs to process.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing teacher plans and restart.",
    )

    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"JSONL file not found: {path}"
        )

    records: list[dict[str, Any]] = []

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
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at "
                    f"{path}:{line_number}"
                ) from error

            records.append(record)

    return records


def load_completed_ids(
    output_path: Path,
) -> set[str]:
    if not output_path.exists():
        return set()

    completed_ids: set[str] = set()

    with output_path.open(
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
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid existing JSON at "
                    f"{output_path}:{line_number}"
                ) from error

            problem_id = record.get(
                "problem_id"
            )

            if problem_id:
                completed_ids.add(
                    problem_id
                )

    return completed_ids


def extract_text(
    response: Any,
) -> str:
    text_parts: list[str] = []

    for block in response.content:
        if getattr(
            block,
            "type",
            None,
        ) == "text":
            text_parts.append(
                block.text
            )

    teacher_plan = "\n".join(
        text_parts
    ).strip()

    if not teacher_plan:
        raise ValueError(
            "Teacher response contained no text."
        )

    return teacher_plan


def append_jsonl(
    path: Path,
    record: dict[str, Any],
) -> None:
    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

        # Ensure the record is persisted immediately.
        f.flush()
        os.fsync(
            f.fileno()
        )


def generate_teacher_plan(
    *,
    client: Anthropic,
    model: str,
    teacher_prompt: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_delay_seconds: float,
) -> tuple[str, Any]:
    last_error: Exception | None = None

    for attempt in range(
        1,
        max_retries + 1,
    ):
        try:
            response = (
                client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                teacher_prompt
                            ),
                        }
                    ],
                )
            )

            teacher_plan = (
                extract_text(
                    response
                )
            )

            return (
                teacher_plan,
                response,
            )

        except Exception as error:
            last_error = error

            print(
                f"  [retry {attempt}/"
                f"{max_retries}] "
                f"{type(error).__name__}: "
                f"{error}"
            )

            if attempt < max_retries:
                time.sleep(
                    retry_delay_seconds
                    * attempt
                )

    assert last_error is not None
    raise last_error


def main() -> None:
    args = parse_args()

    config = load_config(
        args.config
    )

    experiment_config = (
        config["experiment"]
    )
    dataset_config = (
        config["dataset"]
    )
    teacher_config = (
        config["teacher"]
    )
    output_config = (
        config["output"]
    )

    # --------------------------------------------------------------
    # Teacher config
    # --------------------------------------------------------------

    teacher_model = (
        teacher_config["model"]
    )

    plan_version = (
        teacher_config.get(
            "plan_version",
            "v1",
        )
    )

    max_tokens = int(
        teacher_config.get(
            "max_tokens",
            1024,
        )
    )

    temperature = float(
        teacher_config.get(
            "temperature",
            0.0,
        )
    )

    max_retries = int(
        teacher_config.get(
            "max_retries",
            3,
        )
    )

    retry_delay_seconds = float(
        teacher_config.get(
            "retry_delay_seconds",
            2.0,
        )
    )

    if max_tokens <= 0:
        raise ValueError(
            "teacher.max_tokens must "
            "be greater than 0."
        )

    if temperature < 0:
        raise ValueError(
            "teacher.temperature must "
            "be greater than or equal to 0."
        )

    if max_retries <= 0:
        raise ValueError(
            "teacher.max_retries must "
            "be greater than 0."
        )

    # --------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------

    output_root = Path(
        output_config["root"]
    )

    dataset_family = (
        dataset_config.get(
            "output_name",
            "livecodebench_v6",
        )
    )

    teacher_run_name = (
        f"{teacher_model}_"
        f"{plan_version}"
    )

    output_dir = (
        output_root
        / dataset_family
        / teacher_run_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_limit = (
        dataset_config.get(
            "limit"
        )
    )

    suffix = (
        str(dataset_limit)
        if dataset_limit is not None
        else "all"
    )

    input_path = (
        output_dir
        / f"teacher_inputs_{suffix}.jsonl"
    )

    output_path = (
        output_dir
        / f"teacher_plans_{suffix}.jsonl"
    )

    # --------------------------------------------------------------
    # Load teacher inputs
    # --------------------------------------------------------------

    inputs = load_jsonl(
        input_path
    )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "--limit must be greater than 0."
            )

        inputs = inputs[
            :args.limit
        ]

    # --------------------------------------------------------------
    # Resume
    # --------------------------------------------------------------

    if args.no_resume:
        if output_path.exists():
            output_path.unlink()

        completed_ids: set[str] = set()

    else:
        completed_ids = (
            load_completed_ids(
                output_path
            )
        )

    # --------------------------------------------------------------
    # API key / client
    # --------------------------------------------------------------

    api_key = os.getenv(
        "ANTHROPIC_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set."
        )

    client = Anthropic(
        api_key=api_key
    )

    # --------------------------------------------------------------
    # Header
    # --------------------------------------------------------------

    print("=" * 80)
    print("Teacher Plan Generation")
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
        f"Teacher    : "
        f"{teacher_model}"
    )
    print(
        f"Plan ver.  : "
        f"{plan_version}"
    )
    print(
        f"Inputs     : "
        f"{len(inputs)}"
    )
    print(
        f"Completed  : "
        f"{len(completed_ids)}"
    )
    print(
        f"Input file : "
        f"{input_path}"
    )
    print(
        f"Output     : "
        f"{output_path}"
    )
    print()

    # --------------------------------------------------------------
    # Generation loop
    # --------------------------------------------------------------

    processed = 0
    skipped = 0
    failed = 0

    for index, record in enumerate(
        inputs,
        start=1,
    ):
        problem_id = record[
            "problem_id"
        ]

        if (
            problem_id
            in completed_ids
        ):
            skipped += 1

            print(
                f"[{index}/{len(inputs)}] "
                f"{problem_id} "
                "[SKIP]"
            )

            continue

        teacher_prompt = record.get(
            "teacher_prompt",
            ""
        )

        if not teacher_prompt.strip():
            raise ValueError(
                f"Empty teacher_prompt: "
                f"{problem_id}"
            )

        print()
        print("-" * 80)
        print(
            f"[{index}/{len(inputs)}] "
            f"{problem_id} | "
            f"{record.get('difficulty')} | "
            f"{record.get('title')}"
        )
        print("-" * 80)

        start_time = (
            time.perf_counter()
        )

        try:
            (
                teacher_plan,
                response,
            ) = generate_teacher_plan(
                client=client,
                model=teacher_model,
                teacher_prompt=(
                    teacher_prompt
                ),
                max_tokens=max_tokens,
                temperature=temperature,
                max_retries=max_retries,
                retry_delay_seconds=(
                    retry_delay_seconds
                ),
            )

        except Exception as error:
            failed += 1

            print(
                "[ERROR] "
                f"{problem_id}: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            # Stop rather than silently producing
            # an incomplete teacher-plan dataset.
            raise

        generation_time = (
            time.perf_counter()
            - start_time
        )

        usage = getattr(
            response,
            "usage",
            None,
        )

        input_tokens = getattr(
            usage,
            "input_tokens",
            None,
        )

        output_tokens = getattr(
            usage,
            "output_tokens",
            None,
        )

        stop_reason = getattr(
            response,
            "stop_reason",
            None,
        )

        output_record = {
            # Problem identity
            "problem_id": (
                problem_id
            ),
            "dataset": (
                record.get(
                    "dataset"
                )
            ),
            "title": (
                record.get(
                    "title"
                )
            ),
            "difficulty": (
                record.get(
                    "difficulty"
                )
            ),
            "rating": (
                record.get(
                    "rating"
                )
            ),
            "platform": (
                record.get(
                    "platform"
                )
            ),
            "contest_date": (
                record.get(
                    "contest_date"
                )
            ),

            # Teacher plan
            "teacher_plan": (
                teacher_plan
            ),

            # Provenance
            "teacher_model": (
                teacher_model
            ),
            "plan_version": (
                plan_version
            ),

            # The plan has been generated,
            # but not independently verified yet.
            "verified": False,

            # Teacher generation metadata
            "teacher_prompt": (
                teacher_prompt
            ),
            "input_tokens": (
                input_tokens
            ),
            "output_tokens": (
                output_tokens
            ),
            "generation_time": (
                generation_time
            ),
            "stop_reason": (
                stop_reason
            ),
        }

        append_jsonl(
            output_path,
            output_record,
        )

        completed_ids.add(
            problem_id
        )

        processed += 1

        print(
            f"Plan chars : "
            f"{len(teacher_plan)}"
        )
        print(
            f"Tokens     : "
            f"{input_tokens} -> "
            f"{output_tokens}"
        )
        print(
            f"Time       : "
            f"{generation_time:.2f}s"
        )
        print(
            f"Stop       : "
            f"{stop_reason}"
        )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print()
    print("=" * 80)
    print("Teacher Plan Generation Summary")
    print("=" * 80)

    print(
        f"Selected  : "
        f"{len(inputs)}"
    )
    print(
        f"Generated : "
        f"{processed}"
    )
    print(
        f"Skipped   : "
        f"{skipped}"
    )
    print(
        f"Failed    : "
        f"{failed}"
    )
    print(
        f"Output    : "
        f"{output_path}"
    )

    print()
    print(
        "[DONE] Teacher plan generation completed."
    )


if __name__ == "__main__":
    main()