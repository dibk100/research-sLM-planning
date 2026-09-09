# phase4_method_discovery/vanilla_planning_rlvr/evaluation/checkpoint_dataset.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.schemas import ProblemExample


def _to_python(value: Any) -> Any:
    """
    Convert numpy/pandas container values into ordinary Python values
    where necessary.
    """
    if hasattr(value, "tolist"):
        return value.tolist()

    return value


def _extract_extra_info(row: pd.Series) -> dict[str, Any]:
    extra_info = row["extra_info"]

    if hasattr(extra_info, "as_py"):
        extra_info = extra_info.as_py()

    extra_info = _to_python(extra_info)

    if not isinstance(extra_info, dict):
        try:
            extra_info = dict(extra_info)
        except Exception as exc:
            raise TypeError(
                "extra_info must be dict-like, "
                f"got {type(extra_info).__name__}"
            ) from exc

    return extra_info


def _load_problem_payload(
    extra_info: dict[str, Any],
) -> dict[str, Any]:
    if "problem_json" not in extra_info:
        raise KeyError(
            "extra_info does not contain problem_json."
        )

    problem_json = extra_info["problem_json"]

    if isinstance(problem_json, str):
        payload = json.loads(problem_json)

    elif isinstance(problem_json, dict):
        payload = problem_json

    else:
        raise TypeError(
            "problem_json must be str or dict, "
            f"got {type(problem_json).__name__}"
        )

    if not isinstance(payload, dict):
        raise TypeError(
            "Decoded problem_json must be a dict."
        )

    return payload


def _build_problem_example(
    payload: dict[str, Any],
) -> ProblemExample:
    """
    Reconstruct the exact ProblemExample serialized during
    Phase 4 dataset construction.
    """

    return ProblemExample(
        problem_id=str(
            payload["problem_id"]
        ),
        title=str(
            payload.get(
                "title",
                payload["problem_id"],
            )
        ),
        problem=str(
            payload["problem"]
        ),
        starter_code=str(
            payload.get(
                "starter_code",
                "",
            )
            or ""
        ),
        dataset=str(
            payload.get(
                "dataset",
                "deepcoder_taco",
            )
        ),
        platform=str(
            payload.get(
                "platform",
                "taco",
            )
        ),
        difficulty=(
            payload.get("difficulty")
        ),
        rating=(
            payload.get("rating")
        ),
        contest_date=str(
            payload.get(
                "contest_date",
                "",
            )
            or ""
        ),
        evaluation_type=str(
            payload.get(
                "evaluation_type",
                "stdin",
            )
        ),
        public_tests=list(
            payload.get(
                "public_tests",
                [],
            )
            or []
        ),
        private_tests=list(
            payload.get(
                "private_tests",
                [],
            )
            or []
        ),
        time_limit=(
            payload.get("time_limit")
        ),
        memory_limit=(
            payload.get("memory_limit")
        ),
        function_name=(
            payload.get("function_name")
        ),
    )


def load_checkpoint_eval_dataset(
    parquet_path: str | Path,
    *,
    limit: int | None = None,
) -> list[ProblemExample]:
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Evaluation parquet not found: "
            f"{parquet_path}"
        )

    df = pd.read_parquet(
        parquet_path
    )

    required_columns = {
        "prompt",
        "extra_info",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Evaluation parquet is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "limit must be > 0."
            )

        df = df.iloc[:limit]

    examples: list[
        ProblemExample
    ] = []

    seen_ids: set[str] = set()

    for row_index, row in df.iterrows():
        extra_info = (
            _extract_extra_info(row)
        )

        payload = (
            _load_problem_payload(
                extra_info
            )
        )

        example = (
            _build_problem_example(
                payload
            )
        )

        if example.problem_id in seen_ids:
            raise ValueError(
                "Duplicate problem_id in "
                "evaluation parquet: "
                f"{example.problem_id}"
            )

        seen_ids.add(
            example.problem_id
        )

        if not example.problem.strip():
            raise ValueError(
                "Empty problem statement: "
                f"{example.problem_id}"
            )

        total_tests = (
            len(example.public_tests)
            + len(example.private_tests)
        )

        if total_tests == 0:
            raise ValueError(
                "No evaluation tests reconstructed "
                f"for {example.problem_id}"
            )

        examples.append(
            example
        )

    print(
        "[CheckpointDataset] "
        f"loaded {len(examples)} examples "
        f"from {parquet_path}"
    )

    return examples