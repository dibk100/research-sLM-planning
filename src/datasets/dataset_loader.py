# src/datasets/dataset_loader.py

from pathlib import Path

from src.schemas import ProblemExample
from src.datasets.livecodebench import load_livecodebench
from src.datasets.codeforces import load_codeforces


def load_dataset(
    dataset_name: str,
    data_path: str | Path,
    limit: int | None = None,
) -> list[ProblemExample]:

    if dataset_name == "livecodebench_v6_stdin":
        problems = load_livecodebench(
            data_path=data_path,
            evaluation_type="stdin",
        )

    elif dataset_name == "livecodebench_v6_functional":
        problems = load_livecodebench(
            data_path=data_path,
            evaluation_type="functional",
        )

    elif dataset_name == "codeforces":
        problems = load_codeforces(
            data_path=data_path,
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if limit is not None:
        problems = problems[:limit]

    return problems