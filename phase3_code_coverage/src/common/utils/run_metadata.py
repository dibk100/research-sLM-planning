"""Utilities for saving experiment configuration and runtime metadata."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import transformers
import yaml


def save_run_config(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    filename: str = "run_config.yaml",
    overwrite: bool = False,
) -> Path:
    """실험에 실제 사용된 설정을 YAML 파일로 저장한다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    if output_path.exists() and not overwrite:
        return output_path

    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    return output_path


def copy_run_config(
    source_config_path: str | Path,
    output_dir: str | Path,
    *,
    filename: str = "run_config.yaml",
    overwrite: bool = False,
) -> Path:
    """원본 config 파일을 output 디렉터리에 그대로 복사한다."""
    source_path = Path(source_config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {source_path}"
        )

    output_path = output_dir / filename

    if output_path.exists() and not overwrite:
        return output_path

    shutil.copy2(source_path, output_path)

    return output_path


def collect_run_metadata(
    config: dict[str, Any],
) -> dict[str, Any]:
    """실험 재현성을 위한 실행 환경 metadata를 수집한다."""
    experiment_config = config.get("experiment", {})
    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    generation_config = config.get("generation", {})

    metadata: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_name": experiment_config.get("name"),
        "seed": experiment_config.get("seed"),
        "dataset_name": dataset_config.get("name"),
        "dataset_split": dataset_config.get("split"),
        "dataset_limit": dataset_config.get("limit"),
        "dataset_release_version": dataset_config.get(
            "release_version",
            "release_v6"
            if dataset_config.get("name") == "livecodebench_v6"
            else None,
        ),
        "model_name_or_path": model_config.get("name_or_path"),
        "dtype": model_config.get("dtype"),
        "device_map": model_config.get("device_map"),
        "max_new_tokens": generation_config.get(
            "max_new_tokens"
        ),
        "temperature": generation_config.get("temperature"),
        "top_p": generation_config.get("top_p"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": _get_gpu_names(),
    }

    return metadata


def save_run_metadata(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    filename: str = "run_metadata.json",
    overwrite: bool = False,
) -> Path:
    """실행 환경 metadata를 JSON 파일로 저장한다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    if output_path.exists() and not overwrite:
        return output_path

    metadata = collect_run_metadata(config)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def _get_gpu_names() -> list[str]:
    """현재 환경에서 감지된 GPU 이름 목록을 반환한다."""
    if not torch.cuda.is_available():
        return []

    return [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ]