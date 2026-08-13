"""YAML configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(
    config_path: str | Path,
) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Config must be a dictionary: {path}"
        )

    return config