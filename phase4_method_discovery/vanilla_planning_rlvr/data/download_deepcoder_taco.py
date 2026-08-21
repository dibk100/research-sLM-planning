"""
PYTHONPATH=. python \
  phase4_method_discovery/vanilla_planning_rlvr/data/download_deepcoder_taco.py

[Download] loaded=7436
[Download] columns=['problem', 'tests', 'solutions']

[Save] Exporting JSONL -> /mnt/hdd/project_sLM_planning/data/deepcoder_taco/raw/deepcoder_taco_train.jsonl

================================================================================
Download Complete
================================================================================
examples : 7436
jsonl    : /mnt/hdd/project_sLM_planning/data/deepcoder_taco/raw/deepcoder_taco_train.jsonl
metadata : /mnt/hdd/project_sLM_planning/data/deepcoder_taco/raw/deepcoder_taco_metadata.json
HF cache : /mnt/hdd/hf_cache
================================================================================
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


# ======================================================================
# Dataset
# ======================================================================

DATASET_ID = "agentica-org/DeepCoder-Preview-Dataset"
CONFIG_NAME = "taco"
SPLIT = "train"


# ======================================================================
# Paths
# ======================================================================

HF_CACHE_DIR = Path(
    "/mnt/hdd/hf_cache"
)

DEFAULT_DATA_DIR = Path(
    "/mnt/hdd/project_sLM_planning/data/deepcoder_taco/raw"
)

DEFAULT_OUTPUT_NAME = "deepcoder_taco_train.jsonl"
DEFAULT_METADATA_NAME = "deepcoder_taco_metadata.json"


# ======================================================================
# Serialization
# ======================================================================

def to_json_serializable(
    value: Any,
) -> Any:
    """
    Recursively convert values to JSON-serializable Python objects.
    """

    if isinstance(value, dict):
        return {
            str(key): to_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_json_serializable(item)
            for item in value
        ]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def dump_jsonl(
    dataset: Any,
    output_path: Path,
) -> None:
    """
    Export Hugging Face dataset to JSONL without modifying
    the original dataset fields.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for item in dataset:
            payload = to_json_serializable(
                dict(item)
            )

            f.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                )
            )
            f.write("\n")


# ======================================================================
# Metadata
# ======================================================================

def save_metadata(
    *,
    dataset: Any,
    metadata_path: Path,
) -> None:
    """
    Save lightweight metadata for reproducibility.
    """

    metadata = {
        "dataset_id": DATASET_ID,
        "config": CONFIG_NAME,
        "split": SPLIT,
        "num_examples": len(dataset),
        "columns": list(dataset.column_names),
        "hf_cache_dir": str(HF_CACHE_DIR),
    }

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ======================================================================
# Inspection
# ======================================================================

def print_sample(
    dataset: Any,
) -> None:
    """
    Print the first example for a lightweight schema sanity check.
    """

    if len(dataset) == 0:
        print("[Sample] Dataset is empty.")
        return

    sample = dict(dataset[0])

    print()
    print("=" * 80)
    print("First Sample")
    print("=" * 80)

    for key, value in sample.items():
        print()
        print(f"[{key}]")

        if isinstance(value, str):
            preview = value[:1500]

            print(preview)

            if len(value) > 1500:
                print(
                    f"... <truncated, total chars={len(value)}>"
                )

        elif isinstance(value, list):
            print(
                f"type=list, length={len(value)}"
            )

            if value:
                print(
                    repr(value[0])[:1500]
                )

        elif isinstance(value, dict):
            print(
                json.dumps(
                    value,
                    indent=2,
                    ensure_ascii=False,
                )[:1500]
            )

        else:
            print(
                repr(value)[:1500]
            )


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the DeepCoder TACO training dataset "
            "and export it to JSONL."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=(
            "Directory for exported research data. "
            f"Default: {DEFAULT_DATA_DIR}"
        ),
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional number of examples to export. "
            "Useful for a small download/export sanity check."
        ),
    )

    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="Print the first example after loading.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if (
        args.max_samples is not None
        and args.max_samples <= 0
    ):
        raise ValueError(
            "--max-samples must be greater than 0."
        )

    data_dir = Path(args.data_dir)

    output_path = (
        data_dir
        / args.output_name
    )

    metadata_path = (
        data_dir
        / DEFAULT_METADATA_NAME
    )

    # ------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------

    print("=" * 80)
    print("DeepCoder TACO Dataset Download")
    print("=" * 80)

    print(
        f"dataset        : {DATASET_ID}"
    )

    print(
        f"config         : {CONFIG_NAME}"
    )

    print(
        f"split          : {SPLIT}"
    )

    print(
        f"HF cache       : {HF_CACHE_DIR}"
    )

    print(
        f"research data  : {data_dir}"
    )

    # ------------------------------------------------------------------
    # Download / load
    # ------------------------------------------------------------------

    print()
    print("[Download] Loading dataset...")

    dataset = load_dataset(
        DATASET_ID,
        CONFIG_NAME,
        split=SPLIT,
        cache_dir=str(HF_CACHE_DIR),
    )

    print(
        f"[Download] loaded={len(dataset)}"
    )

    print(
        f"[Download] columns={dataset.column_names}"
    )

    # ------------------------------------------------------------------
    # Optional subset
    # ------------------------------------------------------------------

    if args.max_samples is not None:
        limit = min(
            args.max_samples,
            len(dataset),
        )

        dataset = dataset.select(
            range(limit)
        )

        print(
            f"[Debug] selected first "
            f"{len(dataset)} examples"
        )

    # ------------------------------------------------------------------
    # Sample inspection
    # ------------------------------------------------------------------

    if args.show_sample:
        print_sample(
            dataset
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    print()
    print(
        f"[Save] Exporting JSONL -> {output_path}"
    )

    dump_jsonl(
        dataset,
        output_path,
    )

    save_metadata(
        dataset=dataset,
        metadata_path=metadata_path,
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("Download Complete")
    print("=" * 80)

    print(
        f"examples : {len(dataset)}"
    )

    print(
        f"jsonl    : {output_path}"
    )

    print(
        f"metadata : {metadata_path}"
    )

    print(
        f"HF cache : {HF_CACHE_DIR}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()