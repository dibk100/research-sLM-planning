"""
Download Datasets.

Usage:
    python download_dataset.py                
    python download_dataset.py livecode-v6
    python download_dataset.py codeforces_train
"""
import os
import sys
from datasets import load_dataset


# ============================================================
# Hugging Face Cache
# ============================================================

HF_CACHE = "/mnt/hdd/hf_cache"

os.environ["HF_DATASETS_CACHE"] = f"{HF_CACHE}/datasets"
os.environ["HF_HUB_CACHE"] = f"{HF_CACHE}/hub"


# ============================================================
# Dataset Sources
# ============================================================

SOURCES = {
    "livecode-v6": {
        "path": "livecodebench/code_generation_lite",
        "version_tag": "release_v6",
        "split": "test",
    },

    "codeforces_test": {
        "path": "open-r1/codeforces",
        "name": "default",
        "split": "test",
    },
    
    "codeforces_train": {
        "path": "open-r1/codeforces",
        "name": "default",
        "split": "train",
    },

    # 앞으로 추가
    # "apps": {
    #     "path": "...",
    #     "name": "...",
    #     "split": "test",
    # },
}


# ============================================================
# Load Dataset
# ============================================================

def download_dataset(dataset_name):

    if dataset_name not in SOURCES:
        print(f"[ERROR] Unknown dataset: {dataset_name}")
        print()
        print("Available datasets:")
        for name in SOURCES:
            print(f"  - {name}")
        sys.exit(1)

    config = SOURCES[dataset_name]

    print("=" * 70)
    print(f"Downloading dataset: {dataset_name}")
    print("=" * 70)

    print(f"Path      : {config['path']}")
    print(f"Split     : {config['split']}")
    print(f"Cache dir : {HF_CACHE}")

    kwargs = {
        "path": config["path"],
        "split": config["split"],
        "cache_dir": HF_CACHE,
    }

    if "name" in config:
        kwargs["name"] = config["name"]

    if "version_tag" in config:
        kwargs["version_tag"] = config["version_tag"]

    dataset = load_dataset(**kwargs)

    print()
    print("=" * 70)
    print("Download completed")
    print("=" * 70)

    print(f"Dataset          : {dataset_name}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Columns          : {dataset.column_names}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print()
        print("  python download_dataset.py <dataset_name>")
        print()
        print("Available datasets:")

        for name in SOURCES:
            print(f"  - {name}")

        sys.exit(1)

    dataset_name = sys.argv[1]

    download_dataset(dataset_name)