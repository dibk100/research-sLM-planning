"""
echo 'export HF_HOME=/mnt/hdd/hf_cache' >> ~/.bashrc
source ~/.bashrc

Usage:
python -m scripts.inspect_livecodebench
"""

import os

os.environ["HF_DATASETS_CACHE"] = "/mnt/hdd/hf_cache/datasets"
os.environ["HF_HUB_CACHE"] = "/mnt/hdd/hf_cache/hub"

from datasets import load_dataset

dataset = load_dataset(
    "livecodebench/code_generation_lite",
    version_tag="release_v6",
    split="test",
    trust_remote_code=True,
)

print(dataset)
print(dataset.column_names)
print(dataset.features)

row = dataset[0]

print()
print("First row keys:")
for key in row:
    print(
        f"- {key}: "
        f"{type(row[key]).__name__}"
    )

print()
print("First problem:")
print(row)