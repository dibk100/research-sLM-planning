"""
echo 'export HF_HOME=/mnt/hdd/hf_cache' >> ~/.bashrc
source ~/.bashrc

Usage:
python -m src.datasets.inspect_livecodebench

dataset(노션에 기록함) :
- 전체 문제 수: 1055
- split: test
- 모든 주요 필드는 문자열로 저장됨
- public_test_cases: JSON 문자열
- private_test_cases: Base64 + zlib 압축 문자열
- metadata: JSON 문자열
- starter_code는 빈 문자열일 수 있음
- 문제 식별자는 question_id
- 난이도는 easy, medium, hard


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