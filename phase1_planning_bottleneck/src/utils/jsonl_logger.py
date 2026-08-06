"""
JSONL 기록/읽기 유틸.

completed_problem_ids()를 구현하여 실험이 중단되어도 이어서 실행할 수 있다.
"""

import json
from pathlib import Path
from threading import Lock


class JSONLLogger:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._lock = Lock()

    def append(self, record: dict) -> None:
        serialized = json.dumps(
            record,
            ensure_ascii=False,
        )

        with self._lock:
            with self.output_path.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(serialized + "\n")

    def completed_problem_ids(self) -> set[str]:
        if not self.output_path.exists():
            return set()

        completed = set()

        with self.output_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)
                completed.add(record["problem_id"])

        return completed