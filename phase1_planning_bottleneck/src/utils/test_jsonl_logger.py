"""JSONLLogger unit tests.

Usage:
    python -m src.utils.test_jsonl_logger
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.utils.jsonl_logger import JSONLLogger


def make_record(
    problem_id: str,
    *,
    passed: bool,
) -> dict:
    return {
        "problem_id": problem_id,
        "dataset": "livecodebench_v6",
        "strategy": "direct",
        "model_name": "test-model",
        "seed": 42,
        "passed": passed,
        "status": "PASS" if passed else "WRONG_ANSWER",
    }


def test_create_and_append(
    temp_dir: Path,
) -> None:
    output_path = temp_dir / "results.jsonl"
    logger = JSONLLogger(output_path)

    logger.append(
        make_record(
            "1873_A",
            passed=True,
        )
    )

    assert output_path.exists()
    assert logger.count() == 1


def test_multiple_records(
    temp_dir: Path,
) -> None:
    output_path = temp_dir / "multiple.jsonl"
    logger = JSONLLogger(output_path)

    logger.append_many(
        [
            make_record("1873_A", passed=True),
            make_record("1873_B", passed=False),
            make_record("1873_D", passed=True),
        ]
    )

    records = logger.load_records()

    assert len(records) == 3
    assert records[0]["problem_id"] == "1873_A"
    assert records[1]["problem_id"] == "1873_B"
    assert records[2]["problem_id"] == "1873_D"


def test_completed_ids(
    temp_dir: Path,
) -> None:
    output_path = temp_dir / "completed.jsonl"
    logger = JSONLLogger(output_path)

    logger.append(
        make_record(
            "1873_A",
            passed=True,
        )
    )
    logger.append(
        make_record(
            "1873_B",
            passed=False,
        )
    )

    completed = logger.completed_ids()

    assert completed == {
        "1873_A",
        "1873_B",
    }

    # 실패한 문제도 생성 및 평가가 완료됐으므로 completed에 포함한다.
    assert logger.contains("1873_A")
    assert logger.contains("1873_B")
    assert not logger.contains("1873_D")


def test_unicode_content(
    temp_dir: Path,
) -> None:
    output_path = temp_dir / "unicode.jsonl"
    logger = JSONLLogger(output_path)

    record = make_record(
        "한글_문제",
        passed=True,
    )
    record["prompt"] = "한글 프롬프트입니다."
    record["error_message"] = None

    logger.append(record)

    loaded = logger.load_records()

    assert loaded[0]["prompt"] == "한글 프롬프트입니다."


def test_nested_content(
    temp_dir: Path,
) -> None:
    output_path = temp_dir / "nested.jsonl"
    logger = JSONLLogger(output_path)

    record = make_record(
        "1873_A",
        passed=False,
    )
    record["test_results"] = [
        {
            "test_index": 0,
            "passed": False,
            "status": "WRONG_ANSWER",
            "expected_output": "YES\n",
            "actual_output": "NO\n",
        }
    ]

    logger.append(record)

    loaded = logger.load_records()

    assert isinstance(
        loaded[0]["test_results"],
        list,
    )
    assert (
        loaded[0]["test_results"][0]["status"]
        == "WRONG_ANSWER"
    )


def test_empty_logger(
    temp_dir: Path,
) -> None:
    output_path = temp_dir / "empty.jsonl"
    logger = JSONLLogger(output_path)

    assert logger.is_empty()
    assert logger.count() == 0
    assert logger.completed_ids() == set()


def test_missing_problem_id(
    temp_dir: Path,
) -> None:
    output_path = temp_dir / "missing_id.jsonl"
    logger = JSONLLogger(output_path)

    try:
        logger.append(
            {
                "status": "PASS",
            }
        )
    except ValueError as error:
        assert "problem_id" in str(error)
    else:
        raise AssertionError(
            "Missing problem_id did not raise ValueError."
        )


def test_invalid_json_line(
    temp_dir: Path,
) -> None:
    output_path = temp_dir / "invalid.jsonl"

    output_path.write_text(
        '{"problem_id": "1873_A"}\n'
        'invalid json\n',
        encoding="utf-8",
    )

    logger = JSONLLogger(output_path)

    try:
        logger.load_records()
    except ValueError as error:
        assert "Invalid JSONL line 2" in str(error)
    else:
        raise AssertionError(
            "Invalid JSON line did not raise ValueError."
        )

    valid_records = logger.load_records(
        ignore_invalid_lines=True,
    )

    assert len(valid_records) == 1
    assert valid_records[0]["problem_id"] == "1873_A"


def test_physical_line_count(
    temp_dir: Path,
) -> None:
    output_path = temp_dir / "line_count.jsonl"
    logger = JSONLLogger(output_path)

    logger.append(
        make_record(
            "1873_A",
            passed=True,
        )
    )
    logger.append(
        make_record(
            "1873_B",
            passed=False,
        )
    )

    lines = output_path.read_text(
        encoding="utf-8",
    ).splitlines()

    assert len(lines) == 2

    for line in lines:
        parsed = json.loads(line)
        assert isinstance(parsed, dict)


def main() -> None:
    print("=" * 80)
    print("JSONLLogger Test")
    print("=" * 80)

    with tempfile.TemporaryDirectory(
        prefix="phase1_logger_test_"
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        tests = [
            (
                "create and append",
                test_create_and_append,
            ),
            (
                "multiple records",
                test_multiple_records,
            ),
            (
                "completed IDs",
                test_completed_ids,
            ),
            (
                "Unicode content",
                test_unicode_content,
            ),
            (
                "nested content",
                test_nested_content,
            ),
            (
                "empty logger",
                test_empty_logger,
            ),
            (
                "missing problem ID",
                test_missing_problem_id,
            ),
            (
                "invalid JSON line",
                test_invalid_json_line,
            ),
            (
                "physical line count",
                test_physical_line_count,
            ),
        ]

        for name, test_function in tests:
            test_function(temp_dir)
            print(f"[PASS] {name}")

    print()
    print("[PASS] All JSONLLogger tests passed.")


if __name__ == "__main__":
    main()