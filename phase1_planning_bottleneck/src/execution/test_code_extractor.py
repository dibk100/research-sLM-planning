"""CodeExtractor unit test.

Usage:
    python -m src.execution.test_code_extractor
"""

from __future__ import annotations

from src.execution.code_extractor import (
    CodeExtractionError,
    CodeExtractor,
)


EXPECTED_CODE = """import sys

def solve():
    value = int(sys.stdin.readline())
    print(value * 2)

if __name__ == "__main__":
    solve()"""


def test_python_code_block(
    extractor: CodeExtractor,
) -> None:
    raw_output = (
        "Here is the solution:\n"
        "\n"
        "```python\n"
        f"{EXPECTED_CODE}\n"
        "```\n"
    )

    extracted = extractor.extract(raw_output)

    assert extracted == EXPECTED_CODE


def test_py_code_block(
    extractor: CodeExtractor,
) -> None:
    raw_output = (
        "```py\n"
        f"{EXPECTED_CODE}\n"
        "```"
    )

    extracted = extractor.extract(raw_output)

    assert extracted == EXPECTED_CODE


def test_generic_code_block(
    extractor: CodeExtractor,
) -> None:
    raw_output = (
        "```\n"
        f"{EXPECTED_CODE}\n"
        "```"
    )

    extracted = extractor.extract(raw_output)

    assert extracted == EXPECTED_CODE


def test_plain_code(
    extractor: CodeExtractor,
) -> None:
    extracted = extractor.extract(EXPECTED_CODE)

    assert extracted == EXPECTED_CODE


def test_multiple_code_blocks(
    extractor: CodeExtractor,
) -> None:
    """블록이 여러 개면 가장 긴 블록을 선택한다."""
    raw_output = (
        "Example:\n"
        "\n"
        "```python\n"
        'print("example")\n'
        "```\n"
        "\n"
        "Final solution:\n"
        "\n"
        "```python\n"
        f"{EXPECTED_CODE}\n"
        "```\n"
    )

    extracted = extractor.extract(raw_output)

    assert extracted == EXPECTED_CODE


def test_windows_newlines(
    extractor: CodeExtractor,
) -> None:
    raw_output = (
        "```python\r\n"
        "print('hello')\r\n"
        "```\r\n"
    )

    extracted = extractor.extract(raw_output)

    assert extracted == "print('hello')"


def test_empty_output(
    extractor: CodeExtractor,
) -> None:
    try:
        extractor.extract("   ")
    except CodeExtractionError as error:
        assert "empty" in str(error).lower()
    else:
        raise AssertionError(
            "Empty output did not raise CodeExtractionError."
        )


def test_empty_code_block(
    extractor: CodeExtractor,
) -> None:
    """코드 블록은 있으나 내용이 비어 있는 경우."""
    try:
        extractor.extract("```python\n\n```")
    except CodeExtractionError as error:
        assert "empty" in str(error).lower()
    else:
        raise AssertionError(
            "Empty code block did not raise CodeExtractionError."
        )


def test_invalid_type(
    extractor: CodeExtractor,
) -> None:
    try:
        extractor.extract(None)  # type: ignore[arg-type]
    except TypeError as error:
        assert "raw_output must be a string" in str(error)
    else:
        raise AssertionError(
            "Non-string output did not raise TypeError."
        )


TESTS = [
    ("python code block", test_python_code_block),
    ("py code block", test_py_code_block),
    ("generic code block", test_generic_code_block),
    ("plain code", test_plain_code),
    ("multiple code blocks", test_multiple_code_blocks),
    ("Windows newlines", test_windows_newlines),
    ("empty output", test_empty_output),
    ("empty code block", test_empty_code_block),
    ("invalid input type", test_invalid_type),
]


def main() -> int:
    extractor = CodeExtractor()

    print("=" * 80)
    print("CodeExtractor Test")
    print("=" * 80)

    failed = 0

    for name, test_function in TESTS:
        try:
            test_function(extractor)
        except AssertionError as error:
            failed += 1
            print(f"[FAIL] {name}: {error}")
        else:
            print(f"[PASS] {name}")

    print()

    if failed:
        print(f"[FAIL] {failed}/{len(TESTS)} CodeExtractor tests failed.")
        return 1

    print("[PASS] All CodeExtractor tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
