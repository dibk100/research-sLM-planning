# tests/test_code_parser.py

import pytest

from src.parsing.code_parser import (
    CodeParser,
    CodeParsingError,
)
from src.schemas import CodeParseResult


@pytest.fixture
def parser() -> CodeParser:
    return CodeParser()


def test_python_code_block(
    parser: CodeParser,
):
    raw_output = """
Here is the solution:

```python
n = int(input())
print(n * 2)
```

This solves the problem.
"""
    result = parser.parse(raw_output)

    assert isinstance(result, CodeParseResult)
    assert result.status == "SUCCESS"
    assert result.extraction_method == "python_code_block"

    assert result.code == (
        "n = int(input())\n"
        "print(n * 2)"
    )


def test_py_code_block(
    parser: CodeParser,
):
    raw_output = """```py
print("hello")
```"""
    result = parser.parse(raw_output)

    assert result.status == "SUCCESS"
    assert result.extraction_method == "python_code_block"
    assert result.code == 'print("hello")'


def test_generic_code_block(
    parser: CodeParser,
):
    raw_output = """
Solution :

```
a, b = map(int, input().split())
print(a + b)
```
"""
    result = parser.parse(raw_output)

    assert result.status == "SUCCESS"
    assert result.extraction_method == "generic_code_block"

    assert result.code == (
        "a, b = map(int, input().split())\n"
        "print(a + b)"
    )


def test_plain_text_fallback(
    parser: CodeParser,
):
    raw_output = """
n = int(input())
print(n)
"""

    result = parser.parse(raw_output)

    assert result.status == "SUCCESS"
    assert result.extraction_method == "plain_text"

    assert result.code == (
        "n = int(input())\n"
        "print(n)"
    )


def test_python_block_has_priority_over_generic_block(
    parser: CodeParser,
):
    raw_output = """
```
print("generic block that is much longer than python")
print("still generic")
```

```python
print("python")
```
"""

    result = parser.parse(raw_output)

    assert result.status == "SUCCESS"

    # Python-labelled blocks must take priority,
    # even if a generic block is longer.
    assert result.extraction_method == "python_code_block"
    assert result.code == 'print("python")'


def test_select_longest_python_block(
    parser: CodeParser,
):
    raw_output = """
```python
print(1)
```

```python
n = int(input())
answer = n * 2
print(answer)
```
"""

    result = parser.parse(raw_output)

    assert result.status == "SUCCESS"
    assert result.extraction_method == "python_code_block"

    assert result.code == (
        "n = int(input())\n"
        "answer = n * 2\n"
        "print(answer)"
    )


def test_select_longest_generic_block(
    parser: CodeParser,
):
    raw_output = """
```
print(1)
```

```
x = int(input())
y = x + 1
print(y)
```
"""

    result = parser.parse(raw_output)

    assert result.status == "SUCCESS"
    assert result.extraction_method == "generic_code_block"

    assert result.code == (
        "x = int(input())\n"
        "y = x + 1\n"
        "print(y)"
    )


def test_empty_output(
    parser: CodeParser,
):
    result = parser.parse("")

    assert result.code == ""
    assert result.status == "EMPTY_OUTPUT"
    assert result.extraction_method == "none"


def test_whitespace_only_output(
    parser: CodeParser,
):
    result = parser.parse(
        " \n\t\n "
    )

    assert result.code == ""
    assert result.status == "EMPTY_OUTPUT"
    assert result.extraction_method == "none"


def test_empty_python_code_block(
    parser: CodeParser,
):
    raw_output = """
```python
```
"""

    result = parser.parse(raw_output)

    assert result.code == ""
    assert result.status == "EMPTY_CODE"
    assert result.extraction_method == "python_code_block"


def test_windows_newline_normalization(
    parser: CodeParser,
):
    raw_output = (
        "```python\r\n"
        "a = 1\r\n"
        "b = 2\r\n"
        "print(a + b)\r\n"
        "```"
    )

    result = parser.parse(raw_output)

    assert result.status == "SUCCESS"

    assert "\r" not in result.code

    assert result.code == (
        "a = 1\n"
        "b = 2\n"
        "print(a + b)"
    )


def test_remove_python_prefix_from_plain_text(
    parser: CodeParser,
):
    raw_output = """
python
x = 10
print(x)
"""

    result = parser.parse(raw_output)

    assert result.status == "SUCCESS"
    assert result.extraction_method == "plain_text"

    assert result.code == (
        "x = 10\n"
        "print(x)"
    )


def test_remove_py_prefix_from_plain_text(
    parser: CodeParser,
):
    raw_output = """
py
print(123)
"""

    result = parser.parse(raw_output)

    assert result.status == "SUCCESS"
    assert result.code == "print(123)"


def test_extract_returns_code_on_success(
    parser: CodeParser,
):
    raw_output = """
```python
print("ok")
```
"""

    code = parser.extract(raw_output)

    assert code == 'print("ok")'


def test_extract_raises_on_empty_output(
    parser: CodeParser,
):
    with pytest.raises(
        CodeParsingError,
        match="EMPTY_OUTPUT",
    ):
        parser.extract("")


def test_extract_raises_on_empty_code(
    parser: CodeParser,
):
    raw_output = """
```python
```
"""

    with pytest.raises(
        CodeParsingError,
        match="EMPTY_CODE",
    ):
        parser.extract(raw_output)


def test_non_string_input_raises_type_error(
    parser: CodeParser,
):
    with pytest.raises(TypeError):
        parser.parse(None)  # type: ignore[arg-type]


def test_parser_does_not_validate_python_syntax(
    parser: CodeParser,
):
    """
    Syntax validity belongs to the evaluator, not the parser.

    The parser should still successfully extract malformed Python
    as long as the output contains a non-empty code candidate.
    """

    raw_output = """
```python
def broken(
    print("invalid syntax")
```
"""

    result = parser.parse(raw_output)

    assert result.status == "SUCCESS"
    assert result.extraction_method == "python_code_block"

    assert "def broken(" in result.code
