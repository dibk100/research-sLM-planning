"""

```python / ```py 코드 블록 우선
generic code block 차선
없으면 plain text 전체 사용
여러 블록이면 가장 긴 블록 선택
줄바꿈 정규화

"""
# src/parsing/code_parser.py

from __future__ import annotations

import re

from src.schemas import CodeParseResult


class CodeParsingError(ValueError):
    """Raised when valid code cannot be parsed from model output."""


class CodeParser:
    """
    Parse Python code from raw LLM output.

    Parsing priority:
    1. Python-labelled Markdown code block
    2. Generic Markdown code block
    3. Entire raw output as plain text
    """

    PYTHON_CODE_BLOCK_PATTERN = re.compile(
        r"```(?:python|py)\s*\n?(.*?)```",
        flags=re.DOTALL | re.IGNORECASE,
    )

    GENERIC_CODE_BLOCK_PATTERN = re.compile(
        r"```\s*\n?(.*?)```",
        flags=re.DOTALL,
    )

    def parse(
        self,
        raw_output: str,
    ) -> CodeParseResult:
        if not isinstance(raw_output, str):
            raise TypeError(
                "raw_output must be str, "
                f"got {type(raw_output).__name__}"
            )

        text = raw_output.strip()

        if not text:
            return CodeParseResult(
                code="",
                status="EMPTY_OUTPUT",
                extraction_method="none",
            )

        code = self._extract_python_code_block(text)

        if code is not None:
            extraction_method = "python_code_block"

        else:
            code = self._extract_generic_code_block(text)

            if code is not None:
                extraction_method = "generic_code_block"

            else:
                code = text
                extraction_method = "plain_text"

        code = self._clean_code(code)

        if not code:
            return CodeParseResult(
                code="",
                status="EMPTY_CODE",
                extraction_method=extraction_method,
            )

        return CodeParseResult(
            code=code,
            status="SUCCESS",
            extraction_method=extraction_method,
        )

    def extract(
        self,
        raw_output: str,
    ) -> str:
        """
        Compatibility helper for existing Phase 1 code.

        Raises CodeParsingError when parsing does not succeed.
        """

        result = self.parse(raw_output)

        if result.status != "SUCCESS":
            raise CodeParsingError(
                f"Code parsing failed: {result.status}"
            )

        return result.code

    def _extract_python_code_block(
        self,
        text: str,
    ) -> str | None:
        matches = self.PYTHON_CODE_BLOCK_PATTERN.findall(
            text
        )

        if not matches:
            return None

        return self._select_best_block(matches)

    def _extract_generic_code_block(
        self,
        text: str,
    ) -> str | None:
        matches = self.GENERIC_CODE_BLOCK_PATTERN.findall(
            text
        )

        if not matches:
            return None

        return self._select_best_block(matches)

    @staticmethod
    def _select_best_block(
        blocks: list[str],
    ) -> str:
        return max(
            blocks,
            key=lambda block: len(block.strip()),
        )

    @staticmethod
    def _clean_code(
        code: str,
    ) -> str:
        cleaned = code.strip()

        if cleaned.lower().startswith("python\n"):
            cleaned = cleaned[len("python\n"):].lstrip()

        if cleaned.lower().startswith("py\n"):
            cleaned = cleaned[len("py\n"):].lstrip()

        cleaned = cleaned.replace("\r\n", "\n")
        cleaned = cleaned.replace("\r", "\n")

        return cleaned.strip()