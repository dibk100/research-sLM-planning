"""
모델 출력에서 실행 가능한 파이썬 코드 추출. 

모델에게 코드만 출력하라고 지시해도 Markdown fence가 포함될 수 있으므로, 이를 제거하고 순수한 코드만 추출하는 기능을 제공하는 역할

아래를 기록하도록 하기 :
- raw_output
- extracted_code

code extraction 오류를 나중에 확인할 수 있도록, 모델 출력이 비어있거나, 코드 블록이 없거나, 코드 블록이 비어있을 경우 ValueError를 발생시킴
"""
from __future__ import annotations

import re


class CodeExtractionError(ValueError):
    """모델 출력에서 유효한 코드를 추출하지 못한 경우."""


class CodeExtractor:
    """LLM raw output에서 Python 코드만 추출한다."""

    PYTHON_CODE_BLOCK_PATTERN = re.compile(
        r"```(?:python|py)\s*\n?(.*?)```",
        flags=re.DOTALL | re.IGNORECASE,
    )

    GENERIC_CODE_BLOCK_PATTERN = re.compile(
        r"```\s*\n?(.*?)```",
        flags=re.DOTALL,
    )

    def extract(self, raw_output: str) -> str:
        """모델 출력에서 가장 적절한 Python 코드 블록을 반환한다."""
        if not isinstance(raw_output, str):
            raise TypeError(
                "raw_output must be a string, "
                f"got {type(raw_output).__name__}."
            )

        text = raw_output.strip()

        if not text:
            raise CodeExtractionError(
                "Model output is empty."
            )

        code = self._extract_python_code_block(text)

        if code is None:
            code = self._extract_generic_code_block(text)

        if code is None:
            code = self._extract_plain_text(text)

        code = self._clean_code(code)

        if not code:
            raise CodeExtractionError(
                "Extracted code is empty."
            )

        return code

    def _extract_python_code_block(
        self,
        text: str,
    ) -> str | None:
        """python 또는 py로 표시된 코드 블록을 우선 추출한다."""
        matches = self.PYTHON_CODE_BLOCK_PATTERN.findall(text)

        if not matches:
            return None

        return self._select_best_block(matches)

    def _extract_generic_code_block(
        self,
        text: str,
    ) -> str | None:
        """언어명이 없는 Markdown 코드 블록을 추출한다."""
        matches = self.GENERIC_CODE_BLOCK_PATTERN.findall(text)

        if not matches:
            return None

        return self._select_best_block(matches)

    @staticmethod
    def _select_best_block(
        blocks: list[str],
    ) -> str:
        """여러 코드 블록 중 가장 긴 블록을 선택한다."""
        return max(
            blocks,
            key=lambda block: len(block.strip()),
        )

    @staticmethod
    def _extract_plain_text(
        text: str,
    ) -> str:
        """코드 블록이 없으면 전체 출력을 코드 후보로 사용한다."""
        return text

    @staticmethod
    def _clean_code(
        code: str,
    ) -> str:
        """잔여 Markdown 및 불필요한 공백을 제거한다."""
        cleaned = code.strip()

        if cleaned.lower().startswith("python\n"):
            cleaned = cleaned[len("python\n"):].lstrip()

        if cleaned.lower().startswith("py\n"):
            cleaned = cleaned[len("py\n"):].lstrip()

        cleaned = cleaned.replace("\r\n", "\n")
        cleaned = cleaned.replace("\r", "\n")

        return cleaned.strip()