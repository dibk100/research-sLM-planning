"""
모델 출력에서 실행 가능한 파이썬 코드 추출. 

모델에게 코드만 출력하라고 지시해도 Markdown fence가 포함될 수 있으므로, 이를 제거하고 순수한 코드만 추출하는 기능을 제공하는 역할

아래를 기록하도록 하기 :
- raw_output
- extracted_code

code extraction 오류를 나중에 확인할 수 있도록, 모델 출력이 비어있거나, 코드 블록이 없거나, 코드 블록이 비어있을 경우 ValueError를 발생시킴
"""



import re

class CodeExtractor:
    PYTHON_BLOCK_PATTERN = re.compile(
        r"```python\s*(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )

    GENERIC_BLOCK_PATTERN = re.compile(
        r"```\s*(.*?)```",
        re.DOTALL,
    )

    def extract(self, text: str) -> str:
        if not text or not text.strip():
            raise ValueError("Model output is empty.")

        python_match = self.PYTHON_BLOCK_PATTERN.search(text)
        if python_match:
            return python_match.group(1).strip()

        generic_match = self.GENERIC_BLOCK_PATTERN.search(text)
        if generic_match:
            return generic_match.group(1).strip()

        code = text.strip()

        if not code:
            raise ValueError("Extracted code is empty.")

        return code