# src/models/model_adapter.py

from __future__ import annotations

from typing import Any


class ModelAdapter:
    """
    Default adapter for chat-based Hugging Face models.
    """

    def build_chat_prompt(
        self,
        tokenizer: Any,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        if hasattr(
            tokenizer,
            "apply_chat_template",
        ):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return user_prompt