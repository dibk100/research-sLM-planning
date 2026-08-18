# src/utils/feedback.py

from __future__ import annotations


def truncate_input_text(
    text: str,
    max_chars: int | None,
) -> str:
    """
    Truncate an oversized test input while preserving
    both the prefix and suffix.

    The returned string is guaranteed to have at most
    max_chars characters.

    If max_chars is None, truncation is disabled.
    """

    if max_chars is None:
        return text

    if max_chars <= 0:
        raise ValueError(
            "max_chars must be greater than 0."
        )

    if len(text) <= max_chars:
        return text

    marker = "\n...[TEST INPUT TRUNCATED]...\n"

    available_chars = (
        max_chars - len(marker)
    )

    if available_chars <= 0:
        raise ValueError(
            "max_chars is too small for "
            "the truncation marker."
        )

    head_chars = (
        available_chars // 2
    )

    tail_chars = (
        available_chars - head_chars
    )

    return (
        text[:head_chars]
        + marker
        + text[-tail_chars:]
    )