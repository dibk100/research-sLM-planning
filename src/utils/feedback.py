from __future__ import annotations


def truncate_input_text(
    text: str,
    tokenizer,
    max_tokens: int | None,
) -> str:
    if max_tokens is None:
        return text

    if max_tokens <= 0:
        raise ValueError(
            "max_tokens must be greater than 0."
        )

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if len(token_ids) <= max_tokens:
        return text

    head_tokens = max_tokens // 2
    tail_tokens = max_tokens - head_tokens

    head_ids = token_ids[:head_tokens]
    tail_ids = token_ids[-tail_tokens:]

    head_text = tokenizer.decode(
        head_ids,
        skip_special_tokens=True,
    )

    tail_text = tokenizer.decode(
        tail_ids,
        skip_special_tokens=True,
    )

    return (
        head_text
        + "\n...[TEST INPUT TRUNCATED]...\n"
        + tail_text
    )