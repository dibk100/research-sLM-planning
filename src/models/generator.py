# src/models/generator.py

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.models.model_adapter import ModelAdapter
from src.schemas import GenerationOutput


DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class ModelGenerator:
    """
    Shared Hugging Face causal language model generator.

    Responsibilities:
    - load tokenizer/model
    - delegate chat prompt formatting to ModelAdapter
    - perform greedy or sampling-based generation
    - return raw model completion and generation statistics

    Code parsing and evaluation are handled by separate pipeline stages.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        adapter: ModelAdapter | None = None,
    ) -> None:
        if dtype not in DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype: {dtype}. "
                f"Choose from {list(DTYPE_MAP.keys())}."
            )

        self.model_name_or_path = model_name_or_path
        self.dtype = dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code

        self.adapter = adapter or ModelAdapter()

        print(
            "[ModelGenerator] loading tokenizer: "
            f"{model_name_or_path}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

        print(
            "[ModelGenerator] loading model: "
            f"{model_name_or_path}"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=DTYPE_MAP[dtype],
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id
            )

        print("[ModelGenerator] model loaded.")

    def build_chat_prompt(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """
        Delegate chat-template handling to the configured model adapter.
        """

        if not isinstance(user_prompt, str):
            raise TypeError(
                "user_prompt must be str, "
                f"got {type(user_prompt).__name__}"
            )

        if (
            system_prompt is not None
            and not isinstance(system_prompt, str)
        ):
            raise TypeError(
                "system_prompt must be str or None, "
                f"got {type(system_prompt).__name__}"
            )

        return self.adapter.build_chat_prompt(
            tokenizer=self.tokenizer,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> GenerationOutput:
        """
        Generate one raw completion.

        temperature == 0:
            greedy decoding

        temperature > 0:
            stochastic sampling using temperature/top_p

        GenerationOutput.text is intentionally not parsed or cleaned beyond
        stripping leading/trailing whitespace.
        """

        self._validate_generation_args(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        formatted_prompt = self.build_chat_prompt(
            user_prompt=prompt,
            system_prompt=system_prompt,
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
        )

        model_device = next(
            self.model.parameters()
        ).device

        inputs = {
            key: value.to(model_device)
            for key, value in inputs.items()
        }

        generation_kwargs = (
            self._build_generation_kwargs(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )

        start_time = time.perf_counter()

        outputs = self.model.generate(
            **inputs,
            **generation_kwargs,
        )

        generation_time = (
            time.perf_counter() - start_time
        )

        prompt_length = int(
            inputs["input_ids"].shape[1]
        )

        generated_ids = outputs[
            0,
            prompt_length:,
        ]

        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return GenerationOutput(
            text=generated_text.strip(),
            prompt_tokens=prompt_length,
            completion_tokens=int(
                generated_ids.shape[0]
            ),
            generation_time=generation_time,
        )

    def _build_generation_kwargs(
        self,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": (
                self.tokenizer.pad_token_id
            ),
            "eos_token_id": (
                self.tokenizer.eos_token_id
            ),
        }

        if temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p

        return kwargs

    @staticmethod
    def _validate_generation_args(
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than 0."
            )

        if temperature < 0:
            raise ValueError(
                "temperature must be greater than "
                "or equal to 0."
            )

        if not 0 < top_p <= 1:
            raise ValueError(
                "top_p must be in (0, 1]."
            )