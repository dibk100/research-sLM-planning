"""
모델 로딩 및 텍스트 생성 래퍼.

초기 설정 :
- Student model: Qwen2.5-Coder-3B-Instruct
- temperature: 0.0
- do_sample: false
- max_new_tokens: 1024

LiveCodeBench 문제를 입력받아 코드 생성을 위한 raw text를 반환한다.

"""
from __future__ import annotations

import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.schemas import GenerationOutput


DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class ModelGenerator:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        if dtype not in DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype: {dtype}. "
                f"Choose from {list(DTYPE_MAP.keys())}."
            )

        self.model_name_or_path = model_name_or_path
        self.dtype = dtype
        self.device_map = device_map

        print(f"[ModelGenerator] loading tokenizer: {model_name_or_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

        print(f"[ModelGenerator] loading model: {model_name_or_path}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=DTYPE_MAP[dtype],                   # torch_dtype -> dtype (Transformers 최신 버전으로 수정)
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

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return user_prompt

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
        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than 0."
            )

        if temperature < 0:
            raise ValueError(
                "temperature must be greater than or equal to 0."
            )
            
        if not 0.0 < top_p <= 1.0:
            raise ValueError(
                f"top_p must be in (0, 1], got {top_p}."
            )

        formatted_prompt = self.build_chat_prompt(
            user_prompt=prompt,
            system_prompt=system_prompt,
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
        )

        model_device = next(self.model.parameters()).device

        inputs = {
            key: value.to(model_device)
            for key, value in inputs.items()
        }

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        start_time = time.perf_counter()

        outputs = self.model.generate(
            **inputs,
            **generation_kwargs,
        )

        generation_time = (
            time.perf_counter() - start_time
        )

        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0, prompt_length:]

        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return GenerationOutput(
            text=generated_text.strip(),
            prompt_tokens=prompt_length,
            completion_tokens=generated_ids.shape[0],
            generation_time=generation_time,
        )