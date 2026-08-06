"""
모델 로딩 및 텍스트 생성 래퍼.

초기 설정 :
- Student model: Qwen2.5-Coder-3B-Instruct
- temperature: 0.0
- do_sample: false
- max_new_tokens: 1024

"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.schemas import GenerationOutput


class ModelGenerator:
    def __init__(
        self,
        model_name_or_path: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        self.model_name_or_path = model_name_or_path

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerationOutput:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
        ).to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }

        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        start_time = time.perf_counter()

        outputs = self.model.generate(
            **inputs,
            **generation_kwargs,
        )

        generation_time = time.perf_counter() - start_time

        generated_ids = outputs[0, inputs["input_ids"].shape[1] :]
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return GenerationOutput(
            text=generated_text,
            prompt_tokens=inputs["input_ids"].shape[1],
            completion_tokens=generated_ids.shape[0],
            generation_time=generation_time,
        )