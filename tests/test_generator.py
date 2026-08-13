# tests/test_generator.py

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.generator import ModelGenerator

from src.models.model_adapter import ModelAdapter
from src.schemas import GenerationOutput


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()

    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2

    tokenizer.apply_chat_template.return_value = (
        "<formatted prompt>"
    )

    tokenizer.return_value = {
        "input_ids": torch.tensor(
            [[10, 11, 12]],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1]],
            dtype=torch.long,
        ),
    }

    tokenizer.decode.return_value = (
        '```python\nprint("hello")\n```'
    )

    return tokenizer


@pytest.fixture
def mock_model():
    model = MagicMock()

    parameter = MagicMock()
    parameter.device = torch.device("cpu")

    model.parameters.return_value = iter(
        [parameter]
    )

    # prompt tokens = 3
    # generated tokens = 3
    model.generate.return_value = torch.tensor(
        [[10, 11, 12, 20, 21, 22]],
        dtype=torch.long,
    )

    return model


@pytest.fixture
def generator(
    mock_tokenizer,
    mock_model,
):
    """
    Construct ModelGenerator without loading a real HF model.
    """

    with (
        patch(
            "src.models.generator."
            "AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "src.models.generator."
            "AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ),
    ):
        generator = ModelGenerator(
            "mock-model",
            dtype="bfloat16",
            device_map="auto",
            trust_remote_code=False,
        )

    return generator



def test_invalid_dtype_raises():
    with pytest.raises(
        ValueError,
        match="Unsupported dtype",
    ):
        ModelGenerator(
            "mock-model",
            dtype="invalid-dtype",
        )


def test_tokenizer_and_model_loaded():
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 2

    mock_model = MagicMock()

    with (
        patch(
            "src.models.generator."
            "AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ) as tokenizer_loader,
        patch(
            "src.models.generator."
            "AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ) as model_loader,
    ):
        ModelGenerator(
            "test-model",
            dtype="bfloat16",
            device_map="auto",
            trust_remote_code=False,
        )

    tokenizer_loader.assert_called_once_with(
        "test-model",
        trust_remote_code=False,
    )

    model_loader.assert_called_once_with(
        "test-model",
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )

    mock_model.eval.assert_called_once()


def test_pad_token_falls_back_to_eos():
    tokenizer = MagicMock()

    tokenizer.pad_token_id = None
    tokenizer.eos_token_id = 42

    model = MagicMock()

    with (
        patch(
            "src.models.generator."
            "AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ),
        patch(
            "src.models.generator."
            "AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ),
    ):
        generator = ModelGenerator(
            "mock-model",
        )

    assert generator.tokenizer.pad_token_id == 42


def test_default_adapter_created(
    generator: ModelGenerator,
):
    assert isinstance(
        generator.adapter,
        ModelAdapter,
    )


def test_custom_adapter_is_used(
    mock_tokenizer,
    mock_model,
):
    adapter = MagicMock(
        spec=ModelAdapter
    )

    with (
        patch(
            "src.models.generator."
            "AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "src.models.generator."
            "AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ),
    ):
        generator = ModelGenerator(
            "mock-model",
            adapter=adapter,
        )

    assert generator.adapter is adapter


def test_build_chat_prompt_delegates_to_adapter(
    generator: ModelGenerator,
):
    adapter = MagicMock(
        spec=ModelAdapter
    )

    adapter.build_chat_prompt.return_value = (
        "<adapter formatted>"
    )

    generator.adapter = adapter

    formatted = generator.build_chat_prompt(
        "Solve this problem.",
        system_prompt="You are a coder.",
    )

    assert formatted == "<adapter formatted>"

    adapter.build_chat_prompt.assert_called_once_with(
        tokenizer=generator.tokenizer,
        user_prompt="Solve this problem.",
        system_prompt="You are a coder.",
    )


def test_generate_returns_generation_output(
    generator: ModelGenerator,
):
    result = generator.generate(
        "Write Python code.",
        max_new_tokens=128,
        temperature=0.0,
    )

    assert isinstance(
        result,
        GenerationOutput,
    )

    assert result.text == (
        '```python\nprint("hello")\n```'
    )

    assert result.prompt_tokens == 3
    assert result.completion_tokens == 3
    assert result.generation_time >= 0.0


def test_generate_preserves_raw_completion(
    generator: ModelGenerator,
):
    """
    Generator must not perform code parsing.

    Markdown fences must remain in GenerationOutput.text.
    """

    result = generator.generate(
        "Write code.",
    )

    assert result.text.startswith(
        "```python"
    )

    assert result.text.endswith(
        "```"
    )


def test_generate_uses_greedy_settings(
    generator: ModelGenerator,
):
    generator.generate(
        "Write code.",
        max_new_tokens=256,
        temperature=0.0,
        top_p=0.95,
    )

    kwargs = (
        generator.model.generate.call_args.kwargs
    )

    assert kwargs["max_new_tokens"] == 256
    assert kwargs["do_sample"] is False

    assert (
        "temperature" not in kwargs
    )

    assert "top_p" not in kwargs


def test_generate_uses_sampling_settings(
    generator: ModelGenerator,
):
    generator.generate(
        "Write code.",
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )

    kwargs = (
        generator.model.generate.call_args.kwargs
    )

    assert kwargs["max_new_tokens"] == 512
    assert kwargs["do_sample"] is True

    assert kwargs["temperature"] == pytest.approx(
        0.7
    )

    assert kwargs["top_p"] == pytest.approx(
        0.95
    )


@pytest.mark.parametrize(
    "max_new_tokens",
    [
        0,
        -1,
        -100,
    ],
)
def test_invalid_max_new_tokens(
    generator: ModelGenerator,
    max_new_tokens: int,
):
    with pytest.raises(
        ValueError,
        match="max_new_tokens",
    ):
        generator.generate(
            "test",
            max_new_tokens=max_new_tokens,
        )


def test_negative_temperature_raises(
    generator: ModelGenerator,
):
    with pytest.raises(
        ValueError,
        match="temperature",
    ):
        generator.generate(
            "test",
            temperature=-0.1,
        )


@pytest.mark.parametrize(
    "top_p",
    [
        0.0,
        -0.1,
        1.1,
        2.0,
    ],
)
def test_invalid_top_p_raises(
    generator: ModelGenerator,
    top_p: float,
):
    with pytest.raises(
        ValueError,
        match="top_p",
    ):
        generator.generate(
            "test",
            temperature=0.7,
            top_p=top_p,
        )


def test_top_p_one_is_valid(
    generator: ModelGenerator,
):
    generator.generate(
        "test",
        temperature=0.7,
        top_p=1.0,
    )

    kwargs = (
        generator.model.generate.call_args.kwargs
    )

    assert kwargs["top_p"] == 1.0


def test_input_is_moved_to_model_device(
    generator: ModelGenerator,
):
    generator.generate(
        "test",
    )

    kwargs = (
        generator.model.generate.call_args.kwargs
    )

    assert "input_ids" in kwargs
    assert "attention_mask" in kwargs

    assert (
        kwargs["input_ids"].device.type
        == "cpu"
    )


def test_generation_token_count_excludes_prompt(
    generator: ModelGenerator,
):
    """
    Output sequence contains:

        prompt:    [10, 11, 12]
        completion:[20, 21, 22]

    completion_tokens must therefore be 3.
    """

    result = generator.generate(
        "test",
    )

    assert result.prompt_tokens == 3
    assert result.completion_tokens == 3


def test_tokenizer_decode_receives_only_generated_tokens(
    generator: ModelGenerator,
):
    generator.generate(
        "test",
    )

    decode_args = (
        generator.tokenizer.decode.call_args
    )

    generated_ids = decode_args.args[0]

    assert torch.equal(
        generated_ids,
        torch.tensor(
            [20, 21, 22],
            dtype=torch.long,
        ),
    )

    assert (
        decode_args.kwargs[
            "skip_special_tokens"
        ]
        is True
    )