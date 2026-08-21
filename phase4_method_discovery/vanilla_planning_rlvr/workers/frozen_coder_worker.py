"""
GPU에서 frozen coder inference만 담당
prompt를 받아 frozen model로 completion을 생성하는 GPU inference service

"""
# phase4_method_discovery/vanilla_planning_rlvr/workers/frozen_coder_worker.py

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.models.generator import ModelGenerator


# ======================================================================
# Paths
# ======================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "phase4_method_discovery"
    / "vanilla_planning_rlvr"
    / "configs"
    / "vanilla_planning_rlvr_qwen25coder3b.yaml"
)


# ======================================================================
# Config
# ======================================================================

def load_experiment_config(
    config_path: str | Path | None = None,
) -> DictConfig:
    path = (
        Path(config_path)
        if config_path is not None
        else DEFAULT_CONFIG_PATH
    )

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Planning-RLVR config not found: {path}"
        )

    config = OmegaConf.load(path)

    if "coder" not in config:
        raise KeyError(
            "Missing 'coder' section in experiment config."
        )

    coder_cfg = config.coder

    if not str(coder_cfg.model_path).strip():
        raise ValueError(
            "coder.model_path must not be empty."
        )

    if not bool(coder_cfg.frozen):
        raise ValueError(
            "FrozenCoderWorker requires coder.frozen=true."
        )

    generation_cfg = coder_cfg.generation

    if int(generation_cfg.max_new_tokens) <= 0:
        raise ValueError(
            "coder.generation.max_new_tokens must be > 0."
        )

    if float(generation_cfg.temperature) != 0.0:
        raise ValueError(
            "Vanilla Planning-RLVR baseline requires "
            "coder.generation.temperature=0.0."
        )

    if bool(generation_cfg.do_sample):
        raise ValueError(
            "Vanilla Planning-RLVR baseline requires "
            "coder.generation.do_sample=false."
        )

    if not (
        0.0
        < float(generation_cfg.top_p)
        <= 1.0
    ):
        raise ValueError(
            "coder.generation.top_p must be in (0, 1]."
        )

    return config


# ======================================================================
# Frozen coder Ray actor
# ======================================================================

class FrozenCoderWorker:
    """
    GPU inference service for the frozen downstream coder.

    Responsibilities
    ----------------
    - Load exactly one frozen coder model in this Ray actor.
    - Keep the model in eval mode.
    - Disable gradients.
    - Receive already-built code-generation prompts.
    - Return raw model completions and generation metadata.

    It intentionally does NOT know about:
    - ProblemExample
    - planning prompts
    - TACO tests
    - CodeParser
    - execution reward
    - GRPO

    Those remain outside this GPU worker.

    Ray actors execute methods serially by default, which also prevents
    multiple reward requests from concurrently calling generate() on
    the same Hugging Face model.
    """

    def __init__(
        self,
        config_path: str | None = None,
    ) -> None:
        self.config_path = (
            str(config_path)
            if config_path is not None
            else str(DEFAULT_CONFIG_PATH)
        )

        self.config = load_experiment_config(
            self.config_path
        )

        self.model: ModelGenerator | None = None

        self.model_path: str | None = None

        self.max_new_tokens: int = 0
        self.temperature: float = 0.0
        self.top_p: float = 1.0

    # ==================================================================
    # Model lifecycle
    # ==================================================================

    def init_model(self) -> dict[str, Any]:
        """
        Load the frozen coder exactly once.

        Returns diagnostic information so the trainer can verify
        which checkpoint/config was actually loaded.
        """

        if self.model is not None:
            return self.get_status()

        coder_cfg = self.config.coder
        generation_cfg = coder_cfg.generation

        self.model_path = str(
            coder_cfg.model_path
        )

        dtype = str(
            coder_cfg.dtype
        )

        self.max_new_tokens = int(
            generation_cfg.max_new_tokens
        )

        self.temperature = float(
            generation_cfg.temperature
        )

        self.top_p = float(
            generation_cfg.top_p
        )

        print()
        print("=" * 80)
        print(
            "[FrozenCoderWorker] Initializing frozen coder"
        )
        print("=" * 80)

        print(
            f"model           : {self.model_path}"
        )
        print(
            f"dtype           : {dtype}"
        )
        print(
            f"max_new_tokens  : {self.max_new_tokens}"
        )
        print(
            f"temperature     : {self.temperature}"
        )
        print(
            f"top_p           : {self.top_p}"
        )

        # Ray assigns CUDA_VISIBLE_DEVICES to this actor because the
        # actor is created with a fractional GPU resource inside the
        # global placement group.
        #
        # Therefore device_map="auto" sees only the GPU assigned to
        # this actor.
        model = ModelGenerator(
            self.model_path,
            dtype=dtype,
            device_map="auto",
        )

        model.model.eval()

        for parameter in model.model.parameters():
            parameter.requires_grad_(False)

        self.model = model

        print(
            "[FrozenCoderWorker] model initialized."
        )
        print("=" * 80)
        print()

        return self.get_status()

    # ==================================================================
    # Generation
    # ==================================================================

    def generate_code(
        self,
        prompt: str,
    ) -> dict[str, Any]:
        """
        Generate one deterministic plan-conditioned code completion.

        Parameters
        ----------
        prompt:
            Fully constructed Self-Plan -> Code prompt.

        Returns
        -------
        dict
            {
                "text": str,
                "prompt_tokens": int,
                "completion_tokens": int,
                "generation_time": float,
            }
        """

        if self.model is None:
            raise RuntimeError(
                "Frozen coder is not initialized. "
                "Call init_model() first."
            )

        if (
            not isinstance(prompt, str)
            or not prompt.strip()
        ):
            raise ValueError(
                "prompt must be a non-empty string."
            )

        generation = self.model.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        text = generation.text

        if not isinstance(text, str):
            raise TypeError(
                "ModelGenerator.generate().text "
                "must be str."
            )

        if not text.strip():
            raise RuntimeError(
                "Frozen coder returned empty output."
            )

        return {
            "text": text,
            "prompt_tokens": int(
                generation.prompt_tokens
            ),
            "completion_tokens": int(
                generation.completion_tokens
            ),
            "generation_time": float(
                generation.generation_time
            ),
        }

    # ==================================================================
    # Diagnostics
    # ==================================================================

    def get_status(self) -> dict[str, Any]:
        import os

        return {
            "initialized": (
                self.model is not None
            ),
            "model_path": self.model_path,
            "config_path": self.config_path,
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }