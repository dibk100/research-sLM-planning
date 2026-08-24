"""
GPU에서 frozen coder inference만 담당
prompt를 받아 frozen model로 completion을 생성하는 GPU inference service

Frozen downstream coder inference worker.

The model is loaded once, but does not permanently occupy GPU memory.

Lifecycle
---------
init_model()
    -> load frozen coder
    -> move model to CPU
    -> GPU memory released

wake_up()
    -> move frozen coder to CUDA

generate_code()
    -> deterministic code generation

sleep()
    -> move frozen coder back to CPU
    -> release CUDA cache

This allows the single GPU to be shared with:
- trainable planner
- vLLM planner rollout
- frozen downstream coder
"""

# phase4_method_discovery/vanilla_planning_rlvr/workers/frozen_coder_worker.py

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import torch
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
    Frozen downstream coder inference service.

    The model is loaded once and kept frozen.

    Unlike the previous implementation, the coder is not permanently
    resident on GPU. It can be moved between CPU and CUDA so that the
    single RTX 5090 can be shared with the planner/FSDP/vLLM stack.

    Ray actors execute methods serially by default, so wake/generate/sleep
    calls on this worker are serialized.
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

        self.device: str = "cpu"

    # ==================================================================
    # Internal helpers
    # ==================================================================

    @staticmethod
    def _clear_cuda_cache() -> None:
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _require_initialized(self) -> ModelGenerator:
        if self.model is None:
            raise RuntimeError(
                "Frozen coder is not initialized. "
                "Call init_model() first."
            )

        return self.model

    # ==================================================================
    # Model lifecycle
    # ==================================================================

    def init_model(self) -> dict[str, Any]:
        """
        Load the frozen coder exactly once.

        Important
        ---------
        The model is initially loaded on CPU.

        GPU placement is controlled explicitly by wake_up()/sleep().
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

        # ------------------------------------------------------------------
        # Critical difference from previous implementation:
        #
        # Do NOT use device_map="auto".
        #
        # device_map="auto" immediately places the whole frozen coder on
        # the Ray actor's visible CUDA device and keeps it resident there.
        #
        # We deliberately load on CPU so the planner/FSDP/vLLM stack can
        # initialize and synchronize its weights first.
        # ------------------------------------------------------------------

        model = ModelGenerator(
            self.model_path,
            dtype=dtype,
            device_map="cpu",
        )

        model.model.eval()

        for parameter in model.model.parameters():
            parameter.requires_grad_(False)

        self.model = model
        self.device = "cpu"

        self._clear_cuda_cache()

        print(
            "[FrozenCoderWorker] model initialized on CPU."
        )
        print("=" * 80)
        print()

        return self.get_status()

    def wake_up(self) -> dict[str, Any]:
        """
        Move the frozen coder to CUDA.

        This must only be called when the planner/vLLM stack has released
        enough GPU memory for reward inference.
        """

        model = self._require_initialized()

        if self.device == "cuda":
            return self.get_status()

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available in FrozenCoderWorker."
            )

        print(
            "[FrozenCoderWorker] waking coder on GPU..."
        )

        model.model.to("cuda")

        model.model.eval()

        self.device = "cuda"

        torch.cuda.synchronize()

        print(
            "[FrozenCoderWorker] coder is on GPU."
        )

        return self.get_status()

    def sleep(self) -> dict[str, Any]:
        """
        Move the frozen coder back to CPU and release CUDA memory.
        """

        model = self._require_initialized()

        if self.device == "cpu":
            self._clear_cuda_cache()
            return self.get_status()

        print(
            "[FrozenCoderWorker] moving coder to CPU..."
        )

        model.model.to("cpu")

        self.device = "cpu"

        self._clear_cuda_cache()

        print(
            "[FrozenCoderWorker] coder is on CPU."
        )

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

        wake_up() must be called before generation.
        """

        model = self._require_initialized()

        if self.device != "cuda":
            raise RuntimeError(
                "Frozen coder is sleeping on CPU. "
                "Call wake_up() before generate_code()."
            )

        if (
            not isinstance(prompt, str)
            or not prompt.strip()
        ):
            raise ValueError(
                "prompt must be a non-empty string."
            )

        with torch.inference_mode():
            generation = model.generate(
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
        cuda_allocated_gb = 0.0
        cuda_reserved_gb = 0.0

        if torch.cuda.is_available():
            cuda_allocated_gb = (
                torch.cuda.memory_allocated()
                / (1024 ** 3)
            )
            cuda_reserved_gb = (
                torch.cuda.memory_reserved()
                / (1024 ** 3)
            )

        return {
            "initialized": (
                self.model is not None
            ),
            "device": self.device,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "cuda_allocated_gb": round(
                cuda_allocated_gb,
                3,
            ),
            "cuda_reserved_gb": round(
                cuda_reserved_gb,
                3,
            ),
        }