"""
verl에서 GRPO를 구현 및 실행함.

PlanningRewardManager 클래스는 아래 흐름으로 reward환경을 verl에 연결하는 역할

planner rollout
    ↓
response token decode
    ↓
PLAN 문자열
    ↓
compute_score()
    ↓
frozen coder
    ↓
TACO execution
    ↓
0 / 1

PYTHONPATH="$HOME/workspace/project_sLM_planning:$HOME/workspace/verl" \
python \
  phase4_method_discovery/vanilla_planning_rlvr/scripts/smoke_test_planning_reward_manager.py \
  --input /mnt/hdd/project_sLM_planning/data/deepcoder_taco/processed/vanilla_planning_rlvr/train.parquet \
  --row-index 0 \
  --show-response


"""
# phase4_method_discovery/vanilla_planning_rlvr/reward/planning_reward_manager.py

# phase4_method_discovery/vanilla_planning_rlvr/reward/planning_reward_manager.py

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import (
    RewardManagerBase,
)

from phase4_method_discovery.vanilla_planning_rlvr.reward.planning_execution_reward import (
    initialize_reward_runtime,
)
from src.models.generator import ModelGenerator


# ======================================================================
# Project paths
# ======================================================================

THIS_FILE = Path(__file__).resolve()

PROJECT_ROOT = THIS_FILE.parents[3]

DEFAULT_EXPERIMENT_CONFIG_PATH = (
    PROJECT_ROOT
    / "phase4_method_discovery"
    / "vanilla_planning_rlvr"
    / "configs"
    / "vanilla_planning_rlvr_qwen25coder3b.yaml"
)

DEFAULT_CODE_PROMPT_PATH = (
    PROJECT_ROOT
    / "prompt_templates"
    / "self_plan_code.txt"
)


# ======================================================================
# Experiment config loading
# ======================================================================

def load_planning_rlvr_config(
    path: str | Path | None = None,
) -> DictConfig:
    """
    Load the project-level Vanilla Planning-RLVR configuration.

    This YAML is intentionally separate from verl's internal Hydra
    configuration. It is the source of truth for the research-specific
    planner/coder/reward settings.

    Default:
        phase4_method_discovery/
        vanilla_planning_rlvr/
        configs/
        vanilla_planning_rlvr_qwen25coder3b.yaml
    """

    config_path = (
        Path(path)
        if path is not None
        else DEFAULT_EXPERIMENT_CONFIG_PATH
    )

    if not config_path.is_absolute():
        config_path = (
            PROJECT_ROOT
            / config_path
        )

    if not config_path.exists():
        raise FileNotFoundError(
            "Planning-RLVR experiment config not found: "
            f"{config_path}"
        )

    config = OmegaConf.load(
        config_path
    )

    _validate_planning_rlvr_config(
        config
    )

    return config


def _validate_planning_rlvr_config(
    config: DictConfig,
) -> None:
    """
    Validate the subset of the project config required by the
    reward manager.
    """

    required_top_level = (
        "experiment",
        "planner",
        "coder",
        "grpo",
        "reward",
        "data",
    )

    for key in required_top_level:
        if key not in config:
            raise KeyError(
                f"Missing config section: {key}"
            )

    # ------------------------------------------------------------------
    # Coder
    # ------------------------------------------------------------------

    coder = config.coder

    if not str(
        coder.model_path
    ).strip():
        raise ValueError(
            "coder.model_path must not be empty."
        )

    if not bool(
        coder.frozen
    ):
        raise ValueError(
            "Vanilla Planning-RLVR requires "
            "coder.frozen=true."
        )

    if str(coder.dtype) not in {
        "float16",
        "bfloat16",
        "float32",
    }:
        raise ValueError(
            "Unsupported coder.dtype: "
            f"{coder.dtype!r}"
        )

    generation = (
        coder.generation
    )

    if int(
        generation.max_new_tokens
    ) <= 0:
        raise ValueError(
            "coder.generation.max_new_tokens "
            "must be > 0."
        )

    if float(
        generation.temperature
    ) < 0:
        raise ValueError(
            "coder.generation.temperature "
            "must be >= 0."
        )

    if not (
        0
        < float(
            generation.top_p
        )
        <= 1
    ):
        raise ValueError(
            "coder.generation.top_p "
            "must be in (0, 1]."
        )

    # Frozen coder should be deterministic in the baseline.
    if bool(
        generation.do_sample
    ):
        raise ValueError(
            "Vanilla Planning-RLVR baseline expects "
            "coder.generation.do_sample=false."
        )

    if float(
        generation.temperature
    ) != 0.0:
        raise ValueError(
            "Vanilla Planning-RLVR baseline expects "
            "coder.generation.temperature=0.0."
        )

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    reward = config.reward

    if str(
        reward.type
    ) != "binary_execution":
        raise ValueError(
            "Unsupported reward.type: "
            f"{reward.type!r}"
        )

    if int(
        reward.max_tests
    ) <= 0:
        raise ValueError(
            "reward.max_tests must be > 0."
        )

    if int(
        reward.timeout_seconds
    ) <= 0:
        raise ValueError(
            "reward.timeout_seconds must be > 0."
        )

    if int(
        reward.num_workers
    ) <= 0:
        raise ValueError(
            "reward.num_workers must be > 0."
        )


# ======================================================================
# Custom RewardManager
# ======================================================================

class PlanningRewardManager(
    RewardManagerBase
):
    """
    verl V1 reward manager for Vanilla Planning-RLVR.

    Research trajectory
    -------------------

        coding problem x
            ->
        trainable planner
            ->
        plan P
            ->
        frozen coder
            ->
        generated code C
            ->
        TACO execution
            ->
        binary reward R(C) in {0, 1}

    Responsibilities
    ----------------
    1. Load the research-level YAML configuration.
    2. Load one frozen coder per reward-worker process.
    3. Initialize planning_execution_reward runtime once.
    4. Decode planner responses exactly as verl does.
    5. Serialize reward execution inside each reward worker.
    6. Return scalar reward to verl.

    GRPO itself is entirely handled by verl.
    """

    # ------------------------------------------------------------------
    # Process-local class state
    # ------------------------------------------------------------------

    _class_initialized: bool = False

    _frozen_coder: (
        ModelGenerator | None
    ) = None

    _experiment_config: (
        DictConfig | None
    ) = None

    # ==================================================================
    # Construction
    # ==================================================================

    def __init__(
        self,
        config: DictConfig,
        tokenizer: Any,
        compute_score: Any,
        reward_router_address: str | None = None,
        reward_model_tokenizer: Any = None,
    ) -> None:
        """
        Construct one reward-manager instance.

        RewardManagerBase.__init__() calls init_class(), which performs
        process-local frozen-coder initialization.
        """

        if compute_score is None:
            raise ValueError(
                "PlanningRewardManager requires "
                "a custom compute_score function."
            )

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            compute_score=compute_score,
        )

        self.compute_score = (
            compute_score
        )

        self.is_async_reward_score = (
            inspect.iscoroutinefunction(
                self.compute_score
            )
        )

        self.reward_router_address = (
            reward_router_address
        )

        self.reward_model_tokenizer = (
            reward_model_tokenizer
        )

        # verl may schedule several run_single() coroutines concurrently.
        # The frozen HF coder is intentionally accessed serially.
        self._reward_lock = (
            asyncio.Lock()
        )

    # ==================================================================
    # Process-local runtime initialization
    # ==================================================================

    @classmethod
    def init_class(
        cls,
        config: DictConfig,
        tokenizer: Any,
    ) -> None:
        """
        Initialize the Planning-RLVR reward runtime once per process.

        Important:
            reward.num_workers=N means there may be N independent
            reward-worker processes, and therefore up to N frozen
            coder instances.

        Initial single-GPU experiments should use:
            reward.num_workers = 1
        """

        del config
        del tokenizer

        if cls._class_initialized:
            return

        # --------------------------------------------------------------
        # 1. Load research config
        # --------------------------------------------------------------

        experiment_config = (
            load_planning_rlvr_config()
        )

        cls._experiment_config = (
            experiment_config
        )

        coder_cfg = (
            experiment_config.coder
        )

        coder_generation_cfg = (
            coder_cfg.generation
        )

        reward_cfg = (
            experiment_config.reward
        )

        # --------------------------------------------------------------
        # 2. Resolve coder settings
        # --------------------------------------------------------------

        coder_model_path = str(
            coder_cfg.model_path
        )

        coder_dtype = str(
            coder_cfg.dtype
        )

        coder_max_new_tokens = int(
            coder_generation_cfg.max_new_tokens
        )

        coder_temperature = float(
            coder_generation_cfg.temperature
        )

        coder_top_p = float(
            coder_generation_cfg.top_p
        )

        max_reward_tests = int(
            reward_cfg.max_tests
        )

        timeout_seconds = int(
            reward_cfg.timeout_seconds
        )

        # Current ModelGenerator supports device_map directly.
        # Keep "auto" fixed for the initial single-GPU baseline.
        coder_device_map = "auto"

        # --------------------------------------------------------------
        # 3. Diagnostics
        # --------------------------------------------------------------

        print()
        print("=" * 80)

        print(
            "[PlanningRewardManager] "
            "Initializing reward runtime"
        )

        print("=" * 80)

        print(
            f"experiment        : "
            f"{experiment_config.experiment.name}"
        )

        print(
            f"frozen coder      : "
            f"{coder_model_path}"
        )

        print(
            f"coder dtype       : "
            f"{coder_dtype}"
        )

        print(
            f"coder device_map  : "
            f"{coder_device_map}"
        )

        print(
            f"coder max tokens  : "
            f"{coder_max_new_tokens}"
        )

        print(
            f"coder temperature : "
            f"{coder_temperature}"
        )

        print(
            f"coder top_p       : "
            f"{coder_top_p}"
        )

        print(
            f"reward type       : "
            f"{reward_cfg.type}"
        )

        print(
            f"reward tests      : "
            f"{max_reward_tests}"
        )

        print(
            f"execution timeout : "
            f"{timeout_seconds}s"
        )

        print(
            f"reward workers    : "
            f"{reward_cfg.num_workers}"
        )

        print(
            f"code prompt       : "
            f"{DEFAULT_CODE_PROMPT_PATH}"
        )

        # --------------------------------------------------------------
        # 4. Load frozen coder
        # --------------------------------------------------------------

        frozen_coder = ModelGenerator(
            coder_model_path,
            dtype=coder_dtype,
            device_map=(
                coder_device_map
            ),
        )

        frozen_coder.model.eval()

        for parameter in (
            frozen_coder
            .model
            .parameters()
        ):
            parameter.requires_grad_(
                False
            )

        cls._frozen_coder = (
            frozen_coder
        )

        # --------------------------------------------------------------
        # 5. Initialize production reward pipeline
        # --------------------------------------------------------------

        initialize_reward_runtime(
            frozen_coder=(
                frozen_coder
            ),
            code_prompt_path=(
                DEFAULT_CODE_PROMPT_PATH
            ),
            timeout_seconds=(
                timeout_seconds
            ),
            debug=False,
            coder_max_new_tokens=(
                coder_max_new_tokens
            ),
            coder_temperature=(
                coder_temperature
            ),
            coder_top_p=(
                coder_top_p
            ),
            max_reward_tests=(
                max_reward_tests
            ),
        )

        # Set only after all initialization succeeds.
        cls._class_initialized = True

        print(
            "[PlanningRewardManager] "
            "reward runtime initialized."
        )

        print("=" * 80)
        print()

    # ==================================================================
    # Single rollout reward
    # ==================================================================

    async def run_single(
        self,
        data: DataProto,
    ) -> dict[str, Any]:
        """
        Compute reward for one planner rollout.

        Input
        -----
        data:
            verl DataProto containing one or more sequences.

        The final sequence is used, following verl's experimental
        NaiveRewardManager convention.

        Output
        ------
        {
            "reward_score": float,
            "reward_extra_info": dict
        }
        """

        # --------------------------------------------------------------
        # 1. Use final sequence only
        # --------------------------------------------------------------

        data = data[-1:]

        data_item = data[0]

        # --------------------------------------------------------------
        # 2. Extract valid planner response
        # --------------------------------------------------------------

        response_ids = (
            data_item.batch[
                "responses"
            ]
        )

        response_length = (
            response_ids.shape[-1]
        )

        valid_response_length = (
            data_item.batch[
                "attention_mask"
            ][
                -response_length:
            ]
            .sum()
        )

        if hasattr(
            valid_response_length,
            "item",
        ):
            valid_response_length_int = int(
                valid_response_length.item()
            )
        else:
            valid_response_length_int = int(
                valid_response_length
            )

        if (
            valid_response_length_int
            <= 0
        ):
            response_str = ""

        else:
            valid_response_ids = (
                response_ids[
                    :valid_response_length_int
                ]
            )

            response_str = (
                await self.loop.run_in_executor(
                    None,
                    lambda: (
                        self.tokenizer.decode(
                            valid_response_ids,
                            skip_special_tokens=True,
                        )
                    ),
                )
            )

        # --------------------------------------------------------------
        # 3. Extract dataset fields
        # --------------------------------------------------------------

        data_source = (
            data_item
            .non_tensor_batch[
                "data_source"
            ]
        )

        ground_truth = (
            data_item
            .non_tensor_batch[
                "reward_model"
            ][
                "ground_truth"
            ]
        )

        raw_extra_info = (
            data_item
            .non_tensor_batch
            .get(
                "extra_info",
                {},
            )
        )

        extra_info = dict(
            raw_extra_info
        )

        # --------------------------------------------------------------
        # 4. Preserve verl auxiliary fields
        # --------------------------------------------------------------

        tool_extra_fields = (
            data_item
            .non_tensor_batch
            .get(
                "tool_extra_fields",
                None,
            )
        )

        if (
            tool_extra_fields
            is not None
        ):
            extra_info.update(
                tool_extra_fields.items()
            )

        num_turns = (
            data_item
            .non_tensor_batch
            .get(
                "__num_turns__",
                None,
            )
        )

        rollout_reward_scores = (
            data_item
            .non_tensor_batch
            .get(
                "reward_scores",
                {},
            )
        )

        extra_info[
            "num_turns"
        ] = num_turns

        extra_info[
            "rollout_reward_scores"
        ] = (
            rollout_reward_scores
        )

        # --------------------------------------------------------------
        # 5. Optional verl reward-router fields
        # --------------------------------------------------------------

        if (
            self.reward_router_address
            is not None
        ):
            extra_reward_kwargs = {
                "reward_router_address": (
                    self.reward_router_address
                ),

                "reward_model_tokenizer": (
                    self.reward_model_tokenizer
                ),
            }

        else:
            extra_reward_kwargs = {}

        # --------------------------------------------------------------
        # 6. Serialized Planning-RLVR reward execution
        #
        # RewardLoopWorker can schedule multiple run_single() calls
        # concurrently. Only one is allowed to use the process-local
        # frozen coder at a time.
        # --------------------------------------------------------------

        async with self._reward_lock:

            if (
                self.is_async_reward_score
            ):
                result = (
                    await self.compute_score(
                        data_source=(
                            data_source
                        ),
                        solution_str=(
                            response_str
                        ),
                        ground_truth=(
                            ground_truth
                        ),
                        extra_info=(
                            extra_info
                        ),
                        **extra_reward_kwargs,
                    )
                )

            else:
                result = (
                    await self.loop.run_in_executor(
                        None,
                        lambda: (
                            self.compute_score(
                                data_source=(
                                    data_source
                                ),
                                solution_str=(
                                    response_str
                                ),
                                ground_truth=(
                                    ground_truth
                                ),
                                extra_info=(
                                    extra_info
                                ),
                                **extra_reward_kwargs,
                            )
                        ),
                    )
                )

        # --------------------------------------------------------------
        # 7. Normalize reward
        # --------------------------------------------------------------

        reward_extra_info: dict[
            str,
            Any,
        ] = {}

        if isinstance(
            result,
            dict,
        ):
            if "score" not in result:
                raise KeyError(
                    "Dictionary reward result "
                    "must contain 'score'."
                )

            score = float(
                result[
                    "score"
                ]
            )

            for key, value in (
                result.items()
            ):
                reward_extra_info[
                    key
                ] = value

        else:
            score = float(
                result
            )

            # Same convention as verl's experimental
            # NaiveRewardManager.
            reward_extra_info[
                "acc"
            ] = score

        # --------------------------------------------------------------
        # 8. Vanilla Planning-RLVR requires binary reward
        # --------------------------------------------------------------

        if score not in {
            0.0,
            1.0,
        }:
            raise ValueError(
                "Vanilla Planning-RLVR expects "
                "binary reward, got "
                f"{score}."
            )

        return {
            "reward_score": score,

            "reward_extra_info": (
                reward_extra_info
            ),
        }