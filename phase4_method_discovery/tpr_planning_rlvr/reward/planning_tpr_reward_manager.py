# phase4_method_discovery/tpr_planning_rlvr/reward/planning_tpr_reward_manager.py

from __future__ import annotations

import asyncio
import inspect
import math
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import (
    RewardManagerBase,
)


class PlanningTPRRewardManager(
    RewardManagerBase
):
    """
    Custom verl V1 reward manager for TPR Planning-RLVR.

    Research trajectory
    -------------------

        coding problem
            ->
        trainable planner
            ->
        plan
            ->
        FrozenCoderWorker (Ray GPU actor)
            ->
        generated code
            ->
        DeepCoder/TACO non-fail-fast execution
            ->
        TPR reward [0, 1]

    Reward definition
    -----------------

        reward = passed_reward_tests / total_reward_tests

    Examples:

        0 / 15  -> 0.0
        1 / 15  -> 0.0667
        7 / 15  -> 0.4667
        15 / 15 -> 1.0

    Responsibilities
    ----------------
    This class does NOT load the frozen coder model.

    The frozen coder is initialized separately by PPOTrainer as a
    GPU-backed Ray actor and its ActorHandle is propagated through:

        PPOTrainer
            ->
        RewardLoopManager
            ->
        RewardLoopWorker
            ->
        load_reward_manager(...)
            ->
        PlanningTPRRewardManager

    PlanningTPRRewardManager is responsible only for:

    1. decoding the planner rollout response,
    2. collecting verl dataset metadata,
    3. forwarding the FrozenCoderWorker handle to compute_score(),
    4. normalizing the returned reward,
    5. enforcing continuous TPR reward in [0, 1].

    GRPO optimization itself remains entirely inside verl.
    """

    def __init__(
        self,
        config: Any,
        tokenizer: Any,
        compute_score: Any,
        reward_router_address: str | None = None,
        reward_model_tokenizer: Any = None,
        frozen_coder_handle: Any = None,
    ) -> None:
        """
        Initialize the PlanningTPRRewardManager.

        Parameters
        ----------
        config:
            verl PPO configuration.

        tokenizer:
            Planner tokenizer supplied by RewardLoopWorker.

        compute_score:
            Custom reward function loaded from
            planning_tpr_reward.py.

        reward_router_address:
            Optional verl reward-model router address.
            Not used in TPR Planning-RLVR.

        reward_model_tokenizer:
            Optional reward-model tokenizer.
            Not used in TPR Planning-RLVR.

        frozen_coder_handle:
            Ray ActorHandle for FrozenCoderWorker.
        """

        if compute_score is None:
            raise ValueError(
                "PlanningTPRRewardManager requires "
                "a custom compute_score function."
            )

        if frozen_coder_handle is None:
            raise ValueError(
                "PlanningTPRRewardManager requires "
                "frozen_coder_handle."
            )

        # RewardManagerBase stores:
        #   self.config
        #   self.tokenizer
        #   self.compute_score
        #   self.loop
        #
        # It also calls init_class(), but we intentionally do not
        # override init_class() because the frozen coder is loaded
        # by FrozenCoderWorker, not by this process.
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            compute_score=compute_score,
        )

        self.compute_score = compute_score

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

        self.frozen_coder_handle = (
            frozen_coder_handle
        )

        # RewardLoopWorker.compute_score_batch() schedules multiple
        # run_single() calls concurrently.
        #
        # FrozenCoderWorker itself is a normal Ray actor and therefore
        # serializes generate_code() calls by default, but keeping this
        # lock also prevents the local reward pipeline from concurrently
        # entering potentially non-thread-safe execution/evaluation code.
        self._reward_lock = asyncio.Lock()

    # ==================================================================
    # Planner response decoding
    # ==================================================================

    async def _decode_response(
        self,
        data_item: Any,
    ) -> str:
        response_ids = (
            data_item.batch[
                "responses"
            ]
        )

        response_length = int(
            response_ids.shape[-1]
        )

        if response_length <= 0:
            return ""

        attention_mask = (
            data_item.batch[
                "attention_mask"
            ]
        )

        valid_response_length = (
            attention_mask[
                -response_length:
            ]
            .sum()
        )

        if hasattr(
            valid_response_length,
            "item",
        ):
            valid_response_length = (
                valid_response_length.item()
            )

        valid_response_length = int(
            valid_response_length
        )

        if valid_response_length <= 0:
            return ""

        valid_response_ids = (
            response_ids[
                :valid_response_length
            ]
        )

        # Always use the event loop currently executing run_single().
        running_loop = asyncio.get_running_loop()

        response_str = (
            await running_loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(
                    valid_response_ids,
                    skip_special_tokens=True,
                ),
            )
        )

        return response_str.strip()

    # ==================================================================
    # Reward input construction
    # ==================================================================

    def _build_reward_inputs(
        self,
        data_item: Any,
    ) -> tuple[
        Any,
        Any,
        dict[str, Any],
        dict[str, Any],
    ]:
        """
        Extract the standard verl reward-function inputs.

        Returns
        -------
        data_source
        ground_truth
        extra_info
        extra_reward_kwargs
        """

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

        # Do not mutate the original DataProto payload.
        extra_info = dict(
            raw_extra_info
        )

        # --------------------------------------------------------------
        # Preserve optional tool metadata used by verl.
        # --------------------------------------------------------------

        tool_extra_fields = (
            data_item
            .non_tensor_batch
            .get(
                "tool_extra_fields",
                None,
            )
        )

        if tool_extra_fields is not None:
            extra_info.update(
                tool_extra_fields.items()
            )

        # --------------------------------------------------------------
        # Preserve standard verl rollout metadata.
        # --------------------------------------------------------------

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
        ] = rollout_reward_scores

        # --------------------------------------------------------------
        # TPR Planning-RLVR-specific dependency.
        #
        # planning_tpr_reward.compute_score() uses this handle to invoke
        # the frozen downstream coder.
        # --------------------------------------------------------------

        extra_reward_kwargs: dict[
            str,
            Any,
        ] = {
            "frozen_coder_handle": (
                self.frozen_coder_handle
            )
        }

        # Keep compatibility with verl reward-router based reward
        # functions. These fields are normally absent because
        # reward_model.enable=false.
        if self.reward_router_address is not None:
            extra_reward_kwargs.update(
                {
                    "reward_router_address": (
                        self.reward_router_address
                    ),
                    "reward_model_tokenizer": (
                        self.reward_model_tokenizer
                    ),
                }
            )

        return (
            data_source,
            ground_truth,
            extra_info,
            extra_reward_kwargs,
        )

    # ==================================================================
    # Single rollout reward
    # ==================================================================

    async def run_single(
        self,
        data: DataProto,
    ) -> dict[str, Any]:
        """
        Compute TPR execution reward for one planner rollout.

        verl's RewardLoopWorker normally passes a one-item DataProto,
        but following verl's NaiveRewardManager convention we explicitly
        select the final item if multiple sequences are present.

        Returns
        -------
        {
            "reward_score": float,
            "reward_extra_info": dict
        }
        """

        if len(data) <= 0:
            raise ValueError(
                "PlanningTPRRewardManager received "
                "an empty DataProto."
            )

        # For multi-sequence trajectories, reward the final sequence.
        data = data[-1:]

        data_item = data[0]

        # --------------------------------------------------------------
        # 1. Decode planner output
        # --------------------------------------------------------------

        plan = await self._decode_response(
            data_item
        )

        (
            data_source,
            ground_truth,
            extra_info,
            extra_reward_kwargs,
        ) = self._build_reward_inputs(
            data_item
        )

        # --------------------------------------------------------------
        # 2. Execute TPR Planning-RLVR reward
        #
        # RewardLoopWorker.compute_score_batch() creates concurrent
        # asyncio tasks. We serialize each complete reward transaction
        # within this RewardManager instance.
        # --------------------------------------------------------------

        async with self._reward_lock:

            if self.is_async_reward_score:
                result = (
                    await self.compute_score(
                        data_source=data_source,
                        solution_str=plan,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                        **extra_reward_kwargs,
                    )
                )

            else:
                running_loop = (
                    asyncio.get_running_loop()
                )

                result = (
                    await running_loop.run_in_executor(
                        None,
                        lambda: self.compute_score(
                            data_source=data_source,
                            solution_str=plan,
                            ground_truth=ground_truth,
                            extra_info=extra_info,
                            **extra_reward_kwargs,
                        ),
                    )
                )

        # --------------------------------------------------------------
        # 3. Normalize reward result
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
                    "TPR Planning-RLVR compute_score() "
                    "returned a dict without required "
                    "key 'score'."
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

            reward_extra_info[
                "acc"
            ] = score

        # --------------------------------------------------------------
        # 4. Enforce TPR Planning-RLVR reward definition
        #
        # Unlike the vanilla baseline, fractional rewards are valid.
        #
        # Examples:
        #   0/15  -> 0.0
        #   1/15  -> 0.066...
        #   7/15  -> 0.466...
        #   15/15 -> 1.0
        # --------------------------------------------------------------

        if not math.isfinite(
            score
        ):
            raise ValueError(
                "TPR Planning-RLVR requires "
                "a finite reward score, "
                f"but compute_score() returned {score}."
            )

        if not (
            0.0 <= score <= 1.0
        ):
            raise ValueError(
                "TPR Planning-RLVR requires "
                "execution reward in [0, 1], "
                f"but compute_score() returned {score}."
            )

        # --------------------------------------------------------------
        # Optional consistency checks for the TPR reward implementation.
        # --------------------------------------------------------------

        if isinstance(
            result,
            dict,
        ):
            if (
                "test_pass_ratio" in result
            ):
                reported_tpr = float(
                    result[
                        "test_pass_ratio"
                    ]
                )

                if not math.isclose(
                    score,
                    reported_tpr,
                    rel_tol=1e-7,
                    abs_tol=1e-7,
                ):
                    raise ValueError(
                        "TPR reward inconsistency: "
                        f"score={score}, "
                        "test_pass_ratio="
                        f"{reported_tpr}."
                    )

            if (
                "passed_tests" in result
                and "reward_tests" in result
            ):
                passed_tests = int(
                    result[
                        "passed_tests"
                    ]
                )

                reward_tests = int(
                    result[
                        "reward_tests"
                    ]
                )

                if passed_tests < 0:
                    raise ValueError(
                        "passed_tests must be >= 0, "
                        f"got {passed_tests}."
                    )

                if reward_tests < 0:
                    raise ValueError(
                        "reward_tests must be >= 0, "
                        f"got {reward_tests}."
                    )

                if (
                    passed_tests
                    > reward_tests
                ):
                    raise ValueError(
                        "passed_tests cannot exceed "
                        "reward_tests: "
                        f"{passed_tests} > "
                        f"{reward_tests}."
                    )

                expected_tpr = (
                    passed_tests
                    / reward_tests
                    if reward_tests > 0
                    else 0.0
                )

                if not math.isclose(
                    score,
                    expected_tpr,
                    rel_tol=1e-7,
                    abs_tol=1e-7,
                ):
                    raise ValueError(
                        "TPR reward does not match "
                        "passed_tests/reward_tests: "
                        f"score={score}, "
                        f"{passed_tests}/"
                        f"{reward_tests}="
                        f"{expected_tpr}."
                    )

        # --------------------------------------------------------------
        # 5. Metrics
        # --------------------------------------------------------------

        # IMPORTANT:
        #
        # In vanilla, `acc == score` because score is binary.
        #
        # In TPR training, `score` is continuous. Calling TPR itself
        # "accuracy" is semantically misleading.
        #
        # However verl commonly expects an `acc` field in reward metrics.
        # Keep it for compatibility, but also expose explicit TPR metrics.
        reward_extra_info.setdefault(
            "acc",
            score,
        )

        reward_extra_info.setdefault(
            "tpr",
            score,
        )
        # print(
        #     "[TPR RewardManager] "
        #     f"reward_score={score:.6f}",
        #     flush=True,
        # )

        return {
            "reward_score": score,
            "reward_extra_info": (
                reward_extra_info
            ),
        }