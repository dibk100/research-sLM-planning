"""
verl에서 GRPO를 구현 및 실행함.
PlanningRewardManager 클래스는 아래 흐름으로 reward환경을 verl에 연결하는 역할

planning_execution_reward.py
    문제 + plan → code prompt 생성

frozen_coder_worker.py
    code prompt → raw code output 생성

planning_execution_reward.py
    raw output → CodeParser → TACOEvaluator → reward
    
    


"""
# phase4_method_discovery/vanilla_planning_rlvr/reward/planning_reward_manager.py

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import (
    RewardManagerBase,
)


class PlanningRewardManager(
    RewardManagerBase
):
    """
    Custom verl V1 reward manager for Vanilla Planning-RLVR.

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
        DeepCoder/TACO execution
            ->
        binary reward {0, 1}

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
        PlanningRewardManager

    PlanningRewardManager is responsible only for:

    1. decoding the planner rollout response,
    2. collecting verl dataset metadata,
    3. forwarding the FrozenCoderWorker handle to compute_score(),
    4. normalizing the returned reward,
    5. enforcing binary reward for the Vanilla Planning-RLVR baseline.

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
        Initialize the PlanningRewardManager.

        Parameters
        ----------
        config:
            verl PPO configuration.

        tokenizer:
            Planner tokenizer supplied by RewardLoopWorker.

        compute_score:
            Custom reward function loaded from
            planning_execution_reward.py.

        reward_router_address:
            Optional verl reward-model router address.
            Not used in the Vanilla Planning-RLVR baseline.

        reward_model_tokenizer:
            Optional reward-model tokenizer.
            Not used in the Vanilla Planning-RLVR baseline.

        frozen_coder_handle:
            Ray ActorHandle for FrozenCoderWorker.
        """

        if compute_score is None:
            raise ValueError(
                "PlanningRewardManager requires "
                "a custom compute_score function."
            )

        if frozen_coder_handle is None:
            raise ValueError(
                "PlanningRewardManager requires "
                "frozen_coder_handle."
            )

        # RewardManagerBase stores:
        #   self.config
        #   self.tokenizer
        #   self.compute_score
        #   self.loop
        #
        # It also calls init_class(), but we intentionally do not
        # override init_class() anymore because the frozen coder is
        # loaded by FrozenCoderWorker, not by this process.
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
        """
        Decode the valid planner response from one verl DataProto item.
        """

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

        response_str = (
            await self.loop.run_in_executor(
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
        # Planning-RLVR-specific dependency.
        #
        # planning_execution_reward.compute_score() will use this handle
        # to invoke:
        #
        #   frozen_coder_handle.generate_code.remote(...)
        #
        # The Ray ActorHandle itself is passed, not the model.
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
        # functions. These fields are normally absent in our baseline
        # because reward_model.enable=false.
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
        Compute execution reward for one planner rollout.

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
                "PlanningRewardManager received "
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

        # An empty planner response cannot produce a meaningful plan.
        #
        # We still let compute_score() decide the actual reward so that
        # reward logging/error handling remains centralized in
        # planning_execution_reward.py.
        # --------------------------------------------------------------

        (
            data_source,
            ground_truth,
            extra_info,
            extra_reward_kwargs,
        ) = self._build_reward_inputs(
            data_item
        )

        # --------------------------------------------------------------
        # 2. Execute Planning-RLVR reward
        #
        # RewardLoopWorker.compute_score_batch() creates concurrent
        # asyncio tasks. We serialize each complete reward transaction
        # within this RewardManager instance.
        #
        # This protects both:
        #   - FrozenCoderWorker request ordering
        #   - local TACO execution/evaluator state
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
                result = (
                    await self.loop.run_in_executor(
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
                    "Planning-RLVR compute_score() returned "
                    "a dict without required key 'score'."
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

            # Match verl's NaiveRewardManager convention.
            reward_extra_info[
                "acc"
            ] = score

        # --------------------------------------------------------------
        # 4. Enforce Vanilla Planning-RLVR reward definition
        # --------------------------------------------------------------

        if score not in {
            0.0,
            1.0,
        }:
            raise ValueError(
                "Vanilla Planning-RLVR requires "
                "binary execution reward {0, 1}, "
                f"but compute_score() returned {score}."
            )

        # Guarantee an `acc` metric even when compute_score() returns
        # richer diagnostic information.
        reward_extra_info.setdefault(
            "acc",
            score,
        )

        return {
            "reward_score": score,
            "reward_extra_info": (
                reward_extra_info
            ),
        }
