# phase4_method_discovery/vanilla_planning_rlvr/logging/rollout_logger.py
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from phase4_method_discovery.vanilla_planning_rlvr.reward.planning_execution_reward import (
    PlanningRewardResult,
)


@dataclass
class RolloutRecord:
    """
    One rollout record for Vanilla Planning-RLVR.

    A rollout corresponds to:

        problem
        -> planner response (plan)
        -> frozen coder response
        -> execution reward

    Token-level fields are optional because they will be populated
    later from the verl rollout/trainer batch.
    """

    # ------------------------------------------------------------
    # Run / rollout identity
    # ------------------------------------------------------------

    global_step: int
    group_id: str
    sample_id: int

    problem_id: str
    dataset: str

    # ------------------------------------------------------------
    # Planner trajectory
    # ------------------------------------------------------------

    plan: str

    plan_token_count: int = 0

    plan_tokens: list[str] = field(
        default_factory=list
    )

    plan_token_ids: list[int] = field(
        default_factory=list
    )

    token_logprobs: list[float] = field(
        default_factory=list
    )

    # Optional sequence-level quantity.
    plan_logprob_sum: float | None = None
    plan_logprob_mean: float | None = None

    # ------------------------------------------------------------
    # Frozen coder generation
    # ------------------------------------------------------------

    raw_code_output: str = ""
    generated_code: str = ""
    code_extraction_method: str = ""

    coder_prompt_tokens: int = 0
    coder_completion_tokens: int = 0
    coder_generation_time: float = 0.0

    # ------------------------------------------------------------
    # Execution / reward
    # ------------------------------------------------------------

    reward: float = 0.0

    passed: bool = False
    status: str = ""

    passed_tests: int = 0
    total_tests: int = 0

    execution_time: float = 0.0

    error_message: str | None = None

    # ------------------------------------------------------------
    # Additional metadata
    # ------------------------------------------------------------

    model_name: str = ""
    seed: int | None = None

    extra: dict[str, Any] = field(
        default_factory=dict
    )

    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RolloutLogger:
    """
    Append-only JSONL logger for Vanilla Planning-RLVR rollouts.

    Notes
    -----
    - One RolloutLogger instance should normally be used per process.
    - Thread locking protects concurrent writes inside one process.
    - This is NOT sufficient for multiple independent processes writing
      to the same file simultaneously.

    For distributed verl/Ray workers, prefer:
        - one file per worker/rank, or
        - centralized logging on the driver.
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        flush_every_write: bool = True,
    ) -> None:
        self.output_path = Path(output_path)
        self.flush_every_write = flush_every_write

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = threading.Lock()

    def log(
        self,
        record: RolloutRecord,
    ) -> None:
        """
        Append one rollout record as one JSON line.
        """

        if not isinstance(record, RolloutRecord):
            raise TypeError(
                "record must be RolloutRecord, "
                f"got {type(record).__name__}"
            )

        self._validate_record(record)

        payload = record.to_dict()

        line = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
        )

        with self._lock:
            with self.output_path.open(
                "a",
                encoding="utf-8",
            ) as f:
                f.write(line)
                f.write("\n")

                if self.flush_every_write:
                    f.flush()

    @staticmethod
    def from_reward_result(
        *,
        reward_result: PlanningRewardResult,
        global_step: int,
        group_id: str,
        sample_id: int,
        dataset: str,
        model_name: str = "",
        seed: int | None = None,
        plan_tokens: Sequence[str] | None = None,
        plan_token_ids: Sequence[int] | None = None,
        token_logprobs: Sequence[float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> RolloutRecord:
        """
        Build a RolloutRecord from PlanningRewardResult.

        Token-wise information is optional for now and can later be
        supplied directly from verl's rollout batch.
        """

        tokens = (
            list(plan_tokens)
            if plan_tokens is not None
            else []
        )

        token_ids = (
            [int(x) for x in plan_token_ids]
            if plan_token_ids is not None
            else []
        )

        logprobs = (
            [float(x) for x in token_logprobs]
            if token_logprobs is not None
            else []
        )

        plan_token_count = RolloutLogger._infer_token_count(
            tokens=tokens,
            token_ids=token_ids,
            logprobs=logprobs,
        )

        if logprobs:
            logprob_sum = sum(logprobs)
            logprob_mean = (
                logprob_sum / len(logprobs)
            )
        else:
            logprob_sum = None
            logprob_mean = None

        return RolloutRecord(
            global_step=global_step,
            group_id=group_id,
            sample_id=sample_id,

            problem_id=reward_result.problem_id,
            dataset=dataset,

            plan=reward_result.plan,

            plan_token_count=plan_token_count,
            plan_tokens=tokens,
            plan_token_ids=token_ids,
            token_logprobs=logprobs,

            plan_logprob_sum=logprob_sum,
            plan_logprob_mean=logprob_mean,

            raw_code_output=(
                reward_result.raw_code_output
            ),
            generated_code=(
                reward_result.generated_code
            ),
            code_extraction_method=(
                reward_result.code_extraction_method
            ),

            coder_prompt_tokens=(
                reward_result.coder_prompt_tokens
            ),
            coder_completion_tokens=(
                reward_result.coder_completion_tokens
            ),
            coder_generation_time=(
                reward_result.coder_generation_time
            ),

            reward=reward_result.reward,

            passed=reward_result.passed,
            status=reward_result.status,

            passed_tests=reward_result.passed_tests,
            total_tests=reward_result.total_tests,

            execution_time=reward_result.execution_time,

            error_message=reward_result.error_message,

            model_name=model_name,
            seed=seed,

            extra=dict(extra or {}),
        )

    @staticmethod
    def _infer_token_count(
        *,
        tokens: list[str],
        token_ids: list[int],
        logprobs: list[float],
    ) -> int:
        """
        Infer response token count from available token-level arrays.
        """

        lengths = [
            len(values)
            for values in (
                tokens,
                token_ids,
                logprobs,
            )
            if values
        ]

        if not lengths:
            return 0

        if len(set(lengths)) != 1:
            raise ValueError(
                "Token-level fields have inconsistent lengths: "
                f"tokens={len(tokens)}, "
                f"token_ids={len(token_ids)}, "
                f"logprobs={len(logprobs)}"
            )

        return lengths[0]

    @staticmethod
    def _validate_record(
        record: RolloutRecord,
    ) -> None:
        if record.global_step < 0:
            raise ValueError(
                "global_step must be >= 0."
            )

        if record.sample_id < 0:
            raise ValueError(
                "sample_id must be >= 0."
            )

        if not record.problem_id:
            raise ValueError(
                "problem_id must not be empty."
            )

        if not record.dataset:
            raise ValueError(
                "dataset must not be empty."
            )

        if record.reward not in {
            0.0,
            1.0,
        }:
            raise ValueError(
                "Vanilla Planning-RLVR reward must "
                f"be binary, got {record.reward}."
            )

        token_lengths = [
            len(values)
            for values in (
                record.plan_tokens,
                record.plan_token_ids,
                record.token_logprobs,
            )
            if values
        ]

        if (
            token_lengths
            and len(set(token_lengths)) != 1
        ):
            raise ValueError(
                "plan_tokens, plan_token_ids and "
                "token_logprobs must have equal lengths "
                "when provided."
            )

        if (
            token_lengths
            and record.plan_token_count
            != token_lengths[0]
        ):
            raise ValueError(
                "plan_token_count does not match "
                "token-level data."
            )