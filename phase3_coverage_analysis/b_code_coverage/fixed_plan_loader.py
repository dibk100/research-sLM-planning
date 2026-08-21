"""
Phase 3-B fixed-plan loader.

Phase 1 Self-Plan 결과(results.jsonl)의 strategy_trace에서
name == "plan_generation"인 step의 raw_output을 추출한다.

Phase 3-B에서는 이 plan을 문제별로 고정한 뒤,
동일 plan으로부터 code만 N번 stochastic sampling한다.

Phase 1 schema:
    strategy_trace
      ├─ name="plan_generation"
      │    └─ raw_output = generated self-plan
      └─ name="code_generation"
           ├─ formatted_prompt = actual code prompt
           └─ raw_output = generated code
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.schemas import ProblemExample


@dataclass(frozen=True)
class FixedPlanRecord:
    """Phase 1 Self-Plan에서 가져온 하나의 고정 plan."""

    problem_id: str
    plan: str

    title: str | None = None
    difficulty: str | None = None

    # provenance
    source_line: int | None = None

    # Phase 1 당시 실제 code-generation prompt.
    # 이후 Phase 3-B reconstructed prompt와 동일성 검증 가능.
    phase1_code_prompt: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.plan.strip()


def extract_unique_strategy_step(
    raw: dict[str, Any],
    *,
    step_name: str,
    line_number: int,
) -> dict[str, Any]:
    """
    strategy_trace에서 특정 name을 가진 step을 정확히 하나 추출한다.
    """

    strategy_trace = raw.get("strategy_trace")

    if not isinstance(strategy_trace, list):
        raise ValueError(
            "Missing or invalid strategy_trace "
            f"at line {line_number}."
        )

    matched_steps = [
        step
        for step in strategy_trace
        if isinstance(step, dict)
        and step.get("name") == step_name
    ]

    if not matched_steps:
        raise ValueError(
            f"'{step_name}' step not found "
            f"at line {line_number}."
        )

    if len(matched_steps) > 1:
        raise ValueError(
            f"Multiple '{step_name}' steps found "
            f"at line {line_number}."
        )

    return matched_steps[0]


def extract_plan_from_record(
    raw: dict[str, Any],
    *,
    line_number: int,
) -> str:
    """
    Phase 1 Self-Plan 결과에서 실제 generated plan을 추출한다.

    source:
        strategy_trace[
            name == "plan_generation"
        ]["raw_output"]
    """

    plan_step = extract_unique_strategy_step(
        raw,
        step_name="plan_generation",
        line_number=line_number,
    )

    raw_output = plan_step.get(
        "raw_output"
    )

    if not isinstance(raw_output, str):
        raise ValueError(
            "plan_generation.raw_output "
            f"is not a string at line {line_number}."
        )

    plan = raw_output.strip()

    if not plan:
        raise ValueError(
            "plan_generation.raw_output "
            f"is empty at line {line_number}."
        )

    return plan


def extract_phase1_code_prompt(
    raw: dict[str, Any],
    *,
    line_number: int,
) -> str:
    """
    Phase 1 당시 code_generation에서 실제로 사용된 prompt를 추출한다.

    Phase 3-B에서 reconstructed code prompt와 동일한지
    sanity check할 때 사용한다.
    """

    code_step = extract_unique_strategy_step(
        raw,
        step_name="code_generation",
        line_number=line_number,
    )

    formatted_prompt = code_step.get(
        "formatted_prompt"
    )

    if not isinstance(
        formatted_prompt,
        str,
    ):
        raise ValueError(
            "code_generation.formatted_prompt "
            f"is not a string at line {line_number}."
        )

    prompt = formatted_prompt.strip()

    if not prompt:
        raise ValueError(
            "code_generation.formatted_prompt "
            f"is empty at line {line_number}."
        )

    return prompt


class FixedPlanLoader:
    """
    Phase 1 Self-Plan 결과에서 문제별 fixed plan을 로드한다.

    실제 연결은 항상 problem_id lookup으로 수행한다.
    Phase 1 / Phase 3 순서 일치는 별도의 sanity check로만 검증한다.
    """

    def __init__(
        self,
        results_path: str | Path,
        *,
        allow_empty_plans: bool = False,
    ) -> None:
        self.results_path = Path(
            results_path
        )

        self.allow_empty_plans = (
            allow_empty_plans
        )

        if not self.results_path.exists():
            raise FileNotFoundError(
                "Phase 1 Self-Plan results "
                f"not found: {self.results_path}"
            )

        if not self.results_path.is_file():
            raise ValueError(
                "Phase 1 Self-Plan results path "
                f"is not a file: {self.results_path}"
            )

        self._records = self._load()

        if not self._records:
            raise ValueError(
                "No fixed plans loaded from "
                f"{self.results_path}"
            )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(
        self,
    ) -> dict[str, FixedPlanRecord]:
        records: dict[
            str,
            FixedPlanRecord,
        ] = {}

        with self.results_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):
                stripped = line.strip()

                if not stripped:
                    continue

                try:
                    raw = json.loads(
                        stripped
                    )

                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Invalid JSONL in "
                        "Phase 1 Self-Plan results: "
                        f"{self.results_path}:"
                        f"{line_number}"
                    ) from error

                if not isinstance(raw, dict):
                    raise ValueError(
                        "JSONL record must be an object "
                        f"at line {line_number}."
                    )

                record = self._parse_record(
                    raw,
                    line_number=line_number,
                )

                if (
                    record.problem_id
                    in records
                ):
                    raise ValueError(
                        "Duplicate problem_id in "
                        "Phase 1 Self-Plan results: "
                        f"{record.problem_id}"
                    )

                if (
                    record.is_empty
                    and not self.allow_empty_plans
                ):
                    raise ValueError(
                        "Empty fixed plan found: "
                        f"problem_id="
                        f"{record.problem_id}, "
                        f"line={line_number}"
                    )

                records[
                    record.problem_id
                ] = record

        return records

    @staticmethod
    def _parse_record(
        raw: dict[str, Any],
        *,
        line_number: int,
    ) -> FixedPlanRecord:

        # --------------------------------------------------------------
        # identity
        # --------------------------------------------------------------

        if "problem_id" not in raw:
            raise ValueError(
                "Missing problem_id in "
                "Phase 1 Self-Plan results "
                f"at line {line_number}."
            )

        problem_id = str(
            raw["problem_id"]
        ).strip()

        if not problem_id:
            raise ValueError(
                "Empty problem_id in "
                "Phase 1 Self-Plan results "
                f"at line {line_number}."
            )

        # --------------------------------------------------------------
        # strategy validation
        # --------------------------------------------------------------

        strategy = str(
            raw.get(
                "strategy",
                "",
            )
        ).strip()

        if strategy != "self_plan":
            raise ValueError(
                "Expected strategy='self_plan', "
                f"got strategy='{strategy}' "
                f"for problem_id={problem_id}, "
                f"line={line_number}."
            )

        # --------------------------------------------------------------
        # plan
        # --------------------------------------------------------------

        plan = extract_plan_from_record(
            raw,
            line_number=line_number,
        )

        # --------------------------------------------------------------
        # original Phase 1 code prompt
        # --------------------------------------------------------------

        phase1_code_prompt = (
            extract_phase1_code_prompt(
                raw,
                line_number=line_number,
            )
        )

        # --------------------------------------------------------------
        # optional metadata
        # --------------------------------------------------------------

        title = raw.get("title")
        difficulty = raw.get(
            "difficulty"
        )

        return FixedPlanRecord(
            problem_id=problem_id,
            plan=plan,
            title=(
                str(title)
                if title is not None
                else None
            ),
            difficulty=(
                str(difficulty)
                if difficulty is not None
                else None
            ),
            source_line=line_number,
            phase1_code_prompt=(
                phase1_code_prompt
            ),
        )

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(
        self,
        problem_id: str,
    ) -> FixedPlanRecord:

        normalized_id = str(
            problem_id
        ).strip()

        try:
            return self._records[
                normalized_id
            ]

        except KeyError as error:
            raise KeyError(
                "Fixed plan not found for "
                f"problem_id={normalized_id}"
            ) from error

    def get_plan(
        self,
        problem_id: str,
    ) -> str:
        return self.get(
            problem_id
        ).plan

    def __contains__(
        self,
        problem_id: object,
    ) -> bool:
        return (
            str(problem_id).strip()
            in self._records
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self._records
        )

    @property
    def problem_ids(
        self,
    ) -> list[str]:
        """
        Phase 1 results.jsonl에 등장한 순서를 유지한다.
        dict insertion order를 사용한다.
        """
        return list(
            self._records.keys()
        )

    # ------------------------------------------------------------------
    # Dataset validation
    # ------------------------------------------------------------------

    def validate_examples(
        self,
        examples: Iterable[
            ProblemExample
        ],
        *,
        require_exact_match: bool = False,
    ) -> dict[str, Any]:

        examples_list = list(
            examples
        )

        example_ids = [
            str(
                example.problem_id
            ).strip()
            for example in examples_list
        ]

        if (
            len(example_ids)
            != len(set(example_ids))
        ):
            raise ValueError(
                "Duplicate problem IDs found "
                "in Phase 3 dataset."
            )

        fixed_ids = set(
            self._records
        )

        example_id_set = set(
            example_ids
        )

        missing_plans = sorted(
            example_id_set
            - fixed_ids
        )

        unexpected_plans = sorted(
            fixed_ids
            - example_id_set
        )

        if missing_plans:
            raise ValueError(
                "Some Phase 3 problems do not "
                "have Phase 1 fixed plans. "
                f"missing_count="
                f"{len(missing_plans)}, "
                f"examples="
                f"{missing_plans[:20]}"
            )

        if (
            require_exact_match
            and unexpected_plans
        ):
            raise ValueError(
                "Fixed-plan source contains "
                "unexpected problem IDs. "
                f"unexpected_count="
                f"{len(unexpected_plans)}, "
                f"examples="
                f"{unexpected_plans[:20]}"
            )

        empty_plan_ids = [
            problem_id
            for problem_id
            in example_ids
            if self._records[
                problem_id
            ].is_empty
        ]

        if (
            empty_plan_ids
            and not self.allow_empty_plans
        ):
            raise ValueError(
                "Empty fixed plans found "
                "for Phase 3 problems: "
                f"{empty_plan_ids[:20]}"
            )

        return {
            "num_phase3_problems": (
                len(example_ids)
            ),
            "num_fixed_plans": len(
                self._records
            ),
            "matched": len(
                example_id_set
                & fixed_ids
            ),
            "missing_plans": len(
                missing_plans
            ),
            "unexpected_plans": len(
                unexpected_plans
            ),
            "empty_plans": len(
                empty_plan_ids
            ),
        }

    # ------------------------------------------------------------------
    # Sequence validation
    # ------------------------------------------------------------------

    def validate_sequence(
        self,
        examples: Iterable[
            ProblemExample
        ],
    ) -> None:
        """
        Phase 1 source와 Phase 3 dataset의 앞부분 순서가 동일한지 확인.

        pilot:
            first 10 Phase 3 examples
            ==
            first 10 Phase 1 records

        full:
            all 500 examples
            ==
            all 500 Phase 1 records
        """

        example_ids = [
            str(
                example.problem_id
            ).strip()
            for example in examples
        ]

        fixed_ids = (
            self.problem_ids
        )

        if (
            len(example_ids)
            > len(fixed_ids)
        ):
            raise ValueError(
                "Phase 3 dataset contains "
                "more problems than "
                "Phase 1 fixed-plan source."
            )

        expected_ids = fixed_ids[
            : len(example_ids)
        ]

        if (
            example_ids
            == expected_ids
        ):
            return

        for index, (
            phase3_id,
            phase1_id,
        ) in enumerate(
            zip(
                example_ids,
                expected_ids,
            )
        ):
            if (
                phase3_id
                != phase1_id
            ):
                raise ValueError(
                    "Phase 1 / Phase 3 problem "
                    "ordering mismatch at "
                    f"index={index}: "
                    f"phase3={phase3_id}, "
                    f"phase1={phase1_id}"
                )

        raise ValueError(
            "Phase 1 / Phase 3 problem "
            "sequences differ."
        )