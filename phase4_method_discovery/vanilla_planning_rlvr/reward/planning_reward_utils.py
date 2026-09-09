from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from src.schemas import ProblemExample


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

DEFAULT_CODE_PROMPT_PATH = (
    PROJECT_ROOT
    / "prompt_templates"
    / "self_plan_code.txt"
)

_CODE_PROMPT_TEMPLATE: str | None = None


def _ensure_code_prompt_template() -> str:
    global _CODE_PROMPT_TEMPLATE

    if _CODE_PROMPT_TEMPLATE is not None:
        return _CODE_PROMPT_TEMPLATE

    if not DEFAULT_CODE_PROMPT_PATH.exists():
        raise FileNotFoundError(
            "Self-plan code prompt not found: "
            f"{DEFAULT_CODE_PROMPT_PATH}"
        )

    template = DEFAULT_CODE_PROMPT_PATH.read_text(
        encoding="utf-8"
    )

    if not template.strip():
        raise ValueError(
            "Code prompt template is empty."
        )

    _CODE_PROMPT_TEMPLATE = template

    return template


def build_code_prompt(
    *,
    problem_text: str,
    plan: str,
    starter_code: str = "",
) -> str:

    if not isinstance(problem_text, str):
        raise TypeError(
            "problem_text must be str."
        )

    if not problem_text.strip():
        raise ValueError(
            "problem_text must not be empty."
        )

    if not isinstance(plan, str):
        raise TypeError(
            "plan must be str."
        )

    if not plan.strip():
        raise ValueError(
            "plan must not be empty."
        )

    if starter_code is None:
        starter_code = ""

    if not isinstance(starter_code, str):
        raise TypeError(
            "starter_code must be str."
        )

    starter_code = starter_code.strip()

    if starter_code:
        starter_code_section = (
            "\nStarter Code:\n"
            "```python\n"
            f"{starter_code}\n"
            "```\n"
        )
    else:
        starter_code_section = ""

    template = (
        _ensure_code_prompt_template()
    )

    try:
        prompt = template.format(
            problem=problem_text,
            plan=plan,
            starter_code=starter_code,
            starter_code_section=(
                starter_code_section
            ),
        )

    except KeyError as exc:
        raise KeyError(
            "Code prompt template placeholder mismatch. "
            "Supported placeholders: "
            "{problem}, {plan}, {starter_code}, "
            "{starter_code_section}. "
            f"Missing key: {exc}"
        ) from exc

    if not prompt.strip():
        raise RuntimeError(
            "Constructed code prompt is empty."
        )

    return prompt


def _test_input_length(
    test_input: Any,
) -> int:

    if isinstance(test_input, str):
        return len(test_input)

    if isinstance(test_input, list):
        text = "\n".join(
            str(item)
            for item in test_input
        )
        return len(text)

    return len(
        str(test_input)
    )


def select_reward_tests(
    problem: ProblemExample,
    *,
    max_tests: int,
) -> ProblemExample:

    if not isinstance(
        problem,
        ProblemExample,
    ):
        raise TypeError(
            "problem must be ProblemExample."
        )

    if max_tests <= 0:
        raise ValueError(
            "max_tests must be > 0."
        )

    private_tests = list(
        problem.private_tests
    )

    if not private_tests:
        return copy.deepcopy(problem)

    if len(private_tests) <= max_tests:
        return copy.deepcopy(problem)

    ranked_tests = sorted(
        enumerate(private_tests),
        key=lambda pair: (
            -_test_input_length(
                pair[1].get(
                    "input",
                    "",
                )
            ),
            pair[0],
        ),
    )

    selected_indices = [
        index
        for index, _
        in ranked_tests[:max_tests]
    ]

    selected_tests = [
        copy.deepcopy(
            private_tests[index]
        )
        for index in selected_indices
    ]

    reward_problem = copy.deepcopy(
        problem
    )

    reward_problem.private_tests = (
        selected_tests
    )

    return reward_problem