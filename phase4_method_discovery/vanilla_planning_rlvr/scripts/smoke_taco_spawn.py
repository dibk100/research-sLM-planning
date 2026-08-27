# phase4_method_discovery/vanilla_planning_rlvr/scripts/smoke_taco_spawn.py
"""
PYTHONPATH="$HOME/workspace/project_sLM_planning" \
/mnt/hdd/conda_envs/planning_rlvr/bin/python \
  phase4_method_discovery/vanilla_planning_rlvr/scripts/smoke_taco_spawn.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd


# ==============================================================================
# Project path
# ==============================================================================

PROJECT_ROOT = Path(
    "/home/dibaeck/workspace/project_sLM_planning"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from phase4_method_discovery.vanilla_planning_rlvr.reward.planning_execution_reward import (
    _problem_from_extra_info,
)
from src.execution.taco_evaluator import (
    TACOEvaluator,
)


# ==============================================================================
# Config
# ==============================================================================

TRAIN_PARQUET = Path(
    "/mnt/hdd/project_sLM_planning/data/deepcoder_taco/"
    "processed/vanilla_planning_rlvr/train.parquet"
)

ROW_INDEX = 0

TIMEOUT_SECONDS = 6


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    print("=" * 90)
    print("TACOEvaluator spawn smoke test")
    print("=" * 90)

    # --------------------------------------------------------------------------
    # 1. Load one training problem
    # --------------------------------------------------------------------------

    if not TRAIN_PARQUET.exists():
        raise FileNotFoundError(
            f"Training parquet not found: {TRAIN_PARQUET}"
        )

    df = pd.read_parquet(
        TRAIN_PARQUET
    )

    if len(df) == 0:
        raise RuntimeError(
            "Training parquet is empty."
        )

    row = df.iloc[
        ROW_INDEX
    ]

    print(
        f"[INFO] parquet rows : {len(df)}"
    )
    print(
        f"[INFO] row index    : {ROW_INDEX}"
    )

    # --------------------------------------------------------------------------
    # 2. Restore extra_info
    # --------------------------------------------------------------------------

    if "extra_info" not in row:
        raise KeyError(
            "train.parquet row does not contain extra_info."
        )

    extra_info = row[
        "extra_info"
    ]

    if extra_info is None:
        raise ValueError(
            "extra_info is None."
        )

    # Depending on pandas/pyarrow conversion this should normally
    # already be a Python dict.
    if hasattr(
        extra_info,
        "as_py",
    ):
        extra_info = (
            extra_info.as_py()
        )

    if not isinstance(
        extra_info,
        dict,
    ):
        try:
            extra_info = dict(
                extra_info
            )
        except Exception as exc:
            raise TypeError(
                "Could not convert extra_info to dict: "
                f"{type(extra_info).__name__}"
            ) from exc

    # --------------------------------------------------------------------------
    # 3. Reconstruct ProblemExample using the same Phase-4 helper
    # --------------------------------------------------------------------------

    problem = (
        _problem_from_extra_info(
            extra_info
        )
    )

    print(
        f"[INFO] problem id   : {problem.problem_id}"
    )
    print(
        f"[INFO] dataset      : {problem.dataset}"
    )
    print(
        f"[INFO] eval type    : {problem.evaluation_type}"
    )
    print(
        f"[INFO] private tests: {len(problem.private_tests)}"
    )

    # --------------------------------------------------------------------------
    # 4. Use a deliberately simple candidate program.
    #
    # The purpose of this smoke test is NOT solving the problem.
    # We only verify spawn -> evaluator -> Pipe -> parent result.
    # --------------------------------------------------------------------------

    candidate_code = """
def main():
    x = []
    while True:
        x.append("x" * (100 * 1024 * 1024))

if __name__ == "__main__":
    main()
""".strip()

    evaluator = TACOEvaluator(
        timeout_seconds=TIMEOUT_SECONDS,
        debug=True,
    )

    # --------------------------------------------------------------------------
    # 5. Evaluate
    # --------------------------------------------------------------------------

    print()
    print("[INFO] Starting evaluation...")

    start = time.perf_counter()

    result = evaluator.evaluate(
        problem,
        candidate_code,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------------------------
    # 6. Report
    # --------------------------------------------------------------------------

    print()
    print("=" * 90)
    print("Evaluation result")
    print("=" * 90)

    print(
        f"passed          : {result.passed}"
    )
    print(
        f"status          : {result.status}"
    )
    print(
        f"passed tests    : {result.passed_tests}"
    )
    print(
        f"total tests     : {result.total_tests}"
    )
    print(
        f"execution time  : {result.execution_time}"
    )
    print(
        f"wall time       : {elapsed:.3f}s"
    )
    print(
        f"error message   : {result.error_message}"
    )

    print()
    print(
        f"returned test results: {len(result.test_results)}"
    )

    for test_result in (
        result.test_results[:5]
    ):
        print(
            "  "
            f"index={test_result.test_index} "
            f"passed={test_result.passed} "
            f"status={test_result.status}"
        )

    print()
    print("=" * 90)

    # --------------------------------------------------------------------------
    # The candidate is intentionally wrong, so PASS is not expected.
    # What matters is that the evaluator returned normally instead of:
    #
    # - hanging
    # - pickling error
    # - spawn import error
    # - EOFError
    # - Manager/Ray memory explosion
    # --------------------------------------------------------------------------

    if result.status == "EVALUATION_ERROR":
        print(
            "[FAIL] Evaluator returned EVALUATION_ERROR."
        )
        sys.exit(
            1
        )

    print(
        "[PASS] spawn-based TACO evaluation completed normally."
    )


if __name__ == "__main__":
    main()