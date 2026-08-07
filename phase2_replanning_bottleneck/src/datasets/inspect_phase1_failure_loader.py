"""
Phase1FailureLoader sanity check.

확인 항목:
1. FailureCase 3개가 정상 생성되는가?
2. ProblemExample 테스트 케이스가 정상 복원되는가?
3. ExecutionFeedback이 정상 생성되는가?
4. feedback_text가 의도한 형태인가?

Usage:
    python -m src.datasets.inspect_phase1_failure_loader
"""

from src.datasets.phase1_failure_loader import Phase1FailureLoader


RESULTS_PATH = (
    "/mnt/hdd/project_sLM_planning/"
    "output/direct_500_stdin/results.jsonl"
)


def preview_text(
    text: str | None,
    max_chars: int = 500,
) -> str:
    """긴 문자열을 inspection용으로 짧게 출력한다."""

    if text is None:
        return "<None>"

    text = str(text)

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + f"\n...[preview truncated, total={len(text)} chars]"
    )


def main() -> None:
    loader = Phase1FailureLoader(
        RESULTS_PATH,
        limit=3,
        max_feedback_chars=2000,
    )

    failure_cases = list(
        loader.load()
    )

    # ---------------------------------------------------------------
    # 1. FailureCase 개수 확인
    # ---------------------------------------------------------------

    print("=" * 100)
    print("Phase1FailureLoader Sanity Check")
    print("=" * 100)

    print()
    print(
        f"Loaded FailureCases: {len(failure_cases)}"
    )

    assert len(failure_cases) == 3, (
        "Expected exactly 3 FailureCases, "
        f"but got {len(failure_cases)}"
    )

    print("[PASS] 3 FailureCases loaded successfully.")

    # ---------------------------------------------------------------
    # 개별 FailureCase inspection
    # ---------------------------------------------------------------

    for index, case in enumerate(
        failure_cases,
        start=1,
    ):
        example = case.example
        feedback = case.feedback

        print()
        print("=" * 100)
        print(f"FailureCase #{index}")
        print("=" * 100)

        # -----------------------------------------------------------
        # 2. Basic trajectory
        # -----------------------------------------------------------

        print()
        print("[Initial Trajectory]")

        print(
            "problem_id        :",
            example.problem_id,
        )
        print(
            "title             :",
            example.title,
        )
        print(
            "difficulty        :",
            example.difficulty,
        )
        print(
            "initial_status    :",
            case.initial_status,
        )
        print(
            "initial_test_score:",
            (
                f"{case.initial_passed_tests}/"
                f"{case.initial_total_tests}"
            ),
        )
        print(
            "initial_code chars:",
            len(case.initial_code),
        )

        # -----------------------------------------------------------
        # 3. Reconstructed tests
        # -----------------------------------------------------------

        print()
        print("[Reconstructed Tests]")

        num_tests = len(
            example.public_tests
        )

        print(
            "number of tests:",
            num_tests,
        )

        assert num_tests > 0, (
            f"No tests reconstructed for "
            f"{example.problem_id}"
        )

        print(
            "[PASS] Test cases reconstructed."
        )

        # 첫 번째 test만 inspection
        first_test = example.public_tests[0]

        print()
        print("First test keys:")
        print(
            list(first_test.keys())
        )

        print()
        print("First test input:")
        print(
            preview_text(
                first_test.get("input"),
                max_chars=300,
            )
        )

        print()
        print("First test expected output:")
        print(
            preview_text(
                first_test.get("output"),
                max_chars=300,
            )
        )

        # -----------------------------------------------------------
        # 4. Structured ExecutionFeedback
        # -----------------------------------------------------------

        print()
        print("[ExecutionFeedback]")

        print(
            "status            :",
            feedback.status,
        )
        print(
            "passed_tests      :",
            feedback.passed_tests,
        )
        print(
            "total_tests       :",
            feedback.total_tests,
        )
        print(
            "failed_test_index :",
            feedback.failed_test_index,
        )

        print(
            "failed_input chars:",
            len(feedback.failed_input or ""),
        )
        print(
            "expected chars    :",
            len(feedback.expected_output or ""),
        )
        print(
            "actual chars      :",
            len(feedback.actual_output or ""),
        )
        print(
            "stderr chars      :",
            len(feedback.stderr or ""),
        )

        assert feedback.status, (
            f"Empty feedback status: "
            f"{example.problem_id}"
        )

        assert feedback.feedback_text, (
            f"Empty feedback_text: "
            f"{example.problem_id}"
        )

        print(
            "[PASS] ExecutionFeedback created."
        )

        # -----------------------------------------------------------
        # 5. feedback_text 확인
        # -----------------------------------------------------------

        print()
        print("[feedback_text]")
        print("-" * 100)

        print(
            feedback.feedback_text
        )

        print("-" * 100)

        print(
            "feedback_text chars:",
            len(feedback.feedback_text),
        )

        # -----------------------------------------------------------
        # 기본 feedback 형식 검증
        # -----------------------------------------------------------

        assert "Execution Status:" in (
            feedback.feedback_text
        ), (
            "feedback_text does not contain "
            "'Execution Status:'"
        )

        if feedback.failed_input is not None:
            assert "Input:" in (
                feedback.feedback_text
            )

        if feedback.expected_output is not None:
            assert "Expected Output:" in (
                feedback.feedback_text
            )

        if feedback.actual_output is not None:
            assert "Actual Output:" in (
                feedback.feedback_text
            )
            
        assert (
            len(example.public_tests)
            == case.initial_total_tests
        ), (
            f"Reconstructed test count mismatch: "
            f"{example.problem_id} "
            f"{len(example.public_tests)} != "
            f"{case.initial_total_tests}"
        )

        print(
            "[PASS] feedback_text format looks valid."
        )

    # ---------------------------------------------------------------
    # Final
    # ---------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "[SUCCESS] Phase1FailureLoader sanity check passed."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()