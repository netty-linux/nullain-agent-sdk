"""Eval task: fix a bug so a failing test passes — the harness's own
Gauntlet-shaped scenario (a programmatic pass/fail signal, not a text
heuristic). The grader runs pytest itself directly against the finished
workspace rather than trusting the agent's own claim that tests pass."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from nullain_evals.task import EvalTask, GradeContext, GradeResult

_BUGGY_SOURCE = '''\
def is_even(n):
    """Return True if n is even."""
    return n % 2 == 1  # bug: inverted condition
'''

_TEST_SOURCE = """\
from is_even import is_even


def test_is_even_true_for_even_numbers():
    assert is_even(4) is True
    assert is_even(0) is True


def test_is_even_false_for_odd_numbers():
    assert is_even(3) is False
    assert is_even(7) is False
"""


def _setup(workspace: Path) -> None:
    (workspace / "is_even.py").write_text(_BUGGY_SOURCE, encoding="utf-8")
    (workspace / "test_is_even.py").write_text(_TEST_SOURCE, encoding="utf-8")


def _grade(ctx: GradeContext) -> GradeResult:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "test_is_even.py", "-q"],
        cwd=ctx.workspace,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        tail = (result.stdout + result.stderr)[-500:]
        return GradeResult(passed=False, reason=f"pytest still failing:\n{tail}")
    return GradeResult(passed=True, reason="pytest passes after the fix")


def build() -> list[EvalTask]:
    return [
        EvalTask(
            task_id="bugfix_failing_test",
            description="Fix a one-line logic bug so the existing failing test suite passes.",
            prompt=(
                "The tests in test_is_even.py are failing. Find and fix the bug in "
                "is_even.py so all tests pass. Do not modify the test file."
            ),
            setup=_setup,
            grader=_grade,
            forbidden_paths=("test_is_even.py",),
        ),
    ]


__all__ = ["build"]
