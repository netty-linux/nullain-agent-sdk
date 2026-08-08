"""Eval task: add defensive error handling to an existing function — tests
whether the agent adds a targeted try/except without rewriting unrelated
logic."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from nullain_evals.task import EvalTask, GradeContext, GradeResult

_SOURCE = """\
def parse_int(text):
    return int(text)
"""

_TEST_SOURCE = """\
from parser_mod import parse_int


def test_parse_int_valid():
    assert parse_int("42") == 42


def test_parse_int_invalid_returns_none():
    assert parse_int("not a number") is None
"""


def _setup(workspace: Path) -> None:
    (workspace / "parser_mod.py").write_text(_SOURCE, encoding="utf-8")
    (workspace / "test_parser_mod.py").write_text(_TEST_SOURCE, encoding="utf-8")


def _grade(ctx: GradeContext) -> GradeResult:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "test_parser_mod.py", "-q"],
        cwd=ctx.workspace,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        tail = (result.stdout + result.stderr)[-500:]
        return GradeResult(passed=False, reason=f"pytest failing after edit:\n{tail}")
    return GradeResult(passed=True, reason="parse_int now handles invalid input gracefully")


def build() -> list[EvalTask]:
    return [
        EvalTask(
            task_id="add_error_handling",
            description="Add error handling so a function degrades gracefully on bad input.",
            prompt=(
                "parser_mod.py's parse_int(text) currently raises ValueError on invalid "
                "input. Change it to return None instead of raising when the input can't "
                "be parsed as an integer. Do not modify the test file."
            ),
            setup=_setup,
            grader=_grade,
            forbidden_paths=("test_parser_mod.py",),
        ),
    ]


__all__ = ["build"]
