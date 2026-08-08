"""Eval task: rename a function and update its call site — tests edit_file
precision (must not touch unrelated code) more than write_file's blunt
overwrite."""

from __future__ import annotations

from pathlib import Path

from nullain_evals.task import EvalTask, GradeContext, GradeResult

_SOURCE = """\
def compute_total(items):
    return sum(items)


def unrelated_helper(x):
    return x * 2
"""


def _setup(workspace: Path) -> None:
    (workspace / "calc.py").write_text(_SOURCE, encoding="utf-8")


def _grade(ctx: GradeContext) -> GradeResult:
    target = ctx.workspace / "calc.py"
    if not target.exists():
        return GradeResult(passed=False, reason="calc.py was deleted")
    content = target.read_text(encoding="utf-8")
    if "def compute_total" in content:
        return GradeResult(passed=False, reason="old function name compute_total still present")
    if "def compute_sum" not in content:
        return GradeResult(passed=False, reason="new function name compute_sum not found")
    if "def unrelated_helper(x):\n    return x * 2" not in content:
        return GradeResult(
            passed=False, reason="unrelated_helper was modified — edit was not surgical"
        )
    return GradeResult(passed=True, reason="renamed compute_total -> compute_sum surgically")


def build() -> list[EvalTask]:
    return [
        EvalTask(
            task_id="rename_function",
            description="Rename a function in-place without touching unrelated code.",
            prompt=(
                "In calc.py, rename the function compute_total to compute_sum. "
                "Do not change unrelated_helper or anything else in the file."
            ),
            setup=_setup,
            grader=_grade,
        ),
    ]


__all__ = ["build"]
