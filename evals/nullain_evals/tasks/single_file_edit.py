"""Eval tasks: single-file edits — the simplest real coding task shape."""

from __future__ import annotations

from pathlib import Path

from nullain_evals.task import EvalTask, GradeContext, GradeResult


def _setup_create(workspace: Path) -> None:
    pass  # empty workspace — the agent creates the file from scratch


def _grade_create(ctx: GradeContext) -> GradeResult:
    target = ctx.workspace / "greet.py"
    if not target.exists():
        return GradeResult(passed=False, reason="greet.py was never created")
    content = target.read_text(encoding="utf-8")
    if "def greet" not in content:
        return GradeResult(passed=False, reason="greet.py has no greet() function")
    return GradeResult(passed=True, reason="greet.py created with a greet() function")


def _setup_append(workspace: Path) -> None:
    (workspace / "constants.py").write_text("PI = 3.14159\n", encoding="utf-8")


def _grade_append(ctx: GradeContext) -> GradeResult:
    target = ctx.workspace / "constants.py"
    content = target.read_text(encoding="utf-8") if target.exists() else ""
    if "PI = 3.14159" not in content:
        return GradeResult(passed=False, reason="pre-existing PI constant was removed")
    if "E = 2.71828" not in content and "E =" not in content:
        return GradeResult(passed=False, reason="no Euler's number constant was added")
    return GradeResult(passed=True, reason="constants.py extended with E while keeping PI")


def build() -> list[EvalTask]:
    return [
        EvalTask(
            task_id="single_file_edit_create",
            description="Create a new single-file Python module from scratch.",
            prompt=(
                "Create a file called greet.py with a function greet(name) that "
                "returns the string 'Hello, {name}!'."
            ),
            setup=_setup_create,
            grader=_grade_create,
        ),
        EvalTask(
            task_id="single_file_edit_append",
            description="Extend an existing single-file module without breaking it.",
            prompt=(
                "In constants.py, add a constant E set to Euler's number "
                "(2.71828), without removing or changing the existing PI constant."
            ),
            setup=_setup_append,
            grader=_grade_append,
        ),
    ]


__all__ = ["build"]
