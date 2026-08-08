"""Nullain Agent SDK evals — task definition and grading protocol."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


#: A grader inspects the finished workspace (and the agent's RunResult,
#: passed as `final_text`/`success` via GradeContext) and returns a
#: GradeResult. Programmatic graders are strongly preferred (issue #45's
#: scope: "LLM-judge only where a programmatic grader is impossible") because
#: they're deterministic and free to run in CI; every task in this suite uses
#: one.
@dataclass(frozen=True)
class GradeContext:
    """What a grader gets to inspect after a task run finishes."""

    workspace: Path
    final_text: str
    run_success: bool
    steps: int


@dataclass(frozen=True)
class GradeResult:
    """A grader's verdict: pass/fail plus a human-readable reason, always
    populated (even on pass) so the report is legible without re-running."""

    passed: bool
    reason: str


class Grader(Protocol):
    def __call__(self, ctx: GradeContext) -> GradeResult: ...


@dataclass(frozen=True)
class EvalTask:
    """A single self-contained coding task.

    ``setup`` populates a fresh temporary workspace (called with the
    workspace root) before the agent runs — e.g. writing a starter file, a
    failing test, or a small existing project. ``prompt`` is the exact text
    handed to ``Agent.run``. ``grader`` inspects the finished workspace and
    the run's outcome and returns a verdict. ``forbidden_paths`` are
    relative paths the task's own grading run additionally checks were
    never modified (a cross-cutting safety property most tasks care about,
    factored out so individual graders don't have to repeat it).
    """

    task_id: str
    description: str
    prompt: str
    setup: Callable[[Path], None]
    grader: Grader
    forbidden_paths: tuple[str, ...] = ()
    #: Optional per-task override of the default max_steps budget — most
    #: tasks are fine with the harness default; a multi-file task may need
    #: more headroom.
    max_steps: int | None = None


#: Type alias for a task-provider function, used by tasks/__init__.py's
#: registry — kept here so task modules don't need to import from the
#: registry module (avoiding a circular import).
TaskFactory = Callable[[], EvalTask]

#: Async variant, for graders/setups that need to be awaited (none of the
#: current tasks do, but the type is here so a future task can use it
#: without a signature change to the harness).
AsyncTaskFactory = Callable[[], Awaitable[EvalTask]]

__all__ = [
    "AsyncTaskFactory",
    "EvalTask",
    "GradeContext",
    "GradeResult",
    "Grader",
    "TaskFactory",
]
