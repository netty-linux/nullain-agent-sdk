"""Nullain Agent SDK evals — task registry.

Each task lives in its own module (``tasks/<module>.py``) exposing a
``build() -> list[EvalTask]`` factory (usually returning one task, sometimes
a couple of closely related variants); this module collects them into
``ALL_TASKS`` so the runner and CLI don't need to know the task list ahead
of time. Adding a task means: write (or extend) a module under ``tasks/``,
add it to ``_TASK_MODULES`` below, and record its fixture (see
``evals/README.md``).
"""

from __future__ import annotations

from nullain_evals.task import EvalTask
from nullain_evals.tasks import (
    add_error_handling,
    bugfix_failing_test,
    explore_and_explain,
    multi_file_feature,
    refactor_extract_function,
    rename_function,
    single_file_edit,
)

_TASK_MODULES = (
    single_file_edit,
    rename_function,
    bugfix_failing_test,
    multi_file_feature,
    explore_and_explain,
    add_error_handling,
    refactor_extract_function,
)

ALL_TASKS: list[EvalTask] = [task for module in _TASK_MODULES for task in module.build()]

__all__ = ["ALL_TASKS"]
