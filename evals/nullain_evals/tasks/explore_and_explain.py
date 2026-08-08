"""Eval tasks: explore-and-explain — a read-only task shape (the agent must
answer a question about existing code without editing anything). Grades the
final answer text, and uses forbidden_paths to also assert nothing in the
workspace was touched, which most other task graders don't need to check
explicitly because a write is the whole point of those tasks."""

from __future__ import annotations

from pathlib import Path

from nullain_evals.task import EvalTask, GradeContext, GradeResult

_SOURCE = """\
def fib(n, cache={}):
    if n in cache:
        return cache[n]
    if n <= 1:
        result = n
    else:
        result = fib(n - 1) + fib(n - 2)
    cache[n] = result
    return result
"""


def _setup(workspace: Path) -> None:
    (workspace / "fib.py").write_text(_SOURCE, encoding="utf-8")


def _grade(ctx: GradeContext) -> GradeResult:
    text = ctx.final_text.lower()
    # The function memoizes results in a mutable default-argument cache —
    # a real (if commonly flagged) Python idiom. A correct explanation
    # should mention memoization/caching; this is a coarse but programmatic
    # check, not a text-similarity/LLM-judge call.
    mentions_caching = any(word in text for word in ("cache", "memoiz", "memo"))
    if not mentions_caching:
        return GradeResult(
            passed=False,
            reason="explanation didn't mention caching/memoization, the key behavior of fib()",
        )
    return GradeResult(passed=True, reason="explanation correctly identifies the caching behavior")


def build() -> list[EvalTask]:
    return [
        EvalTask(
            task_id="explore_and_explain_fib",
            description="Answer a question about existing code without editing anything.",
            prompt=(
                "Explain what the fib() function in fib.py does and why it uses a "
                "dict as a default argument. Do not modify any files — just answer."
            ),
            setup=_setup,
            grader=_grade,
            forbidden_paths=("fib.py",),
        ),
    ]


__all__ = ["build"]
