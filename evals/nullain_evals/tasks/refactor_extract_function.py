"""Eval task: extract duplicated logic into a shared helper function —
a refactor task shape distinct from rename_function (structural change,
not just a name swap), graded by both structure and behavior preservation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from nullain_evals.task import EvalTask, GradeContext, GradeResult

_SOURCE = """\
def total_price_usd(items):
    total = 0
    for item in items:
        total += item["price"] * item["quantity"]
    return round(total, 2)


def total_price_eur(items):
    total = 0
    for item in items:
        total += item["price"] * item["quantity"]
    return round(total * 0.92, 2)
"""

_TEST_SOURCE = """\
from pricing import total_price_usd, total_price_eur

ITEMS = [{"price": 10.0, "quantity": 2}, {"price": 5.5, "quantity": 1}]


def test_total_price_usd():
    assert total_price_usd(ITEMS) == 25.5


def test_total_price_eur():
    assert total_price_eur(ITEMS) == round(25.5 * 0.92, 2)
"""


def _setup(workspace: Path) -> None:
    (workspace / "pricing.py").write_text(_SOURCE, encoding="utf-8")
    (workspace / "test_pricing.py").write_text(_TEST_SOURCE, encoding="utf-8")


def _grade(ctx: GradeContext) -> GradeResult:
    target = ctx.workspace / "pricing.py"
    content = target.read_text(encoding="utf-8") if target.exists() else ""
    # Structural check: the duplicated summation loop should now appear
    # only once (extracted into a shared helper), not twice.
    if content.count("for item in items:") > 1:
        return GradeResult(passed=False, reason="summation loop is still duplicated, not extracted")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "test_pricing.py", "-q"],
        cwd=ctx.workspace,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        tail = (result.stdout + result.stderr)[-500:]
        return GradeResult(passed=False, reason=f"behavior changed, pytest failing:\n{tail}")
    return GradeResult(passed=True, reason="duplication extracted, behavior preserved")


def build() -> list[EvalTask]:
    return [
        EvalTask(
            task_id="refactor_extract_function",
            description="Extract duplicated logic into a shared helper without changing behavior.",
            prompt=(
                "total_price_usd and total_price_eur in pricing.py duplicate the same "
                "summation loop. Refactor to extract the shared logic into a helper "
                "function, without changing either function's return value. Do not "
                "modify the test file."
            ),
            setup=_setup,
            grader=_grade,
            forbidden_paths=("test_pricing.py",),
            max_steps=15,
        ),
    ]


__all__ = ["build"]
