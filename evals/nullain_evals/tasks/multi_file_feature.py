"""Eval tasks: multi-file feature work — the shape of task this SDK's
max_steps/max_tokens defaults (docs/configuration.md) are explicitly sized
for. Requires reading one file to inform an edit to another, not just a
single blind write."""

from __future__ import annotations

from pathlib import Path

from nullain_evals.task import EvalTask, GradeContext, GradeResult

_MODELS_SOURCE = """\
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
"""

_MAIN_SOURCE = """\
from models import User

user = User("Ada", "ada@example.com")
print(user.name)
"""


def _setup(workspace: Path) -> None:
    (workspace / "models.py").write_text(_MODELS_SOURCE, encoding="utf-8")
    (workspace / "main.py").write_text(_MAIN_SOURCE, encoding="utf-8")


def _grade(ctx: GradeContext) -> GradeResult:
    models = ctx.workspace / "models.py"
    main = ctx.workspace / "main.py"
    if not models.exists() or not main.exists():
        return GradeResult(passed=False, reason="models.py or main.py is missing")

    models_content = models.read_text(encoding="utf-8")
    main_content = main.read_text(encoding="utf-8")

    if "self.age" not in models_content and "age" not in models_content:
        return GradeResult(passed=False, reason="User class was not extended with an age field")
    if "self.name = name" not in models_content or "self.email = email" not in models_content:
        return GradeResult(passed=False, reason="existing User fields (name/email) were removed")
    if "age" not in main_content:
        return GradeResult(
            passed=False, reason="main.py's User(...) construction was not updated for age"
        )
    return GradeResult(
        passed=True, reason="User gained an age field and main.py was updated consistently"
    )


def build() -> list[EvalTask]:
    return [
        EvalTask(
            task_id="multi_file_feature_add_field",
            description=(
                "Add a field to a class defined in one file and update its usage in another."
            ),
            prompt=(
                "Add an 'age' field to the User class in models.py (keep the existing "
                "name and email fields). Then update main.py's User(...) construction "
                "to also pass an age value, e.g. 34."
            ),
            setup=_setup,
            grader=_grade,
            max_steps=15,
        ),
    ]


__all__ = ["build"]
