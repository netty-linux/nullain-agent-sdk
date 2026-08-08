"""Nullain Agent SDK evals — JSON report schema.

Kept as plain Pydantic models (not ad-hoc dicts) so the schema is
version-stable and testable — issue #45's acceptance criteria explicitly
calls for "report schema stability" as something the runner's own tests
verify.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

#: Bump when a field is removed or its meaning changes (additive fields
#: don't require a bump — a consumer diffing by task_id still works).
REPORT_SCHEMA_VERSION = 1


class TaskResult(BaseModel):
    """Outcome of running one task once, against one provider/model."""

    task_id: str
    passed: bool
    reason: str
    steps: int
    wall_time_seconds: float
    #: None when the provider doesn't report usage (shouldn't happen for
    #: OllamaCloudProvider or ReplayProvider replaying a real recording, but
    #: kept optional rather than assumed).
    total_tokens: int | None = None
    error: str | None = None


class EvalReport(BaseModel):
    """The full output of one `make evals` / `make evals-live` run."""

    schema_version: int = Field(default=REPORT_SCHEMA_VERSION)
    mode: str  # "offline" | "live"
    provider: str
    model: str
    results: list[TaskResult] = Field(default_factory=list[TaskResult])

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total_count(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return self.pass_count / self.total_count if self.total_count else 0.0


__all__ = ["REPORT_SCHEMA_VERSION", "EvalReport", "TaskResult"]
