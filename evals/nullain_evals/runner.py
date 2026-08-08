"""Nullain Agent SDK evals — task runner.

Executes tasks through the public :class:`nullain.Agent` API — never through
internal harness pieces directly — so a change to any layer between the
facade and the model (routing, context compaction, tool batching) is
exercised exactly as a real user would experience it. Two modes:

- **offline** (:func:`run_offline`): loads each task's recorded fixture from
  ``evals/fixtures/<task_id>.json`` into a :class:`ReplayProvider`. No
  network access, fully deterministic, safe for CI.
- **live** (:func:`run_live`): runs against a real provider (currently
  ``OllamaCloudProvider``; the interface takes any ``LLMProvider`` so a
  future multi-provider addition — issue #40 — plugs in without changing
  this module), wrapped in a :class:`RecordingProvider` so the run can
  optionally be saved as a new/updated fixture for offline replay.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from nullain.agent import Agent
from nullain.config import NullainSettings
from nullain.events import EventStore
from nullain.llm.provider import LLMProvider

from nullain_evals.recorder import RecordingProvider
from nullain_evals.replay import ReplayExhaustedError, ReplayProvider
from nullain_evals.report import EvalReport, TaskResult
from nullain_evals.task import EvalTask, GradeContext

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _fixture_path(task_id: str) -> Path:
    return FIXTURES_DIR / f"{task_id}.json"


async def _run_one_task(
    task: EvalTask,
    *,
    provider: LLMProvider,
    model: str,
) -> TaskResult:
    """Run a single task in an isolated temp workspace and grade the result.

    Uses a fresh ``tempfile.mkdtemp`` per task (not a shared eval-suite
    workspace) so tasks can never see each other's files — the same
    isolation property a real multi-user deployment needs, exercised here
    for free.
    """
    with tempfile.TemporaryDirectory(prefix=f"nullain_eval_{task.task_id}_") as workspace_dir:
        workspace = Path(workspace_dir)
        task.setup(workspace)

        before_forbidden = {
            path: (workspace / path).read_bytes() if (workspace / path).exists() else None
            for path in task.forbidden_paths
        }

        async def _auto_approve(tool_name: str, description: str) -> bool:
            # Every eval task runs in a throwaway temp workspace created
            # fresh for this one task (see the caller) — there is nothing an
            # ASK-level tool call could put at risk here that the workspace
            # isolation doesn't already contain. Without this, every write/
            # edit tool call is denied fail-closed (Agent's default with no
            # callback configured), which silently turned every write-based
            # task into an automatic failure the first time this harness was
            # run end-to-end — found live, not by inspection.
            return True

        agent = Agent(
            settings=NullainSettings(),
            provider=provider,
            workspace_root=workspace,
            model=model,
            max_steps=task.max_steps,
            event_store=EventStore(":memory:"),
            permission_callback=_auto_approve,
        )

        start = time.monotonic()
        error: str | None = None
        try:
            result = await agent.run(task.prompt, session_id=f"eval-{task.task_id}")
            final_text = result.final_text
            run_success = result.success
            steps = result.steps
        except ReplayExhaustedError as exc:
            # A stale/incomplete fixture — always a harness/fixture bug, not
            # a legitimate grading outcome. Surfaced as a distinct error
            # field rather than a plain fail so a report reader can tell
            # "the agent produced a wrong answer" apart from "the fixture
            # needs re-recording" at a glance.
            final_text = ""
            run_success = False
            steps = 0
            error = f"replay exhausted: {exc}"
        except Exception as exc:
            final_text = ""
            run_success = False
            steps = 0
            error = f"{type(exc).__name__}: {exc}"
        wall_time = time.monotonic() - start

        if error is not None:
            return TaskResult(
                task_id=task.task_id,
                passed=False,
                reason=error,
                steps=steps,
                wall_time_seconds=wall_time,
                error=error,
            )

        for path, before in before_forbidden.items():
            after_file = workspace / path
            after = after_file.read_bytes() if after_file.exists() else None
            if after != before:
                return TaskResult(
                    task_id=task.task_id,
                    passed=False,
                    reason=f"forbidden path modified: {path}",
                    steps=steps,
                    wall_time_seconds=wall_time,
                )

        grade = task.grader(
            GradeContext(
                workspace=workspace,
                final_text=final_text,
                run_success=run_success,
                steps=steps,
            )
        )
        return TaskResult(
            task_id=task.task_id,
            passed=grade.passed,
            reason=grade.reason,
            steps=steps,
            wall_time_seconds=wall_time,
        )


async def run_offline(tasks: list[EvalTask], *, model: str = "offline-replay") -> EvalReport:
    """Run every task against its recorded fixture. Raises if a task has no
    fixture yet — offline mode is meant to be exhaustive, not best-effort;
    a missing fixture means the suite is out of sync with the task list."""
    results: list[TaskResult] = []
    for task in tasks:
        fixture = _fixture_path(task.task_id)
        if not fixture.exists():
            results.append(
                TaskResult(
                    task_id=task.task_id,
                    passed=False,
                    reason=f"no recorded fixture at {fixture}",
                    steps=0,
                    wall_time_seconds=0.0,
                    error="missing fixture",
                )
            )
            continue
        provider = ReplayProvider.from_fixture(fixture)
        results.append(await _run_one_task(task, provider=provider, model=model))
    return EvalReport(mode="offline", provider="replay", model=model, results=results)


async def run_live(
    tasks: list[EvalTask],
    *,
    provider: LLMProvider,
    provider_name: str,
    model: str,
    save_fixtures: bool = False,
) -> EvalReport:
    """Run every task against a real provider. When ``save_fixtures`` is
    True, each task's recorded responses are written to
    ``evals/fixtures/<task_id>.json`` after a passing run — so
    `make evals-live` doubles as the fixture-recording workflow (never
    records a failing run's responses as a fixture, since replaying a
    known-bad trajectory forever would defeat the point of the baseline).
    """
    results: list[TaskResult] = []
    for task in tasks:
        recording = RecordingProvider(provider)
        result = await _run_one_task(task, provider=recording, model=model)
        results.append(result)
        if save_fixtures and result.passed:
            recording.save(_fixture_path(task.task_id))
    return EvalReport(mode="live", provider=provider_name, model=model, results=results)


__all__ = ["FIXTURES_DIR", "run_live", "run_offline"]
