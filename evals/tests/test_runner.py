"""End-to-end tests for the offline runner against the real recorded
fixtures — proves the whole harness (runner -> Agent -> tools -> grader)
actually works, and that replaying twice gives identical results
(determinism, issue #45's acceptance criteria)."""

from __future__ import annotations

import pytest
from nullain_evals.runner import run_offline
from nullain_evals.tasks import ALL_TASKS


@pytest.mark.asyncio
async def test_all_tasks_pass_against_their_recorded_fixtures() -> None:
    """The flagship proof: every task in the suite, run through the real
    Agent against its own recorded fixture, passes its own grader."""
    report = await run_offline(ALL_TASKS)
    failures = [(r.task_id, r.reason) for r in report.results if not r.passed]
    assert not failures, f"tasks failing against their own fixtures: {failures}"
    assert report.total_count == len(ALL_TASKS)


@pytest.mark.asyncio
async def test_offline_replay_is_deterministic() -> None:
    """Running the same fixture twice must give the same pass/fail per task
    — the whole point of offline mode being CI-safe."""
    report1 = await run_offline(ALL_TASKS)
    report2 = await run_offline(ALL_TASKS)
    results1 = {r.task_id: r.passed for r in report1.results}
    results2 = {r.task_id: r.passed for r in report2.results}
    assert results1 == results2


@pytest.mark.asyncio
async def test_missing_fixture_fails_that_task_not_the_whole_run() -> None:
    from nullain_evals.task import EvalTask, GradeResult

    fake_task = EvalTask(
        task_id="__nonexistent_task_for_test__",
        description="test-only",
        prompt="does not matter",
        setup=lambda workspace: None,
        grader=lambda ctx: GradeResult(passed=True, reason="unreachable"),
    )
    report = await run_offline([fake_task])
    assert report.total_count == 1
    assert report.pass_count == 0
    assert report.results[0].error == "missing fixture"


@pytest.mark.asyncio
async def test_report_mode_and_provider_labeled_correctly() -> None:
    report = await run_offline(ALL_TASKS)
    assert report.mode == "offline"
    assert report.provider == "replay"
