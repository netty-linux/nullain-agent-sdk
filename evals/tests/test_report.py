"""Tests for the eval report schema (issue #45: "report schema stability")."""

from __future__ import annotations

import pytest
from nullain_evals.report import REPORT_SCHEMA_VERSION, EvalReport, TaskResult


def test_pass_rate_computed_correctly() -> None:
    report = EvalReport(
        mode="offline",
        provider="replay",
        model="m",
        results=[
            TaskResult(task_id="a", passed=True, reason="ok", steps=1, wall_time_seconds=0.1),
            TaskResult(task_id="b", passed=False, reason="no", steps=1, wall_time_seconds=0.1),
            TaskResult(task_id="c", passed=True, reason="ok", steps=1, wall_time_seconds=0.1),
        ],
    )
    assert report.pass_count == 2
    assert report.total_count == 3
    assert report.pass_rate == pytest.approx(2 / 3)


def test_empty_report_pass_rate_is_zero_not_a_divide_error() -> None:
    report = EvalReport(mode="offline", provider="replay", model="m", results=[])
    assert report.pass_rate == 0.0


def test_schema_version_defaults_to_current() -> None:
    report = EvalReport(mode="offline", provider="replay", model="m", results=[])
    assert report.schema_version == REPORT_SCHEMA_VERSION


def test_report_round_trips_through_json() -> None:
    report = EvalReport(
        mode="live",
        provider="ollama",
        model="glm-5.2:cloud",
        results=[
            TaskResult(
                task_id="t1",
                passed=True,
                reason="ok",
                steps=3,
                wall_time_seconds=1.23,
                total_tokens=900,
            ),
        ],
    )
    restored = EvalReport.model_validate_json(report.model_dump_json())
    assert restored == report


def test_task_result_error_field_defaults_to_none() -> None:
    result = TaskResult(task_id="t", passed=False, reason="x", steps=0, wall_time_seconds=0.0)
    assert result.error is None
