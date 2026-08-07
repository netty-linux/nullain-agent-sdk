"""Unit tests for the git workspace tools (git_status, git_diff, git_commit).

Exercises real `git` subprocess calls against a throwaway repo under
tmp_path (matching this project's preference for testing against real
behavior over mocking, since these tools exist specifically to wrap git
correctly) — no network involved, git is always available in CI (every
runner has it for checkout).
"""

from pathlib import Path
from typing import Any

import pytest
from nullain_tools import create_git_tools


def _tools(tmp_path: Path, **kwargs: Any):
    return {t.name: t for t in create_git_tools(tmp_path, **kwargs)}


async def _run_git(*args: str, cwd: Path) -> None:
    """Set up a real git repo for the tools under test to operate on."""
    from nullain.tools import execute_subprocess

    await execute_subprocess(["git", *args], cwd=cwd)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    return tmp_path


@pytest.mark.asyncio
async def test_git_status_reports_untracked_file(git_repo: Path) -> None:
    await _run_git("init", cwd=git_repo)
    await _run_git("config", "user.email", "test@example.com", cwd=git_repo)
    await _run_git("config", "user.name", "Test", cwd=git_repo)
    (git_repo / "new.txt").write_text("hello")

    tools = _tools(git_repo)
    out = await tools["git_status"].func()
    assert "new.txt" in out


@pytest.mark.asyncio
async def test_git_diff_shows_uncommitted_change(git_repo: Path) -> None:
    await _run_git("init", cwd=git_repo)
    await _run_git("config", "user.email", "test@example.com", cwd=git_repo)
    await _run_git("config", "user.name", "Test", cwd=git_repo)
    tracked = git_repo / "a.txt"
    tracked.write_text("line1\n")
    await _run_git("add", "a.txt", cwd=git_repo)
    await _run_git("commit", "-m", "initial", cwd=git_repo)
    tracked.write_text("line1\nline2\n")

    tools = _tools(git_repo)
    out = await tools["git_diff"].func()
    assert "line2" in out


@pytest.mark.asyncio
async def test_git_diff_reports_no_changes_when_clean(git_repo: Path) -> None:
    await _run_git("init", cwd=git_repo)
    await _run_git("config", "user.email", "test@example.com", cwd=git_repo)
    await _run_git("config", "user.name", "Test", cwd=git_repo)

    tools = _tools(git_repo)
    out = await tools["git_diff"].func()
    assert out == "No uncommitted changes."


@pytest.mark.asyncio
async def test_git_commit_stages_and_commits_all_by_default(git_repo: Path) -> None:
    await _run_git("init", cwd=git_repo)
    await _run_git("config", "user.email", "test@example.com", cwd=git_repo)
    await _run_git("config", "user.name", "Test", cwd=git_repo)
    (git_repo / "a.txt").write_text("hello")

    tools = _tools(git_repo)
    out = await tools["git_commit"].func(message="add a.txt")
    assert "Git commit successful" in out

    status = await tools["git_status"].func()
    assert "a.txt" not in status  # committed, no longer showing as untracked


@pytest.mark.asyncio
async def test_git_commit_stages_only_specified_files(git_repo: Path) -> None:
    await _run_git("init", cwd=git_repo)
    await _run_git("config", "user.email", "test@example.com", cwd=git_repo)
    await _run_git("config", "user.name", "Test", cwd=git_repo)
    (git_repo / "a.txt").write_text("a")
    (git_repo / "b.txt").write_text("b")

    tools = _tools(git_repo)
    out = await tools["git_commit"].func(message="add a only", files=["a.txt"])
    assert "Git commit successful" in out

    status = await tools["git_status"].func()
    assert "b.txt" in status  # still untracked — not committed


@pytest.mark.asyncio
async def test_git_commit_reports_failure_on_empty_commit(git_repo: Path) -> None:
    """Committing with nothing staged (no changes at all) fails — git_commit
    surfaces that as a ToolResult error rather than a false "successful"."""
    await _run_git("init", cwd=git_repo)
    await _run_git("config", "user.email", "test@example.com", cwd=git_repo)
    await _run_git("config", "user.name", "Test", cwd=git_repo)

    tools = _tools(git_repo)
    result = await tools["git_commit"].func(message="nothing to commit")
    assert result.is_error is True
    assert "Git commit failed" in result.output
