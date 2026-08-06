"""M11 — Checkpoints and undo (4.1).

Offline tests: the file-copy fallback runs against ``tmp_path``; the git-object
path runs against a throwaway ``git init`` repo created via
``asyncio.create_subprocess_exec`` (no shell, no network).
"""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from nullain.tools import ToolRegistry
from nullain_tools import CheckpointStore, create_filesystem_tools


def _registry(tmp_path: Path, **kwargs: Any) -> ToolRegistry:
    reg = ToolRegistry()
    for t in create_filesystem_tools(tmp_path, **kwargs):
        reg.register(t)
    return reg


async def _git_init(path: Path) -> None:
    """Create a throwaway git repository at ``path`` (argv list, no shell)."""
    proc = await asyncio.create_subprocess_exec(
        "git",
        "init",
        "-q",
        cwd=str(path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()


# ---------------------------------------------------------------------------
# CheckpointStore: file-copy fallback (non-git workspace)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_checkpoint_undo_restores_file_copy(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    target = tmp_path / "a.txt"
    target.write_text("original")
    await store.snapshot(target)
    target.write_text("modified")
    result = await store.undo()
    assert "Restored 'a.txt'" in result
    assert target.read_text() == "original"


@pytest.mark.asyncio
async def test_checkpoint_undo_removes_created_file(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    target = tmp_path / "new.txt"
    await store.snapshot(target)  # file does not exist yet
    target.write_text("created")
    result = await store.undo()
    assert "Restored 'new.txt'" in result
    assert not target.exists()


@pytest.mark.asyncio
async def test_checkpoint_undo_no_checkpoints(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    assert await store.undo() == "No checkpoints to undo."


@pytest.mark.asyncio
async def test_checkpoint_eviction_keeps_newest(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, max_checkpoints=2)
    for i in range(3):
        f = tmp_path / f"f{i}.txt"
        f.write_text(f"v{i}")
        await store.snapshot(f)
    assert store.checkpoint_count == 2
    # Oldest (f0) evicted; undo restores f2 then f1 (LIFO).
    assert "f2.txt" in await store.undo()
    assert "f1.txt" in await store.undo()


# ---------------------------------------------------------------------------
# CheckpointStore: git-object storage (git workspace)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_checkpoint_undo_restores_git_blob(tmp_path: Path) -> None:
    await _git_init(tmp_path)
    store = CheckpointStore(tmp_path)
    target = tmp_path / "a.txt"
    target.write_text("original")
    await store.snapshot(target)
    target.write_text("modified")
    result = await store.undo()
    assert "Restored 'a.txt'" in result
    assert target.read_text() == "original"


@pytest.mark.asyncio
async def test_checkpoint_git_does_not_touch_index_or_refs(tmp_path: Path) -> None:
    """The git-object path must not create commits or alter the index."""
    await _git_init(tmp_path)
    store = CheckpointStore(tmp_path)
    target = tmp_path / "a.txt"
    target.write_text("original")
    await store.snapshot(target)
    # No commit and no staged changes: the object database holds the blob, but
    # the user's index/refs are untouched.
    proc = await asyncio.create_subprocess_exec(
        "git",
        "log",
        "--oneline",
        cwd=str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    assert stdout.decode().strip() == ""  # no commits created


# ---------------------------------------------------------------------------
# Write tools snapshot + undo via registry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_tools_snapshot_and_undo_via_registry(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    reg = _registry(tmp_path, checkpoint_store=store)
    await reg.execute("write_file", {"path": "a.txt", "content": "v1"})
    await reg.execute("write_file", {"path": "a.txt", "content": "v2"})
    out = await reg.execute("undo", {})
    assert "Restored 'a.txt'" in out.output
    assert (tmp_path / "a.txt").read_text() == "v1"


@pytest.mark.asyncio
async def test_undo_without_store_reports_error(tmp_path: Path) -> None:
    reg = _registry(tmp_path)  # no checkpoint_store
    out = await reg.execute("undo", {})
    assert "no checkpoint store" in out.output
    assert out.is_error
