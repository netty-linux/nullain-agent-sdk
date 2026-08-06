"""M8 — coding-grade file tools: paged reads, safe edits, atomic multi_edit, rg grep, todo.

All tests are offline: filesystem tools run against ``tmp_path`` and the grep
ripgrep path is exercised via monkeypatched ``shutil.which`` / ``_grep_with_rg``
so no subprocess or network is involved.
"""

from pathlib import Path
from typing import Any

import pytest
from nullain.events import BaseEvent, EventBus, EventStore, TodoEvent, TodoItem
from nullain.tools import ToolRegistry
from nullain_tools import FileAccessTracker, create_filesystem_tools


def _tools(tmp_path: Path, **kwargs: Any):
    return {t.name: t for t in create_filesystem_tools(tmp_path, **kwargs)}


def _registry(tmp_path: Path, **kwargs: Any) -> ToolRegistry:
    reg = ToolRegistry()
    for t in create_filesystem_tools(tmp_path, **kwargs):
        reg.register(t)
    return reg


def _which_none(_: str) -> None:
    """Monkeypatch for ``shutil.which`` reporting ripgrep is absent."""
    return None


def _which_rg(_: str) -> str:
    """Monkeypatch for ``shutil.which`` reporting ripgrep is present."""
    return "/usr/bin/rg"


# ---------------------------------------------------------------------------
# read_file: pagination + numbering
# ---------------------------------------------------------------------------


def test_read_file_returns_numbered_lines(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")
    tools = _tools(tmp_path)
    out = tools["read_file"].func(path="a.txt")
    assert out == "1\tone\n2\ttwo\n3\tthree"


def test_read_file_pages_with_offset_and_limit(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("\n".join(f"line{i}" for i in range(10)))
    tools = _tools(tmp_path)
    out = tools["read_file"].func(path="a.txt", offset=2, limit=3)
    assert "3\tline2\n4\tline3\n5\tline4" in out
    assert "5 more lines" in out


def test_read_file_announces_remaining_lines(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("\n".join(f"line{i}" for i in range(10)))
    tools = _tools(tmp_path)
    out = tools["read_file"].func(path="a.txt", offset=0, limit=3)
    assert "7 more lines" in out
    assert "offset=3" in out


def test_read_file_truncates_absurdly_long_line(tmp_path: Path) -> None:
    long_line = "x" * 5000
    (tmp_path / "a.txt").write_text(long_line)
    tools = _tools(tmp_path)
    out = tools["read_file"].func(path="a.txt")
    assert "[truncated" in out
    assert "x" * 2000 in out


def test_read_file_rejects_negative_offset(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    tools = _tools(tmp_path)
    assert "offset must be >= 0" in tools["read_file"].func(path="a.txt", offset=-1).output


# ---------------------------------------------------------------------------
# edit_file: safe edits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_file_requires_prior_read(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello world")
    tools = _tools(tmp_path)
    out = await tools["edit_file"].func(path="a.txt", old_str="hello", new_str="goodbye")
    assert "has not been read" in out.output
    # After reading, the edit succeeds.
    tools["read_file"].func(path="a.txt")
    out = await tools["edit_file"].func(path="a.txt", old_str="hello", new_str="goodbye")
    assert "Edited" in out
    assert (tmp_path / "a.txt").read_text() == "goodbye world"


@pytest.mark.asyncio
async def test_edit_file_rejects_ambiguous_match(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("foo foo foo")
    tools = _tools(tmp_path)
    tools["read_file"].func(path="a.txt")
    out = await tools["edit_file"].func(path="a.txt", old_str="foo", new_str="bar")
    assert "appears 3 times" in out.output
    # File unchanged.
    assert (tmp_path / "a.txt").read_text() == "foo foo foo"


@pytest.mark.asyncio
async def test_edit_file_replace_all(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("foo foo foo")
    tools = _tools(tmp_path)
    tools["read_file"].func(path="a.txt")
    out = await tools["edit_file"].func(
        path="a.txt", old_str="foo", new_str="bar", replace_all=True
    )
    assert "Edited" in out
    assert (tmp_path / "a.txt").read_text() == "bar bar bar"


@pytest.mark.asyncio
async def test_edit_file_rejects_noop(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello")
    tools = _tools(tmp_path)
    tools["read_file"].func(path="a.txt")
    out = await tools["edit_file"].func(path="a.txt", old_str="hello", new_str="hello")
    assert "no-op" in out.output


@pytest.mark.asyncio
async def test_edit_file_returns_numbered_snippet(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("line1\nline2\nline3\n")
    tools = _tools(tmp_path)
    tools["read_file"].func(path="a.txt")
    out = await tools["edit_file"].func(path="a.txt", old_str="line2", new_str="CHANGED")
    assert "CHANGED" in out
    assert "2\tCHANGED" in out


# ---------------------------------------------------------------------------
# multi_edit: atomic, chained (via registry to exercise dict->FileEdit coercion)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_edit_applies_all_edits(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a b c")
    reg = _registry(tmp_path)
    await reg.execute("read_file", {"path": "a.txt"})
    out = await reg.execute(
        "multi_edit",
        {
            "path": "a.txt",
            "edits": [
                {"old_str": "a", "new_str": "1"},
                {"old_str": "b", "new_str": "2"},
                {"old_str": "c", "new_str": "3"},
            ],
        },
    )
    assert "Applied 3 edits" in out.output
    assert (tmp_path / "a.txt").read_text() == "1 2 3"


@pytest.mark.asyncio
async def test_multi_edit_rolls_back_on_failure(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a b c")
    reg = _registry(tmp_path)
    await reg.execute("read_file", {"path": "a.txt"})
    out = await reg.execute(
        "multi_edit",
        {
            "path": "a.txt",
            "edits": [
                {"old_str": "a", "new_str": "1"},
                {"old_str": "zzz", "new_str": "2"},  # not present -> fails
            ],
        },
    )
    assert "edit #2" in out.output
    # Nothing was written: the first edit is rolled back.
    assert (tmp_path / "a.txt").read_text() == "a b c"


@pytest.mark.asyncio
async def test_multi_edit_chains_edits(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    reg = _registry(tmp_path)
    await reg.execute("read_file", {"path": "a.txt"})
    # Second edit's old_str is produced by the first edit (chained).
    out = await reg.execute(
        "multi_edit",
        {
            "path": "a.txt",
            "edits": [
                {"old_str": "a", "new_str": "ab"},
                {"old_str": "ab", "new_str": "abc"},
            ],
        },
    )
    assert "Applied 2 edits" in out.output
    assert (tmp_path / "a.txt").read_text() == "abc"


@pytest.mark.asyncio
async def test_multi_edit_requires_prior_read(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    reg = _registry(tmp_path)
    out = await reg.execute(
        "multi_edit", {"path": "a.txt", "edits": [{"old_str": "a", "new_str": "b"}]}
    )
    assert "has not been read" in out.output


# ---------------------------------------------------------------------------
# grep: ripgrep detection + fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grep_uses_rg_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    pass\n")
    tools = _tools(tmp_path)
    monkeypatch.setattr("nullain_tools.filesystem.shutil.which", _which_rg)

    async def fake_rg(*args: object, **kwargs: object) -> str:
        return "a.py:1:def foo():"

    monkeypatch.setattr("nullain_tools.filesystem._grep_with_rg", fake_rg)
    out = await tools["grep"].func(pattern="foo")
    assert out == "a.py:1:def foo():"


@pytest.mark.asyncio
async def test_grep_fallback_content_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    pass\n")
    (tmp_path / "b.py").write_text("x = 1\n")
    tools = _tools(tmp_path)
    monkeypatch.setattr("nullain_tools.filesystem.shutil.which", _which_none)
    out = await tools["grep"].func(pattern="foo")
    assert "a.py:1:def foo():" in out
    assert "b.py" not in out


@pytest.mark.asyncio
async def test_grep_fallback_files_with_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.py").write_text("foo\n")
    (tmp_path / "b.py").write_text("bar\n")
    tools = _tools(tmp_path)
    monkeypatch.setattr("nullain_tools.filesystem.shutil.which", _which_none)
    out = await tools["grep"].func(pattern="foo", output_mode="files_with_matches")
    assert "a.py" in out
    assert "b.py" not in out


@pytest.mark.asyncio
async def test_grep_fallback_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a.py").write_text("foo\nfoo\nbar\n")
    tools = _tools(tmp_path)
    monkeypatch.setattr("nullain_tools.filesystem.shutil.which", _which_none)
    out = await tools["grep"].func(pattern="foo", output_mode="count")
    assert "a.py:2" in out


@pytest.mark.asyncio
async def test_grep_fallback_announces_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.py").write_text("foo\nfoo\nfoo\n")
    tools = _tools(tmp_path)
    monkeypatch.setattr("nullain_tools.filesystem.shutil.which", _which_none)
    out = await tools["grep"].func(pattern="foo", head_limit=1)
    assert "truncated: showing 1 of 3" in out


@pytest.mark.asyncio
async def test_grep_fallback_case_insensitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.py").write_text("FOO\n")
    tools = _tools(tmp_path)
    monkeypatch.setattr("nullain_tools.filesystem.shutil.which", _which_none)
    assert "a.py:1:FOO" in await tools["grep"].func(pattern="foo", case_insensitive=True)
    assert "No matches" in await tools["grep"].func(pattern="foo")


# ---------------------------------------------------------------------------
# todo_write: single in_progress + TodoEvent emission (via registry)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_todo_write_rejects_multiple_in_progress(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    out = await reg.execute(
        "todo_write",
        {
            "items": [
                {"content": "a", "status": "in_progress"},
                {"content": "b", "status": "in_progress"},
            ]
        },
    )
    assert "at most one item may be 'in_progress'" in out.output


@pytest.mark.asyncio
async def test_todo_write_emits_todo_event(tmp_path: Path) -> None:
    bus = EventBus()
    received: list[TodoEvent] = []

    async def track(ev: BaseEvent) -> None:
        if isinstance(ev, TodoEvent):
            received.append(ev)

    bus.subscribe("todo", track)
    reg = _registry(tmp_path, event_bus=bus, session_id="sess-1")
    out = await reg.execute(
        "todo_write",
        {
            "items": [
                {"content": "a", "status": "in_progress"},
                {"content": "b", "status": "pending"},
            ]
        },
    )
    assert "Todo list updated" in out.output
    assert len(received) == 1
    assert received[0].session_id == "sess-1"
    assert received[0].items[0].content == "a"
    assert received[0].items[0].status == "in_progress"


@pytest.mark.asyncio
async def test_todo_write_without_bus_still_validates(tmp_path: Path) -> None:
    reg = _registry(tmp_path)  # no event_bus
    out = await reg.execute("todo_write", {"items": [{"content": "a", "status": "completed"}]})
    assert "Todo list updated" in out.output


# ---------------------------------------------------------------------------
# TodoEvent round-trip through EventStore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_todo_event_roundtrip_through_event_store() -> None:
    store = EventStore()
    await store.initialize()
    ev = TodoEvent(
        session_id="s1",
        items=(TodoItem(content="a", status="in_progress"),),
    )
    await store.append(ev)
    loaded = await store.get_session_events("s1")
    assert len(loaded) == 1
    assert isinstance(loaded[0], TodoEvent)
    assert loaded[0].items[0].content == "a"
    await store.close()


# ---------------------------------------------------------------------------
# FileAccessTracker
# ---------------------------------------------------------------------------


def test_file_access_tracker_marks_and_checks(tmp_path: Path) -> None:
    tracker = FileAccessTracker()
    p = tmp_path / "a.txt"
    assert not tracker.was_read(p)
    tracker.mark_read(p)
    assert tracker.was_read(p)
