"""Nullain Tools — File system operations (read/write/edit/multi_edit/grep/glob/todo).

M8 turns the file tools into a coding-grade toolset: paged/numbered reads,
safe edits (ambiguous-match and no-op rejection, read-before-edit), atomic
multi-hunk edits, ripgrep-backed search with a consistent fallback, and a
todo list that mirrors progress to the client via ``TodoEvent``.
"""

import asyncio
import contextlib
import os
import re
import shutil
from pathlib import Path
from typing import Literal

from nullain.authority import Capability
from nullain.events import EventBus, TodoEvent, TodoItem
from nullain.tools import RegisteredTool, redact_secrets, resolve_and_validate_path, tool
from nullain.tools.result import ToolResult
from pydantic import BaseModel

from nullain_tools.checkpoints import CheckpointStore

#: Individual lines longer than this are truncated with an explicit marker so a
#: minified file cannot blow the model's context window.
MAX_LINE_LEN = 2000
#: Default page size for :func:`read_file`.
DEFAULT_READ_LIMIT = 2000
#: Default cap on grep results before truncation is announced.
DEFAULT_GREP_HEAD_LIMIT = 200
#: Subprocess timeout for ripgrep (seconds).
RG_TIMEOUT = 30.0

_IGNORED_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".hypothesis",
}


class FileAccessTracker:
    """Tracks which workspace files have been read in a session (M8) and
    caches read_file's rendered page (M14).

    ``read_file`` marks a resolved path as read; ``edit_file``/``multi_edit``
    require the target to have been read first, so the model only edits files it
    has actually seen. A fresh instance is created per toolset by default; the
    daemon injects one per session so the read-set is scoped to the session.

    The read cache (M14) keys a rendered page on ``(path, offset, limit,
    mtime_ns, size)``: a repeated ``read_file`` call with the same arguments
    hits the cache without touching disk, but the mtime/size componenets of
    the key mean any change to the file — whether from this session's own
    ``write_file``/``edit_file``/``multi_edit``, a ``bash`` command, or an
    external process — silently misses the cache and falls through to a
    real read, rather than requiring every writer to remember to invalidate
    a separate cache explicitly.

    A cache hit (M20) returns a short pointer ("unchanged since your last
    read...") instead of re-serving the cached body verbatim — the earlier
    call already put that exact content in the model's context, so
    resending it costs real tokens for information the model already has.
    This mirrors Claude Code's session-scoped file read cache, which avoids
    re-paying the token/latency cost of re-reading files the model has
    already seen and that have not changed.
    """

    def __init__(self) -> None:
        self._read: set[str] = set()
        self._page_cache: dict[tuple[str, int, int, int, int], str] = {}

    def mark_read(self, path: Path) -> None:
        """Record that ``path`` (already resolved) was read this session."""
        self._read.add(str(path))

    def was_read(self, path: Path) -> bool:
        """Whether ``path`` (already resolved) was read this session."""
        return str(path) in self._read

    @staticmethod
    def _page_key(
        path: Path, offset: int, limit: int, stat: "os.stat_result"
    ) -> tuple[str, int, int, int, int]:
        return (str(path), offset, limit, stat.st_mtime_ns, stat.st_size)

    def get_page(self, path: Path, offset: int, limit: int) -> str | None:
        """Return the cached rendered page for this exact read, if still fresh.

        Returns ``None`` on a cache miss — including when the file's mtime or
        size has changed since it was cached, or when ``path`` no longer
        exists (stat failure), so the caller always falls back to a real read
        rather than risking a stale result.
        """
        try:
            stat = path.stat()
        except OSError:
            return None
        return self._page_cache.get(self._page_key(path, offset, limit, stat))

    def put_page(self, path: Path, offset: int, limit: int, rendered: str) -> None:
        """Cache a rendered page, keyed on the file's mtime/size at cache time."""
        try:
            stat = path.stat()
        except OSError:
            return
        self._page_cache[self._page_key(path, offset, limit, stat)] = rendered


class FileEdit(BaseModel):
    """A single hunk in an atomic :func:`multi_edit` call (M8)."""

    old_str: str
    new_str: str
    replace_all: bool = False


def _numbered_snippet(content: str, anchor: str, context: int = 3) -> str:
    """Return a numbered window of lines around the first line containing ``anchor``.

    Used to show the model a visual confirmation of what an edit changed.
    """
    lines = content.splitlines()
    anchor_idx = next((i for i, ln in enumerate(lines) if anchor in ln), None)
    if anchor_idx is None:
        return ""
    start = max(0, anchor_idx - context)
    end = min(len(lines), anchor_idx + context + 1)
    return "\n".join(f"{i + 1}\t{lines[i]}" for i in range(start, end))


def _truncate_line(line: str) -> str:
    """Truncate an absurdly long line with an explicit marker."""
    if len(line) <= MAX_LINE_LEN:
        return line
    return f"{line[:MAX_LINE_LEN]} ... [truncated {len(line) - MAX_LINE_LEN} chars]"


def _grep_python(
    pattern: str,
    search_dir: Path,
    root: Path,
    output_mode: Literal["content", "files_with_matches", "count"],
    context_lines: int,
    glob_filter: str | None,
    case_insensitive: bool,
    head_limit: int,
) -> str | ToolResult:
    """Pure-Python grep fallback producing the same observable format as ripgrep.

    Content lines are ``relpath:linenum:line``; context lines use ``-`` as the
    separator (matching ``rg --no-heading``). Truncation is always announced.
    """
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as err:
        return ToolResult(
            output=f"Error: Invalid regex pattern '{pattern}': {err}",
            is_error=True,
            error_type="ToolError",
        )

    glob_re = re.compile(glob_filter) if glob_filter else None
    matches: list[str] = []
    total = 0
    for file_path in search_dir.rglob("*"):
        if any(part in _IGNORED_DIRS for part in file_path.parts):
            continue
        if not file_path.is_file() or file_path.name.startswith("."):
            continue
        if glob_re is not None and not glob_re.search(str(file_path)):
            continue
        rel = file_path.relative_to(root)
        content = ""
        with contextlib.suppress(Exception):
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()
        if output_mode == "count":
            count = sum(1 for ln in lines if regex.search(ln))
            if count:
                total += count
                matches.append(f"{rel}:{count}")
            continue
        if output_mode == "files_with_matches":
            if any(regex.search(ln) for ln in lines):
                total += 1
                matches.append(str(rel).replace("\\", "/"))
            continue
        # content mode
        for i, ln in enumerate(lines):
            if not regex.search(ln):
                continue
            total += 1
            if len(matches) >= head_limit:
                continue
            for c in range(max(0, i - context_lines), i):
                matches.append(f"{rel}-{c + 1}-{_truncate_line(lines[c])}")
            matches.append(f"{rel}:{i + 1}:{_truncate_line(ln)}")
            for c in range(i + 1, min(len(lines), i + 1 + context_lines)):
                matches.append(f"{rel}-{c + 1}-{_truncate_line(lines[c])}")

    if not matches:
        return f"No matches found for pattern '{pattern}'."
    body = "\n".join(matches[:head_limit])
    if total > head_limit:
        body += f"\n... (truncated: showing {head_limit} of {total} matches)"
    return redact_secrets(body)


async def _grep_with_rg(
    rg: str,
    pattern: str,
    search_dir: Path,
    root: Path,
    output_mode: Literal["content", "files_with_matches", "count"],
    context_lines: int,
    glob_filter: str | None,
    case_insensitive: bool,
    head_limit: int,
) -> str | ToolResult:
    """Run ripgrep by explicit argv (never shell) and format like the fallback."""
    args = [rg, "--no-heading", "--line-number"]
    if case_insensitive:
        args.append("-i")
    if context_lines > 0:
        args.extend(["-C", str(context_lines)])
    if glob_filter:
        args.extend(["--glob", glob_filter])
    if output_mode == "files_with_matches":
        args.append("-l")
    elif output_mode == "count":
        args.append("-c")
    args.append(pattern)
    args.append(str(search_dir))

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=RG_TIMEOUT)
    except TimeoutError:
        return ToolResult(
            output=f"Error: ripgrep timed out after {RG_TIMEOUT}s for pattern '{pattern}'.",
            is_error=True,
            error_type="ToolError",
        )
    except OSError as err:
        return ToolResult(
            output=f"Error: failed to run ripgrep: {err}",
            is_error=True,
            error_type="ToolError",
        )

    if proc.returncode not in (0, 1):
        err_text = stderr.decode("utf-8", errors="replace").strip()
        return ToolResult(
            output=f"Error: ripgrep failed (exit {proc.returncode}): {err_text}",
            is_error=True,
            error_type="ToolError",
        )

    text = stdout.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln]
    if not lines:
        return f"No matches found for pattern '{pattern}'."

    # Normalise paths to be relative to the workspace root (rg prints absolute).
    rel_lines: list[str] = []
    for ln in lines:
        if output_mode == "files_with_matches":
            rel_lines.append(_rel_path(ln, root))
        elif output_mode == "count":
            path_part, _, count = ln.rpartition(":")
            rel_lines.append(f"{_rel_path(path_part, root)}:{count}")
        else:
            rel_lines.append(_rel_path(ln, root))

    body = "\n".join(rel_lines[:head_limit])
    if len(rel_lines) > head_limit:
        body += f"\n... (truncated: showing {head_limit} of {len(rel_lines)} matches)"
    return redact_secrets(body)


def _rel_path(line: str, root: Path) -> str:
    """Rewrite an absolute path prefix in a grep output line to be workspace-relative."""
    try:
        path_part, _, rest = line.partition(":")
        p = Path(path_part)
        if p.is_absolute():
            try:
                rel = p.relative_to(root)
                return f"{rel.as_posix()}:{rest}"
            except ValueError:
                return line
        return line
    except Exception:
        return line


def create_filesystem_tools(
    workspace_root: str | Path,
    file_access_tracker: FileAccessTracker | None = None,
    event_bus: EventBus | None = None,
    session_id: str = "default",
    checkpoint_store: CheckpointStore | None = None,
) -> list[RegisteredTool]:
    """Create the built-in filesystem tools.

    Args:
        workspace_root: Workspace root; all paths are validated against it.
        file_access_tracker: Session-scoped read tracker (M8). When None, a
            fresh instance is created per call (backward compatible).
        event_bus: Optional event bus; when provided, ``todo_write`` emits a
            ``TodoEvent`` on each update. Without it, the tool validates and
            returns without emitting (graceful degradation).
        session_id: Session id stamped on emitted ``TodoEvent``s.
        checkpoint_store: Session-scoped checkpoint store (M11). When provided,
            ``write_file``/``edit_file``/``multi_edit`` snapshot the pre-write
            state and an ``undo`` tool is registered. When None, writes proceed
            without snapshots and ``undo`` reports no checkpoints (backward
            compatible).
    """
    root = Path(workspace_root).resolve()
    tracker = file_access_tracker or FileAccessTracker()

    @tool(
        name="read_file",
        description=(
            "Read a text file within the workspace, returning numbered lines "
            "(cat -n format). Use offset/limit to page through large files."
        ),
        read_only=True,
        requires=frozenset({Capability.READ}),
    )
    def read_file(path: str, offset: int = 0, limit: int = DEFAULT_READ_LIMIT) -> str | ToolResult:
        target = resolve_and_validate_path(root, path)
        if not target.exists():
            return ToolResult(
                output=f"Error: File '{path}' does not exist.",
                is_error=True,
                error_type="ToolError",
            )
        if not target.is_file():
            return ToolResult(
                output=f"Error: Path '{path}' is not a regular file.",
                is_error=True,
                error_type="ToolError",
            )
        if offset < 0:
            return ToolResult(
                output=f"Error: offset must be >= 0, got {offset}.",
                is_error=True,
                error_type="ToolError",
            )
        if limit < 1:
            return ToolResult(
                output=f"Error: limit must be >= 1, got {limit}.",
                is_error=True,
                error_type="ToolError",
            )

        tracker.mark_read(target)

        cached = tracker.get_page(target, offset, limit)
        if cached is not None:
            # The model already has this exact page's content earlier in
            # this same conversation (the disk-read cache proves the file
            # hasn't changed since). Re-sending the full body again costs
            # real tokens for content the model has already seen and
            # doesn't need repeated — return a short pointer instead,
            # mirroring how a human developer would say "same as before"
            # rather than re-pasting a file they already showed you.
            return (
                f"(unchanged since your last read_file('{path}', offset={offset}, "
                f"limit={limit}) in this conversation — see that earlier result "
                "for the content)"
            )

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        start = offset
        end = min(offset + limit, total)

        numbered = [f"{i + 1}\t{_truncate_line(lines[i])}" for i in range(start, end)]
        body = redact_secrets("\n".join(numbered))
        remaining = total - end
        if remaining > 0:
            body += f"\n... ({remaining} more lines; call read_file with offset={end} to continue)"
        tracker.put_page(target, offset, limit, body)
        return body

    @tool(
        name="write_file",
        description="Write text content to a file in the workspace.",
        requires=frozenset({Capability.WRITE}),
    )
    async def write_file(path: str, content: str) -> str | ToolResult:
        target = resolve_and_validate_path(root, path)
        if checkpoint_store is None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        else:
            # One undo unit per tool call: the write and its snapshot revert
            # together even if the tool later grows to touch several files.
            async with checkpoint_store.operation():
                await checkpoint_store.snapshot(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to '{path}'."

    @tool(
        name="edit_file",
        description=(
            "Perform exact string replacement (old_str with new_str) in a "
            "workspace file. Fails if old_str is ambiguous (appears more than "
            "once) unless replace_all is set, and rejects no-op edits."
        ),
        requires=frozenset({Capability.WRITE}),
    )
    async def edit_file(
        path: str, old_str: str, new_str: str, replace_all: bool = False
    ) -> str | ToolResult:
        target = resolve_and_validate_path(root, path)
        if not target.exists():
            return ToolResult(
                output=f"Error: File '{path}' does not exist.",
                is_error=True,
                error_type="ToolError",
            )
        if not tracker.was_read(target):
            return ToolResult(
                output=(
                    f"Error: File '{path}' has not been read in this session. Call read_file first."
                ),
                is_error=True,
                error_type="ToolError",
            )
        if old_str == new_str:
            return ToolResult(
                output="Error: old_str and new_str are identical — no-op edit rejected.",
                is_error=True,
                error_type="ToolError",
            )

        content = target.read_text(encoding="utf-8")
        count = content.count(old_str)
        if count == 0:
            return ToolResult(
                output=f"Error: Target string not found in '{path}'.",
                is_error=True,
                error_type="ToolError",
            )
        if count > 1 and not replace_all:
            return ToolResult(
                output=(
                    f"Error: Target string appears {count} times in '{path}'. "
                    "Set replace_all=True to replace all, or widen old_str to make "
                    "it unique."
                ),
                is_error=True,
                error_type="ToolError",
            )

        if replace_all:
            updated = content.replace(old_str, new_str)
        else:
            updated = content.replace(old_str, new_str, 1)
        if checkpoint_store is None:
            target.write_text(updated, encoding="utf-8")
        else:
            async with checkpoint_store.operation():
                await checkpoint_store.snapshot(target)
                target.write_text(updated, encoding="utf-8")
        snippet = redact_secrets(_numbered_snippet(updated, new_str))
        return f"Edited '{path}'.\n{snippet}"

    @tool(
        name="multi_edit",
        description=(
            "Apply a list of edits to a file atomically. All edits are applied "
            "in sequence over in-memory content; the file is written only if "
            "every edit succeeds. Each edit operates on the result of the "
            "previous one (chained edits)."
        ),
        requires=frozenset({Capability.WRITE}),
    )
    async def multi_edit(path: str, edits: list[FileEdit]) -> str | ToolResult:
        # Coerce raw dicts (from direct .func calls) to FileEdit models; the
        # registry passes arguments through without pydantic validation.
        edits = [FileEdit.model_validate(e) for e in edits]
        target = resolve_and_validate_path(root, path)
        if not target.exists():
            return ToolResult(
                output=f"Error: File '{path}' does not exist.",
                is_error=True,
                error_type="ToolError",
            )
        if not tracker.was_read(target):
            return ToolResult(
                output=(
                    f"Error: File '{path}' has not been read in this session. Call read_file first."
                ),
                is_error=True,
                error_type="ToolError",
            )
        if not edits:
            return ToolResult(
                output="Error: edits list is empty.",
                is_error=True,
                error_type="ToolError",
            )

        content = target.read_text(encoding="utf-8")
        for idx, edit in enumerate(edits):
            if edit.old_str == edit.new_str:
                return ToolResult(
                    output=f"Error: edit #{idx + 1} is a no-op (old_str == new_str).",
                    is_error=True,
                    error_type="ToolError",
                )
            count = content.count(edit.old_str)
            if count == 0:
                return ToolResult(
                    output=f"Error: edit #{idx + 1} target string not found in '{path}'.",
                    is_error=True,
                    error_type="ToolError",
                )
            if count > 1 and not edit.replace_all:
                return ToolResult(
                    output=(
                        f"Error: edit #{idx + 1} target string appears {count} times "
                        f"in '{path}'. Set replace_all=True or widen old_str."
                    ),
                    is_error=True,
                    error_type="ToolError",
                )
            content = (
                content.replace(edit.old_str, edit.new_str)
                if edit.replace_all
                else content.replace(edit.old_str, edit.new_str, 1)
            )

        if checkpoint_store is None:
            target.write_text(content, encoding="utf-8")
        else:
            # All hunks of one multi_edit are a single undo unit — reverting
            # only part of an atomic edit would defeat its purpose.
            async with checkpoint_store.operation():
                await checkpoint_store.snapshot(target)
                target.write_text(content, encoding="utf-8")
        snippet = redact_secrets(_numbered_snippet(content, edits[-1].new_str))
        return f"Applied {len(edits)} edits to '{path}'.\n{snippet}"

    @tool(
        name="grep",
        description=(
            "Search for a regex pattern across files in the workspace. Uses "
            "ripgrep when available (same output format as the fallback). "
            "output_mode: content | files_with_matches | count."
        ),
        read_only=True,
        requires=frozenset({Capability.READ}),
    )
    async def grep(
        pattern: str,
        relative_dir: str = ".",
        output_mode: Literal["content", "files_with_matches", "count"] = "content",
        context_lines: int = 0,
        glob_filter: str | None = None,
        case_insensitive: bool = False,
        head_limit: int = DEFAULT_GREP_HEAD_LIMIT,
    ) -> str | ToolResult:
        search_dir = resolve_and_validate_path(root, relative_dir)
        if not search_dir.is_dir():
            return ToolResult(
                output=f"Error: Path '{relative_dir}' is not a directory.",
                is_error=True,
                error_type="ToolError",
            )
        if output_mode not in ("content", "files_with_matches", "count"):
            return ToolResult(
                output=f"Error: invalid output_mode '{output_mode}'.",
                is_error=True,
                error_type="ToolError",
            )
        if context_lines < 0:
            return ToolResult(
                output=f"Error: context_lines must be >= 0, got {context_lines}.",
                is_error=True,
                error_type="ToolError",
            )
        if head_limit < 1:
            return ToolResult(
                output=f"Error: head_limit must be >= 1, got {head_limit}.",
                is_error=True,
                error_type="ToolError",
            )

        rg = shutil.which("rg")
        if rg is not None:
            return await _grep_with_rg(
                rg,
                pattern,
                search_dir,
                root,
                output_mode,
                context_lines,
                glob_filter,
                case_insensitive,
                head_limit,
            )
        return _grep_python(
            pattern,
            search_dir,
            root,
            output_mode,
            context_lines,
            glob_filter,
            case_insensitive,
            head_limit,
        )

    @tool(
        name="glob",
        description=(
            "Find files in the workspace matching a glob pattern "
            "(e.g. '**/*.py', 'src/**/*.ts'). Returns matching paths, one per line."
        ),
        read_only=True,
        requires=frozenset({Capability.READ}),
    )
    def glob(pattern: str, relative_dir: str = ".") -> str | ToolResult:
        search_dir = resolve_and_validate_path(root, relative_dir)
        if not search_dir.is_dir():
            return ToolResult(
                output=f"Error: Path '{relative_dir}' is not a directory.",
                is_error=True,
                error_type="ToolError",
            )

        matches: list[str] = []
        try:
            for file_path in search_dir.glob(pattern):
                if any(part in _IGNORED_DIRS for part in file_path.parts):
                    continue
                if file_path.is_file() and not file_path.name.startswith("."):
                    rel = file_path.relative_to(root)
                    matches.append(str(rel).replace("\\", "/"))
                    if len(matches) >= 200:
                        break
        except (ValueError, OSError) as err:
            return ToolResult(
                output=f"Error: glob failed for pattern '{pattern}': {err}",
                is_error=True,
                error_type="ToolError",
            )

        if not matches:
            return f"No files matched pattern '{pattern}'."
        return "\n".join(matches)

    @tool(
        name="list_directory",
        description=(
            "List entries in a workspace directory. Each line is 'name/' for "
            "directories and 'name' for files."
        ),
        read_only=True,
        requires=frozenset({Capability.READ}),
    )
    def list_directory(relative_dir: str = ".") -> str | ToolResult:
        target = resolve_and_validate_path(root, relative_dir)
        if not target.exists():
            return ToolResult(
                output=f"Error: Path '{relative_dir}' does not exist.",
                is_error=True,
                error_type="ToolError",
            )
        if not target.is_dir():
            return ToolResult(
                output=f"Error: Path '{relative_dir}' is not a directory.",
                is_error=True,
                error_type="ToolError",
            )

        entries: list[str] = []
        for entry in sorted(target.iterdir()):
            if entry.name in _IGNORED_DIRS or entry.name.startswith("."):
                continue
            entries.append(f"{entry.name}/" if entry.is_dir() else entry.name)
        if not entries:
            return f"Directory '{relative_dir}' is empty."
        return "\n".join(entries)

    @tool(
        name="todo_write",
        description=(
            "Update the agent's todo list. Each item has content and a status "
            "of pending | in_progress | completed. At most one item may be "
            "in_progress at a time."
        ),
        read_only=False,
        requires=frozenset(),
    )
    async def todo_write(items: list[TodoItem]) -> str | ToolResult:
        # Coerce raw dicts (from direct .func calls) to TodoItem models.
        items = [TodoItem.model_validate(it) for it in items]
        in_progress = [it for it in items if it.status == "in_progress"]
        if len(in_progress) > 1:
            return ToolResult(
                output=f"Error: at most one item may be 'in_progress', got {len(in_progress)}.",
                is_error=True,
                error_type="ToolError",
            )
        if event_bus is not None:
            await event_bus.publish(TodoEvent(session_id=session_id, items=tuple(items)))
        rendered = "\n".join(f"- [{it.status}] {it.content}" for it in items)
        return f"Todo list updated:\n{rendered}"

    @tool(
        name="undo",
        description=(
            "Restore the most recent pre-write checkpoint of a file. Reverts "
            "the last write_file/edit_file/multi_edit. Returns a summary of what "
            "was restored, or a message if there is nothing to undo."
        ),
        requires=frozenset({Capability.WRITE}),
    )
    async def undo() -> str | ToolResult:
        if checkpoint_store is None:
            return ToolResult(
                output="Error: no checkpoint store configured for this session.",
                is_error=True,
                error_type="ToolError",
            )
        return await checkpoint_store.undo()

    return [
        read_file,
        write_file,
        edit_file,
        multi_edit,
        grep,
        glob,
        list_directory,
        todo_write,
        undo,
    ]


__all__ = ["FileAccessTracker", "create_filesystem_tools"]
