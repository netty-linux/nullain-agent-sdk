"""Nullain Tools — Checkpoint and undo for safe file writes (M11).

Before each write operation (``write_file``, ``edit_file``, ``multi_edit``) a
restorable snapshot of the file's pre-write state is taken. The ``undo`` tool
restores the most recent *operation*.

Operations, not files, are the unit of undo: a single logical write may touch
several files, and reverting only the last one would leave the workspace in a
half-reverted state the model cannot reason about. Snapshots taken inside one
``operation`` context share an ``operation_id`` and are restored together, in
reverse order.

Storage mechanism (M11.1): when the workspace is a git repository, the pre-write
content is stored as a git blob via ``git hash-object -w`` and restored via
``git cat-file blob``. This writes to the object database only — it never
touches the user's index, refs, or commit history (a hard requirement). When the
workspace is not a git repository, the content is copied under
``.nullain/checkpoints/``. Retention is bounded by ``max_checkpoints`` and
``max_bytes``; the oldest checkpoints are evicted.

Subprocesses are always launched via ``asyncio.create_subprocess_exec`` with an
explicit argv list — never ``shell=True`` (AGENTS.md rule 6).
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal

from nullain.errors import ToolError
from nullain.telemetry import get_logger
from pydantic import BaseModel

logger = get_logger(__name__)


class Checkpoint(BaseModel):
    """A single restorable snapshot of a file's pre-write state (M11).

    Attributes:
        id: Monotonic snapshot identifier.
        operation_id: Identifier of the logical write operation this snapshot
            belongs to. All snapshots taken inside one ``operation`` context
            share it and are undone together.
        relpath: Workspace-relative path of the snapshotted file.
        storage: Where the pre-write content lives (git object db or a copy).
        ref: Git blob hash, or the path of the copied file.
        size: Size of the snapshotted content in bytes.
        existed: Whether the file existed before the write (False means undo
            deletes it rather than restoring content).
    """

    id: int
    operation_id: int
    relpath: str
    storage: Literal["git", "copy"]
    ref: str
    size: int
    existed: bool


class CheckpointStore:
    """Stores restorable snapshots taken before each file write.

    Args:
        workspace_root: Workspace root; all paths are validated against it.
        max_checkpoints: Maximum number of checkpoints retained before the
            oldest are evicted.
        max_bytes: Maximum total checkpoint bytes retained before the oldest
            are evicted.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        max_checkpoints: int = 20,
        max_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        self._root = Path(workspace_root).resolve()
        self._checkpoints_dir = self._root / ".nullain" / "checkpoints"
        self._max_checkpoints = max_checkpoints
        self._max_bytes = max_bytes
        self._stack: list[Checkpoint] = []
        self._counter = 0
        self._operation_counter = 0
        self._current_operation: int | None = None
        # Number of whole operations dropped by retention. Surfaced by ``undo``
        # so a silently-truncated history is never mistaken for an empty one.
        self._evicted_operations = 0
        self._git = shutil.which("git") if self._is_git_repo() else None

    @property
    def checkpoint_count(self) -> int:
        """Number of checkpoints currently retained (after eviction)."""
        return len(self._stack)

    @property
    def operation_count(self) -> int:
        """Number of distinct undoable operations currently retained."""
        return len({c.operation_id for c in self._stack})

    @property
    def evicted_operations(self) -> int:
        """Number of operations dropped by retention and no longer undoable."""
        return self._evicted_operations

    @contextlib.asynccontextmanager
    async def operation(self) -> AsyncGenerator[None]:
        """Group every snapshot taken inside this context into one undo unit.

        Nested use is a no-op: the outermost context defines the operation, so
        a tool that snapshots several files under one logical write yields a
        single undoable operation. Without this context each snapshot is its
        own operation, preserving the previous per-file behaviour for callers
        that never opt in.
        """
        if self._current_operation is not None:
            yield
            return
        self._operation_counter += 1
        self._current_operation = self._operation_counter
        try:
            yield
        finally:
            self._current_operation = None

    def _is_git_repo(self) -> bool:
        """Whether the workspace (or an ancestor) is a git repository."""
        cur = self._root
        while True:
            if (cur / ".git").exists():
                return True
            if cur.parent == cur:
                return False
            cur = cur.parent

    async def _run_git(self, args: list[str], input_bytes: bytes | None = None) -> bytes:
        """Run a git subcommand by argv list, returning its stdout bytes."""
        if self._git is None:
            raise ToolError("git is not available for checkpoint storage")
        stdin = asyncio.subprocess.PIPE if input_bytes is not None else asyncio.subprocess.DEVNULL
        proc = await asyncio.create_subprocess_exec(
            self._git,
            *args,
            stdin=stdin,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._root),
        )
        stdout, stderr = await proc.communicate(input_bytes)
        if proc.returncode != 0:
            raise ToolError(f"git {' '.join(args)} failed: {stderr.decode(errors='replace')}")
        return stdout

    async def snapshot(self, path: Path) -> None:
        """Record the current content of ``path`` (already resolved) as a checkpoint.

        When called inside an :meth:`operation` context, the snapshot joins that
        operation's undo unit; otherwise it forms an operation of its own.
        """
        relpath = path.relative_to(self._root).as_posix()
        existed = path.exists()
        content = path.read_bytes() if existed else b""
        if self._git is not None:
            blob = (await self._run_git(["hash-object", "-w", "--stdin"], content)).decode().strip()
            storage: Literal["git", "copy"] = "git"
            ref = blob
        else:
            cp = self._checkpoints_dir / str(self._counter) / relpath
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_bytes(content)
            storage = "copy"
            ref = str(cp)
        if self._current_operation is not None:
            operation_id = self._current_operation
        else:
            self._operation_counter += 1
            operation_id = self._operation_counter
        self._stack.append(
            Checkpoint(
                id=self._counter,
                operation_id=operation_id,
                relpath=relpath,
                storage=storage,
                ref=ref,
                size=len(content),
                existed=existed,
            )
        )
        self._counter += 1
        self._evict()

    async def undo(self) -> str:
        """Restore the most recent operation and return a summary string.

        Every file touched by that operation is restored, in reverse order of
        snapshotting, so a multi-file write reverts as one unit. When retention
        has dropped older operations, the summary says so explicitly rather than
        letting a truncated history read as a complete one.
        """
        if not self._stack:
            if self._evicted_operations:
                return (
                    "No checkpoints to undo. "
                    f"({self._evicted_operations} older operation(s) were dropped by "
                    "checkpoint retention and can no longer be undone.)"
                )
            return "No checkpoints to undo."

        operation_id = self._stack[-1].operation_id
        restored: list[Checkpoint] = []
        while self._stack and self._stack[-1].operation_id == operation_id:
            restored.append(self._stack.pop())

        for cp in restored:
            content = await self._restore(cp)
            target = self._root / cp.relpath
            if not cp.existed:
                target.unlink(missing_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)

        paths = ", ".join(f"'{cp.relpath}'" for cp in restored)
        summary = (
            f"Restored {len(restored)} file(s) from operation #{operation_id}: {paths}."
            if len(restored) > 1
            else f"Restored '{restored[0].relpath}' from operation #{operation_id}."
        )
        if not self._stack and self._evicted_operations:
            summary += (
                f" (No further undo available: {self._evicted_operations} older "
                "operation(s) were dropped by checkpoint retention.)"
            )
        return summary

    async def _restore(self, cp: Checkpoint) -> bytes:
        """Read the stored content of a checkpoint."""
        if cp.storage == "git":
            return await self._run_git(["cat-file", "blob", cp.ref])
        return Path(cp.ref).read_bytes()

    def _evict(self) -> None:
        """Evict the oldest operations beyond the retention bounds.

        Eviction drops whole operations, never a partial one: removing some of
        an operation's files would leave an undo that half-reverts. Each drop is
        logged and counted so :meth:`undo` can tell the caller that history was
        truncated instead of silently reporting nothing to undo.
        """
        while self._over_limits() and self._stack:
            oldest_operation = self._stack[0].operation_id
            dropped: list[Checkpoint] = []
            while self._stack and self._stack[0].operation_id == oldest_operation:
                dropped.append(self._stack.pop(0))
            for cp in dropped:
                if cp.storage == "copy":
                    with contextlib.suppress(FileNotFoundError):
                        Path(cp.ref).unlink()
                # git blobs are left in the object database; git GC reclaims them.
            self._evicted_operations += 1
            logger.info(
                "checkpoint_operation_evicted",
                operation_id=oldest_operation,
                files=len(dropped),
                bytes=sum(c.size for c in dropped),
                reason=(
                    "max_checkpoints"
                    if len(self._stack) + len(dropped) > self._max_checkpoints
                    else "max_bytes"
                ),
                remaining_operations=self.operation_count,
            )

    def _over_limits(self) -> bool:
        """Whether the retained checkpoints exceed either retention bound."""
        return (
            len(self._stack) > self._max_checkpoints
            or sum(c.size for c in self._stack) > self._max_bytes
        )


__all__ = ["Checkpoint", "CheckpointStore"]
