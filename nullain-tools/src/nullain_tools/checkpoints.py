"""Nullain Tools — Checkpoint and undo for safe file writes (M11).

Before each write operation (``write_file``, ``edit_file``, ``multi_edit``) a
restorable snapshot of the file's pre-write state is taken. The ``undo`` tool
restores the most recent snapshot.

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
from pathlib import Path
from typing import Literal

from nullain.errors import ToolError
from pydantic import BaseModel


class Checkpoint(BaseModel):
    """A single restorable snapshot of a file's pre-write state (M11)."""

    id: int
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
        self._git = shutil.which("git") if self._is_git_repo() else None

    @property
    def checkpoint_count(self) -> int:
        """Number of checkpoints currently retained (after eviction)."""
        return len(self._stack)

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
        """Record the current content of ``path`` (already resolved) as a checkpoint."""
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
        self._stack.append(
            Checkpoint(
                id=self._counter,
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
        """Restore the most recent checkpoint and return a summary string."""
        if not self._stack:
            return "No checkpoints to undo."
        cp = self._stack.pop()
        content = await self._restore(cp)
        target = self._root / cp.relpath
        if not cp.existed:
            target.unlink(missing_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return f"Restored '{cp.relpath}' from checkpoint #{cp.id}."

    async def _restore(self, cp: Checkpoint) -> bytes:
        """Read the stored content of a checkpoint."""
        if cp.storage == "git":
            return await self._run_git(["cat-file", "blob", cp.ref])
        return Path(cp.ref).read_bytes()

    def _evict(self) -> None:
        """Evict the oldest checkpoints beyond the retention bounds."""
        while (
            len(self._stack) > self._max_checkpoints
            or sum(c.size for c in self._stack) > self._max_bytes
        ):
            oldest = self._stack.pop(0)
            if oldest.storage == "copy":
                with contextlib.suppress(FileNotFoundError):
                    Path(oldest.ref).unlink()
            # git blobs are left in the object database; git GC reclaims them.


__all__ = ["Checkpoint", "CheckpointStore"]
