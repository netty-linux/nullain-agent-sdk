"""Nullain Agent SDK — Persistent Auto-Memory.

A Claude-Code-style persistent memory store: a ``MEMORY.md`` index file with a
soft byte cap plus one topic file per fact. The index is re-injected into the
system prompt so accumulated facts survive context compaction.

Layout (under ``memory_dir``, default ``<workspace>/.nullain/memory``)::

    MEMORY.md            # index: one "- [name](name.md) — description" per fact
    <name>.md            # one topic file per fact, with frontmatter + body

The index is capped at ``MAX_INDEX_BYTES`` (25 KB). When a write pushes the
index over the cap, the oldest entries are evicted (their index lines removed
and their topic files deleted) until it fits again — bounding total memory
without losing the most recent context.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from nullain.telemetry import get_logger

logger = get_logger(__name__)

#: Soft cap on the MEMORY.md index file size in bytes.
MAX_INDEX_BYTES = 25_000

_INDEX_HEADER = "# Memory Index\n\n"

#: Valid memory slugs: lowercase kebab-case, 1-64 chars.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class MemoryType(StrEnum):
    """Category of a stored fact, mirroring Claude's memory taxonomy."""

    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"


class MemoryEntry(BaseModel):
    """One persisted fact. ``body`` holds the raw markdown content."""

    name: str
    description: str
    type: MemoryType = MemoryType.REFERENCE
    body: str = ""

    def to_topic_file(self) -> str:
        """Serialize this entry to a topic file (frontmatter + body)."""
        return (
            "---\n"
            f"name: {self.name}\n"
            f"description: {self.description}\n"
            "metadata:\n"
            f"  type: {self.type.value}\n"
            "---\n"
            f"{self.body.strip()}\n"
        )


def _parse_topic_file(text: str) -> MemoryEntry | None:
    """Parse a topic file into a MemoryEntry, or None if malformed."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    try:
        close = lines.index("---", 1)
    except ValueError:
        return None
    front = lines[1:close]
    body = "\n".join(lines[close + 1 :]).strip()

    name = ""
    description = ""
    mtype: MemoryType = MemoryType.REFERENCE
    in_metadata = False
    for line in front:
        if not line.strip():
            continue
        if line.strip() == "metadata:":
            in_metadata = True
            continue
        if in_metadata:
            if line.startswith("  type:"):
                raw = line.split(":", 1)[1].strip()
                try:
                    mtype = MemoryType(raw)
                except ValueError:
                    mtype = MemoryType.REFERENCE
            continue
        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("description:"):
            description = line.split(":", 1)[1].strip()
    if not name:
        return None
    return MemoryEntry(name=name, description=description, type=mtype, body=body)


def _index_line(entry: MemoryEntry) -> str:
    return f"- [{entry.name}]({entry.name}.md) — {entry.description}"


def _slug_from_index_line(line: str) -> str | None:
    """Extract the slug from an index line like ``- [x](x.md) — desc``."""
    m = re.match(r"- \[([^\]]+)\]\(([^\)]+)\)", line)
    if not m:
        return None
    link = m.group(2)
    if link.endswith(".md"):
        return link[:-3]
    return link


class PersistentMemory:
    """File-backed persistent memory with a capped MEMORY.md index.

    All methods are synchronous: the files are tiny and the operations are
    cheap, so blocking the event loop briefly is preferable to wrapping every
    call in a thread executor.
    """

    def __init__(
        self,
        memory_dir: str | Path | None = None,
        *,
        workspace_root: str | Path | None = None,
        max_index_bytes: int = MAX_INDEX_BYTES,
    ) -> None:
        if memory_dir is not None:
            self.memory_dir = Path(memory_dir).resolve()
        elif workspace_root is not None:
            self.memory_dir = (Path(workspace_root).resolve() / ".nullain" / "memory").resolve()
        else:
            self.memory_dir = Path(".nullain", "memory").resolve()
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.memory_dir / "MEMORY.md"
        self.max_index_bytes = max_index_bytes

    # ------------------------------------------------------------------ index

    def _read_index_entries(self) -> list[str]:
        """Return the raw bullet lines of the index in file order."""
        if not self.index_file.exists():
            return []
        return [
            ln
            for ln in self.index_file.read_text(encoding="utf-8").splitlines()
            if ln.startswith("- ")
        ]

    def _write_index(self, lines: list[str]) -> None:
        body = _INDEX_HEADER + "\n".join(lines) + ("\n" if lines else "")
        self.index_file.write_text(body, encoding="utf-8")

    def _enforce_cap(self, lines: list[str]) -> list[str]:
        """Evict oldest index lines (and their topic files) until under cap."""
        while lines and len(self._render_index(lines).encode()) > self.max_index_bytes:
            evicted = lines.pop(0)
            slug = _slug_from_index_line(evicted)
            if slug:
                self._delete_topic_file(slug)
                logger.info("memory_evicted", slug=slug, reason="index_cap")
        return lines

    def _render_index(self, lines: list[str]) -> str:
        return _INDEX_HEADER + "\n".join(lines) + ("\n" if lines else "")

    # --------------------------------------------------------------- topic i/o

    def _topic_path(self, name: str) -> Path:
        # Defense in depth: the slug regex already forbids separators, but
        # resolve + is_relative_to guarantees no escape if the regex is ever
        # relaxed.
        path = (self.memory_dir / f"{name}.md").resolve()
        if not path.is_relative_to(self.memory_dir):
            raise ValueError(f"memory name escapes memory_dir: {name!r}")
        return path

    def _delete_topic_file(self, name: str) -> None:
        path = self._topic_path(name)
        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------ public

    def write(self, entry: MemoryEntry) -> None:
        """Create or update a memory entry and its index pointer.

        Raises ValueError if ``entry.name`` is not a valid slug.
        """
        if not _SLUG_RE.match(entry.name):
            raise ValueError(
                f"memory name must be lowercase kebab-case (a-z0-9-), got: {entry.name!r}"
            )
        self._topic_path(entry.name).write_text(entry.to_topic_file(), encoding="utf-8")

        lines = self._read_index_entries()
        # Replace existing line for this slug, else append.
        replaced = False
        for i, ln in enumerate(lines):
            if _slug_from_index_line(ln) == entry.name:
                lines[i] = _index_line(entry)
                replaced = True
                break
        if not replaced:
            lines.append(_index_line(entry))

        lines = self._enforce_cap(lines)
        self._write_index(lines)

    def read(self, name: str) -> str | None:
        """Return the body of a memory entry by slug, or None if absent."""
        path = self._topic_path(name)
        if not path.exists():
            return None
        entry = _parse_topic_file(path.read_text(encoding="utf-8"))
        return entry.body if entry else None

    def delete(self, name: str) -> bool:
        """Delete a memory entry and its index line. Returns True if removed."""
        path = self._topic_path(name)
        existed = path.exists()
        self._delete_topic_file(name)
        lines = self._read_index_entries()
        new_lines = [ln for ln in lines if _slug_from_index_line(ln) != name]
        if new_lines != lines:
            self._write_index(new_lines)
        return existed or new_lines != lines

    def list_entries(self) -> list[str]:
        """Return slugs in index order."""
        return [s for s in (_slug_from_index_line(ln) for ln in self._read_index_entries()) if s]

    def to_context(self) -> str:
        """Render the index as a system-prompt block, or empty string if none."""
        lines = self._read_index_entries()
        if not lines:
            return ""
        return "# PERSISTENT MEMORY INDEX\n" + "\n".join(lines) + "\n"


__all__ = [
    "MAX_INDEX_BYTES",
    "MemoryEntry",
    "MemoryType",
    "PersistentMemory",
]
