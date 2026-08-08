"""Nullain Agent SDK — OS-level sandbox port (hexagonal boundary).

A ``Sandbox`` adapter confines a subprocess (filesystem + network) at the
operating-system level. The runner (:mod:`nullain.tools.sandbox.runner`) is
sandbox-agnostic: it asks an injected adapter for the extra
:func:`asyncio.create_subprocess_exec` kwargs (``preexec_fn`` on POSIX,
``creationflags`` on Windows) and merges them. Adapters live behind this
``Protocol`` so the core never imports platform specifics (AGENTS.md rule 1).

Fail-closed contract: when an adapter reports ``required`` and is not
``available()`` on the current platform/kernel, the runner raises
:class:`~nullain.errors.SandboxUnavailableError` rather than executing the
subprocess without isolation (AGENTS.md threat model: sandbox by default).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class SandboxOptions:
    """Configuration handed to an adapter when preparing a subprocess.

    ``workspace_root`` is the single path the child may read/write freely;
    ``allow_paths`` are extra read-only or read-write paths the caller has
    explicitly approved (e.g. a pytest cache). ``allow_read_paths`` are extra
    paths granted READ access only (e.g. a shared reference dataset the
    child must not modify) — currently consumed by the seatbelt adapter's
    read-isolation profile (#42); adapters that allow reads globally (or
    don't isolate reads at all) may ignore it. ``deny_network`` requests
    network isolation when the adapter supports it.
    """

    workspace_root: Path
    allow_paths: Sequence[Path] = field(default_factory=tuple)
    allow_read_paths: Sequence[Path] = field(default_factory=tuple)
    deny_network: bool = True


@runtime_checkable
class Sandbox(Protocol):
    """Port: an OS-level subprocess isolation adapter."""

    @property
    def name(self) -> str:
        """Short adapter identifier (``"landlock"``, ``"seatbelt"``, ...)."""
        ...

    @property
    def required(self) -> bool:
        """Whether the runner must fail-closed when this adapter is unavailable.

        ``True`` for the configured platform adapter; ``False`` for
        :class:`~nullain.tools.sandbox.adapters.none.NoSandbox` (explicitly
        disabled) so the runner stays permissive when isolation is opt-out.
        """
        ...

    def available(self) -> bool:
        """Whether this adapter can apply REAL isolation on the current OS/kernel.

        ``False`` on a non-matching platform, an older kernel, or when the
        required syscalls/builtins are unavailable. The runner treats
        ``required and not available()`` as a fail-closed condition.
        """
        ...

    def prepare(self, argv: Sequence[str], opts: SandboxOptions) -> dict[str, Any]:
        """Return extra kwargs to merge into :func:`asyncio.create_subprocess_exec`.

        Typically ``{"preexec_fn": callable}`` on POSIX (applied in the child
        after fork, before exec) or ``{"creationflags": int}`` on Windows. Must
        not mutate ``argv``; the runner still owns the argv list (never shell).
        """
        ...


__all__ = ["Sandbox", "SandboxOptions"]
