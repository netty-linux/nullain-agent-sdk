"""Nullain Agent SDK — NoSandbox adapter (explicit opt-out).

Returned by the selector when isolation is disabled in config, or as a
permissive fallback while a platform adapter is being wired. ``required`` is
always ``False`` so the runner never fail-closes on it — choosing NoSandbox is
an explicit acknowledgment that no OS isolation is applied (the PermissionPolicy
path checks still run).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from nullain.tools.sandbox.port import SandboxOptions


class NoSandbox:
    """A Sandbox adapter that applies no OS-level isolation."""

    def __init__(self, required: bool = False) -> None:
        # ``required`` defaults to False: NoSandbox is the opt-out adapter and
        # must never itself trigger a fail-closed refusal.
        self._required = required

    @property
    def name(self) -> str:
        return "none"

    @property
    def required(self) -> bool:
        return self._required

    def available(self) -> bool:
        # Always "available" — it is the no-op adapter. Availability is only a
        # meaningful question for real adapters; NoSandbox never fail-closes.
        return True

    def prepare(self, argv: Sequence[str], opts: SandboxOptions) -> dict[str, Any]:
        # No confinement kwargs to merge; the child launches unconfined.
        _ = (argv, opts)
        return {}


__all__ = ["NoSandbox"]
