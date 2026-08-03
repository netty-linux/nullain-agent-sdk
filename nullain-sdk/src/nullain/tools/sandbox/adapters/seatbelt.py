"""Nullain Agent SDK — macOS Seatbelt sandbox adapter.

macOS Seatbelt (the same framework behind App Sandbox) confines a process tree
via a Scheme-like profile applied by ``/usr/bin/sandbox-exec``. The adapter wraps
the command in ``sandbox-exec -p <profile> <cmd...>`` (an explicit argv list —
never a shell), so confinement is enforced by the kernel for the wrapped process
and all its children.

v1 isolation contract (honest):
- **Writes** are confined to the workspace (+ configured allow_paths). Writes
  outside are denied — this is what the gated escape test proves.
- **Network** is fully denied when ``deny_network``.
- **Reads** are allowed globally. Stricter read isolation (deny reads outside an
  allowlist, like the landlock adapter does) is possible but fragile on macOS
  (dyld shared cache, frameworks, ``/private/var/folders`` temp, home) and is a
  follow-up, not hidden.

This adapter is gated to macOS. CI runs only Ubuntu, so the escape test is
skipped there and validated manually on a macOS host (or by adding a macOS job
to CI later). The fail-closed contract is proven cross-platform by the unit test.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any

from nullain.tools.sandbox.port import SandboxOptions

_SANDBOX_EXEC = "/usr/bin/sandbox-exec"


def _esc(path: str) -> str:
    """Escape a path for a Seatbelt profile double-quoted string."""
    return path.replace("\\", "\\\\").replace('"', '\\"')


class SeatbeltSandbox:
    """macOS Seatbelt sandbox: write-confined + network-denied via sandbox-exec."""

    def __init__(self, required: bool = True) -> None:
        self._required = required

    @property
    def name(self) -> str:
        return "seatbelt"

    @property
    def required(self) -> bool:
        return self._required

    def available(self) -> bool:
        # Available iff we are on macOS and the sandbox-exec binary exists.
        return sys.platform == "darwin" and os.path.exists(_SANDBOX_EXEC)

    def prepare(self, argv: Sequence[str], opts: SandboxOptions) -> dict[str, Any]:
        if not self.available():
            # Unavailable + not required: run unconfined (runner already declined
            # to fail-closed). Unavailable + required never reaches here — the
            # runner raised SandboxUnavailableError before calling prepare.
            return {}
        workspace = os.path.realpath(os.fspath(opts.workspace_root))
        allow_paths = [os.path.realpath(os.fspath(p)) for p in opts.allow_paths]
        profile = _build_profile(workspace, allow_paths, opts.deny_network)
        # Wrap the command in sandbox-exec; still an explicit argv, never shell.
        return {"argv": [_SANDBOX_EXEC, "-p", profile, *argv]}


def _build_profile(workspace: str, allow_paths: Sequence[str], deny_network: bool) -> str:
    """Build a Seatbelt profile: read-anything, write only the workspace (+allow),
    network denied on request. Later, more specific rules override earlier ones."""
    lines = [
        "(version 1)",
        "(deny default)",
        # sandbox-exec itself must be able to fork+exec the child.
        "(allow process-fork)",
        "(allow process-exec)",
        "(allow signal)",
        # Reads are global (see module docstring for the honesty note).
        "(allow file-read*)",
        # Writes denied by default, then re-allowed only beneath the workspace
        # and any configured allow_paths.
        "(deny file-write*)",
        f'(allow file-write* (subpath "{_esc(workspace)}"))',
    ]
    for p in allow_paths:
        lines.append(f'(allow file-write* (subpath "{_esc(p)}"))')
    if deny_network:
        lines.append("(deny network*)")
    return "\n".join(lines)


__all__ = ["SeatbeltSandbox"]
