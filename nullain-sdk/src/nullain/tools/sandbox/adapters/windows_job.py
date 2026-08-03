"""Nullain Agent SDK — Windows Job Object sandbox adapter.

Honest v1 isolation contract (per the chosen "Win v1 JobObject, AppContainer
depois" scope):

- **Process containment** — the child (and its whole process tree) is placed in
  a Job Object with ``KILL_ON_JOB_CLOSE``. If the launcher or its parent dies,
  the tree is torn down; the child cannot break out of the job because it is
  assigned atomically (``CREATE_SUSPENDED`` + assign + resume), not after start.
- **Privilege drop** — the child runs with a restricted token carrying **no
  privileges** (``CreateRestrictedToken(DISABLE_MAX_PRIVILEGES)``). Integrity is
  unchanged, so the child can still write the workspace the parent created.
- **Fail-closed** — ``available()`` is True on Windows (kernel32 is always
  present), and the runner refuse-to-execute path is proven cross-platform by
  the unit test. The launcher itself dies (non-zero) if any kernel call fails,
  so a misconfigured host surfaces as a failed run rather than silent unsandboxed
  execution.

What this adapter does NOT do in v1 (follow-ups, not hidden):
- **Filesystem isolation** — handled by the existing ``PermissionPolicy``
  (resolve + ``is_relative_to(workspace_root)`` with symlinks resolved), applied
  by the bash/git tools before launch and already covered by
  ``test_tools_security.py``. Real token/fs confinement (low-integrity token +
  workspace mandatory label, or AppContainer) is the follow-up.
- **Network isolation** — Job Object does not restrict network; ``deny_network``
  is accepted but not yet enforced on Windows v1.

The confinement is realized by wrapping the command in a launcher
(:mod:`nullain.tools.sandbox.adapters._win_launcher`) invoked via ``python -m``;
the real command + options are passed as a JSON argv element (never a shell
string), and the launcher does the ctypes dance.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Any

from nullain.tools.sandbox.port import SandboxOptions

_LAUNCHER_MODULE = "nullain.tools.sandbox.adapters._win_launcher"


class WindowsJobSandbox:
    """Windows Job Object + privilege-drop sandbox (process containment)."""

    def __init__(self, required: bool = True) -> None:
        self._required = required

    @property
    def name(self) -> str:
        return "windows_job"

    @property
    def required(self) -> bool:
        return self._required

    def available(self) -> bool:
        # kernel32 + advapi32 are always present on Windows; the launcher dies
        # with a non-zero exit if a needed call fails, surfacing a bad host.
        return sys.platform == "win32"

    def prepare(self, argv: Sequence[str], opts: SandboxOptions) -> dict[str, Any]:
        if not self.available():
            # Unavailable + not required: run unconfined. Unavailable + required
            # never reaches here (runner raised SandboxUnavailableError first).
            return {}
        cfg = json.dumps({"cmd": list(argv), "deny_network": opts.deny_network})
        # Wrap the command in the launcher; still an explicit argv, never shell.
        return {"argv": [sys.executable, "-m", _LAUNCHER_MODULE, cfg]}


__all__ = ["WindowsJobSandbox"]
