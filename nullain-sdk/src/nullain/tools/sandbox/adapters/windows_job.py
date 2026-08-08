"""Nullain Agent SDK — Windows AppContainer sandbox adapter.

Isolation contract (issue #41 — both gaps the original P4.23 v1 explicitly
deferred are closed here):

- **Process containment** — the child (and its whole process tree) is placed in
  a Job Object with ``KILL_ON_JOB_CLOSE``. If the launcher or its parent dies,
  the tree is torn down; the child cannot break out of the job because it is
  assigned atomically (``CREATE_SUSPENDED`` + assign + resume), not after start.
- **Real network isolation** — the child launches inside a Windows AppContainer
  (:mod:`nullain.tools.sandbox.adapters._win_launcher`) with the
  ``INTERNET_CLIENT`` / ``INTERNET_CLIENT_SERVER`` capabilities omitted when
  ``deny_network=True``. An AppContainer process with neither capability
  cannot open a socket at all — enforced by Windows' firewall/WFP integration
  for every AppContainer process unconditionally, not a best-effort check.
  Proven live: a denied child's ``socket.connect()`` raises ``WinError 10013``
  before ever reaching the network.
- **Real filesystem isolation** — an AppContainer token implies NO file access
  by default, even to files the launching user owns. The launcher explicitly
  ACLs (``SetNamedSecurityInfoW``) only the workspace, ``allow_paths``, and the
  paths the interpreter itself needs to start (its venv and, since a venv's
  ``python.exe`` is a shim, ``sys.base_prefix``); every other path on disk
  stays inaccessible to the child. Proven live: a write outside the granted
  paths raises ``PermissionError`` with no file created, while a write inside
  the workspace succeeds normally.
- **Fail-closed** — ``available()`` is True on Windows (kernel32/advapi32/
  userenv are always present), and the runner refuse-to-execute path is
  proven cross-platform by the unit test. The launcher dies non-zero if any
  AppContainer/ACL/job call fails, so a misconfigured host surfaces as a
  failed run rather than silent unsandboxed (or partially sandboxed)
  execution — there is no fallback path that drops isolation quietly.

The confinement is realized by wrapping the command in a launcher
(:mod:`nullain.tools.sandbox.adapters._win_launcher`) invoked via ``python -m``;
the real command + options (including which paths to grant) are passed as a
JSON argv element (never a shell string), and the launcher does the ctypes
dance.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nullain.tools.sandbox.port import SandboxOptions

_LAUNCHER_MODULE = "nullain.tools.sandbox.adapters._win_launcher"


def _interpreter_grant_paths() -> list[str]:
    """Paths the sandboxed interpreter itself needs read+execute access to
    just to start — mirrors what the Landlock adapter grants read-only for
    the same reason (site.py, stdlib, shared libs), adapted for Windows'
    venv-shim indirection: ``sys.executable`` is typically a small shim in
    ``<venv>/Scripts/python.exe`` that re-execs the real interpreter under
    ``sys.base_prefix`` — both trees must be granted or the child can't even
    reach the point where our own confinement logic would matter.
    """
    paths = {os.path.dirname(os.path.dirname(sys.executable)), sys.base_prefix}
    return [p for p in paths if p and os.path.exists(p)]


class WindowsJobSandbox:
    """Windows AppContainer + Job Object sandbox: real network + filesystem
    isolation (issue #41), plus process containment."""

    def __init__(self, required: bool = True) -> None:
        self._required = required

    @property
    def name(self) -> str:
        return "windows_job"

    @property
    def required(self) -> bool:
        return self._required

    def available(self) -> bool:
        # kernel32/advapi32/userenv are always present on Windows; the
        # launcher dies with a non-zero exit if a needed call fails,
        # surfacing a bad host rather than silently degrading isolation.
        return sys.platform == "win32"

    def prepare(self, argv: Sequence[str], opts: SandboxOptions) -> dict[str, Any]:
        if not self.available():
            # Unavailable + not required: run unconfined. Unavailable + required
            # never reaches here (runner raised SandboxUnavailableError first).
            return {}
        grant_paths = [
            *_interpreter_grant_paths(),
            str(Path(opts.workspace_root).resolve()),
            *(str(Path(p).resolve()) for p in opts.allow_paths),
        ]
        cfg = json.dumps(
            {
                "cmd": list(argv),
                "deny_network": opts.deny_network,
                "grant_paths": grant_paths,
            }
        )
        # Wrap the command in the launcher; still an explicit argv, never shell.
        return {"argv": [sys.executable, "-m", _LAUNCHER_MODULE, cfg]}


__all__ = ["WindowsJobSandbox"]
