"""Nullain Agent SDK — macOS Seatbelt sandbox adapter.

macOS Seatbelt (the same framework behind App Sandbox) confines a process tree
via a Scheme-like profile applied by ``/usr/bin/sandbox-exec``. The adapter wraps
the command in ``sandbox-exec -p <profile> <cmd...>`` (an explicit argv list —
never a shell), so confinement is enforced by the kernel for the wrapped process
and all its children.

v2 isolation contract (issue #42 — read isolation added; honest about what's
still unverified):
- **Writes** are confined to the workspace (+ configured ``allow_paths``).
  Writes outside are denied — this is what the gated escape test proves.
- **Network** is fully denied when ``deny_network``.
- **Reads** are now deny-by-default too, confined to: the workspace,
  ``allow_paths``, ``allow_read_paths`` (the documented escape hatch for
  callers whose tooling needs more — ``[sandbox] allow_read_paths`` in
  ``nullain.toml``), and a fixed bootstrap allowlist covering what a CPython
  process needs to start (dyld shared cache, ``/System/Library/Frameworks``,
  ``/usr/lib``, ``/dev``, per-user ``var/folders`` temp — see
  ``_BOOTSTRAP_READ_PATHS`` below for the exact list and the reasoning behind
  each entry).

  **Important limitation, stated plainly**: the issue's own test plan calls
  for *empirically* enumerating the bootstrap allowlist by tracing a real
  ``sandbox-exec`` run on macOS (``(trace ...)`` or deny+logging) rather than
  guessing from documentation — this implementation was written and unit
  tested without access to macOS hardware, so the bootstrap list below is
  built from public Apple/Seatbelt documentation and the equivalent
  landlock/AppContainer bootstrap lists in this same sandbox package, NOT
  from a trace. It compiles, the darwin-gated tests exist and will run on any
  macOS host or CI job, but **this profile has not been run on real macOS and
  needs that empirical validation — the trace-driven check the issue asks
  for — before being trusted as the shipped default.** If Python's startup
  needs paths not listed here, the gated
  ``test_seatbelt_allows_interpreter_startup`` test will fail loudly on a
  real Mac (dyld/framework read denied) rather than silently under-isolating,
  and the fix is to add the missing path to ``_BOOTSTRAP_READ_PATHS`` with a
  comment explaining why, exactly as landlock documents its own bootstrap
  grants.

This adapter is gated to macOS. CI runs only Ubuntu, so the escape tests are
skipped there — validate manually on a macOS host (or by adding a macOS job
to CI; see the note above) before treating the read-isolation half of this
adapter as proven. The fail-closed contract is proven cross-platform by the
unit test.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any

from nullain.tools.sandbox.port import SandboxOptions

_SANDBOX_EXEC = "/usr/bin/sandbox-exec"

#: Paths a CPython interpreter needs READ access to just to start, beyond the
#: workspace/allow_paths — mirrors the same "what does the runtime need to
#: exec+load itself" concern the Linux Landlock adapter documents for its own
#: system-tree grants, adapted to macOS's dyld/framework layout. Each entry:
#: - "/usr/lib" — dyld shared cache and system libraries (libSystem, etc.)
#:   every process links against.
#: - "/System/Library/Frameworks" — Foundation/CoreFoundation and friends,
#:   which the system Python (and often a venv built from it) loads at
#:   startup even for pure-stdlib scripts.
#: - "/System/Library/PrivateFrameworks" — some CPython builds (notably the
#:   python.org installer and Homebrew builds linking against system
#:   frameworks) transitively touch private frameworks during interpreter
#:   init; listed defensively since a missing read here fails CLOSED (a
#:   startup crash) rather than open, so the cost of including it is low.
#: - "/private/var/db/dyld" — the dyld shared cache itself on modern macOS
#:   (moved out of /usr/lib/dyld in newer OS versions).
#: - "/dev" — /dev/null, /dev/urandom, tty devices; needed by the interpreter
#:   and by essentially any subprocess.
#: NOT included: the user's home directory, /Users/*, /private/etc (mail/
#:   passwd/etc.), Keychain-adjacent paths, browser profile directories — the
#:   entire point of #42 is that these stay denied.
_BOOTSTRAP_READ_PATHS: tuple[str, ...] = (
    "/usr/lib",
    "/System/Library/Frameworks",
    "/System/Library/PrivateFrameworks",
    "/private/var/db/dyld",
    "/dev",
)


def _esc(path: str) -> str:
    """Escape a path for a Seatbelt profile double-quoted string."""
    return path.replace("\\", "\\\\").replace('"', '\\"')


def _per_user_tmp_dir() -> str | None:
    """The per-user ``/private/var/folders/.../T`` temp directory CPython
    (and many tools) use for ``tempfile.gettempdir()`` — distinct from
    ``/tmp`` and required for normal interpreter operation (bytecode cache
    scratch, some stdlib modules). Read via ``$TMPDIR``, the standard macOS
    mechanism for discovering this path (there's no fixed, predictable name —
    it's derived from a per-boot random component)."""
    return os.environ.get("TMPDIR")


class SeatbeltSandbox:
    """macOS Seatbelt sandbox: write-confined + network-denied + read-isolated
    (workspace/allow_paths/bootstrap only) via sandbox-exec."""

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
        allow_read_paths = [os.path.realpath(os.fspath(p)) for p in opts.allow_read_paths]
        profile = _build_profile(workspace, allow_paths, allow_read_paths, opts.deny_network)
        # Wrap the command in sandbox-exec; still an explicit argv, never shell.
        return {"argv": [_SANDBOX_EXEC, "-p", profile, *argv]}


def _build_profile(
    workspace: str,
    allow_paths: Sequence[str],
    allow_read_paths: Sequence[str],
    deny_network: bool,
) -> str:
    """Build a Seatbelt profile: deny-by-default reads AND writes, re-allowed
    only for the workspace, allow_paths/allow_read_paths, and the fixed
    bootstrap allowlist a Python interpreter needs to start. Network denied
    on request. Later, more specific rules override earlier ones."""
    read_grant_paths = [workspace, *allow_paths, *allow_read_paths, *_BOOTSTRAP_READ_PATHS]
    tmp_dir = _per_user_tmp_dir()
    if tmp_dir:
        read_grant_paths.append(tmp_dir)

    lines = [
        "(version 1)",
        "(deny default)",
        # sandbox-exec itself must be able to fork+exec the child.
        "(allow process-fork)",
        "(allow process-exec)",
        "(allow signal)",
        # Reads denied by default, then re-allowed only for the workspace,
        # allow_paths/allow_read_paths, and the bootstrap trees a Python
        # interpreter needs to start (see _BOOTSTRAP_READ_PATHS).
        "(deny file-read*)",
    ]
    for p in read_grant_paths:
        lines.append(f'(allow file-read* (subpath "{_esc(p)}"))')
    lines.extend(
        [
            # Writes denied by default, then re-allowed only beneath the
            # workspace and any configured allow_paths (never allow_read_paths
            # — that grant is read-only by name and by design).
            "(deny file-write*)",
            f'(allow file-write* (subpath "{_esc(workspace)}"))',
        ]
    )
    for p in allow_paths:
        lines.append(f'(allow file-write* (subpath "{_esc(p)}"))')
    if tmp_dir:
        # The interpreter's own scratch temp dir needs write too (bytecode
        # cache, tempfile-based stdlib usage), not just read.
        lines.append(f'(allow file-write* (subpath "{_esc(tmp_dir)}"))')
    if deny_network:
        lines.append("(deny network*)")
    return "\n".join(lines)


__all__ = ["SeatbeltSandbox"]
