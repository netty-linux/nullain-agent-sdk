"""Nullain Agent SDK — Filesystem & Command Execution Sandbox.

Subprocess execution (argv list, never ``shell=True``), output truncation,
secret redaction, workspace path validation, and OS-level sandbox isolation
behind the :class:`Sandbox` port. Adapters are platform-specific; the runner is
sandbox-agnostic and fail-closed (a required-but-unavailable adapter refuses to
execute rather than degrading to unsandboxed).
"""

from nullain.tools.sandbox.adapters.none import NoSandbox
from nullain.tools.sandbox.port import Sandbox, SandboxOptions
from nullain.tools.sandbox.runner import (
    execute_subprocess,
    redact_secrets,
    resolve_and_validate_path,
)
from nullain.tools.sandbox.selector import select_sandbox

__all__ = [
    "NoSandbox",
    "Sandbox",
    "SandboxOptions",
    "execute_subprocess",
    "redact_secrets",
    "resolve_and_validate_path",
    "select_sandbox",
]
