"""Nullain Agent SDK — Sandbox runner: subprocess execution with OS isolation.

The runner launches a subprocess via :func:`asyncio.create_subprocess_exec`
with an explicit argv list (never ``shell=True``), applies output truncation
and secret redaction, and — when an :class:`~nullain.tools.sandbox.port.Sandbox`
adapter is injected — merges the adapter's confinement kwargs. Fail-closed:
a required-but-unavailable adapter raises before any process is spawned.

The path validator and secret redactor live here too; they are reused by the
file tools and by the sandbox adapters themselves.
"""

import asyncio
import contextlib
import re
from pathlib import Path
from typing import Any

from nullain.errors import SandboxUnavailableError, ToolExecutionError, ToolPermissionError
from nullain.tools.sandbox.port import Sandbox, SandboxOptions

SECRET_PATTERNS = [
    r"sk-[a-zA-Z0-9]{32,}",
    r"ollama_[a-zA-Z0-9]{32,}",
    r"ghp_[a-zA-Z0-9]{36}",
    r"Bearer\s+[a-zA-Z0-9_\-\.]{32,}",
]


def redact_secrets(text: str) -> str:
    """Redact sensitive patterns (API keys, tokens) from text output."""
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = re.sub(pattern, "[REDACTED_SECRET]", redacted)
    return redacted


def resolve_and_validate_path(workspace_root: str | Path, target_path: str | Path) -> Path:
    """Resolve symlinks and ensure path is strictly contained within workspace_root.

    Raises:
        ToolPermissionError if target path escapes the workspace root.
    """
    root = Path(workspace_root).resolve()
    target = Path(target_path)

    if not target.is_absolute():
        target = root / target

    # Resolve symlinks completely
    try:
        resolved_target = target.resolve()
    except Exception as e:
        raise ToolPermissionError(f"Invalid path or link resolution failed: {e}") from e

    # Symlink / parent traversal check
    if not resolved_target.is_relative_to(root):
        raise ToolPermissionError(
            f"Access denied: path '{target_path}' resolves outside workspace '{workspace_root}'"
        )

    return resolved_target


def _fail_closed(sandbox: Sandbox) -> None:
    """Raise if a required sandbox adapter is unavailable on this platform.

    The whole point of the security differentiator: never silently fall back to
    unsandboxed execution when isolation was requested. The user opted into the
    safe default; honoring it means refusing rather than degrading.
    """
    if sandbox.required and not sandbox.available():
        raise SandboxUnavailableError(
            f"Sandbox '{sandbox.name}' is required but unavailable on this platform",
            details={"sandbox": sandbox.name},
        )


async def execute_subprocess(
    cmd_args: list[str],
    cwd: str | Path,
    timeout: float = 120.0,
    max_output_bytes: int = 100_000,
    sandbox: Sandbox | None = None,
    sandbox_opts: SandboxOptions | None = None,
) -> tuple[int, str]:
    """Execute a command as a list of args using asyncio.create_subprocess_exec.

    NEVER uses ``shell=True``. When a ``sandbox`` adapter is provided, its
    confinement kwargs are merged into the launch; a required-but-unavailable
    adapter raises :class:`~nullain.errors.SandboxUnavailableError` (fail-closed)
    before any process is spawned. Truncates output beyond ``max_output_bytes``
    and redacts secrets.

    Args:
        cmd_args: Explicit argv list (``cmd_args[0]`` is the executable).
        cwd: Working directory; resolved to an absolute path.
        timeout: Wall-clock timeout in seconds.
        max_output_bytes: Captured output is truncated above this size.
        sandbox: Optional OS-level isolation adapter.
        sandbox_opts: Options handed to the adapter (workspace root, allow
            paths, deny_network). Required when ``sandbox`` is provided.

    Returns:
        tuple[returncode, output_text]
    """
    if not cmd_args:
        raise ToolExecutionError("Command arguments list cannot be empty")

    cwd_path = Path(cwd).resolve()

    launch_kwargs: dict[str, Any] = {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.STDOUT,
        "cwd": str(cwd_path),
    }
    if sandbox is not None:
        _fail_closed(sandbox)
        opts = sandbox_opts if sandbox_opts is not None else SandboxOptions(workspace_root=cwd_path)
        # Adapter returns platform-specific kwargs (preexec_fn / creationflags),
        # and may also wrap the command in a confining launcher (e.g. macOS
        # sandbox-exec) by returning an ``argv`` list. Merge defensively: adapter
        # keys win, but never overwrite our stdio/cwd; ``argv`` is consumed here
        # (never passed as a subprocess kwarg) and remains an explicit argv list,
        # never a shell string.
        extra = sandbox.prepare(list(cmd_args), opts)
        wrapped_argv = extra.pop("argv", None)
        for key, value in extra.items():
            if key in ("stdout", "stderr", "stdin", "cwd", "env"):
                continue
            launch_kwargs[key] = value
        if wrapped_argv is not None:
            cmd_args = list(wrapped_argv)

    try:
        proc = await asyncio.create_subprocess_exec(cmd_args[0], *cmd_args[1:], **launch_kwargs)
    except FileNotFoundError as err:
        raise ToolExecutionError(f"Executable not found: {cmd_args[0]}") from err
    except Exception as err:
        raise ToolExecutionError(f"Failed to launch process: {err}") from err

    try:
        stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError as err:
        with contextlib.suppress(Exception):
            proc.kill()
            await proc.wait()
        raise ToolExecutionError(f"Command execution timed out after {timeout} seconds") from err

    output = stdout_bytes.decode("utf-8", errors="replace")
    if len(stdout_bytes) > max_output_bytes:
        truncated_text = output[:max_output_bytes]
        output = f"{truncated_text}\n\n[OUTPUT TRUNCATED: Exceeded {max_output_bytes} bytes]"

    return proc.returncode or 0, redact_secrets(output)


__all__ = ["execute_subprocess", "redact_secrets", "resolve_and_validate_path"]
