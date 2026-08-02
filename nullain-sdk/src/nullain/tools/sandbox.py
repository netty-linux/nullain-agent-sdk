"""Nullain Agent SDK — File System and Command Execution Sandbox."""

import asyncio
import contextlib
import re
from pathlib import Path

from nullain.errors import ToolExecutionError, ToolPermissionError

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


async def execute_subprocess(
    cmd_args: list[str],
    cwd: str | Path,
    timeout: float = 120.0,
    max_output_bytes: int = 100_000,
) -> tuple[int, str]:
    """Execute a command as a list of args using asyncio.create_subprocess_exec.

    NEVER uses shell=True. Truncates output if exceeding max_output_bytes and redacts secrets.

    Returns:
        tuple[returncode, output_text]
    """
    if not cmd_args:
        raise ToolExecutionError("Command arguments list cannot be empty")

    cwd_path = Path(cwd).resolve()

    try:
        proc = await asyncio.create_subprocess_exec(
            cmd_args[0],
            *cmd_args[1:],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd_path),
        )
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


__all__ = [
    "execute_subprocess",
    "redact_secrets",
    "resolve_and_validate_path",
]
