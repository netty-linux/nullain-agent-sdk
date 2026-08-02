"""Security unit tests and property tests for Tools and Sandbox."""

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from nullain.errors import ToolExecutionError, ToolPermissionError
from nullain.tools import (
    PermissionLevel,
    PermissionPolicy,
    execute_subprocess,
    redact_secrets,
    resolve_and_validate_path,
)


def test_path_traversal_prevention(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret content")

    # Attempt path traversal
    with pytest.raises(ToolPermissionError):
        resolve_and_validate_path(workspace, "../secret.txt")


def test_symlink_escape_prevention(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    target_file = outside_dir / "target.txt"
    target_file.write_text("outside data")

    symlink_in_workspace = workspace / "link_outside"
    try:
        symlink_in_workspace.symlink_to(target_file)
    except OSError:
        pytest.skip("Symlinks not supported on this OS environment")

    with pytest.raises(ToolPermissionError):
        resolve_and_validate_path(workspace, "link_outside")


def test_secret_redaction() -> None:
    raw_text = (
        "API Key: sk-abcdef12345678901234567890123456 and token "
        "ghp_123456789012345678901234567890123456"
    )
    cleaned = redact_secrets(raw_text)
    assert "sk-abcdef" not in cleaned
    assert "ghp_12" not in cleaned
    assert "[REDACTED_SECRET]" in cleaned


def test_permission_policy_deny_patterns(tmp_path: Path) -> None:
    policy = PermissionPolicy(workspace_root=str(tmp_path))

    assert policy.evaluate_command(["rm", "-rf", "/"]) == PermissionLevel.DENY
    assert policy.evaluate_command(["git", "push", "--force"]) == PermissionLevel.DENY
    assert policy.evaluate_file_access(".env", is_write=False) == PermissionLevel.DENY
    assert policy.evaluate_command(["ls", "-la"]) == PermissionLevel.ASK


@pytest.mark.asyncio
async def test_subprocess_timeout(tmp_path: Path) -> None:
    with pytest.raises(ToolExecutionError, match="timed out"):
        await execute_subprocess(
            ["python", "-c", "import time; time.sleep(5)"],
            cwd=tmp_path,
            timeout=0.5,
        )


@given(subpath=st.text(min_size=1, max_size=50))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_hypothesis_path_safety(subpath: str, tmp_path: Path) -> None:
    """Hypothesis property test: Any path resolved inside workspace never raises if relative."""
    workspace = (tmp_path / "fake_ws").resolve()
    workspace.mkdir(exist_ok=True)

    # Filter out null bytes or illegal Windows characters in path
    if "\x00" in subpath or any(c in subpath for c in '<>:"|?*'):
        return

    # Filter Windows reserved device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9):
    # resolving paths over them is pathologically slow on Windows and they are
    # not valid workspace subpaths.
    upper = subpath.upper().split(".")[0].split("\\")[0].split("/")[0]
    reserved = (
        {"CON", "PRN", "AUX", "NUL"}
        | {f"COM{i}" for i in range(1, 10)}
        | {f"LPT{i}" for i in range(1, 10)}
    )
    if upper in reserved:
        return

    clean = subpath.replace("\\", "/").lstrip("/")
    resolved = workspace / clean
    if resolved.is_relative_to(workspace):
        try:
            res = resolve_and_validate_path(workspace, clean)
            assert res.is_relative_to(workspace)
        except ToolPermissionError:
            pass


@pytest.mark.asyncio
async def test_tool_registry_file_permission_enforcement(tmp_path: Path) -> None:
    from nullain.tools import ToolRegistry
    from nullain_tools import create_filesystem_tools

    policy = PermissionPolicy(workspace_root=str(tmp_path))
    registry = ToolRegistry(permission_policy=policy)
    for t in create_filesystem_tools(tmp_path):
        registry.register(t)

    # Accessing .env should be denied by policy
    with pytest.raises(ToolPermissionError, match="denied by permission policy"):
        await registry.execute("write_file", {"path": ".env", "content": "SECRET=123"})

    with pytest.raises(ToolPermissionError, match="denied by permission policy"):
        await registry.execute("read_file", {"path": ".env"})
