"""Unit tests for the OS-level sandbox port, runner fail-closed behavior, and selector.

These tests are fully offline and cross-platform: the fail-closed logic and
adapter-kwargs merge are exercised with a fake adapter + monkeypatched
``asyncio.create_subprocess_exec`` (no real confinement, no platform dependency).
Platform-specific adapters (landlock/seatbelt/windows_job) get their own gated
tests in follow-up commits.
"""

import asyncio
import contextlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from nullain.config import NullainSettings, SandboxConfig
from nullain.errors import SandboxUnavailableError
from nullain.tools.sandbox import (
    NoSandbox,
    SandboxOptions,
    execute_subprocess,
    select_sandbox,
)


class _FakeSandbox:
    """In-memory Sandbox adapter for offline runner/selector tests."""

    def __init__(
        self,
        *,
        required: bool,
        available: bool,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._required = required
        self._available = available
        self._extra = extra or {}
        self.prepared: list[tuple[list[str], SandboxOptions]] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def required(self) -> bool:
        return self._required

    def available(self) -> bool:
        return self._available

    def prepare(self, argv: Sequence[str], opts: SandboxOptions) -> dict[str, Any]:
        self.prepared.append((list(argv), opts))
        return dict(self._extra)


class _FakeProc:
    def __init__(self) -> None:
        self.returncode = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        return (b"ok", b"")


# ---------------------------------------------------------------------------
# NoSandbox adapter
# ---------------------------------------------------------------------------


def test_no_sandbox_is_permissive_and_available() -> None:
    sb = NoSandbox()
    assert sb.name == "none"
    assert sb.required is False
    assert sb.available() is True
    assert sb.prepare(["ls"], SandboxOptions(workspace_root=Path("."))) == {}


def test_no_sandbox_required_flag_is_respected() -> None:
    # Even NoSandbox carries the flag so the selector can communicate intent;
    # the runner only fail-closes when required AND not available, and
    # NoSandbox.available() is always True, so it never fail-closes.
    assert NoSandbox(required=True).required is True


# ---------------------------------------------------------------------------
# Runner fail-closed behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fail_closed_raises_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A required adapter that is unavailable refuses to launch — the security
    differentiator. No subprocess is spawned; the runner raises immediately."""
    launched = False

    async def _spy(*args: Any, **kwargs: Any) -> _FakeProc:
        nonlocal launched
        launched = True
        return _FakeProc()

    # Patch the symbol the runner actually calls; monkeypatch restores it after.
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _spy)

    sb = _FakeSandbox(required=True, available=False)
    with pytest.raises(SandboxUnavailableError) as exc:
        await execute_subprocess(
            [sys.executable, "-c", "print('hi')"],
            cwd=tmp_path,
            sandbox=sb,
            sandbox_opts=SandboxOptions(workspace_root=tmp_path),
        )
    assert "fake" in str(exc.value)
    assert launched is False, "fail-closed must not spawn any subprocess"


@pytest.mark.asyncio
async def test_not_required_unavailable_still_runs(tmp_path: Path) -> None:
    """required=False means the unavailable adapter does NOT fail-closed: the
    caller opted out of strict isolation, so execution proceeds."""
    sb = _FakeSandbox(required=False, available=False, extra={})
    code, output = await execute_subprocess(
        [sys.executable, "-c", "print('ran')"],
        cwd=tmp_path,
        sandbox=sb,
        sandbox_opts=SandboxOptions(workspace_root=tmp_path),
    )
    assert code == 0
    assert "ran" in output
    assert sb.prepared, "adapter.prepare must still be called when not required"


@pytest.mark.asyncio
async def test_adapter_kwargs_are_merged_into_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    async def _spy(*args: Any, **kwargs: Any) -> _FakeProc:
        captured.update(kwargs)
        captured["_argv"] = args
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _spy)

    sb = _FakeSandbox(required=True, available=True, extra={"_marker": True})
    await execute_subprocess(
        [sys.executable, "-c", "pass"],
        cwd=tmp_path,
        sandbox=sb,
        sandbox_opts=SandboxOptions(workspace_root=tmp_path),
    )
    assert captured.get("_marker") is True, "adapter extra kwargs must be merged"
    # The runner still owns the executable argv (never shell).
    assert captured["_argv"][0] == sys.executable


@pytest.mark.asyncio
async def test_adapter_cannot_override_runner_stdio_or_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An adapter must not be able to redirect stdio or change cwd past the
    runner's controlled workspace — those are the runner's to set."""
    captured: dict[str, Any] = {}

    async def _spy(*args: Any, **kwargs: Any) -> _FakeProc:
        captured.update(kwargs)
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _spy)

    sb = _FakeSandbox(
        required=True,
        available=True,
        extra={"cwd": "/evil", "stdout": None, "stdin": None, "_ok": 1},
    )
    await execute_subprocess(
        [sys.executable, "-c", "pass"],
        cwd=tmp_path,
        sandbox=sb,
        sandbox_opts=SandboxOptions(workspace_root=tmp_path),
    )
    assert captured["cwd"] == str(tmp_path.resolve()), "cwd must stay the workspace"
    assert captured["stdout"] is asyncio.subprocess.PIPE, "stdout must stay PIPE"
    assert "stdin" not in captured, "adapter stdin override must be dropped"
    assert captured.get("_ok") == 1, "non-protected keys still merge"


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


def test_select_sandbox_disabled_returns_nosandbox() -> None:
    sb = select_sandbox(SandboxConfig(enabled=False))
    assert sb.name == "none"
    assert sb.required is False


def test_select_sandbox_enabled_returns_adapter() -> None:
    # Foundation: no real platform adapter wired yet, so the selector returns
    # the permissive NoSandbox fallback (regression-free). Follow-up commits
    # replace this with landlock/seatbelt/windows_job on their platforms.
    sb = select_sandbox(SandboxConfig(enabled=True, required=True))
    assert sb.available(), "fallback adapter must be usable"
    # The fallback never fail-closes; real isolation activates with adapters.


def test_select_sandbox_preserves_disabled_required_false() -> None:
    # Disabling isolation is an explicit opt-out: required is False so even a
    # real-but-unavailable adapter downstream would not block execution.
    sb = select_sandbox(SandboxConfig(enabled=False, required=True))
    assert sb.required is False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_sandbox_config_defaults() -> None:
    cfg = SandboxConfig()
    assert cfg.enabled is True
    assert cfg.required is True
    assert cfg.allow_paths == []
    assert cfg.deny_network is True


def test_nullain_settings_has_sandbox_default() -> None:
    settings = NullainSettings()
    assert isinstance(settings.sandbox, SandboxConfig)
    assert settings.sandbox.enabled is True


def test_load_settings_sandbox_from_toml(tmp_path: Path) -> None:
    from nullain.config import load_settings

    toml = tmp_path / "nullain.toml"
    toml.write_text(
        """
[sandbox]
enabled = true
required = false
allow_paths = ["/opt/nullain/cache"]
deny_network = false
"""
    )
    settings = load_settings(toml)
    assert settings.sandbox.enabled is True
    assert settings.sandbox.required is False
    assert settings.sandbox.allow_paths == ["/opt/nullain/cache"]
    assert settings.sandbox.deny_network is False


# ---------------------------------------------------------------------------
# Linux Landlock adapter (platform-gated; CI Ubuntu validates the escape test)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="landlock is Linux-only")
@pytest.mark.asyncio
async def test_landlock_blocks_write_outside_workspace(tmp_path: Path) -> None:
    """The flagship security proof: a sandboxed child CANNOT write outside its
    workspace. Runs only on Linux where landlock is supported; CI (Ubuntu)
    validates it. Skipped on hosts without landlock rather than failing — the
    cross-platform fail-closed unit test already guards the security property.
    """
    from nullain.tools.sandbox.adapters.landlock import LandlockSandbox

    sb = LandlockSandbox(required=True)
    if not sb.available():
        pytest.skip("landlock not available on this kernel")

    # A target OUTSIDE the workspace, under the shared pytest tmp root.
    outside_dir = tmp_path.parent / f"nullain_landlock_escape_{tmp_path.name}"
    outside_dir.mkdir(exist_ok=True)
    target = outside_dir / "evil.txt"
    if target.exists():
        target.unlink()

    try:
        code, output = await execute_subprocess(
            [sys.executable, "-c", f"open(r'{target}', 'w').write('x')"],
            cwd=tmp_path,
            sandbox=sb,
            sandbox_opts=SandboxOptions(workspace_root=tmp_path, deny_network=True),
        )
        assert code != 0, "writing outside the workspace must be denied"
        assert not target.exists(), "no file must be created outside the workspace"
        assert "PermissionError" in output or "Permission denied" in output, output
    finally:
        with contextlib.suppress(FileNotFoundError):
            target.unlink()
        with contextlib.suppress(OSError):
            outside_dir.rmdir()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="landlock is Linux-only")
@pytest.mark.asyncio
async def test_landlock_allows_write_inside_workspace(tmp_path: Path) -> None:
    """Isolation must not over-restrict: the child can still write inside its
    workspace. This complements the escape test so a regression that blocks
    everything is caught, not just one that lets everything through."""
    from nullain.tools.sandbox.adapters.landlock import LandlockSandbox

    sb = LandlockSandbox(required=True)
    if not sb.available():
        pytest.skip("landlock not available on this kernel")

    inside = tmp_path / "inside.txt"
    code, output = await execute_subprocess(
        [sys.executable, "-c", f"open(r'{inside}', 'w').write('x')"],
        cwd=tmp_path,
        sandbox=sb,
        sandbox_opts=SandboxOptions(workspace_root=tmp_path),
    )
    assert code == 0, f"writing inside the workspace must succeed: {output}"
    assert inside.read_text() == "x"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="landlock is Linux-only")
def test_landlock_selector_wires_on_linux() -> None:
    """select_sandbox on Linux returns the landlock adapter (not the NoSandbox
    fallback) when isolation is enabled and required."""
    from nullain.tools.sandbox.adapters.landlock import LandlockSandbox

    sb = select_sandbox(SandboxConfig(enabled=True, required=True))
    assert isinstance(sb, LandlockSandbox)
    assert sb.required is True


# ---------------------------------------------------------------------------
# macOS Seatbelt adapter (platform-gated; CI is Ubuntu-only, so this is skipped
# there and validated manually on a macOS host)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "darwin", reason="seatbelt is macOS-only")
@pytest.mark.asyncio
async def test_seatbelt_blocks_write_outside_workspace(tmp_path: Path) -> None:
    """A sandboxed child cannot write outside its workspace on macOS. Gated to
    macOS because CI runs Ubuntu; validated on a macOS dev box. Skipped (not
    failed) elsewhere — the cross-platform fail-closed unit test guards the
    security property regardless of platform."""
    from nullain.tools.sandbox.adapters.seatbelt import SeatbeltSandbox

    sb = SeatbeltSandbox(required=True)
    if not sb.available():
        pytest.skip("sandbox-exec not available on this host")

    outside_dir = tmp_path.parent / f"nullain_seatbelt_escape_{tmp_path.name}"
    outside_dir.mkdir(exist_ok=True)
    target = outside_dir / "evil.txt"
    if target.exists():
        target.unlink()

    try:
        code, output = await execute_subprocess(
            [sys.executable, "-c", f"open(r'{target}', 'w').write('x')"],
            cwd=tmp_path,
            sandbox=sb,
            sandbox_opts=SandboxOptions(workspace_root=tmp_path, deny_network=True),
        )
        assert code != 0, "writing outside the workspace must be denied"
        assert not target.exists(), "no file must be created outside the workspace"
        assert "Operation not permitted" in output or "Permission denied" in output, output
    finally:
        with contextlib.suppress(FileNotFoundError):
            target.unlink()
        with contextlib.suppress(OSError):
            outside_dir.rmdir()


@pytest.mark.skipif(sys.platform != "darwin", reason="seatbelt is macOS-only")
@pytest.mark.asyncio
async def test_seatbelt_allows_write_inside_workspace(tmp_path: Path) -> None:
    """Complement to the escape test: the child can still write inside its
    workspace, so a profile that blocks everything is caught, not just one that
    lets everything through."""
    from nullain.tools.sandbox.adapters.seatbelt import SeatbeltSandbox

    sb = SeatbeltSandbox(required=True)
    if not sb.available():
        pytest.skip("sandbox-exec not available on this host")

    inside = tmp_path / "inside.txt"
    code, output = await execute_subprocess(
        [sys.executable, "-c", f"open(r'{inside}', 'w').write('x')"],
        cwd=tmp_path,
        sandbox=sb,
        sandbox_opts=SandboxOptions(workspace_root=tmp_path),
    )
    assert code == 0, f"writing inside the workspace must succeed: {output}"
    assert inside.read_text() == "x"


@pytest.mark.skipif(sys.platform != "darwin", reason="seatbelt is macOS-only")
def test_seatbelt_selector_wires_on_darwin() -> None:
    from nullain.tools.sandbox.adapters.seatbelt import SeatbeltSandbox

    sb = select_sandbox(SandboxConfig(enabled=True, required=True))
    assert isinstance(sb, SeatbeltSandbox)
    assert sb.required is True


# ---------------------------------------------------------------------------
# Windows Job Object adapter (platform-gated; runs on a Windows host — CI is
# Ubuntu-only so this is skipped there and validated on a Windows box)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="job object is Windows-only")
@pytest.mark.asyncio
async def test_windows_job_runs_child_and_captures_output(tmp_path: Path) -> None:
    """End-to-end launcher integration: a sandboxed child execs, its stdout
    reaches the runner's pipe (inherited handles), and its exit code is
    forwarded. This proves the Job Object + restricted-token + suspended-assign
    path works, not an fs escape (Windows v1 fs isolation is PermissionPolicy)."""
    from nullain.tools.sandbox.adapters.windows_job import WindowsJobSandbox

    sb = WindowsJobSandbox(required=True)
    if not sb.available():
        pytest.skip("job object not available on this host")

    code, output = await execute_subprocess(
        [sys.executable, "-c", "import sys; sys.stdout.write('ran'); sys.exit(0)"],
        cwd=tmp_path,
        sandbox=sb,
        sandbox_opts=SandboxOptions(workspace_root=tmp_path),
    )
    assert code == 0, f"sandboxed child must run to completion: {output}"
    assert "ran" in output, "child stdout must reach the runner via inherited handles"


@pytest.mark.skipif(sys.platform != "win32", reason="job object is Windows-only")
@pytest.mark.asyncio
async def test_windows_job_forwards_nonzero_exit(tmp_path: Path) -> None:
    """A non-zero child exit code must propagate through the launcher."""
    from nullain.tools.sandbox.adapters.windows_job import WindowsJobSandbox

    sb = WindowsJobSandbox(required=True)
    if not sb.available():
        pytest.skip("job object not available on this host")

    code, _output = await execute_subprocess(
        [sys.executable, "-c", "import sys; sys.exit(7)"],
        cwd=tmp_path,
        sandbox=sb,
        sandbox_opts=SandboxOptions(workspace_root=tmp_path),
    )
    assert code == 7


@pytest.mark.skipif(sys.platform != "win32", reason="job object is Windows-only")
def test_windows_job_selector_wires_on_win32() -> None:
    from nullain.tools.sandbox.adapters.windows_job import WindowsJobSandbox

    sb = select_sandbox(SandboxConfig(enabled=True, required=True))
    assert isinstance(sb, WindowsJobSandbox)


# ---------------------------------------------------------------------------
# Windows AppContainer isolation (issue #41): real filesystem + network
# confinement when deny_network=True, mirroring the landlock escape/allow
# pair. Gated to win32; validated live on a Windows host.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="AppContainer is Windows-only")
@pytest.mark.asyncio
async def test_windows_job_blocks_write_outside_workspace(tmp_path: Path) -> None:
    """The flagship security proof (issue #41): with deny_network=True, a
    sandboxed child CANNOT write outside its workspace + allow_paths — the
    AppContainer token implies no file access by default, and only granted
    paths are ACL'd. Complements the landlock/seatbelt escape tests."""
    from nullain.tools.sandbox.adapters.windows_job import WindowsJobSandbox

    sb = WindowsJobSandbox(required=True)
    if not sb.available():
        pytest.skip("AppContainer not available on this host")

    outside_dir = tmp_path.parent / f"nullain_appcontainer_escape_{tmp_path.name}"
    outside_dir.mkdir(exist_ok=True)
    target = outside_dir / "evil.txt"
    if target.exists():
        target.unlink()

    try:
        code, output = await execute_subprocess(
            [sys.executable, "-c", f"open(r'{target}', 'w').write('x')"],
            cwd=tmp_path,
            sandbox=sb,
            sandbox_opts=SandboxOptions(workspace_root=tmp_path, deny_network=True),
        )
        assert code != 0, "writing outside the workspace must be denied"
        assert not target.exists(), "no file must be created outside the workspace"
        assert "PermissionError" in output or "Permission denied" in output, output
    finally:
        with contextlib.suppress(FileNotFoundError):
            target.unlink()
        with contextlib.suppress(OSError):
            outside_dir.rmdir()


@pytest.mark.skipif(sys.platform != "win32", reason="AppContainer is Windows-only")
@pytest.mark.asyncio
async def test_windows_job_allows_write_inside_workspace(tmp_path: Path) -> None:
    """Complement to the escape test: isolation must not over-restrict — the
    child can still write inside its granted workspace."""
    from nullain.tools.sandbox.adapters.windows_job import WindowsJobSandbox

    sb = WindowsJobSandbox(required=True)
    if not sb.available():
        pytest.skip("AppContainer not available on this host")

    inside = tmp_path / "inside.txt"
    code, output = await execute_subprocess(
        [sys.executable, "-c", f"open(r'{inside}', 'w').write('x')"],
        cwd=tmp_path,
        sandbox=sb,
        sandbox_opts=SandboxOptions(workspace_root=tmp_path, deny_network=True),
    )
    assert code == 0, f"writing inside the workspace must succeed: {output}"
    assert inside.read_text() == "x"


@pytest.mark.skipif(sys.platform != "win32", reason="AppContainer is Windows-only")
@pytest.mark.asyncio
async def test_windows_job_denies_network_when_requested(tmp_path: Path) -> None:
    """With deny_network=True, a sandboxed child cannot establish a TCP
    connection — real AppContainer confinement (no capabilities attached),
    not a best-effort check. Proven live via WinError 10013 (WSAEACCES)."""
    from nullain.tools.sandbox.adapters.windows_job import WindowsJobSandbox

    sb = WindowsJobSandbox(required=True)
    if not sb.available():
        pytest.skip("AppContainer not available on this host")

    code, _output = await execute_subprocess(
        [
            sys.executable,
            "-c",
            "import socket; s=socket.socket(); s.settimeout(3); s.connect(('8.8.8.8', 53))",
        ],
        cwd=tmp_path,
        sandbox=sb,
        sandbox_opts=SandboxOptions(workspace_root=tmp_path, deny_network=True),
    )
    assert code != 0, "network connection must be denied when deny_network=True"


@pytest.mark.skipif(sys.platform != "win32", reason="AppContainer is Windows-only")
@pytest.mark.asyncio
async def test_windows_job_allows_network_when_not_denied(tmp_path: Path) -> None:
    """deny_network=False must not silently break network access: an
    AppContainer's capability grant does not actually restore network for an
    ad-hoc (non-package-registered) container (see _win_launcher's module
    docstring), so this path uses a plain restricted token instead — still
    process-contained, but with normal filesystem + network access."""
    from nullain.tools.sandbox.adapters.windows_job import WindowsJobSandbox

    sb = WindowsJobSandbox(required=True)
    if not sb.available():
        pytest.skip("AppContainer not available on this host")

    result_file = tmp_path / "net_ok.txt"
    code, output = await execute_subprocess(
        [
            sys.executable,
            "-c",
            "import socket; s=socket.socket(); s.settimeout(5); "
            "s.connect(('8.8.8.8', 53)); "
            f"open(r'{result_file}', 'w').write('connected')",
        ],
        cwd=tmp_path,
        sandbox=sb,
        sandbox_opts=SandboxOptions(workspace_root=tmp_path, deny_network=False),
        timeout=15.0,
    )
    if code != 0 and "TimeoutError" in output:
        pytest.skip("no outbound network reachable from this CI/host — cannot prove allow-path")
    assert code == 0, f"network connection must succeed when deny_network=False: {output}"
    assert result_file.exists()
    assert sb.required is True
