"""M9 — CLI tests: exit codes, --json NDJSON output, and TOML editing.

All tests are offline: the ``run``/``chat`` commands are exercised with a fake
``Agent`` (monkeypatched into ``nullain.cli``) so no provider or network is
involved. ``doctor`` and ``mcp`` are tested against a fake config path.

The tests deliberately exercise the CLI's underscore-prefixed internals
(``_run``, ``_build_parser``, ``_edit_mcp_server``, ...) rather than the
``app()`` entry point, so ``reportPrivateUsage`` is disabled for this file.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import nullain.cli as cli
import pytest
from nullain.agent import RunResult
from nullain.events import ModelResponseEvent


class _FakeAgent:
    """Scripted stand-in for the Agent facade."""

    def __init__(self, result: RunResult) -> None:
        self._result = result
        self._stream_items: list[Any] = []
        self.seen_session_ids: list[str | None] = []

    async def run(self, prompt: str, session_id: str | None = None) -> RunResult:
        return self._result

    async def stream(self, prompt: str, session_id: str | None = None):
        self.seen_session_ids.append(session_id)
        for item in self._stream_items:
            yield item
        yield self._result


def _fake_agent_factory(
    result: RunResult,
    stream_items: list[Any] | None = None,
    created: list[_FakeAgent] | None = None,
) -> Callable[..., _FakeAgent]:
    """Return a factory that builds a ``_FakeAgent``, ignoring constructor kwargs.

    ``stream_items`` are yielded before the terminal ``RunResult``, mirroring
    what ``Agent.stream()`` actually emits (e.g. a ``ModelResponseEvent``
    carrying the final answer's text) — the real ``TUIRenderer``-driven
    ``_run``/``_chat`` render text from those events, not from
    ``RunResult.final_text`` directly. When ``created`` is given, every
    ``_FakeAgent`` instance the factory builds is appended to it, so a test
    can inspect (e.g.) which ``session_id`` each ``stream()`` call received
    across multiple ``_run``/``_chat`` invocations.
    """

    def factory(**kw: object) -> _FakeAgent:
        agent = _FakeAgent(result)
        agent._stream_items = list(stream_items or [])
        if created is not None:
            created.append(agent)
        return agent

    return factory


def _success_result(text: str = "done") -> RunResult:
    return RunResult(session_id="s1", status="success", success=True, final_text=text)


def _failure_result() -> RunResult:
    return RunResult(session_id="s1", status="error", success=False, error="boom")


def _scripted_input(monkeypatch: pytest.MonkeyPatch, *answers: str) -> None:
    """Patch ``input()`` to return ``answers`` in order, one per call."""
    it = iter(answers)

    def _fake_input(prompt: str = "") -> str:
        return next(it)

    monkeypatch.setattr("builtins.input", _fake_input)


def _scripted_getpass(monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    """Patch ``getpass.getpass()`` to return a fixed answer."""

    def _fake_getpass(prompt: str = "") -> str:
        return answer

    monkeypatch.setattr("getpass.getpass", _fake_getpass)


# ---------------------------------------------------------------------------
# run: exit codes + --json
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_success_prints_final_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The TUIRenderer renders the final answer from the ModelResponseEvent
    Agent.stream() emits, not from RunResult.final_text directly — so the
    fake stream must include that event to exercise the real render path."""
    final_response = ModelResponseEvent(session_id="s1", model="m", content="hello")
    monkeypatch.setattr(
        cli, "Agent", _fake_agent_factory(_success_result("hello"), [final_response])
    )
    result = await cli._run("hi", model=None, workspace=".", max_steps=25, json_output=False)
    assert result.success
    captured = capsys.readouterr()
    assert "hello" in captured.out


@pytest.mark.asyncio
async def test_run_json_emits_ndjson(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "Agent", _fake_agent_factory(_success_result("done")))
    await cli._run("hi", model=None, workspace=".", max_steps=25, json_output=True)
    lines = [json.loads(ln) for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert lines, "expected at least one NDJSON line"
    assert lines[-1]["type"] == "result"
    assert lines[-1]["status"] == "success"


def test_run_handler_exit_code_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "Agent", _fake_agent_factory(_success_result("ok")))
    args = cli._build_parser().parse_args(["run", "hi", "--workspace", ".", "--max-steps", "5"])
    assert cli._run_handler(args) == cli.EXIT_OK


def test_run_handler_exit_code_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "Agent", _fake_agent_factory(_failure_result()))
    args = cli._build_parser().parse_args(["run", "hi"])
    assert cli._run_handler(args) == cli.EXIT_RUNTIME


# ---------------------------------------------------------------------------
# session persistence (M16): --session / --continue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_session_id_explicit_session_wins(tmp_path: Path) -> None:
    """--session <id> is used as-is, no event store lookup needed."""
    resolved = await cli._resolve_session_id(
        str(tmp_path), session_id="explicit-id", continue_session=True
    )
    assert resolved == "explicit-id"


@pytest.mark.asyncio
async def test_resolve_session_id_no_flags_returns_none(tmp_path: Path) -> None:
    """Neither --session nor --continue: a fresh session id is left to the
    loop to generate, matching the pre-M16 default."""
    resolved = await cli._resolve_session_id(str(tmp_path), session_id=None, continue_session=False)
    assert resolved is None


@pytest.mark.asyncio
async def test_resolve_session_id_continue_with_no_prior_sessions(tmp_path: Path) -> None:
    """--continue with an empty/nonexistent event store falls through to
    None (fresh session) rather than erroring — nothing to continue yet."""
    resolved = await cli._resolve_session_id(str(tmp_path), session_id=None, continue_session=True)
    assert resolved is None


@pytest.mark.asyncio
async def test_resolve_session_id_continue_finds_latest_session(tmp_path: Path) -> None:
    """--continue looks up the most recently appended session in
    <workspace>/.nullain/sessions.db."""
    from nullain.events import EventStore, UserMessageEvent

    store = EventStore(tmp_path / ".nullain" / "sessions.db")
    await store.initialize()
    await store.append(UserMessageEvent(session_id="older-session", content="hi"))
    await store.append(UserMessageEvent(session_id="newer-session", content="hi again"))
    await store.close()

    resolved = await cli._resolve_session_id(str(tmp_path), session_id=None, continue_session=True)
    assert resolved == "newer-session"


@pytest.mark.asyncio
async def test_run_passes_resolved_session_id_to_agent_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """cli._run threads the resolved session id through to agent.stream()."""
    created: list[_FakeAgent] = []
    monkeypatch.setattr(cli, "Agent", _fake_agent_factory(_success_result("ok"), created=created))
    await cli._run(
        "hi",
        model=None,
        workspace=str(tmp_path),
        max_steps=5,
        json_output=True,
        session_id="my-session",
    )
    assert created[0].seen_session_ids == ["my-session"]


@pytest.mark.asyncio
async def test_chat_reuses_one_session_id_across_turns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: each chat turn previously generated its own fresh session
    id internally (session_id=None passed to every stream() call), so turn 2
    never actually saw turn 1's exchange despite being in the "same" chat.
    All turns in one chat process must now share one session id."""
    created: list[_FakeAgent] = []
    monkeypatch.setattr(cli, "Agent", _fake_agent_factory(_success_result("ok"), created=created))

    inputs = iter(["first turn", "second turn", "exit"])

    def _fake_input(prompt: str = "") -> str:
        return next(inputs)

    monkeypatch.setattr("builtins.input", _fake_input)

    await cli._chat(model=None, workspace=str(tmp_path))

    seen = created[0].seen_session_ids
    assert len(seen) == 2
    assert seen[0] == seen[1]
    assert seen[0] is not None


def test_version_handler(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = cli._build_parser().parse_args(["version"])
    assert cli._version_handler(args) == cli.EXIT_OK
    assert "Nullain Agent SDK" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# first-run setup wizard
# ---------------------------------------------------------------------------


def test_needs_setup_true_with_no_config_and_no_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("NULLAIN_OLLAMA_API_KEY", raising=False)
    assert cli._needs_setup(str(tmp_path)) is True


def test_needs_setup_false_with_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "some-key")
    assert cli._needs_setup(str(tmp_path)) is False


def test_needs_setup_false_with_workspace_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("NULLAIN_OLLAMA_API_KEY", raising=False)
    (tmp_path / "nullain.toml").write_text('ollama_api_key = "from-file"\n', encoding="utf-8")
    assert cli._needs_setup(str(tmp_path)) is False


@pytest.mark.asyncio
async def test_setup_wizard_writes_config_and_gitignore(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Happy path: key entered, health check passes, config written, and a
    workspace that is a git repo gets nullain.toml auto-gitignored."""
    (tmp_path / ".git").mkdir()

    _scripted_input(monkeypatch, "", "", "", "")  # base_url, then 3 model-tier prompts — defaulted
    _scripted_getpass(monkeypatch, "secret-key-123")

    async def _healthy(self: object) -> bool:
        return True

    monkeypatch.setattr(cli.OllamaCloudProvider, "health_check", _healthy)

    ok = await cli._run_setup_wizard(str(tmp_path))
    assert ok is True

    config_path = tmp_path / "nullain.toml"
    assert config_path.exists()
    text = config_path.read_text(encoding="utf-8")
    assert "secret-key-123" in text
    assert "glm-5.2:cloud" in text  # default balanced-tier model

    gitignore_text = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert "nullain.toml" in gitignore_text.splitlines()


@pytest.mark.asyncio
async def test_setup_wizard_no_key_aborts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _scripted_input(monkeypatch, "")  # base_url only
    _scripted_getpass(monkeypatch, "")  # empty key

    ok = await cli._run_setup_wizard(str(tmp_path))
    assert ok is False
    assert not (tmp_path / "nullain.toml").exists()


@pytest.mark.asyncio
async def test_setup_wizard_failed_health_check_prompts_and_can_proceed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed health check still lets the user save anyway if they confirm."""
    _scripted_input(monkeypatch, "", "y", "", "", "")  # base_url, confirm-anyway, 3 model tiers
    _scripted_getpass(monkeypatch, "bad-key")

    async def _unhealthy(self: object) -> bool:
        return False

    monkeypatch.setattr(cli.OllamaCloudProvider, "health_check", _unhealthy)

    ok = await cli._run_setup_wizard(str(tmp_path))
    assert ok is True
    assert (tmp_path / "nullain.toml").exists()


@pytest.mark.asyncio
async def test_setup_wizard_failed_health_check_declined_aborts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _scripted_input(monkeypatch, "", "n")  # base_url, decline to save anyway
    _scripted_getpass(monkeypatch, "bad-key")

    async def _unhealthy(self: object) -> bool:
        return False

    monkeypatch.setattr(cli.OllamaCloudProvider, "health_check", _unhealthy)

    ok = await cli._run_setup_wizard(str(tmp_path))
    assert ok is False
    assert not (tmp_path / "nullain.toml").exists()


def test_ensure_gitignored_noop_without_git_repo(tmp_path: Path) -> None:
    cli._ensure_gitignored(tmp_path / "nullain.toml")
    assert not (tmp_path / ".gitignore").exists()


def test_ensure_gitignored_appends_without_duplicating(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")

    cli._ensure_gitignored(tmp_path / "nullain.toml")
    cli._ensure_gitignored(tmp_path / "nullain.toml")  # idempotent

    lines = (tmp_path / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert lines.count("nullain.toml") == 1
    assert "*.log" in lines


@pytest.mark.asyncio
async def test_default_entry_skips_wizard_when_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """nullain (no subcommand) with an existing key goes straight to chat,
    without prompting for setup."""
    monkeypatch.setenv("OLLAMA_API_KEY", "already-configured")

    wizard_called = False

    async def _wizard_should_not_run(workspace: str) -> bool:
        nonlocal wizard_called
        wizard_called = True
        return True

    async def _fake_chat(
        *, model: str | None, workspace: str, continue_session: bool = False
    ) -> int:
        return cli.EXIT_OK

    monkeypatch.setattr(cli, "_run_setup_wizard", _wizard_should_not_run)
    monkeypatch.setattr(cli, "_chat", _fake_chat)

    result = await cli._default_entry(str(tmp_path))
    assert result == cli.EXIT_OK
    assert wizard_called is False


@pytest.mark.asyncio
async def test_default_entry_runs_wizard_then_chat_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("NULLAIN_OLLAMA_API_KEY", raising=False)

    chat_called = False

    async def _fake_wizard(workspace: str) -> bool:
        return True

    async def _fake_chat(
        *, model: str | None, workspace: str, continue_session: bool = False
    ) -> int:
        nonlocal chat_called
        chat_called = True
        return cli.EXIT_OK

    monkeypatch.setattr(cli, "_run_setup_wizard", _fake_wizard)
    monkeypatch.setattr(cli, "_chat", _fake_chat)

    result = await cli._default_entry(str(tmp_path))
    assert result == cli.EXIT_OK
    assert chat_called is True


@pytest.mark.asyncio
async def test_default_entry_aborted_wizard_skips_chat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("NULLAIN_OLLAMA_API_KEY", raising=False)

    chat_called = False

    async def _fake_wizard(workspace: str) -> bool:
        return False

    async def _fake_chat(
        *, model: str | None, workspace: str, continue_session: bool = False
    ) -> int:
        nonlocal chat_called
        chat_called = True
        return cli.EXIT_OK

    monkeypatch.setattr(cli, "_run_setup_wizard", _fake_wizard)
    monkeypatch.setattr(cli, "_chat", _fake_chat)

    result = await cli._default_entry(str(tmp_path))
    assert result == cli.EXIT_USAGE
    assert chat_called is False


def test_unknown_command_exits_usage() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli._build_parser().parse_args(["bogus"])
    assert exc_info.value.code == cli.EXIT_USAGE


# ---------------------------------------------------------------------------
# doctor: exit code reflects failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doctor_ok_when_all_checks_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(cli, "shutil", _Shutil(rg="/usr/bin/rg"))
    assert await cli._doctor() == cli.EXIT_OK
    assert "OK" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_doctor_fails_when_rg_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(cli, "shutil", _Shutil(rg=None))
    assert await cli._doctor() == cli.EXIT_RUNTIME


# ---------------------------------------------------------------------------
# mcp: TOML editing round-trip
# ---------------------------------------------------------------------------


def test_mcp_add_remove_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "nullain.toml"
    cfg.write_text('# header\n\n[router]\nfallback_chain = ["deep"]\n')
    monkeypatch.setenv("NULLAIN_CONFIG", str(cfg))

    cli._edit_mcp_server("filesystem", command="npx", args=["-y", "server-filesystem"])
    text = cfg.read_text()
    assert "[mcp.servers.filesystem]" in text
    assert 'command = "npx"' in text
    assert "# header" in text  # other content preserved
    assert "[router]" in text

    cli._edit_mcp_server("filesystem", remove=True)
    assert "[mcp.servers.filesystem]" not in cfg.read_text()


def test_mcp_parse_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "nullain.toml"
    cfg.write_text('[mcp.servers.git]\ncommand = "uvx"\nargs = ["mcp-server-git"]\n')
    parsed = cli._parse_mcp_servers(cfg.read_text())
    assert parsed["git"]["command"] == "uvx"
    assert parsed["git"]["args"] == ["mcp-server-git"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _Shutil:
    """Minimal stand-in for the ``shutil`` module used by doctor."""

    def __init__(self, rg: str | None) -> None:
        self._rg = rg

    def which(self, name: str) -> str | None:
        return self._rg if name == "rg" else None


def _settings(tmp_path: Path) -> Any:
    """Build a NullainSettings with no MCP servers and a disabled sandbox."""
    from nullain.config import NullainSettings, SandboxConfig

    return NullainSettings(
        ollama_base_url="https://ollama.com",
        sandbox=SandboxConfig(enabled=False, required=False, allow_paths=[], deny_network=True),
    )
