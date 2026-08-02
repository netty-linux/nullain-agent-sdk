"""Unit tests for lifecycle hooks (P3.18)."""

import sys
import textwrap
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from nullain.agent import AgentLoop
from nullain.config import load_settings
from nullain.context import ContextManager
from nullain.errors import ContextWindowExhaustedError
from nullain.events import BaseEvent, CompactionEvent, EventBus, ToolResultEvent
from nullain.hooks import HookConfig, HookLifecycle, HookManager, HooksConfig
from nullain.llm import CompletionChunk, CompletionRequest, LLMProvider, ToolCall
from nullain.tools import ToolRegistry
from nullain_tools import register_default_tools


def _py(script: str) -> list[str]:
    """Build a cross-platform hook command that runs ``script`` via python."""
    return [sys.executable, "-c", textwrap.dedent(script)]


class _FakeProvider(LLMProvider):
    """Scripted fake provider for hook integration tests."""

    def __init__(self, responses: list[CompletionChunk]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        if self.call_count < len(self.responses):
            chunk = self.responses[self.call_count]
            self.call_count += 1
            return chunk
        return CompletionChunk(delta_text="Done")

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
        chunk = await self.generate(request)
        yield chunk

    async def health_check(self) -> bool:
        return True


def test_hook_manager_disabled_when_empty() -> None:
    mgr = HookManager()
    assert mgr.enabled is False

    import asyncio

    outcomes = asyncio.run(mgr.run(HookLifecycle.PRE_TOOL, {"tool_name": "x"}))
    assert outcomes == []


@pytest.mark.asyncio
async def test_hook_manager_runs_and_receives_payload() -> None:
    script = """
    import json, sys
    payload = json.load(sys.stdin)
    assert payload["lifecycle"] == "pre_tool"
    assert payload["tool_name"] == "write_file"
    print("additional-context")
    sys.exit(0)
    """
    cfg = HooksConfig(pre_tool=[HookConfig(command=_py(script), timeout=10.0)])
    mgr = HookManager(cfg)
    assert mgr.enabled is True

    outcomes = await mgr.run(
        HookLifecycle.PRE_TOOL,
        {"session_id": "s1", "call_id": "c1", "tool_name": "write_file", "arguments": {}},
    )
    assert len(outcomes) == 1
    out = outcomes[0]
    assert out.ok is True
    assert out.blocked is False
    assert out.additional_context == "additional-context"


@pytest.mark.asyncio
async def test_hook_exit_2_blocks_and_short_circuits(tmp_path: Path) -> None:
    blocker = HookConfig(command=_py("import sys; sys.exit(2)"))
    marker = tmp_path / "sentinel_ran"
    # The second hook writes a marker file only if it actually executes.
    sentinel = HookConfig(
        command=_py(
            f"""
            import sys
            from pathlib import Path
            Path({str(marker)!r}).write_text("ran")
            sys.exit(0)
            """
        )
    )

    cfg = HooksConfig(pre_tool=[blocker, sentinel])
    mgr = HookManager(cfg)

    outcomes = await mgr.run(HookLifecycle.PRE_TOOL, {"tool_name": "x"})
    assert len(outcomes) == 1
    assert outcomes[0].blocked is True
    assert outcomes[0].additional_context is None
    assert not marker.exists(), "second hook must not run after a block"


@pytest.mark.asyncio
async def test_hook_nonzero_nonblocking_exit_code() -> None:
    cfg = HooksConfig(post_tool=[HookConfig(command=_py("import sys; sys.exit(1)"))])
    mgr = HookManager(cfg)
    outcomes = await mgr.run(HookLifecycle.POST_TOOL, {"tool_name": "x"})
    assert len(outcomes) == 1
    assert outcomes[0].exit_code == 1
    assert outcomes[0].blocked is False


@pytest.mark.asyncio
async def test_hook_missing_command_is_nonblocking() -> None:
    cfg = HooksConfig(stop=[HookConfig(command=["this-binary-does-not-exist-xyz"])])
    mgr = HookManager(cfg)
    outcomes = await mgr.run(HookLifecycle.STOP, {"status": "success"})
    assert len(outcomes) == 1
    assert outcomes[0].exit_code == -1
    assert outcomes[0].blocked is False


@pytest.mark.asyncio
async def test_hook_timeout() -> None:
    cfg = HooksConfig(
        pre_compact=[
            HookConfig(command=_py("import time; time.sleep(5)"), timeout=0.3),
        ]
    )
    mgr = HookManager(cfg)
    outcomes = await mgr.run(HookLifecycle.PRE_COMPACT, {"current_tokens": 100})
    assert len(outcomes) == 1
    assert outcomes[0].exit_code == -1
    assert outcomes[0].stderr == "timeout"


def test_hooks_config_loads_from_toml(tmp_path: Path) -> None:
    toml = tmp_path / "nullain.toml"
    toml.write_text(
        textwrap.dedent(
            """
            [[hooks.pre_tool]]
            command = ["./hooks/pre_tool.sh"]
            timeout = 5.0

            [[hooks.stop]]
            command = ["./hooks/stop.sh"]
            """
        )
    )
    settings = load_settings(toml)
    assert settings.hooks.has_any() is True
    assert len(settings.hooks.pre_tool) == 1
    assert settings.hooks.pre_tool[0].command == ["./hooks/pre_tool.sh"]
    assert settings.hooks.pre_tool[0].timeout == 5.0
    assert settings.hooks.stop[0].command == ["./hooks/stop.sh"]
    assert settings.hooks.post_tool == []


@pytest.mark.asyncio
async def test_agent_loop_pre_tool_hook_blocks_call(tmp_path: Path) -> None:
    """A pre_tool hook exiting 2 vetoes the call: the tool never runs, and a
    ToolResultEvent marks the call as an error with the block reason."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    registry = ToolRegistry()
    register_default_tools(registry, workspace)

    spec_chunk = CompletionChunk(
        delta_text=(
            '{"objective": "Create file", "steps": ["write"], '
            '"target_files": ["OUT.txt"], "acceptance_criteria": ["OUT.txt exists"]}'
        )
    )
    tool_chunk = CompletionChunk(
        tool_calls=[
            ToolCall(
                id="cw1",
                name="write_file",
                arguments={"path": "OUT.txt", "content": "data"},
            )
        ]
    )
    final_chunk = CompletionChunk(delta_text="Done.")
    provider = _FakeProvider([spec_chunk, tool_chunk, final_chunk])

    bus = EventBus()
    events: list[BaseEvent] = []

    async def track(ev: BaseEvent) -> None:
        events.append(ev)

    bus.subscribe("*", track)

    cfg = HooksConfig(pre_tool=[HookConfig(command=_py("import sys; sys.exit(2)"), timeout=10.0)])
    agent = AgentLoop(
        provider=provider,
        tools=registry,
        event_bus=bus,
        hooks=HookManager(cfg),
        max_steps=5,
        workspace_root=workspace,
    )

    await agent.run("Create OUT.txt")

    # The tool was vetoed: file must not exist.
    assert not (workspace / "OUT.txt").exists()

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    blocked = [e for e in tool_results if e.tool_name == "write_file" and e.is_error]
    assert blocked, "expected a blocked ToolResultEvent for write_file"
    assert "blocked by pre_tool hook" in blocked[0].output


@pytest.mark.asyncio
async def test_agent_loop_pre_compact_hook_skips_compaction(tmp_path: Path) -> None:
    """A pre_compact hook exiting 2 vetoes compaction: no CompactionEvent is
    emitted. The thrash counter still advances (a vetoed compaction did not
    free tokens), so the run still terminates with ContextWindowExhaustedError."""
    registry = ToolRegistry()
    register_default_tools(registry, tmp_path)

    tool_chunk = CompletionChunk(
        tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "x.txt"})]
    )
    provider = _FakeProvider([tool_chunk] * 20)
    cm = ContextManager(max_window_tokens=100, compaction_threshold=0.5)

    bus = EventBus()
    events: list[BaseEvent] = []

    async def track(ev: BaseEvent) -> None:
        events.append(ev)

    bus.subscribe("*", track)

    cfg = HooksConfig(
        pre_compact=[HookConfig(command=_py("import sys; sys.exit(2)"), timeout=10.0)]
    )
    agent = AgentLoop(
        provider=provider,
        tools=registry,
        event_bus=bus,
        context_manager=cm,
        hooks=HookManager(cfg),
        max_steps=15,
        max_compaction_attempts=3,
        loop_detection_threshold=100,
    )

    with pytest.raises(ContextWindowExhaustedError):
        await agent.run("Thrash with vetoed compaction")

    compactions = [e for e in events if isinstance(e, CompactionEvent)]
    assert compactions == [], "no CompactionEvent should be emitted when pre_compact blocks"
