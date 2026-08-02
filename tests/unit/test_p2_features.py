"""P2 feature tests: new tools, RunResult, parallel read-only dispatch."""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from nullain.agent import AgentLoop, RunResult
from nullain.events import BaseEvent, EventBus, ToolResultEvent
from nullain.llm import CompletionChunk, CompletionRequest, LLMProvider, TokenUsage, ToolCall
from nullain.tools import ToolRegistry
from nullain_tools import register_default_tools


class _FakeProvider(LLMProvider):
    """Fake provider yielding a scripted response sequence."""

    def __init__(self, responses: list[CompletionChunk]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        if self.call_count < len(self.responses):
            chunk = self.responses[self.call_count]
            self.call_count += 1
            return chunk
        return CompletionChunk(delta_text="")

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
        yield await self.generate(request)

    async def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# New tools: glob, list_directory
# ---------------------------------------------------------------------------


def test_glob_tool_finds_files(tmp_path: Path) -> None:
    from nullain_tools import create_filesystem_tools

    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.py").write_text("y")
    (tmp_path / "README.md").write_text("z")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("w")

    tools = {t.name: t for t in create_filesystem_tools(tmp_path)}
    result = tools["glob"].func(pattern="*.py")
    assert "a.py" in result
    assert "b.py" in result
    assert "README.md" not in result

    recursive = tools["glob"].func(pattern="**/*.py")
    assert "sub/c.py" in recursive.replace("\\", "/")
    assert tools["glob"].func(pattern="**/*.nomatch").startswith("No files")


def test_list_directory_tool(tmp_path: Path) -> None:
    from nullain_tools import create_filesystem_tools

    (tmp_path / "file.txt").write_text("x")
    (tmp_path / "dir").mkdir()
    (tmp_path / ".hidden").write_text("y")

    tools = {t.name: t for t in create_filesystem_tools(tmp_path)}
    result = tools["list_directory"].func(relative_dir=".")
    assert "file.txt" in result
    assert "dir/" in result
    assert ".hidden" not in result


def test_read_only_flags(tmp_path: Path) -> None:
    from nullain_tools import register_default_tools

    registry = ToolRegistry()
    register_default_tools(registry, tmp_path)
    for name in ("read_file", "grep", "glob", "list_directory", "web_fetch"):
        assert registry.is_read_only(name), f"{name} should be read-only"
    # Side-effecting tools are NOT read-only
    assert not registry.is_read_only("write_file")
    assert not registry.is_read_only("edit_file")
    assert not registry.is_read_only("bash")
    assert not registry.is_read_only("ask_user")
    # Unknown tools are treated as not read-only (fail-safe)
    assert not registry.is_read_only("does_not_exist")


# ---------------------------------------------------------------------------
# web_fetch (offline paths)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_fetch_rejects_non_http_url() -> None:
    from nullain_tools.web import create_web_fetch_tool

    tool = create_web_fetch_tool()
    out = await tool.func(url="file:///etc/passwd")
    assert out.startswith("Error: URL must start")


def test_web_fetch_html_to_text_strips_tags() -> None:
    from nullain_tools.web import html_to_text

    html = "<html><body><p>Hello</p><script>x</script><b>world</b></body></html>"
    text = html_to_text(html)
    assert "Hello" in text
    assert "world" in text
    assert "x" not in text  # script content stripped
    assert "<" not in text


# ---------------------------------------------------------------------------
# ask_user
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_user_returns_callback_answer() -> None:
    from nullain_tools import create_ask_user_tool

    async def cb(question: str) -> str:
        return f"answerto:{question}"

    tool = create_ask_user_tool(cb)
    out = await tool.func(question="what next?")
    assert out == "answerto:what next?"


@pytest.mark.asyncio
async def test_ask_user_without_callback_degrades_gracefully() -> None:
    from nullain_tools import create_ask_user_tool

    tool = create_ask_user_tool(None)
    out = await tool.func(question="what next?")
    assert out.startswith("Error: ask_user is unavailable")


# ---------------------------------------------------------------------------
# RunResult structured outcome
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_result_success_status(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_default_tools(registry, tmp_path)

    spec_chunk = CompletionChunk(
        delta_text=(
            '{"objective": "Create file", "steps": ["write"], '
            '"target_files": ["out.txt"], "acceptance_criteria": ["out.txt exists"]}'
        )
    )
    write_chunk = CompletionChunk(
        tool_calls=[
            ToolCall(
                id="w1",
                name="write_file",
                arguments={"path": "out.txt", "content": "hi"},
            )
        ],
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    final_chunk = CompletionChunk(delta_text="Done creating out.txt")

    provider = _FakeProvider([spec_chunk, write_chunk, final_chunk])
    agent = AgentLoop(provider=provider, tools=registry, max_steps=5, workspace_root=tmp_path)

    result = await agent.run_result("Create out.txt")
    assert isinstance(result, RunResult)
    assert result.status == "success"
    assert result.success
    assert "Done creating out.txt" in result.final_text
    assert (tmp_path / "out.txt").exists()


@pytest.mark.asyncio
async def test_run_result_max_steps_status(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_default_tools(registry, tmp_path)

    # Vary the call each step so loop detection does not fire.
    chunks = [
        CompletionChunk(
            tool_calls=[ToolCall(id=f"r{i}", name="read_file", arguments={"path": f"no_{i}.txt"})]
        )
        for i in range(10)
    ]
    provider = _FakeProvider(chunks)
    agent = AgentLoop(provider=provider, tools=registry, max_steps=3)

    result = await agent.run_result("loop forever")
    assert result.status == "max_steps"
    assert not result.success
    # And run() re-raises nothing for max_steps (non-terminal) — returns text
    text = await agent.run("loop forever")
    assert text == ""


# ---------------------------------------------------------------------------
# Parallel read-only dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_read_only_dispatch_executes_all_calls(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_default_tools(registry, tmp_path)
    (tmp_path / "f1.txt").write_text("one")
    (tmp_path / "f2.txt").write_text("two")

    spec_chunk = CompletionChunk(
        delta_text=(
            '{"objective": "read two", "steps": ["read"], '
            '"target_files": [], "acceptance_criteria": []}'
        )
    )
    # A single step requesting TWO read-only tool calls at once.
    parallel_chunk = CompletionChunk(
        tool_calls=[
            ToolCall(id="p1", name="read_file", arguments={"path": "f1.txt"}),
            ToolCall(id="p2", name="read_file", arguments={"path": "f2.txt"}),
        ]
    )
    final_chunk = CompletionChunk(delta_text="read both")
    provider = _FakeProvider([spec_chunk, parallel_chunk, final_chunk])

    bus = EventBus()
    events: list[BaseEvent] = []

    async def track(ev: BaseEvent) -> None:
        events.append(ev)

    bus.subscribe("*", track)

    agent = AgentLoop(provider=provider, tools=registry, event_bus=bus, max_steps=5)
    result = await agent.run_result("read two files")
    assert result.status == "success"

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    # Both calls executed (concurrently, since both are read-only)
    outputs = {e.output for e in tool_results}
    assert {"one", "two"} <= outputs
