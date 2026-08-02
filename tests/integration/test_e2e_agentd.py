"""End-to-end integration tests for the stdio NDJSON agent daemon.

Drives ``run_agentd`` fully in-process: an injected scripted LLM provider, an
``asyncio.StreamReader`` fed NDJSON envelopes, and an ``io.StringIO`` output
capture. No real Ollama, no real subprocess, no real MCP server, and the
SQLite stores use a temp directory — so the suite is fully offline and
side-effect-free.

These tests prove the daemon's wiring end-to-end: NDJSON parsing, session
lifecycle, the permission approval round-trip over the same stdin channel,
MCP tool registration and invocation, and error-envelope emission.
"""

import asyncio
import io
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from nullain.config import MCPConfig, MCPServerConfig, NullainSettings
from nullain.events import EventStore
from nullain.llm import CompletionChunk, CompletionRequest, LLMProvider, TokenUsage, ToolCall
from nullain.mcp import MCPClient
from nullain.memory import EpisodicMemory
from nullain_agentd.main import run_agentd

SPEC_JSON = (
    '{"objective": "do the thing", "steps": ["s"], "target_files": [], "acceptance_criteria": []}'
)


class _ScriptedProvider(LLMProvider):
    """Scripted provider: returns canned CompletionChunks in order."""

    def __init__(self, responses: list[CompletionChunk]) -> None:
        self._responses = list(responses)
        self._n = 0

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        if self._n < len(self._responses):
            chunk = self._responses[self._n]
            self._n += 1
            return chunk
        return CompletionChunk(delta_text="Done")

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
        yield await self.generate(request)

    async def health_check(self) -> bool:
        return True


class _FakeMCPTransport:
    """Minimal in-memory MCP transport for the daemon E2E test."""

    def __init__(self, responses: dict[str, list[Any]]) -> None:
        self._responses = {k: list(v) for k, v in responses.items()}

    async def start(self) -> None:
        pass

    async def send_request(self, request: Any) -> str:
        queue = self._responses.get(request.method, [])
        result: Any = queue.pop(0) if queue else {}
        return json.dumps({"jsonrpc": "2.0", "id": request.id, "result": result})

    async def send_notification(self, notification: Any) -> None:
        pass

    async def close(self) -> None:
        pass


def _envelope(env_type: str, payload: dict[str, Any], eid: str = "1") -> str:
    """Build an NDJSON protocol envelope line."""
    return json.dumps({"v": 1, "type": env_type, "id": eid, "payload": payload})


async def _drive_daemon(
    *,
    provider: LLMProvider,
    lines: list[str],
    tmp_path: Path,
    settings: NullainSettings | None = None,
    mcp_clients: list[MCPClient] | None = None,
) -> list[dict[str, Any]]:
    """Run the daemon over a buffered reader + StringIO output, return envelopes."""
    reader = asyncio.StreamReader()
    out = io.StringIO()
    for ln in lines:
        reader.feed_data(ln.encode("utf-8") + b"\n")
    reader.feed_eof()

    event_store = EventStore(":memory:")
    episodic = EpisodicMemory(tmp_path / "e2e_memory.db")
    await run_agentd(
        provider=provider,
        input_reader=reader,
        output=out,
        settings=settings or NullainSettings(),
        mcp_clients=mcp_clients,
        event_store=event_store,
        episodic_memory=episodic,
    )
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def _envelopes_by_type(envelopes: list[dict[str, Any]], env_type: str) -> list[dict[str, Any]]:
    return [e for e in envelopes if e.get("type") == env_type]


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_session_start_and_message_completes(tmp_path: Path) -> None:
    provider = _ScriptedProvider(
        [CompletionChunk(delta_text=SPEC_JSON), CompletionChunk(delta_text="hello from agent")]
    )
    lines = [
        _envelope("session.start", {"session_id": "s1", "workspace_root": str(tmp_path)}),
        _envelope("user.message", {"session_id": "s1", "prompt": "say hi"}),
    ]

    envelopes = await _drive_daemon(provider=provider, lines=lines, tmp_path=tmp_path)

    started = _envelopes_by_type(envelopes, "session.started")
    assert started and started[0]["payload"]["status"] == "ok"

    ends = _envelopes_by_type(envelopes, "session.end")
    assert ends and ends[0]["payload"]["status"] == "completed"
    assert ends[0]["payload"]["output"] == "hello from agent"


# ---------------------------------------------------------------------------
# Permission approval round-trip over the shared stdin channel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_permission_round_trip_grants_write(tmp_path: Path) -> None:
    provider = _ScriptedProvider(
        [
            CompletionChunk(delta_text=SPEC_JSON),
            CompletionChunk(
                tool_calls=[
                    ToolCall(
                        id="w1",
                        name="write_file",
                        arguments={"path": "out.txt", "content": "hi"},
                    )
                ],
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
            CompletionChunk(delta_text="wrote out.txt"),
        ]
    )
    lines = [
        _envelope("session.start", {"session_id": "s1", "workspace_root": str(tmp_path)}),
        _envelope("user.message", {"session_id": "s1", "prompt": "write out.txt"}),
        _envelope(
            "permission.response",
            {"request_id": "any", "granted": True},
            eid="perm",
        ),
    ]

    envelopes = await _drive_daemon(provider=provider, lines=lines, tmp_path=tmp_path)

    # A permission.request was emitted for the write_file call.
    requests = _envelopes_by_type(envelopes, "permission.request")
    assert requests and requests[0]["payload"]["tool_name"] == "write_file"

    ends = _envelopes_by_type(envelopes, "session.end")
    assert ends and ends[0]["payload"]["status"] == "completed"
    # The granted write actually landed in the workspace.
    assert (tmp_path / "out.txt").read_text() == "hi"


@pytest.mark.asyncio
async def test_e2e_permission_denied_blocks_write(tmp_path: Path) -> None:
    provider = _ScriptedProvider(
        [
            CompletionChunk(delta_text=SPEC_JSON),
            CompletionChunk(
                tool_calls=[
                    ToolCall(
                        id="w1",
                        name="write_file",
                        arguments={"path": "denied.txt", "content": "x"},
                    )
                ]
            ),
            # Self-correction injection then a final answer without the write.
            CompletionChunk(delta_text="could not write"),
        ]
    )
    lines = [
        _envelope("session.start", {"session_id": "s1", "workspace_root": str(tmp_path)}),
        _envelope("user.message", {"session_id": "s1", "prompt": "write denied.txt"}),
        _envelope("permission.response", {"request_id": "any", "granted": False}, eid="perm"),
    ]

    envelopes = await _drive_daemon(provider=provider, lines=lines, tmp_path=tmp_path)

    assert _envelopes_by_type(envelopes, "permission.request")
    # The denied write did NOT create the file.
    assert not (tmp_path / "denied.txt").exists()
    ends = _envelopes_by_type(envelopes, "session.end")
    assert ends and ends[0]["payload"]["status"] == "completed"


# ---------------------------------------------------------------------------
# MCP tool registration + invocation through the daemon
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_mcp_tool_invocable_via_daemon(tmp_path: Path) -> None:
    transport = _FakeMCPTransport(
        {
            "initialize": [
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": {"name": "fake", "version": "0.1"},
                }
            ],
            "tools/list": [
                {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "echo back the text",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                            },
                        }
                    ]
                }
            ],
            "tools/call": [{"content": [{"type": "text", "text": "ECHO:mcp-works"}]}],
        }
    )
    client = MCPClient(transport=transport, name="fake")
    settings = NullainSettings(
        mcp=MCPConfig(servers={"fake": MCPServerConfig(command="noop", auto_approve=True)})
    )

    provider = _ScriptedProvider(
        [
            CompletionChunk(delta_text=SPEC_JSON),
            CompletionChunk(
                tool_calls=[
                    ToolCall(
                        id="m1",
                        name="mcp__fake__echo",
                        arguments={"text": "mcp-works"},
                    )
                ],
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
            CompletionChunk(delta_text="the echo said ECHO:mcp-works"),
        ]
    )
    lines = [
        _envelope("session.start", {"session_id": "s1", "workspace_root": str(tmp_path)}),
        _envelope("user.message", {"session_id": "s1", "prompt": "echo via mcp"}),
    ]

    envelopes = await _drive_daemon(
        provider=provider, lines=lines, tmp_path=tmp_path, settings=settings, mcp_clients=[client]
    )

    ends = _envelopes_by_type(envelopes, "session.end")
    assert ends and ends[0]["payload"]["status"] == "completed"
    # The MCP tool result surfaces in the agent events stream.
    tool_results = [
        e
        for e in envelopes
        if e.get("type") == "agent.event"
        and e.get("payload", {}).get("event_type") == "tool_result"
    ]
    assert tool_results
    assert "ECHO:mcp-works" in tool_results[0]["payload"]["data"]["output"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_invalid_envelope_emits_error(tmp_path: Path) -> None:
    provider = _ScriptedProvider([CompletionChunk(delta_text="Done")])
    envelopes = await _drive_daemon(provider=provider, lines=["not valid json"], tmp_path=tmp_path)
    errors = _envelopes_by_type(envelopes, "error")
    assert errors and "Invalid NDJSON envelope" in errors[0]["payload"]["message"]
