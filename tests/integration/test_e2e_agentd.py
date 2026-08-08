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
        #: Every request this provider was asked to answer, in order — lets
        #: a test inspect exactly what the agent sent (e.g. to prove a
        #: resumed session's history actually made it into the request).
        self.seen_requests: list[CompletionRequest] = []

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        self.seen_requests.append(request)
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
    event_store: EventStore | None = None,
) -> list[dict[str, Any]]:
    """Run the daemon over a buffered reader + StringIO output, return envelopes.

    ``event_store`` defaults to a fresh in-memory store per call; pass an
    explicit file-backed one across two separate calls to simulate a daemon
    restart (the store — unlike everything else run_agentd builds — is the
    one thing that must survive a restart for session resume to work).
    """
    reader = asyncio.StreamReader()
    out = io.StringIO()
    for ln in lines:
        reader.feed_data(ln.encode("utf-8") + b"\n")
    reader.feed_eof()

    store = event_store or EventStore(":memory:")
    await store.initialize()
    episodic = EpisodicMemory(tmp_path / "e2e_memory.db")
    await run_agentd(
        provider=provider,
        input_reader=reader,
        output=out,
        settings=settings or NullainSettings(),
        mcp_clients=mcp_clients,
        event_store=store,
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


@pytest.mark.asyncio
async def test_e2e_user_message_for_unknown_session_errors(tmp_path: Path) -> None:
    """A user.message for a session_id that never had session.start must
    error cleanly, not silently run against some other session's state."""
    provider = _ScriptedProvider([CompletionChunk(delta_text="Done")])
    lines = [_envelope("user.message", {"session_id": "ghost", "prompt": "hi"})]
    envelopes = await _drive_daemon(provider=provider, lines=lines, tmp_path=tmp_path)
    ends = _envelopes_by_type(envelopes, "session.end")
    assert ends and ends[0]["payload"]["status"] == "error"
    assert "ghost" in ends[0]["payload"]["error"]


# ---------------------------------------------------------------------------
# Concurrent session isolation (issue #43)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_concurrent_sessions_do_not_share_workspace(tmp_path: Path) -> None:
    """The flagship regression test for issue #43: starting session B must
    NOT clobber session A's workspace/registry. Before the fix, ws_root/
    registry/policy/persistent_memory were single closure-local variables
    reassigned on every session.start — a user.message for session A sent
    after session B started ran against B's workspace instead of A's."""
    ws_a = tmp_path / "workspace_a"
    ws_b = tmp_path / "workspace_b"
    ws_a.mkdir()
    ws_b.mkdir()

    provider = _ScriptedProvider(
        [
            CompletionChunk(delta_text=SPEC_JSON),  # session A's plan
            CompletionChunk(
                tool_calls=[
                    ToolCall(
                        id="w1",
                        name="write_file",
                        arguments={"path": "from_a.txt", "content": "belongs to A"},
                    )
                ],
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
            CompletionChunk(delta_text="wrote from_a.txt"),
        ]
    )
    lines = [
        _envelope("session.start", {"session_id": "session-a", "workspace_root": str(ws_a)}),
        _envelope("session.start", {"session_id": "session-b", "workspace_root": str(ws_b)}),
        # A message for session A, sent AFTER session B started — must still
        # use A's workspace (ws_a), not B's (the bug this test catches).
        _envelope("user.message", {"session_id": "session-a", "prompt": "write from_a.txt"}),
        _envelope(
            "permission.response",
            {"request_id": "any", "granted": True},
            eid="perm",
        ),
    ]

    await _drive_daemon(provider=provider, lines=lines, tmp_path=tmp_path)

    assert (ws_a / "from_a.txt").exists(), "session A's write must land in session A's workspace"
    assert not (ws_b / "from_a.txt").exists(), "session A's write must NOT leak into session B's"


@pytest.mark.asyncio
async def test_e2e_two_concurrent_sessions_both_complete_independently(tmp_path: Path) -> None:
    """Both sessions started concurrently must be independently addressable
    and complete with their own output — not just "the last one started"."""
    ws_a = tmp_path / "ws_a"
    ws_b = tmp_path / "ws_b"
    ws_a.mkdir()
    ws_b.mkdir()

    provider = _ScriptedProvider(
        [
            CompletionChunk(delta_text=SPEC_JSON),  # A's plan
            CompletionChunk(delta_text="answer for A"),
            CompletionChunk(delta_text=SPEC_JSON),  # B's plan
            CompletionChunk(delta_text="answer for B"),
        ]
    )
    lines = [
        _envelope("session.start", {"session_id": "a", "workspace_root": str(ws_a)}),
        _envelope("session.start", {"session_id": "b", "workspace_root": str(ws_b)}),
        _envelope("user.message", {"session_id": "a", "prompt": "hello a"}),
        _envelope("user.message", {"session_id": "b", "prompt": "hello b"}),
    ]

    envelopes = await _drive_daemon(provider=provider, lines=lines, tmp_path=tmp_path)
    ends = _envelopes_by_type(envelopes, "session.end")
    by_session = {e["payload"]["session_id"]: e["payload"] for e in ends}

    assert by_session["a"]["status"] == "completed"
    assert by_session["a"]["output"] == "answer for A"
    assert by_session["b"]["status"] == "completed"
    assert by_session["b"]["output"] == "answer for B"


# ---------------------------------------------------------------------------
# Session resume after daemon restart (issue #43)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_session_resumes_after_daemon_restart(tmp_path: Path) -> None:
    """The other flagship acceptance criterion: a session's history survives
    a full run_agentd() restart (a new process in production; a second
    run_agentd() call sharing the same file-backed EventStore here), and the
    resumed run's request actually includes the earlier turn's content —
    not just that it doesn't error."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    store = EventStore(tmp_path / "sessions.db")

    # First "process": one turn, then the daemon "restarts" (run_agentd returns).
    provider_1 = _ScriptedProvider(
        [
            CompletionChunk(delta_text=SPEC_JSON),
            CompletionChunk(delta_text="first answer"),
        ]
    )
    lines_1 = [
        _envelope("session.start", {"session_id": "resumable", "workspace_root": str(ws)}),
        _envelope("user.message", {"session_id": "resumable", "prompt": "remember X=42"}),
    ]
    envelopes_1 = await _drive_daemon(
        provider=provider_1, lines=lines_1, tmp_path=tmp_path, event_store=store
    )
    ends_1 = _envelopes_by_type(envelopes_1, "session.end")
    assert ends_1 and ends_1[0]["payload"]["status"] == "completed"

    # Second "process": same session_id, same on-disk store, fresh
    # run_agentd() call (a fresh provider instance too, standing in for a
    # brand-new daemon process) — this is what a restart looks like.
    provider_2 = _ScriptedProvider(
        [
            CompletionChunk(delta_text=SPEC_JSON),
            CompletionChunk(delta_text="second answer"),
        ]
    )
    lines_2 = [
        _envelope("session.start", {"session_id": "resumable", "workspace_root": str(ws)}),
        _envelope("user.message", {"session_id": "resumable", "prompt": "what is X?"}),
    ]
    envelopes_2 = await _drive_daemon(
        provider=provider_2, lines=lines_2, tmp_path=tmp_path, event_store=store
    )
    ends_2 = _envelopes_by_type(envelopes_2, "session.end")
    assert ends_2 and ends_2[0]["payload"]["status"] == "completed"
    assert ends_2[0]["payload"]["output"] == "second answer"

    # The resumed run's actual LLM request must include the first turn's
    # exchange — proving history was really loaded, not just that no error
    # was raised.
    second_request = provider_2.seen_requests[-1]
    contents = [m.content for m in second_request.messages]
    assert any(c and "remember X=42" in c for c in contents)
    assert any(c and "first answer" in c for c in contents)
