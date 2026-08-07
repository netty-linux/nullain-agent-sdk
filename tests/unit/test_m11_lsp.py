"""M11 — LSP client and tools (4.2).

Offline tests: a real Python subprocess acts as a minimal LSP server speaking
Content-Length framing, so the client's handshake, text-document requests, and
fail-soft tool routing are exercised end-to-end with no network. A second group
covers the tools' routing and fail-soft behavior with an in-memory fake client.
"""

import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from nullain.lsp import (
    DiagnosticReport,
    Hover,
    Location,
    LSPClient,
    MarkupContent,
    language_for_path,
    register_lsp_tools,
)
from nullain.tools import ToolRegistry

# ---------------------------------------------------------------------------
# Fake LSP server subprocess (Content-Length framing)
# ---------------------------------------------------------------------------


def _write_fake_server(path: Path) -> None:
    """Write a minimal LSP server script that echoes scripted responses."""
    path.write_text(
        textwrap.dedent(
            """
            import json
            import sys

            def _read_message():
                header = sys.stdin.readline()
                if not header:
                    return None
                length = 0
                while header.strip():
                    if header.lower().startswith("content-length:"):
                        length = int(header.split(":", 1)[1].strip())
                    header = sys.stdin.readline()
                body = sys.stdin.read(length)
                return json.loads(body)

            def _write_message(payload):
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                sys.stdout.write(f"Content-Length: {len(body)}\\r\\n\\r\\n")
                sys.stdout.write(body.decode("utf-8"))
                sys.stdout.flush()

            def main() -> None:
                while True:
                    msg = _read_message()
                    if msg is None:
                        break
                    if "id" not in msg:
                        continue  # notification, no response
                    method = msg.get("method")
                    rid = msg["id"]
                    if method == "initialize":
                        result = {
                            "capabilities": {},
                            "serverInfo": {"name": "py-lsp-fake", "version": "0.1"},
                        }
                    elif method == "textDocument/diagnostic":
                        result = {
                            "kind": "full",
                            "items": [{
                                "range": {"start": {"line": 0, "character": 0},
                                          "end": {"line": 0, "character": 1}},
                                "severity": 1,
                                "message": "syntax error",
                                "source": "fake",
                            }],
                        }
                    elif method == "textDocument/definition":
                        result = {
                            "uri": "file:///defs.py",
                            "range": {"start": {"line": 3, "character": 2},
                                      "end": {"line": 3, "character": 5}},
                        }
                    elif method == "textDocument/references":
                        result = [
                            {"uri": "file:///a.py",
                             "range": {"start": {"line": 1, "character": 0},
                                       "end": {"line": 1, "character": 1}}},
                            {"uri": "file:///b.py",
                             "range": {"start": {"line": 2, "character": 0},
                                       "end": {"line": 2, "character": 1}}},
                        ]
                    elif method == "textDocument/hover":
                        result = {"contents": {"kind": "markdown", "value": "def foo() -> int"}}
                    else:
                        _write_message({"jsonrpc": "2.0", "id": rid,
                                        "error": {"code": -32601, "message": "unknown"}})
                        continue
                    _write_message({"jsonrpc": "2.0", "id": rid, "result": result})

            if __name__ == "__main__":
                main()
            """
        )
    )


@pytest.fixture
def fake_server_script(tmp_path: Path) -> Path:
    script = tmp_path / "fake_lsp_server.py"
    _write_fake_server(script)
    return script


async def _make_client(script: Path) -> LSPClient:
    client = LSPClient(command=sys.executable, args=[str(script)], timeout=15.0)
    await client.initialize()
    return client


# ---------------------------------------------------------------------------
# LSPClient: handshake and text-document requests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_handshake_completes(fake_server_script: Path) -> None:
    client = await _make_client(fake_server_script)
    try:
        assert client.is_initialized is True
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_diagnostics_returns_typed_report(fake_server_script: Path) -> None:
    client = await _make_client(fake_server_script)
    try:
        result = await client.diagnostics("/ws/a.py", "x = 1")
        assert result is not None
        assert result.items[0].message == "syntax error"
        assert result.items[0].severity == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_goto_definition_returns_location(fake_server_script: Path) -> None:
    client = await _make_client(fake_server_script)
    try:
        result = await client.goto_definition("/ws/a.py", "x = 1", 0, 0)
        assert isinstance(result, Location)
        assert result.uri == "file:///defs.py"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_find_references_returns_locations(fake_server_script: Path) -> None:
    client = await _make_client(fake_server_script)
    try:
        result = await client.find_references("/ws/a.py", "x = 1", 0, 0)
        assert result is not None
        assert len(result) == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_hover_returns_contents(fake_server_script: Path) -> None:
    client = await _make_client(fake_server_script)
    try:
        result = await client.hover("/ws/a.py", "x = 1", 0, 0)
        assert result is not None
        assert isinstance(result.contents, MarkupContent)
        assert result.contents.value == "def foo() -> int"
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# LSPClient: error and transport-failure paths
# ---------------------------------------------------------------------------


def test_lsp_client_rejects_empty_command() -> None:
    from nullain.errors import LSPTransportError
    from nullain.lsp.client import LSPClient as _LSPClient

    with pytest.raises(LSPTransportError, match="non-empty command"):
        _LSPClient(command="")


@pytest.mark.asyncio
async def test_lsp_client_start_raises_on_missing_executable() -> None:
    """Spawning a server whose command doesn't exist on PATH surfaces as
    LSPTransportError (FileNotFoundError), not a raw OSError leaking out of
    the client's abstraction."""
    from nullain.errors import LSPTransportError
    from nullain.lsp.client import LSPClient as _LSPClient

    client = _LSPClient(command="nullain-definitely-not-a-real-executable-xyz")
    with pytest.raises(LSPTransportError, match="executable not found"):
        await client.start()


@pytest.mark.asyncio
async def test_lsp_server_error_response_raises_protocol_error(
    fake_server_script: Path,
) -> None:
    """The fake server responds to any unrecognized method with a JSON-RPC
    error object — exercises the client's error-response branch, which
    real malformed/unsupported LSP requests would also trigger."""
    from nullain.errors import LSPProtocolError

    client = await _make_client(fake_server_script)
    try:
        with pytest.raises(LSPProtocolError, match="LSP server error"):
            await client._request("textDocument/completion", {})  # type: ignore[reportPrivateUsage]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_lsp_client_close_is_idempotent(fake_server_script: Path) -> None:
    """Calling close() twice (e.g. once explicitly, once via a caller's
    finally block after an earlier failure already closed it) must not
    raise."""
    client = await _make_client(fake_server_script)
    await client.close()
    await client.close()  # no exception


# ---------------------------------------------------------------------------
# language_for_path routing
# ---------------------------------------------------------------------------


def test_language_for_path_known_extensions() -> None:
    assert language_for_path("a.py") == "python"
    assert language_for_path("a.tsx") == "typescript"
    assert language_for_path("a.go") == "go"


def test_language_for_path_unknown_extension() -> None:
    assert language_for_path("a.xyz") is None
    assert language_for_path("noext") is None


# ---------------------------------------------------------------------------
# Tools: routing and fail-soft behavior (in-memory fake client)
# ---------------------------------------------------------------------------


class FakeLSPClient:
    """In-memory LSP client with scripted responses for tool tests."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    async def diagnostics(self, path: str, text: str) -> DiagnosticReport | None:
        self.calls.append(("diagnostics", path))
        return self._responses.get("diagnostics")

    async def goto_definition(self, path: str, text: str, line: int, col: int) -> Location | None:
        self.calls.append(("definition", path))
        return self._responses.get("definition")

    async def find_references(
        self, path: str, text: str, line: int, col: int
    ) -> list[Location] | None:
        self.calls.append(("references", path))
        return self._responses.get("references")

    async def hover(self, path: str, text: str, line: int, col: int) -> Hover | None:
        self.calls.append(("hover", path))
        return self._responses.get("hover")


def _registry(clients: dict[str, FakeLSPClient]) -> ToolRegistry:
    reg = ToolRegistry()
    register_lsp_tools(reg, clients)  # type: ignore[arg-type]
    return reg


@pytest.mark.asyncio
async def test_lsp_diagnostics_tool_routes_by_extension(tmp_path: Path) -> None:
    fake = FakeLSPClient(
        {
            "diagnostics": DiagnosticReport.model_validate(
                {
                    "items": [
                        {
                            "range": {
                                "start": {"line": 0, "character": 0},
                                "end": {"line": 0, "character": 1},
                            },
                            "severity": 1,
                            "message": "boom",
                            "source": "fake",
                        }
                    ]
                }
            )
        }
    )
    reg = _registry({"python": fake})
    target = tmp_path / "a.py"
    target.write_text("x = 1")

    out = await reg.execute("lsp_diagnostics", {"path": str(target)})

    assert "error:1:1 [fake] boom" in out.output
    assert fake.calls == [("diagnostics", str(target))]


@pytest.mark.asyncio
async def test_lsp_goto_definition_tool_formats_location(tmp_path: Path) -> None:
    fake = FakeLSPClient(
        {
            "definition": Location.model_validate(
                {
                    "uri": "file:///defs.py",
                    "range": {
                        "start": {"line": 3, "character": 2},
                        "end": {"line": 3, "character": 5},
                    },
                }
            )
        }
    )
    reg = _registry({"python": fake})
    target = tmp_path / "a.py"
    target.write_text("x = 1")

    out = await reg.execute("lsp_goto_definition", {"path": str(target), "line": 0, "col": 0})

    assert "file:///defs.py:4:3" in out.output


@pytest.mark.asyncio
async def test_lsp_find_references_tool_formats_locations(tmp_path: Path) -> None:
    fake = FakeLSPClient(
        {
            "references": [
                Location.model_validate(
                    {
                        "uri": "file:///a.py",
                        "range": {
                            "start": {"line": 1, "character": 0},
                            "end": {"line": 1, "character": 1},
                        },
                    }
                )
            ]
        }
    )
    reg = _registry({"python": fake})
    target = tmp_path / "a.py"
    target.write_text("x = 1")

    out = await reg.execute("lsp_find_references", {"path": str(target), "line": 0, "col": 0})

    assert "file:///a.py:2:1" in out.output


@pytest.mark.asyncio
async def test_lsp_hover_tool_formats_contents(tmp_path: Path) -> None:
    fake = FakeLSPClient(
        {"hover": Hover(contents=MarkupContent(kind="markdown", value="def foo()"))}
    )
    reg = _registry({"python": fake})
    target = tmp_path / "a.py"
    target.write_text("x = 1")

    out = await reg.execute("lsp_hover", {"path": str(target), "line": 0, "col": 0})

    assert out.output == "def foo()"


@pytest.mark.asyncio
async def test_lsp_tool_unsupported_file_type_is_fail_soft(tmp_path: Path) -> None:
    reg = _registry({"python": FakeLSPClient({})})
    target = tmp_path / "a.xyz"
    target.write_text("x")

    out = await reg.execute("lsp_diagnostics", {"path": str(target)})

    assert out.is_error
    assert "unsupported file type" in out.output


@pytest.mark.asyncio
async def test_lsp_tool_no_server_configured_is_fail_soft(tmp_path: Path) -> None:
    reg = _registry({})  # no clients
    target = tmp_path / "a.py"
    target.write_text("x = 1")

    out = await reg.execute("lsp_diagnostics", {"path": str(target)})

    assert out.is_error
    assert "no LSP server configured for language 'python'" in out.output


@pytest.mark.asyncio
async def test_lsp_tool_missing_file_is_fail_soft(tmp_path: Path) -> None:
    reg = _registry({"python": FakeLSPClient({})})

    out = await reg.execute("lsp_diagnostics", {"path": str(tmp_path / "missing.py")})

    assert out.is_error
    assert "file not found" in out.output


@pytest.mark.asyncio
async def test_lsp_tools_are_read_only(tmp_path: Path) -> None:
    reg = _registry({"python": FakeLSPClient({})})
    for name in (
        "lsp_diagnostics",
        "lsp_goto_definition",
        "lsp_find_references",
        "lsp_hover",
    ):
        assert reg.is_read_only(name), f"{name} should be read-only"
