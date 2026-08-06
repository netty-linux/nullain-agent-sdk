"""Nullain Agent SDK — LSP Client over stdio JSON-RPC (M11.2).

:class:`LSPClient` speaks the Language Server Protocol (LSP 3.17) over a
spawned server's stdio using Content-Length framing. The framing is inspired by
``nullain.mcp.transport`` but deliberately independent — LSP and MCP are
different protocols and must not be coupled.

The client performs the ``initialize`` → ``initialized`` handshake, then issues
pull-diagnostics and text-document requests (``textDocument/diagnostic``,
``textDocument/definition``, ``textDocument/references``, ``textDocument/hover``).
Before each text-document request the document is opened with
``textDocument/didOpen`` so the server has its content.

Security: the server is ALWAYS launched via ``asyncio.create_subprocess_exec``
with an explicit argv list — never ``shell=True`` (AGENTS.md rule 6). Server
output is untrusted and parsed defensively.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any, cast

from nullain.errors import LSPProtocolError, LSPTransportError
from nullain.lsp.types import (
    DiagnosticReport,
    Hover,
    Location,
    LocationLink,
    LocationResult,
)
from nullain.telemetry import get_logger

logger = get_logger(__name__)

#: LSP 3.17 protocol version advertised in the initialize request.
LSP_PROTOCOL_VERSION = "3.17.0"

#: Methods used by the client.
METHOD_INITIALIZE = "initialize"
METHOD_INITIALIZED = "initialized"
METHOD_DID_OPEN = "textDocument/didOpen"
METHOD_DIAGNOSTIC = "textDocument/diagnostic"
METHOD_DEFINITION = "textDocument/definition"
METHOD_REFERENCES = "textDocument/references"
METHOD_HOVER = "textDocument/hover"


def _validate_optional(model: type[Any], raw: Any) -> Any | None:
    """Validate a raw response into ``model``, or return ``None`` for null."""
    if raw is None:
        return None
    return model.model_validate(raw)


def _validate_location_result(raw: Any) -> LocationResult:
    """Validate a definition result (Location | LocationLink | list | null)."""
    if raw is None:
        return None
    if isinstance(raw, list):
        items = cast(list[Any], raw)
        return [
            LocationLink.model_validate(item)
            if "targetUri" in item
            else Location.model_validate(item)
            for item in items
        ]
    if isinstance(raw, dict) and "targetUri" in raw:
        return LocationLink.model_validate(raw)
    return Location.model_validate(raw)


class LSPClient:
    """JSON-RPC 2.0 client for a single LSP server over stdio.

    The client is single-flight (one outstanding request at a time), matching
    the synchronous stdio request/response model. Server-initiated
    notifications (e.g. ``window/logMessage``, ``textDocument/publishDiagnostics``)
    are read and skipped while waiting for a matching response.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        name: str = "lsp",
        timeout: float = 30.0,
    ) -> None:
        if not command:
            raise LSPTransportError("LSPClient requires a non-empty command")
        self._argv: list[str] = [command, *(args or [])]
        self._env: dict[str, str] = {**dict(os.environ), **(env or {})}
        self._env["PYTHONUNBUFFERED"] = "1"
        self._timeout = timeout
        self.name = name
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Whether the initialize handshake has completed successfully."""
        return self._initialized

    def _next_request_id(self) -> int:
        rid = self._next_id
        self._next_id += 1
        return rid

    async def start(self) -> None:
        """Spawn the LSP server subprocess."""
        if self._proc is not None:
            return
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self._argv[0],
                *self._argv[1:],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env,
            )
        except FileNotFoundError as err:
            raise LSPTransportError(
                f"LSP server executable not found: {self._argv[0]}",
                details={"argv": self._argv},
            ) from err
        except OSError as err:
            raise LSPTransportError(
                f"Failed to launch LSP server: {err}", details={"argv": self._argv}
            ) from err

    async def _ensure_started(self) -> asyncio.subprocess.Process:
        if self._proc is None:
            await self.start()
        assert self._proc is not None
        return self._proc

    async def _write_message(self, payload: dict[str, Any]) -> None:
        """Serialize a JSON-RPC message with Content-Length framing."""
        proc = await self._ensure_started()
        if proc.stdin is None:
            raise LSPTransportError("LSP server stdin is not available")
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        proc.stdin.write(header + body)
        await proc.stdin.drain()

    async def _read_message(self) -> dict[str, Any]:
        """Read one Content-Length-framed JSON-RPC message from the server."""
        proc = await self._ensure_started()
        if proc.stdout is None:
            raise LSPTransportError("LSP server stdout is not available")
        try:
            header = await asyncio.wait_for(proc.stdout.readline(), timeout=self._timeout)
        except TimeoutError as err:
            raise LSPTransportError(
                f"Timed out waiting for LSP server response after {self._timeout}s"
            ) from err
        if not header:
            raise LSPTransportError("LSP server closed stdout (EOF) before responding")
        header_line = header.decode("utf-8", errors="replace").strip()
        if not header_line.startswith("Content-Length:"):
            raise LSPProtocolError(f"LSP server sent unexpected header line: {header_line[:200]}")
        try:
            length = int(header_line.split(":", 1)[1].strip())
        except ValueError as err:
            raise LSPProtocolError(
                f"LSP server sent malformed Content-Length: {header_line[:200]}"
            ) from err
        # Consume the blank line separating headers from the body.
        blank = await asyncio.wait_for(proc.stdout.readline(), timeout=self._timeout)
        if blank.strip():
            raise LSPProtocolError("LSP server sent non-empty header separator")
        body = await asyncio.wait_for(proc.stdout.readexactly(length), timeout=self._timeout)
        try:
            obj = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as err:
            raise LSPProtocolError(f"LSP server returned non-JSON body: {body[:200]!r}") from err
        if not isinstance(obj, dict):
            raise LSPProtocolError("LSP server returned a non-object JSON message")
        return cast(dict[str, Any], obj)

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        """Send a request and return its validated ``result`` field.

        Server-initiated notifications (no ``id``) are skipped while waiting for
        the response whose ``id`` matches the request.

        Raises:
            LSPProtocolError: server returned an error or malformed response.
            LSPTransportError: transport-level failure (EOF, timeout, spawn).
        """
        request_id = self._next_request_id()
        await self._write_message(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        while True:
            msg = await self._read_message()
            if "id" not in msg:
                # Server-initiated notification; skip and keep reading.
                logger.debug("lsp_skip_notification", method=msg.get("method"))
                continue
            if msg.get("id") != request_id:
                logger.debug("lsp_skip_unmatched", id=msg.get("id"))
                continue
            if "error" in msg and msg["error"] is not None:
                err = msg["error"]
                raise LSPProtocolError(
                    f"LSP server error (code {err.get('code')}): {err.get('message')}",
                    details={"method": method},
                )
            return msg.get("result")

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a fire-and-forget notification (no response expected)."""
        await self._write_message({"jsonrpc": "2.0", "method": method, "params": params})

    async def initialize(self) -> Any:
        """Perform the LSP initialize handshake.

        Sends ``initialize``, then emits the ``initialized`` notification.
        Subsequent text-document requests require this to have succeeded.
        """
        result = await self._request(
            METHOD_INITIALIZE,
            {
                "processId": None,
                "clientInfo": {"name": "nullain", "version": "0.1.0"},
                "capabilities": {},
                "rootUri": None,
            },
        )
        self._initialized = True
        await self._notify(METHOD_INITIALIZED, {})
        return result

    async def _open_document(self, path: str, text: str) -> None:
        """Send ``textDocument/didOpen`` so the server has the document content."""
        await self._notify(
            METHOD_DID_OPEN,
            {
                "textDocument": {
                    "uri": Path(path).resolve().as_uri(),
                    "languageId": "plaintext",
                    "version": 1,
                    "text": text,
                }
            },
        )

    async def diagnostics(self, path: str, text: str) -> DiagnosticReport | None:
        """Pull diagnostics for a document (LSP 3.17 ``textDocument/diagnostic``)."""
        await self._open_document(path, text)
        raw = await self._request(
            METHOD_DIAGNOSTIC,
            {"textDocument": {"uri": Path(path).resolve().as_uri()}, "identifier": "nullain"},
        )
        return _validate_optional(DiagnosticReport, raw)

    async def goto_definition(self, path: str, text: str, line: int, col: int) -> LocationResult:
        """Resolve the definition at a position (``textDocument/definition``).

        The result may be a single :class:`Location` or :class:`LocationLink`,
        a list of either, or ``None``.
        """
        await self._open_document(path, text)
        raw = await self._request(
            METHOD_DEFINITION,
            {
                "textDocument": {"uri": Path(path).resolve().as_uri()},
                "position": {"line": line, "character": col},
            },
        )
        return _validate_location_result(raw)

    async def find_references(
        self, path: str, text: str, line: int, col: int
    ) -> list[Location] | None:
        """Find references at a position (``textDocument/references``)."""
        await self._open_document(path, text)
        raw = await self._request(
            METHOD_REFERENCES,
            {
                "textDocument": {"uri": Path(path).resolve().as_uri()},
                "position": {"line": line, "character": col},
                "context": {"includeDeclaration": True},
            },
        )
        if raw is None:
            return None
        return [Location.model_validate(item) for item in raw]

    async def hover(self, path: str, text: str, line: int, col: int) -> Hover | None:
        """Return hover information at a position (``textDocument/hover``)."""
        await self._open_document(path, text)
        raw = await self._request(
            METHOD_HOVER,
            {
                "textDocument": {"uri": Path(path).resolve().as_uri()},
                "position": {"line": line, "character": col},
            },
        )
        return _validate_optional(Hover, raw)

    async def close(self) -> None:
        """Terminate the subprocess and drain stderr for diagnostics."""
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        if proc.stdin is not None:
            with contextlib.suppress(Exception):
                proc.stdin.close()
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except TimeoutError:
            with contextlib.suppress(Exception):
                proc.kill()
                await proc.wait()
        if proc.stderr is not None:
            with contextlib.suppress(Exception):
                err_bytes = await proc.stderr.read()
                if err_bytes:
                    logger.warning(
                        "lsp_server_stderr",
                        stderr=err_bytes.decode("utf-8", errors="replace")[:2000],
                    )


__all__ = [
    "LSP_PROTOCOL_VERSION",
    "METHOD_DEFINITION",
    "METHOD_DIAGNOSTIC",
    "METHOD_HOVER",
    "METHOD_REFERENCES",
    "LSPClient",
]
