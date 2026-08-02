"""Nullain Agent SDK — MCP Transports.

The client depends on the :class:`MCPTransport` protocol (hexagonal port), so
the JSON-RPC framing logic is independent of how messages are physically
exchanged. :class:`StdioTransport` is the reference adapter: it spawns an MCP
server as a subprocess and speaks newline-delimited JSON over its stdio.

Security: subprocesses are ALWAYS launched via ``asyncio.create_subprocess_exec``
with an explicit argv list — never ``shell=True`` (AGENTS.md rule 6). stderr is
captured for diagnostics and surfaced through structured logs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import Any, Protocol, cast, runtime_checkable

from nullain.errors import MCPTransportError
from nullain.mcp.protocol import JSONRPCNotification, JSONRPCRequest
from nullain.telemetry import get_logger

logger = get_logger(__name__)


@runtime_checkable
class MCPTransport(Protocol):
    """Bidirectional newline-delimited JSON transport for MCP JSON-RPC."""

    async def start(self) -> None:
        """Start the transport (e.g. spawn the server subprocess)."""
        ...

    async def send_request(self, request: JSONRPCRequest) -> str:
        """Serialize a request, send it, and return the raw response line.

        The response line is the next JSON object whose ``id`` matches
        ``request.id``; transport-level notifications/log lines without a
        matching id are skipped. Raises :class:`MCPTransportError` on EOF or
        I/O failure.
        """
        ...

    async def send_notification(self, notification: JSONRPCNotification) -> None:
        """Send a fire-and-forget notification (no response expected)."""
        ...

    async def close(self) -> None:
        """Tear the transport down (terminate the subprocess, close streams)."""
        ...


class StdioTransport:
    """MCP transport over a spawned subprocess's stdin/stdout.

    The server is launched with ``[command, *args]`` as an explicit argv list
    (no shell). Each JSON-RPC message is one UTF-8 line terminated by ``\\n``.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not command:
            raise MCPTransportError("StdioTransport requires a non-empty command")
        self._argv: list[str] = [command, *(args or [])]
        # Merge the caller's env into the current environment rather than
        # replacing it, so PATH and other essentials remain available.
        self._env: dict[str, str] = {**dict(os.environ), **(env or {})}
        self._env["PYTHONUNBUFFERED"] = "1"
        self._timeout = timeout
        self._proc: asyncio.subprocess.Process | None = None

    def _argv_for_exec(self) -> list[str]:
        """Return the argv list passed to create_subprocess_exec (no shell)."""
        return list(self._argv)

    async def start(self) -> None:
        """Spawn the MCP server subprocess."""
        if self._proc is not None:
            return
        argv = self._argv_for_exec()
        try:
            self._proc = await asyncio.create_subprocess_exec(
                argv[0],
                *argv[1:],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env,
            )
        except FileNotFoundError as err:
            raise MCPTransportError(
                f"MCP server executable not found: {argv[0]}", details={"argv": argv}
            ) from err
        except OSError as err:
            raise MCPTransportError(
                f"Failed to launch MCP server: {err}", details={"argv": argv}
            ) from err

    async def _ensure_started(self) -> asyncio.subprocess.Process:
        if self._proc is None:
            await self.start()
        assert self._proc is not None
        return self._proc

    async def _write_line(self, line: str) -> None:
        proc = await self._ensure_started()
        if proc.stdin is None:
            raise MCPTransportError("MCP server stdin is not available")
        data = (line + "\n").encode("utf-8")
        proc.stdin.write(data)
        await proc.stdin.drain()

    async def _read_line(self) -> str:
        proc = await self._ensure_started()
        if proc.stdout is None:
            raise MCPTransportError("MCP server stdout is not available")
        try:
            raw = await asyncio.wait_for(proc.stdout.readline(), timeout=self._timeout)
        except TimeoutError as err:
            raise MCPTransportError(
                f"Timed out waiting for MCP server response after {self._timeout}s"
            ) from err
        if not raw:
            raise MCPTransportError("MCP server closed stdout (EOF) before responding")
        return raw.decode("utf-8", errors="replace").strip()

    async def send_notification(self, notification: JSONRPCNotification) -> None:
        """Send a notification; no response is read."""
        await self._write_line(notification.model_dump_json())

    async def send_request(self, request: JSONRPCRequest) -> str:
        """Send a request and read the matching response line.

        Lines that are not JSON objects carrying a matching ``id`` (server-side
        log notifications, progress events, empty lines) are skipped so a chatty
        server cannot desynchronize the request/response pairing.
        """
        await self._write_line(request.model_dump_json())

        while True:
            line = await self._read_line()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("mcp_skip_non_json_line", line=line[:200])
                continue
            if not isinstance(obj, dict):
                logger.debug("mcp_skip_non_object_line", line=line[:200])
                continue
            data = cast(dict[str, Any], obj)
            if data.get("id") != request.id:
                # A notification or unrelated message; skip and keep reading.
                logger.debug("mcp_skip_unmatched_line", line=line[:200])
                continue
            return line

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
                        "mcp_server_stderr",
                        stderr=err_bytes.decode("utf-8", errors="replace")[:2000],
                    )


__all__ = ["MCPTransport", "StdioTransport"]
