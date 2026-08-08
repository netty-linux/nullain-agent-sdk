"""Nullain Agent SDK — Streamable-HTTP MCP Transport.

Composio's hosted MCP endpoint (and other "streamable HTTP" MCP servers)
speak one JSON-RPC message per HTTP POST rather than a long-lived stdio
pipe: each request gets its own response, framed either as a bare JSON
body or as a single SSE ``data:`` line (``Content-Type: text/event-stream``).
There is no persistent connection to keep open between calls, so
:class:`HttpTransport` needs no background reader task — unlike
:class:`~nullain.mcp.transport.StdioTransport`, ``send_request`` is a
single request/response round trip.
"""

from __future__ import annotations

import httpx

from nullain.errors import MCPTransportError
from nullain.mcp.protocol import JSONRPCNotification, JSONRPCRequest


def _parse_streamable_response(text: str) -> str:
    """Extract the JSON-RPC envelope from a streamable-HTTP response body.

    The body is either a bare JSON object, or SSE-framed with one or more
    ``data: <json>`` lines — the last non-empty ``data:`` line carries the
    JSON-RPC envelope. Returns the raw JSON text (not yet parsed into an
    envelope type) so the caller applies the same validation path used for
    stdio responses.
    """
    lines = text.splitlines()
    last_data: str | None = None
    for line in lines:
        if not line.startswith("data:"):
            continue
        raw = line[len("data:") :].strip()
        if not raw or raw == "[DONE]":
            continue
        last_data = raw
    if last_data is not None:
        return last_data
    return text


class HttpTransport:
    """MCP transport over streamable HTTP (one JSON-RPC exchange per POST).

    Args:
        url: The MCP server's HTTP endpoint.
        headers: Extra headers sent on every request (e.g. an API key header
            such as ``x-consumer-api-key`` for Composio).
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 35.0,
    ) -> None:
        if not url:
            raise MCPTransportError("HttpTransport requires a non-empty url")
        self._url = url
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **(headers or {}),
        }
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Create the underlying HTTP client (idempotent)."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout, connect=10.0))

    async def _ensure_started(self) -> httpx.AsyncClient:
        if self._client is None:
            await self.start()
        assert self._client is not None
        return self._client

    async def send_request(self, request: JSONRPCRequest) -> str:
        """POST the request and return the raw JSON-RPC response line."""
        client = await self._ensure_started()
        try:
            res = await client.post(
                self._url,
                headers=self._headers,
                content=request.model_dump_json(),
            )
        except httpx.HTTPError as err:
            raise MCPTransportError(f"MCP HTTP request failed: {err}") from err
        if res.status_code >= 400:
            raise MCPTransportError(
                f"MCP server returned HTTP {res.status_code}",
                details={"body": res.text[:500]},
            )
        # Envelope validation (JSON shape, id match, error field) is the
        # caller's job — MCPClient._request applies the same
        # _parse_jsonrpc_result path to every transport, stdio included.
        # This transport only has to hand back the raw response line.
        return _parse_streamable_response(res.text)

    async def send_notification(self, notification: JSONRPCNotification) -> None:
        """POST a fire-and-forget notification; the response body is discarded."""
        client = await self._ensure_started()
        try:
            await client.post(
                self._url,
                headers=self._headers,
                content=notification.model_dump_json(),
            )
        except httpx.HTTPError as err:
            raise MCPTransportError(f"MCP HTTP notification failed: {err}") from err

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


__all__ = ["HttpTransport"]
