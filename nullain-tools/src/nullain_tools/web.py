"""Nullain Tools — web_fetch: retrieve a URL as raw markdown text.

No extraction model is involved: HTML is converted to plain text with a
minimal tag-stripping pass. This gives the agent page content to reason over
without an LLM summarization round-trip, matching the parity-feature scope.
"""

import re

import httpx
from nullain.authority import Capability
from nullain.tools import RegisteredTool, tool
from nullain.tools.result import ToolResult

_MAX_RESPONSE_CHARS = 50_000
_REQUEST_TIMEOUT = 30.0
_USER_AGENT = "Nullain-Agent-SDK/0.1 (+web_fetch)"

#: Status codes a site commonly returns for bot detection, paywalls, or
#: rate limiting — distinct from a genuine "this URL is broken" error, so
#: the agent can tell "try a different source" apart from "fix the URL".
_BOT_BLOCK_STATUS_CODES = frozenset({401, 403, 429})

# Block-level tags that should introduce a line break when stripped.
_BLOCK_TAGS = re.compile(r"</?(p|div|br|li|h[1-6]|tr|section|article|header|footer)[^>]*>", re.I)
_SCRIPT_STYLE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.DOTALL)
_TAG_STRIP = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"[ \t]+")


def html_to_text(html: str) -> str:
    """Minimal HTML → plain text conversion (no extraction model)."""
    text = _SCRIPT_STYLE.sub("", html)
    text = _BLOCK_TAGS.sub("\n", text)
    text = _TAG_STRIP.sub("", text)
    text = (
        text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    )
    lines = [_WHITESPACE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def create_web_fetch_tool(headers: dict[str, str] | None = None) -> RegisteredTool:
    """Build the ``web_fetch`` tool.

    Args:
        headers: HTTP headers sent on every request, overriding the
            default ``User-Agent``/``Accept``/``Accept-Language``. When
            None, the honest bot-identifying defaults are used (see
            ``nullain.config.WebFetchConfig``'s docstring for why those
            are the default rather than a spoofed browser User-Agent).
    """
    request_headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        **(headers or {}),
    }

    @tool(
        name="web_fetch",
        description=(
            "Fetch a URL and return its content as plain text (HTML is stripped "
            "to text; no summarization). Useful for reading documentation or APIs."
        ),
        read_only=True,
        requires=frozenset({Capability.READ, Capability.NETWORK}),
    )
    async def web_fetch(url: str) -> str | ToolResult:
        if not url.startswith(("http://", "https://")):
            return ToolResult(
                output=f"Error: URL must start with http:// or https:// (got '{url}').",
                is_error=True,
                error_type="ToolError",
            )

        try:
            async with httpx.AsyncClient(
                timeout=_REQUEST_TIMEOUT,
                follow_redirects=True,
                headers=request_headers,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
        except httpx.HTTPStatusError as err:
            status = err.response.status_code
            message = f"Error: HTTP {status} for '{url}'."
            if status in _BOT_BLOCK_STATUS_CODES:
                message += (
                    " This site is likely blocking automated requests (bot "
                    "detection, paywall, or rate limiting) — retrying this "
                    "same URL will not help; use a different source instead."
                )
            return ToolResult(output=message, is_error=True, error_type="ToolError")
        except httpx.RequestError as err:
            return ToolResult(
                output=f"Error: Request failed for '{url}': {err}.",
                is_error=True,
                error_type="ToolError",
            )

        content_type = response.headers.get("content-type", "")
        body = response.text
        if "html" in content_type.lower():
            body = html_to_text(body)

        if len(body) > _MAX_RESPONSE_CHARS:
            body = body[:_MAX_RESPONSE_CHARS] + "\n...[truncated]"
        return body or "Error: Empty response body."

    return web_fetch


__all__ = ["create_web_fetch_tool", "html_to_text"]
