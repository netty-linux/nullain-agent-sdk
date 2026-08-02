"""Nullain Tools — web_fetch: retrieve a URL as raw markdown text.

No extraction model is involved: HTML is converted to plain text with a
minimal tag-stripping pass. This gives the agent page content to reason over
without an LLM summarization round-trip, matching the parity-feature scope.
"""

import re

import httpx
from nullain.tools import RegisteredTool, tool

_MAX_RESPONSE_CHARS = 50_000
_REQUEST_TIMEOUT = 30.0
_USER_AGENT = "Nullain-Agent-SDK/0.1 (+web_fetch)"

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


def create_web_fetch_tool() -> RegisteredTool:
    @tool(
        name="web_fetch",
        description=(
            "Fetch a URL and return its content as plain text (HTML is stripped "
            "to text; no summarization). Useful for reading documentation or APIs."
        ),
        read_only=True,
    )
    async def web_fetch(url: str) -> str:
        if not url.startswith(("http://", "https://")):
            return f"Error: URL must start with http:// or https:// (got '{url}')."

        try:
            async with httpx.AsyncClient(
                timeout=_REQUEST_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
        except httpx.HTTPStatusError as err:
            return f"Error: HTTP {err.response.status_code} for '{url}'."
        except httpx.RequestError as err:
            return f"Error: Request failed for '{url}': {err}."

        content_type = response.headers.get("content-type", "")
        body = response.text
        if "html" in content_type.lower():
            body = html_to_text(body)

        if len(body) > _MAX_RESPONSE_CHARS:
            body = body[:_MAX_RESPONSE_CHARS] + "\n...[truncated]"
        return body or "Error: Empty response body."

    return web_fetch


__all__ = ["create_web_fetch_tool", "html_to_text"]
