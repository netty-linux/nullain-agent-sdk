"""Nullain Tools — ask_user: interactive clarifying-question tool.

``ask_user`` lets the agent request information from the human operator
mid-run. It is backed by an async ``AskUserCallback`` that resolves the user's
answer text. In the stdio daemon this is wired to an NDJSON request/response
pair over the protocol; with no callback configured the tool returns an error
string instead of blocking, so an unattended run degrades gracefully rather
than hanging forever.
"""

from collections.abc import Awaitable, Callable

from nullain.tools import RegisteredTool, tool
from nullain.tools.result import ToolResult

AskUserCallback = Callable[[str], Awaitable[str]]


def create_ask_user_tool(callback: AskUserCallback | None = None) -> RegisteredTool:
    @tool(
        name="ask_user",
        description=(
            "Ask the user a clarifying question and return their answer. "
            "Use when critical information is missing and cannot be inferred."
        ),
    )
    async def ask_user(question: str) -> str | ToolResult:
        if not question.strip():
            return ToolResult(
                output="Error: ask_user requires a non-empty question.",
                is_error=True,
                error_type="ToolError",
            )
        if callback is None:
            return ToolResult(
                output="Error: ask_user is unavailable (no user-interaction channel configured).",
                is_error=True,
                error_type="ToolError",
            )
        try:
            answer = await callback(question)
        except Exception as err:
            return ToolResult(
                output=f"Error: ask_user failed: {err}",
                is_error=True,
                error_type="ToolError",
            )
        if answer.strip():
            return answer
        return ToolResult(
            output="Error: ask_user received an empty answer.",
            is_error=True,
            error_type="ToolError",
        )

    return ask_user


__all__ = ["AskUserCallback", "create_ask_user_tool"]
