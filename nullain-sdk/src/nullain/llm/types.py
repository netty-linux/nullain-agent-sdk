"""Nullain Agent SDK — LLM Data Types and Schemas."""

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """Represents a tool execution request from the model.

    During streaming, ``arguments`` may transiently hold a raw JSON *string
    fragment* (OpenAI-style incremental tool-call deltas) that the loop merges
    across chunks before execution; a complete call always carries a ``dict``.
    """

    id: str = Field(description="Unique identifier for this tool invocation")
    name: str = Field(description="Name of the tool to execute")
    arguments: dict[str, Any] | str = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )


class FunctionSpec(BaseModel):
    """Specification of a tool function schema."""

    name: str = Field(description="Function name")
    description: str = Field(description="Function description")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for arguments"
    )


class ToolSpec(BaseModel):
    """Tool definition wrapper."""

    type: Literal["function"] = "function"
    function: FunctionSpec


class TokenUsage(BaseModel):
    """Token consumption metrics for an LLM request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatMessage(BaseModel):
    """Represents a single message in the conversation history."""

    role: Role
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def to_api_dict(self) -> dict[str, Any]:
        """Convert to dict matching standard API format."""
        data: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            data["content"] = self.content
        if self.name is not None:
            data["name"] = self.name
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        return data


class CompletionRequest(BaseModel):
    """Request payload for model completion generation or streaming."""

    model: str = Field(description="Model identifier")
    messages: list[ChatMessage] = Field(description="Conversation context history")
    tools: list[ToolSpec] | None = Field(default=None, description="Available tools")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = Field(default=True, description="Whether to stream response chunks")


class CompletionChunk(BaseModel):
    """Single response unit from generation or streaming LLM execution."""

    delta_text: str = ""
    tool_calls: list[ToolCall] | None = None
    usage: TokenUsage | None = None
    finish_reason: str | None = None


__all__ = [
    "ChatMessage",
    "CompletionChunk",
    "CompletionRequest",
    "FunctionSpec",
    "Role",
    "TokenUsage",
    "ToolCall",
    "ToolSpec",
]
