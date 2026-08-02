"""Nullain Agent SDK — LLM Response Models."""

from pydantic import BaseModel, ConfigDict, Field


class FunctionCall(BaseModel):
    """Function call data."""

    name: str = ""
    arguments: dict[str, object] | str = Field(default_factory=dict[str, object])

    model_config = ConfigDict(strict=True)


class ToolCallData(BaseModel):
    """Tool call payload."""

    id: str = ""
    type: str = "function"
    function: FunctionCall = Field(default_factory=FunctionCall)

    model_config = ConfigDict(strict=True)


class OllamaMessage(BaseModel):
    """Ollama native message payload."""

    role: str = ""
    content: str = ""
    tool_calls: list[ToolCallData] | None = None

    model_config = ConfigDict(strict=True)


class OllamaNativeResponse(BaseModel):
    """Ollama native response payload."""

    model: str = ""
    created_at: str = ""
    message: OllamaMessage | None = None
    done: bool = False
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None

    model_config = ConfigDict(strict=True)


class OpenAIMessage(BaseModel):
    """OpenAI message/delta payload."""

    role: str | None = None
    content: str | None = None
    tool_calls: list[ToolCallData] | None = None

    model_config = ConfigDict(strict=True)


class OpenAIChoice(BaseModel):
    """OpenAI choice payload."""

    index: int = 0
    message: OpenAIMessage | None = None
    delta: OpenAIMessage | None = None
    finish_reason: str | None = None

    model_config = ConfigDict(strict=True)


class OpenAIUsage(BaseModel):
    """OpenAI usage payload."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    model_config = ConfigDict(strict=True)


class OpenAIResponse(BaseModel):
    """OpenAI compatible response payload."""

    id: str = ""
    object: str = ""
    created: int = 0
    model: str = ""
    choices: list[OpenAIChoice] = Field(default_factory=list[OpenAIChoice])
    usage: OpenAIUsage | None = None

    model_config = ConfigDict(strict=True)
