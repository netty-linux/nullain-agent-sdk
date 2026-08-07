"""Unit tests for LLM Provider layer with respx mocking."""

import httpx
import pytest
import respx
from nullain.errors import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from nullain.llm import ChatMessage, CompletionRequest, OllamaCloudProvider, ToolCall


@pytest.mark.asyncio
@respx.mock
async def test_ollama_generate_success() -> None:
    respx.post("https://ollama.com/v1/chat/completions").respond(
        status_code=200,
        json={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello world!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )

    provider = OllamaCloudProvider(base_url="https://ollama.com", max_retries=1)
    req = CompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Hi")],
    )

    chunk = await provider.generate(req)
    assert chunk.delta_text == "Hello world!"
    assert chunk.finish_reason == "stop"
    assert chunk.usage is not None
    assert chunk.usage.total_tokens == 15


@pytest.mark.asyncio
@respx.mock
async def test_ollama_generate_tool_call() -> None:
    respx.post("https://ollama.com/v1/chat/completions").respond(
        status_code=200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "README.md"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    )

    provider = OllamaCloudProvider(base_url="https://ollama.com", max_retries=1)
    req = CompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Read file")],
    )

    chunk = await provider.generate(req)
    assert chunk.tool_calls is not None
    assert len(chunk.tool_calls) == 1
    tc = chunk.tool_calls[0]
    assert tc.id == "call_123"
    assert tc.name == "read_file"
    assert tc.arguments == {"path": "README.md"}


@pytest.mark.asyncio
@respx.mock
async def test_ollama_rate_limit_error() -> None:
    respx.post("https://ollama.com/v1/chat/completions").respond(
        status_code=429, text="Rate limit exceeded"
    )

    provider = OllamaCloudProvider(base_url="https://ollama.com", max_retries=1)
    req = CompletionRequest(model="test-model", messages=[ChatMessage(role="user", content="Hi")])

    with pytest.raises(ProviderRateLimitError):
        await provider.generate(req)


@pytest.mark.asyncio
@respx.mock
async def test_ollama_auth_error() -> None:
    respx.post("https://ollama.com/v1/chat/completions").respond(
        status_code=401, text="Unauthorized"
    )

    provider = OllamaCloudProvider(base_url="https://ollama.com", max_retries=1)
    req = CompletionRequest(model="test-model", messages=[ChatMessage(role="user", content="Hi")])

    with pytest.raises(ProviderAuthenticationError):
        await provider.generate(req)


@pytest.mark.asyncio
@respx.mock
async def test_ollama_timeout_error() -> None:
    respx.post("https://ollama.com/v1/chat/completions").side_effect = httpx.TimeoutException(
        "Request timeout"
    )

    provider = OllamaCloudProvider(base_url="https://ollama.com", max_retries=1)
    req = CompletionRequest(model="test-model", messages=[ChatMessage(role="user", content="Hi")])

    with pytest.raises(ProviderTimeoutError):
        await provider.generate(req)


@pytest.mark.asyncio
@respx.mock
async def test_ollama_streaming() -> None:
    stream_content = (
        'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
        'data: {"choices": [{"delta": {"content": " world!"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    respx.post("https://ollama.com/v1/chat/completions").respond(
        status_code=200, text=stream_content
    )

    provider = OllamaCloudProvider(base_url="https://ollama.com")
    req = CompletionRequest(
        model="test-model", messages=[ChatMessage(role="user", content="Hi")], stream=True
    )

    chunks: list[str] = []
    async for chunk in provider.stream(req):
        chunks.append(chunk.delta_text)

    assert "".join(chunks) == "Hello world!"


@pytest.mark.asyncio
@respx.mock
async def test_ollama_health_check() -> None:
    respx.get("https://ollama.com/v1/models").respond(status_code=200)

    provider = OllamaCloudProvider(base_url="https://ollama.com")
    assert await provider.health_check() is True


# ---------------------------------------------------------------------------
# Native Ollama API (/api/chat, /api/tags)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_native_generate_success() -> None:
    respx.post("https://ollama.com/api/chat").respond(
        status_code=200,
        json={
            "model": "test-model",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "Hello native!"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        },
    )

    provider = OllamaCloudProvider(
        base_url="https://ollama.com", endpoint_type="native", max_retries=1
    )
    req = CompletionRequest(model="test-model", messages=[ChatMessage(role="user", content="Hi")])

    chunk = await provider.generate(req)
    assert chunk.delta_text == "Hello native!"
    assert chunk.finish_reason == "stop"
    assert chunk.usage is not None
    assert chunk.usage.prompt_tokens == 10
    assert chunk.usage.completion_tokens == 5
    assert chunk.usage.total_tokens == 15


@pytest.mark.asyncio
@respx.mock
async def test_native_generate_tool_call() -> None:
    respx.post("https://ollama.com/api/chat").respond(
        status_code=200,
        json={
            "model": "test-model",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": {"path": "README.md"}},
                    }
                ],
            },
            "done": True,
        },
    )

    provider = OllamaCloudProvider(
        base_url="https://ollama.com", endpoint_type="native", max_retries=1
    )
    req = CompletionRequest(
        model="test-model", messages=[ChatMessage(role="user", content="Read file")]
    )

    chunk = await provider.generate(req)
    assert chunk.tool_calls is not None
    assert len(chunk.tool_calls) == 1
    tc = chunk.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "read_file"
    assert tc.arguments == {"path": "README.md"}


@pytest.mark.asyncio
@respx.mock
async def test_native_streaming() -> None:
    # Native streams raw NDJSON objects (no "data: " prefix), terminated by a
    # final object with done=true carrying token counts.
    stream_content = (
        '{"model":"m","message":{"role":"assistant","content":"Hello"}}\n'
        '{"model":"m","message":{"role":"assistant","content":" native!"}}\n'
        '{"model":"m","message":{"role":"assistant","content":""},'
        '"done":true,"prompt_eval_count":4,"eval_count":2}\n'
    )
    respx.post("https://ollama.com/api/chat").respond(status_code=200, text=stream_content)

    provider = OllamaCloudProvider(base_url="https://ollama.com", endpoint_type="native")
    req = CompletionRequest(
        model="m", messages=[ChatMessage(role="user", content="Hi")], stream=True
    )

    parts: list[str] = []
    usage = None
    async for chunk in provider.stream(req):
        if chunk.delta_text:
            parts.append(chunk.delta_text)
        if chunk.usage:
            usage = chunk.usage

    assert "".join(parts) == "Hello native!"
    assert usage is not None
    assert usage.total_tokens == 6


@pytest.mark.asyncio
@respx.mock
async def test_native_health_check() -> None:
    respx.get("https://ollama.com/api/tags").respond(status_code=200)

    provider = OllamaCloudProvider(base_url="https://ollama.com", endpoint_type="native")
    assert await provider.health_check() is True


@pytest.mark.asyncio
@respx.mock
async def test_ollama_timeout_is_retried_then_succeeds() -> None:
    """Regression: a request timeout must be retried, not immediately fatal.

    Found via live testing against Ollama Cloud: httpx.TimeoutException was
    converted straight to ProviderTimeoutError inside the retried block, so
    tenacity's retry_if_exception_type(TransientHttpError) never matched it
    and a single slow response aborted the whole call even with
    max_retries > 1 configured. A route that times out once then succeeds
    must now return the successful response instead of raising.
    """
    route = respx.post("https://ollama.com/v1/chat/completions")
    route.side_effect = [
        httpx.TimeoutException("Request timeout"),
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Recovered!"},
                        "finish_reason": "stop",
                    }
                ]
            },
        ),
    ]

    provider = OllamaCloudProvider(base_url="https://ollama.com", max_retries=3)
    req = CompletionRequest(model="test-model", messages=[ChatMessage(role="user", content="Hi")])

    chunk = await provider.generate(req)
    assert chunk.delta_text == "Recovered!"
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_ollama_timeout_raises_after_retries_exhausted() -> None:
    """A timeout on every attempt still surfaces as ProviderTimeoutError once
    max_retries is exhausted, distinct from the generic ProviderError used
    for exhausted network-error retries."""
    from nullain.errors import ProviderTimeoutError

    respx.post("https://ollama.com/v1/chat/completions").side_effect = httpx.TimeoutException(
        "Request timeout"
    )

    provider = OllamaCloudProvider(base_url="https://ollama.com", max_retries=2)
    req = CompletionRequest(model="test-model", messages=[ChatMessage(role="user", content="Hi")])

    with pytest.raises(ProviderTimeoutError):
        await provider.generate(req)


@pytest.mark.asyncio
@respx.mock
async def test_ollama_error_response_logs_status_and_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (M20): _handle_error_response only ever raised — no
    structured log line recorded which status code came back or why,
    making a 400/401/429 failure invisible in telemetry until the raised
    exception's message was read from application logs."""
    from nullain.llm import ollama as ollama_module

    respx.post("https://ollama.com/v1/chat/completions").respond(
        status_code=400, json={"error": {"message": "invalid request"}}
    )

    calls: list[tuple[str, dict[str, object]]] = []

    def _spy(event: str, **kwargs: object) -> None:
        calls.append((event, kwargs))

    monkeypatch.setattr(ollama_module.logger, "warning", _spy)

    provider = OllamaCloudProvider(base_url="https://ollama.com", max_retries=1)
    req = CompletionRequest(model="test-model", messages=[ChatMessage(role="user", content="Hi")])

    with pytest.raises(Exception):  # noqa: B017  (ProviderError subclass — any is fine here)
        await provider.generate(req)

    logged = [c for c in calls if c[0] == "llm_http_error_response"]
    assert len(logged) == 1
    assert logged[0][1]["status_code"] == 400


@pytest.mark.asyncio
@respx.mock
async def test_ollama_retries_exhausted_logs_attempt_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (M20): exhausting retries on a persistent timeout raised
    ProviderTimeoutError with no observable record of how many attempts
    were made before giving up."""
    from nullain.llm import ollama as ollama_module

    respx.post("https://ollama.com/v1/chat/completions").side_effect = httpx.TimeoutException(
        "Request timeout"
    )

    calls: list[tuple[str, dict[str, object]]] = []

    def _spy(event: str, **kwargs: object) -> None:
        calls.append((event, kwargs))

    monkeypatch.setattr(ollama_module.logger, "error", _spy)

    provider = OllamaCloudProvider(base_url="https://ollama.com", max_retries=2)
    req = CompletionRequest(model="test-model", messages=[ChatMessage(role="user", content="Hi")])

    with pytest.raises(ProviderTimeoutError):
        await provider.generate(req)

    logged = [c for c in calls if c[0] == "llm_request_retries_exhausted"]
    assert len(logged) == 1
    assert logged[0][1]["attempts"] == 2
    assert logged[0][1]["reason"] == "timeout"


def test_chat_message_to_api_dict_serializes_tool_call_arguments_as_string() -> None:
    """Regression: found via live testing against Ollama Cloud, which rejected
    a follow-up request with HTTP 400 ("cannot unmarshal object into ...
    arguments of type string"). The OpenAI-compatible schema requires
    function.arguments to be a JSON-encoded string, but ChatMessage.to_api_dict
    was passing ToolCall.arguments through as a raw dict when rebuilding an
    assistant turn's tool_calls for the next request's message history —
    breaking any multi-step tool-calling conversation on the second turn."""
    msg = ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(id="call_1", name="write_file", arguments={"path": "a.txt", "content": "hi"})
        ],
    )
    data = msg.to_api_dict()
    args = data["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)
    import json

    assert json.loads(args) == {"path": "a.txt", "content": "hi"}


def test_chat_message_to_api_dict_passes_through_string_arguments_unchanged() -> None:
    """A ToolCall whose arguments are already a string (e.g. mid-stream
    fragment) should not be double-encoded."""
    msg = ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="call_1", name="write_file", arguments='{"path": "a.txt"}')],
    )
    data = msg.to_api_dict()
    assert data["tool_calls"][0]["function"]["arguments"] == '{"path": "a.txt"}'


def test_chat_message_to_api_dict_includes_null_content_for_tool_call_only_assistant_turn() -> None:
    """Regression: found via live testing against Ollama Cloud. A pure
    tool-call assistant turn (content=None, only tool_calls set — the
    common case) used to *omit* the "content" key entirely when replayed
    into a later request's message history. Ollama Cloud's compat shim
    rejected that later request with HTTP 400 ("invalid message content
    type: <nil>") — its server-side unmarshal needs the key present as
    JSON null, matching the real OpenAI spec for this case, rather than
    absent. Every other role's content is always a required, non-None
    string, so only the assistant/content=None case is affected."""
    msg = ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="call_1", name="bash", arguments={"command_args": ["ls"]})],
    )
    data = msg.to_api_dict()
    assert "content" in data
    assert data["content"] is None


def test_chat_message_to_api_dict_omits_content_for_non_assistant_none() -> None:
    """Non-assistant roles keep the prior omit-when-None behavior — every
    other role's content is a required non-None string in practice, so
    this only documents that the assistant-specific fix above is scoped
    correctly and doesn't change other roles' serialization."""
    msg = ChatMessage(role="system", content=None)
    data = msg.to_api_dict()
    assert "content" not in data
