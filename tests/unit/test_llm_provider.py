"""Unit tests for LLM Provider layer with respx mocking."""

import httpx
import pytest
import respx
from nullain.errors import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from nullain.llm import ChatMessage, CompletionRequest, OllamaCloudProvider


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
