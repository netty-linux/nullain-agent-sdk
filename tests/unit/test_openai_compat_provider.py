"""Unit tests for OpenAICompatibleProvider (issue #40).

Mirrors tests/unit/test_llm_provider.py's structure, but exercised against a
distinctly non-Ollama base_url (api.openai.com) to prove the provider is
genuinely generic — no Ollama-specific branch exists in openai_compat.py at
all (OllamaCloudProvider is a thin subclass adding only its own defaults and
optional native endpoint; see nullain/llm/ollama.py).
"""

from pathlib import Path

import httpx
import pytest
import respx
from nullain.errors import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from nullain.llm import ChatMessage, CompletionRequest, OpenAICompatibleProvider


@pytest.mark.asyncio
@respx.mock
async def test_generate_success() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").respond(
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

    provider = OpenAICompatibleProvider(
        api_key="sk-test", base_url="https://api.openai.com", max_retries=1
    )
    req = CompletionRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role="user", content="Hi")],
    )

    chunk = await provider.generate(req)
    assert chunk.delta_text == "Hello world!"
    assert chunk.finish_reason == "stop"
    assert chunk.usage is not None
    assert chunk.usage.total_tokens == 15


@pytest.mark.asyncio
@respx.mock
async def test_generate_sends_bearer_auth_header() -> None:
    route = respx.post("https://api.openai.com/v1/chat/completions").respond(
        status_code=200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    )
    provider = OpenAICompatibleProvider(
        api_key="sk-secret-key", base_url="https://api.openai.com", max_retries=1
    )
    req = CompletionRequest(model="gpt-4o-mini", messages=[ChatMessage(role="user", content="hi")])
    await provider.generate(req)
    assert route.calls.last.request.headers["Authorization"] == "Bearer sk-secret-key"


@pytest.mark.asyncio
@respx.mock
async def test_generate_works_against_an_openrouter_style_base_url() -> None:
    """Proves genericity, not just that api.openai.com happens to work:
    any OpenAI-compatible base_url (OpenRouter here) needs zero code changes."""
    respx.post("https://openrouter.ai/api/v1/chat/completions").respond(
        status_code=200,
        json={"choices": [{"message": {"role": "assistant", "content": "from openrouter"}}]},
    )
    provider = OpenAICompatibleProvider(
        api_key="sk-or-test", base_url="https://openrouter.ai/api", max_retries=1
    )
    req = CompletionRequest(
        model="anthropic/claude-3.5-sonnet", messages=[ChatMessage(role="user", content="hi")]
    )
    chunk = await provider.generate(req)
    assert chunk.delta_text == "from openrouter"


@pytest.mark.asyncio
@respx.mock
async def test_generate_tool_call_with_json_string_arguments() -> None:
    """Issue #40 acceptance criterion: arguments-as-JSON-string case (#10)."""
    respx.post("https://api.openai.com/v1/chat/completions").respond(
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

    provider = OpenAICompatibleProvider(base_url="https://api.openai.com", max_retries=1)
    req = CompletionRequest(
        model="gpt-4o-mini", messages=[ChatMessage(role="user", content="Read file")]
    )

    chunk = await provider.generate(req)
    assert chunk.tool_calls is not None
    tc = chunk.tool_calls[0]
    assert tc.id == "call_123"
    assert tc.name == "read_file"
    assert tc.arguments == {"path": "README.md"}


@pytest.mark.asyncio
@respx.mock
async def test_generate_tool_call_only_turn_has_null_content() -> None:
    """Issue #40 acceptance criterion: the assistant content: null tool-call
    turn case (#21) — a response with tool_calls and no text content must
    still parse to an empty delta_text, not raise on a missing/null field."""
    respx.post("https://api.openai.com/v1/chat/completions").respond(
        status_code=200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "call_1", "function": {"name": "bash", "arguments": "{}"}}
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    )
    provider = OpenAICompatibleProvider(base_url="https://api.openai.com", max_retries=1)
    req = CompletionRequest(model="gpt-4o-mini", messages=[ChatMessage(role="user", content="go")])

    chunk = await provider.generate(req)
    assert chunk.delta_text == ""
    assert chunk.tool_calls is not None
    assert chunk.tool_calls[0].name == "bash"


@pytest.mark.asyncio
@respx.mock
async def test_rate_limit_error() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").respond(
        status_code=429, text="Rate limit exceeded"
    )
    provider = OpenAICompatibleProvider(base_url="https://api.openai.com", max_retries=1)
    req = CompletionRequest(model="gpt-4o-mini", messages=[ChatMessage(role="user", content="Hi")])
    with pytest.raises(ProviderRateLimitError):
        await provider.generate(req)


@pytest.mark.asyncio
@respx.mock
async def test_auth_error() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").respond(
        status_code=401, text="Invalid API key"
    )
    provider = OpenAICompatibleProvider(base_url="https://api.openai.com", max_retries=1)
    req = CompletionRequest(model="gpt-4o-mini", messages=[ChatMessage(role="user", content="Hi")])
    with pytest.raises(ProviderAuthenticationError):
        await provider.generate(req)


@pytest.mark.asyncio
@respx.mock
async def test_server_error_retried_then_raises_provider_error() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").respond(
        status_code=503, text="Service unavailable"
    )
    provider = OpenAICompatibleProvider(base_url="https://api.openai.com", max_retries=2)
    req = CompletionRequest(model="gpt-4o-mini", messages=[ChatMessage(role="user", content="Hi")])
    with pytest.raises(ProviderError):
        await provider.generate(req)


@pytest.mark.asyncio
@respx.mock
async def test_timeout_error() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").side_effect = httpx.TimeoutException(
        "Request timeout"
    )
    provider = OpenAICompatibleProvider(base_url="https://api.openai.com", max_retries=1)
    req = CompletionRequest(model="gpt-4o-mini", messages=[ChatMessage(role="user", content="Hi")])
    with pytest.raises(ProviderTimeoutError):
        await provider.generate(req)


@pytest.mark.asyncio
@respx.mock
async def test_timeout_is_retried_then_succeeds() -> None:
    route = respx.post("https://api.openai.com/v1/chat/completions")
    route.side_effect = [
        httpx.TimeoutException("first attempt times out"),
        httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        ),
    ]
    provider = OpenAICompatibleProvider(base_url="https://api.openai.com", max_retries=3)
    req = CompletionRequest(model="gpt-4o-mini", messages=[ChatMessage(role="user", content="Hi")])
    chunk = await provider.generate(req)
    assert chunk.delta_text == "ok"


@pytest.mark.asyncio
@respx.mock
async def test_streaming() -> None:
    stream_body = (
        b'data: {"choices":[{"delta":{"role":"assistant","content":"Hello"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    respx.post("https://api.openai.com/v1/chat/completions").respond(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        content=stream_body,
    )
    provider = OpenAICompatibleProvider(base_url="https://api.openai.com")
    req = CompletionRequest(
        model="gpt-4o-mini", messages=[ChatMessage(role="user", content="Hi")], stream=True
    )
    parts = [chunk.delta_text async for chunk in provider.stream(req)]
    assert "".join(parts) == "Hello world"


@pytest.mark.asyncio
@respx.mock
async def test_health_check_true_on_200() -> None:
    respx.get("https://api.openai.com/v1/models").respond(status_code=200)
    provider = OpenAICompatibleProvider(base_url="https://api.openai.com")
    assert await provider.health_check() is True


@pytest.mark.asyncio
@respx.mock
async def test_health_check_false_on_error() -> None:
    respx.get("https://api.openai.com/v1/models").respond(status_code=500)
    provider = OpenAICompatibleProvider(base_url="https://api.openai.com")
    assert await provider.health_check() is False


@pytest.mark.asyncio
@respx.mock
async def test_tools_are_serialized_in_openai_function_schema() -> None:
    from nullain.llm import FunctionSpec, ToolSpec

    route = respx.post("https://api.openai.com/v1/chat/completions").respond(
        status_code=200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    )
    provider = OpenAICompatibleProvider(base_url="https://api.openai.com", max_retries=1)
    req = CompletionRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role="user", content="hi")],
        tools=[
            ToolSpec(
                function=FunctionSpec(
                    name="read_file",
                    description="Read a file",
                    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                )
            )
        ],
    )
    await provider.generate(req)
    sent = route.calls.last.request
    import json as _json

    body = _json.loads(sent.content)
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]


@pytest.mark.asyncio
@respx.mock
async def test_agent_runs_end_to_end_against_an_openai_compatible_endpoint(
    tmp_path: Path,
) -> None:
    """Issue #40's flagship acceptance criterion: Agent() through the public
    facade, not just the provider in isolation, completes a real run against
    an OpenAI-compatible endpoint."""
    from nullain.agent import Agent
    from nullain.config import NullainSettings
    from nullain.events import EventStore

    respx.post("https://api.openai.com/v1/chat/completions").respond(
        status_code=200,
        json={
            "choices": [
                {"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}
            ]
        },
    )
    provider = OpenAICompatibleProvider(
        api_key="sk-test", base_url="https://api.openai.com", max_retries=1
    )
    agent = Agent(
        settings=NullainSettings(),
        provider=provider,
        workspace_root=tmp_path,
        model="gpt-4o-mini",
        event_store=EventStore(":memory:"),
    )
    result = await agent.run("format this")
    assert result.status == "success"
    assert result.final_text == "done"
