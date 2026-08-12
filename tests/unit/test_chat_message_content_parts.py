"""Unit tests for `ChatMessage`'s multimodal `content` support (PLAN.md
Fase 2 prerequisite for `VisionProvider`'s ModelRouter-backed adapter).

Covers both `to_api_dict()` directly and `OpenAICompatibleProvider`'s
outgoing HTTP payload, since `_format_request_payload` delegates straight to
`to_api_dict()` — a regression in either layer would show up here.
"""

from __future__ import annotations

import base64

import pytest
import respx
from nullain.llm import (
    ChatMessage,
    CompletionRequest,
    ImagePart,
    OpenAICompatibleProvider,
    TextPart,
)


def test_text_only_content_serializes_as_plain_string() -> None:
    msg = ChatMessage(role="user", content="Hi there")
    assert msg.to_api_dict() == {"role": "user", "content": "Hi there"}


def test_none_content_on_assistant_serializes_as_null() -> None:
    msg = ChatMessage(role="assistant", content=None)
    assert msg.to_api_dict() == {"role": "assistant", "content": None}


def test_multimodal_content_serializes_as_content_blocks() -> None:
    msg = ChatMessage(
        role="user",
        content=[
            TextPart(text="What's in this image?"),
            ImagePart(data=b"\x89PNG...", mime_type="image/png"),
        ],
    )
    data = msg.to_api_dict()
    assert data["role"] == "user"
    blocks = data["content"]
    assert blocks[0] == {"type": "text", "text": "What's in this image?"}
    assert blocks[1]["type"] == "image_url"
    expected_b64 = base64.b64encode(b"\x89PNG...").decode("ascii")
    assert blocks[1]["image_url"]["url"] == f"data:image/png;base64,{expected_b64}"


@pytest.mark.asyncio
@respx.mock
async def test_text_only_request_payload_unchanged() -> None:
    route = respx.post("https://api.openai.com/v1/chat/completions").respond(
        status_code=200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    )
    provider = OpenAICompatibleProvider(
        api_key="sk-test", base_url="https://api.openai.com", max_retries=1
    )
    req = CompletionRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role="user", content="Hi")],
    )
    await provider.generate(req)
    sent = route.calls.last.request
    import json as _json

    body = _json.loads(sent.content)
    assert body["messages"] == [{"role": "user", "content": "Hi"}]


@pytest.mark.asyncio
@respx.mock
async def test_multimodal_request_payload_includes_image_url_block() -> None:
    route = respx.post("https://api.openai.com/v1/chat/completions").respond(
        status_code=200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    )
    provider = OpenAICompatibleProvider(
        api_key="sk-test", base_url="https://api.openai.com", max_retries=1
    )
    req = CompletionRequest(
        model="gpt-4o-mini",
        messages=[
            ChatMessage(
                role="user",
                content=[
                    TextPart(text="Describe this"),
                    ImagePart(data=b"fakepngbytes", mime_type="image/png"),
                ],
            )
        ],
    )
    await provider.generate(req)
    import json as _json

    body = _json.loads(route.calls.last.request.content)
    blocks = body["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "Describe this"}
    expected_b64 = base64.b64encode(b"fakepngbytes").decode("ascii")
    assert blocks[1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{expected_b64}"},
    }
