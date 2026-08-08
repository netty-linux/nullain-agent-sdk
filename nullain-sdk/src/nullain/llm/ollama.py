"""Nullain Agent SDK — Ollama Cloud LLM Provider Adapter.

:class:`OllamaCloudProvider` is :class:`~nullain.llm.openai_compat.OpenAICompatibleProvider`
with Ollama Cloud's defaults (``base_url="https://ollama.com"``), plus support
for Ollama's optional native ``/api/chat`` endpoint (``endpoint_type="native"``)
— Ollama Cloud's *default* mode (``endpoint_type="v1"``) already speaks the
same OpenAI-compatible ``/v1/chat/completions`` schema the base class
implements, so this subclass only needs to override what's genuinely
different: URL selection, health-check path, request-payload shaping for
``options``/``tools`` in native form, and native-response parsing (issue #40
extracted the shared v1 logic into ``openai_compat.py`` rather than
duplicating it here).
"""

import json
from typing import Any, cast

import httpx

from nullain.llm.openai_compat import OpenAICompatibleProvider
from nullain.llm.response_models import OllamaNativeResponse
from nullain.llm.types import (
    CompletionChunk,
    CompletionRequest,
    TokenUsage,
    ToolCall,
)


class OllamaCloudProvider(OpenAICompatibleProvider):
    """Adapter for Ollama Cloud (and, via ``endpoint_type="v1"``, any other
    OpenAI-compatible Chat API reachable at ``base_url``)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://ollama.com",
        endpoint_type: str = "v1",  # "v1" for /v1/chat/completions or "native" for /api/chat
        timeout: float = 60.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )
        self.endpoint_type = endpoint_type

    def _get_url(self) -> str:
        if self.endpoint_type == "native":
            return f"{self.base_url}/api/chat"
        return super()._get_url()

    def _health_check_url(self) -> str:
        if self.endpoint_type == "native":
            return f"{self.base_url}/api/tags"
        return super()._health_check_url()

    def _format_request_payload(self, request: CompletionRequest, stream: bool) -> dict[str, Any]:
        if self.endpoint_type != "native":
            return super()._format_request_payload(request, stream)

        messages = [m.to_api_dict() for m in request.messages]
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": stream,
            "temperature": request.temperature,
        }

        if request.max_tokens is not None:
            options: dict[str, Any] = payload.setdefault("options", {})
            options["num_predict"] = request.max_tokens

        if request.tools:
            payload["tools"] = [t.model_dump() for t in request.tools]
        return payload

    def _parse_chunk_data(self, data: dict[str, Any]) -> CompletionChunk:
        if self.endpoint_type != "native":
            return super()._parse_chunk_data(data)

        delta_text: str = ""
        tool_calls: list[ToolCall] | None = None
        usage: TokenUsage | None = None
        finish_reason: str | None = None

        response_native = OllamaNativeResponse.model_validate(data)

        if response_native.message:
            delta_text = response_native.message.content
            if response_native.message.tool_calls:
                tool_calls = []
                for idx, tc in enumerate(response_native.message.tool_calls):
                    tc_id = tc.id or f"call_{idx}"
                    parsed_args: dict[str, object] | str = {}
                    if isinstance(tc.function.arguments, dict):
                        parsed_args = tc.function.arguments
                    elif tc.function.arguments.strip():
                        try:
                            parsed: object = json.loads(tc.function.arguments)
                            if isinstance(parsed, dict):
                                parsed_dict = cast(dict[str, object], parsed)
                                parsed_args = {str(k): v for k, v in parsed_dict.items()}
                            else:
                                parsed_args = tc.function.arguments
                        except json.JSONDecodeError:
                            # Streaming fragment: keep the raw string so the
                            # loop can merge it with later chunks (M10 D4).
                            parsed_args = tc.function.arguments

                    tool_calls.append(
                        ToolCall(
                            id=tc_id,
                            name=tc.function.name,
                            arguments=parsed_args,
                        )
                    )

        if response_native.done:
            finish_reason = "stop"

        has_eval = (
            response_native.prompt_eval_count is not None or response_native.eval_count is not None
        )
        if has_eval:
            p_tokens = response_native.prompt_eval_count or 0
            c_tokens = response_native.eval_count or 0
            usage = TokenUsage(
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=p_tokens + c_tokens,
            )

        return CompletionChunk(
            delta_text=delta_text,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
        )


__all__ = ["OllamaCloudProvider"]
