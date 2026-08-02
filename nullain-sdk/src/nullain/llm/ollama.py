"""Nullain Agent SDK — Ollama Cloud & OpenAI Compatible LLM Provider Adapter."""

import json
from collections.abc import AsyncGenerator
from typing import Any, cast

import httpx
from structlog.stdlib import BoundLogger
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from nullain.errors import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from nullain.llm.provider import LLMProvider
from nullain.llm.types import (
    CompletionChunk,
    CompletionRequest,
    TokenUsage,
    ToolCall,
)
from nullain.telemetry import get_logger

logger: BoundLogger = get_logger("nullain.llm.ollama")


class TransientHttpError(Exception):
    """Internal exception for retriable HTTP errors."""

    pass


class OllamaCloudProvider(LLMProvider):
    """Adapter for Ollama Cloud and OpenAI-compatible Chat APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://ollama.com",
        endpoint_type: str = "v1",  # "v1" for /v1/chat/completions or "native" for /api/chat
        timeout: float = 60.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.endpoint_type = endpoint_type
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = client

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get_url(self) -> str:
        if self.endpoint_type == "native":
            return f"{self.base_url}/api/chat"
        return f"{self.base_url}/v1/chat/completions"

    async def health_check(self) -> bool:
        """Check availability of the Ollama provider."""
        url = (
            f"{self.base_url}/api/tags"
            if self.endpoint_type == "native"
            else f"{self.base_url}/v1/models"
        )
        client = self._client or httpx.AsyncClient(timeout=10.0)
        should_close = self._client is None
        try:
            response = await client.get(url, headers=self._get_headers())
            return response.status_code == 200
        except Exception as err:
            logger.warning("Health check failed", error=str(err))
            return False
        finally:
            if should_close:
                await client.aclose()

    def _format_request_payload(self, request: CompletionRequest, stream: bool) -> dict[str, Any]:
        messages = [m.to_api_dict() for m in request.messages]
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": stream,
            "temperature": request.temperature,
        }

        if request.max_tokens is not None:
            if self.endpoint_type == "native":
                options: dict[str, Any] = payload.setdefault("options", {})
                options["num_predict"] = request.max_tokens
            else:
                payload["max_tokens"] = request.max_tokens

        if request.tools:
            if self.endpoint_type == "native":
                payload["tools"] = [t.model_dump() for t in request.tools]
            else:
                payload["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": t.function.name,
                            "description": t.function.description,
                            "parameters": t.function.parameters,
                        },
                    }
                    for t in request.tools
                ]
        return payload

    def _parse_chunk_data(self, data: dict[str, Any]) -> CompletionChunk:
        delta_text: str = ""
        tool_calls: list[ToolCall] | None = None
        usage: TokenUsage | None = None
        finish_reason: str | None = None

        if self.endpoint_type == "native":
            # Ollama native API /api/chat
            message = cast(dict[str, Any], data.get("message") or {})
            delta_text = str(message.get("content") or "")
            finish_reason = "stop" if data.get("done") else None

            raw_tcs = cast(list[dict[str, Any]] | None, message.get("tool_calls"))
            if raw_tcs:
                tool_calls = []
                for idx, tc in enumerate(raw_tcs):
                    fn = cast(dict[str, Any], tc.get("function") or {})
                    tc_id = str(tc.get("id") or f"call_{idx}")
                    raw_args: Any = fn.get("arguments", {})
                    parsed_args: dict[str, Any] = {}
                    if isinstance(raw_args, dict):
                        parsed_args = cast(dict[str, Any], raw_args)
                    elif isinstance(raw_args, str) and raw_args.strip():
                        try:
                            parsed_args = cast(dict[str, Any], json.loads(raw_args))
                        except json.JSONDecodeError:
                            parsed_args = {}

                    tool_calls.append(
                        ToolCall(
                            id=tc_id,
                            name=str(fn.get("name") or ""),
                            arguments=parsed_args,
                        )
                    )

            if data.get("prompt_eval_count") is not None or data.get("eval_count") is not None:
                p_tokens = int(data.get("prompt_eval_count") or 0)
                c_tokens = int(data.get("eval_count") or 0)
                usage = TokenUsage(
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=p_tokens + c_tokens,
                )

        else:
            # OpenAI compatible API /v1/chat/completions
            choices = cast(list[dict[str, Any]], data.get("choices") or [])
            if choices:
                choice = choices[0]
                finish_reason = cast(str | None, choice.get("finish_reason"))
                delta = cast(dict[str, Any], choice.get("delta") or choice.get("message") or {})
                delta_text = str(delta.get("content") or "")

                raw_tcs = cast(list[dict[str, Any]] | None, delta.get("tool_calls"))
                if raw_tcs:
                    tool_calls = []
                    for idx, tc in enumerate(raw_tcs):
                        fn = cast(dict[str, Any], tc.get("function") or {})
                        raw_args_v1: Any = fn.get("arguments", {})
                        parsed_args_v1: dict[str, Any] = {}
                        if isinstance(raw_args_v1, dict):
                            parsed_args_v1 = cast(dict[str, Any], raw_args_v1)
                        elif isinstance(raw_args_v1, str) and raw_args_v1.strip():
                            try:
                                parsed_args_v1 = cast(dict[str, Any], json.loads(raw_args_v1))
                            except json.JSONDecodeError:
                                parsed_args_v1 = {}

                        tool_calls.append(
                            ToolCall(
                                id=str(tc.get("id") or f"call_{idx}"),
                                name=str(fn.get("name") or ""),
                                arguments=parsed_args_v1,
                            )
                        )

            raw_usage = cast(dict[str, int] | None, data.get("usage"))
            if raw_usage:
                usage = TokenUsage(
                    prompt_tokens=raw_usage.get("prompt_tokens", 0),
                    completion_tokens=raw_usage.get("completion_tokens", 0),
                    total_tokens=raw_usage.get("total_tokens", 0),
                )

        return CompletionChunk(
            delta_text=delta_text,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
        )

    def _handle_error_response(self, response: httpx.Response) -> None:
        status = response.status_code
        text = response.text
        if status in (401, 403):
            raise ProviderAuthenticationError(
                f"Authentication failed with status {status}", details={"response": text}
            )
        if status == 429:
            raise ProviderRateLimitError("Rate limit exceeded", details={"response": text})
        if status in (500, 502, 503, 504):
            raise TransientHttpError(f"Transient server error {status}: {text}")
        raise ProviderError(
            f"Provider request failed with status {status}", details={"response": text}
        )

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        """Generate complete completion (non-streaming or aggregated)."""
        payload = self._format_request_payload(request, stream=False)
        url = self._get_url()
        headers = self._get_headers()

        async def _make_request() -> CompletionChunk:
            client = self._client or httpx.AsyncClient(timeout=self.timeout)
            should_close = self._client is None
            try:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    self._handle_error_response(response)

                data = cast(dict[str, Any], response.json())
                return self._parse_chunk_data(data)
            except (httpx.TimeoutException, httpx.ConnectTimeout) as e:
                raise ProviderTimeoutError(f"Request timed out: {e}") from e
            except (httpx.NetworkError, httpx.ConnectError) as e:
                raise TransientHttpError(f"Network error: {e}") from e
            finally:
                if should_close:
                    await client.aclose()

        try:
            async for attempt in AsyncRetrying(
                reraise=True,
                stop=stop_after_attempt(self.max_retries),
                wait=wait_random_exponential(min=0.1, max=2.0),
                retry=retry_if_exception_type(TransientHttpError),
            ):
                with attempt:
                    return await _make_request()
        except TransientHttpError as err:
            raise ProviderError(f"Transient retries exhausted: {err}") from err

        raise ProviderError("Failed to complete request")

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
        """Stream completion chunks sequentially."""
        payload = self._format_request_payload(request, stream=True)
        url = self._get_url()
        headers = self._get_headers()

        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        should_close = self._client is None

        try:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    await response.aread()
                    self._handle_error_response(response)

                async for line in response.aiter_lines():
                    line_str = line.strip()
                    if not line_str:
                        continue
                    if line_str.startswith("data: "):
                        line_str = line_str[6:].strip()
                    if line_str == "[DONE]":
                        break

                    try:
                        data = cast(dict[str, Any], json.loads(line_str))
                        yield self._parse_chunk_data(data)
                    except json.JSONDecodeError:
                        logger.warning("Skipping invalid JSON stream chunk", line=line_str)
        except (httpx.TimeoutException, httpx.ConnectTimeout) as e:
            raise ProviderTimeoutError(f"Stream timed out: {e}") from e
        except (httpx.NetworkError, httpx.ConnectError) as e:
            raise ProviderError(f"Stream network error: {e}") from e
        finally:
            if should_close:
                await client.aclose()


__all__ = ["OllamaCloudProvider"]
