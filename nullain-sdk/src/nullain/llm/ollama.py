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
from nullain.llm.response_models import OllamaNativeResponse, OpenAIResponse
from nullain.llm.types import (
    CompletionChunk,
    CompletionRequest,
    TokenUsage,
    ToolCall,
)
from nullain.telemetry import get_logger
from nullain.telemetry import span as telemetry_span

logger: BoundLogger = get_logger("nullain.llm.ollama")


class TransientHttpError(Exception):
    """Internal exception for retriable HTTP errors."""

    pass


class TransientTimeoutError(Exception):
    """Internal exception for retriable request timeouts.

    Kept distinct from :class:`TransientHttpError` so exhausted retries can
    still surface the right domain exception (:class:`ProviderTimeoutError`
    vs. the generic :class:`ProviderError`) while both are retried the same
    way — a single slow response is exactly as recoverable as a single
    network blip and should not abort the call while retry budget remains.
    """

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
                response_native.prompt_eval_count is not None
                or response_native.eval_count is not None
            )
            if has_eval:
                p_tokens = response_native.prompt_eval_count or 0
                c_tokens = response_native.eval_count or 0
                usage = TokenUsage(
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=p_tokens + c_tokens,
                )
        else:
            # OpenAI compatible API /v1/chat/completions
            response_openai = OpenAIResponse.model_validate(data)

            if response_openai.choices:
                choice = response_openai.choices[0]
                finish_reason = choice.finish_reason
                delta = choice.delta or choice.message
                if delta:
                    delta_text = delta.content or ""
                    if delta.tool_calls:
                        tool_calls = []
                        for idx, tc in enumerate(delta.tool_calls):
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

            if response_openai.usage:
                usage = TokenUsage(
                    prompt_tokens=response_openai.usage.prompt_tokens,
                    completion_tokens=response_openai.usage.completion_tokens,
                    total_tokens=response_openai.usage.total_tokens,
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
        # A structured log line per non-200 response — previously this
        # branch only raised, with no observable trace of which status
        # code came back or why. Truncated so a large HTML/JSON error body
        # doesn't blow out a log line.
        logger.warning("llm_http_error_response", status_code=status, body=text[:500])
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
        attempt_count = 0

        async def _make_request() -> CompletionChunk:
            nonlocal attempt_count
            attempt_count += 1
            client = self._client or httpx.AsyncClient(timeout=self.timeout)
            should_close = self._client is None
            # A span per HTTP attempt (distinct from the loop-level
            # llm_request span, which covers the whole call including all
            # retries as one unit): without this, a failure buried inside
            # AsyncRetrying was invisible in traces — only the final
            # exhausted-retries error surfaced, with no way to tell from
            # telemetry alone whether it was one slow request or three
            # retried 500s before it gave up.
            with telemetry_span(
                "llm_http_attempt",
                **{"llm.model": request.model, "llm.attempt": attempt_count},
            ) as attempt_span:
                try:
                    response = await client.post(url, json=payload, headers=headers)
                    attempt_span.set_attribute("http.status_code", response.status_code)
                    if response.status_code != 200:
                        self._handle_error_response(response)

                    data = cast(dict[str, Any], response.json())
                    return self._parse_chunk_data(data)
                except (httpx.TimeoutException, httpx.ConnectTimeout) as e:
                    logger.warning(
                        "llm_http_attempt_timeout", model=request.model, attempt=attempt_count
                    )
                    raise TransientTimeoutError(f"Request timed out: {e}") from e
                except (httpx.NetworkError, httpx.ConnectError) as e:
                    logger.warning(
                        "llm_http_attempt_network_error",
                        model=request.model,
                        attempt=attempt_count,
                        error=str(e),
                    )
                    raise TransientHttpError(f"Network error: {e}") from e
                finally:
                    if should_close:
                        await client.aclose()

        try:
            async for attempt in AsyncRetrying(
                reraise=True,
                stop=stop_after_attempt(self.max_retries),
                wait=wait_random_exponential(min=0.1, max=2.0),
                # Both a slow response (TransientTimeoutError) and a network
                # error (TransientHttpError) are worth retrying — a single
                # timed-out request should not abort the call when
                # max_retries allows another attempt. Retrying on
                # ProviderTimeoutError directly would not work here since
                # that is raised only after retries are exhausted, below.
                retry=retry_if_exception_type((TransientHttpError, TransientTimeoutError)),
            ):
                with attempt:
                    return await _make_request()
        except TransientTimeoutError as err:
            logger.error(
                "llm_request_retries_exhausted",
                model=request.model,
                attempts=attempt_count,
                reason="timeout",
            )
            raise ProviderTimeoutError(f"Retries exhausted: {err}") from err
        except TransientHttpError as err:
            logger.error(
                "llm_request_retries_exhausted",
                model=request.model,
                attempts=attempt_count,
                reason="network_error",
            )
            raise ProviderError(f"Transient retries exhausted: {err}") from err

        # Unreachable: AsyncRetrying(reraise=True) either returns from the loop
        # body above or raises. This guard only exists to satisfy the type
        # checker that all code paths return a CompletionChunk.
        raise ProviderError("generate: retries exhausted without producing a result")

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
            logger.error("llm_stream_timeout", model=request.model)
            raise ProviderTimeoutError(f"Stream timed out: {e}") from e
        except (httpx.NetworkError, httpx.ConnectError) as e:
            logger.error("llm_stream_network_error", model=request.model, error=str(e))
            raise ProviderError(f"Stream network error: {e}") from e
        finally:
            if should_close:
                await client.aclose()


__all__ = ["OllamaCloudProvider"]
