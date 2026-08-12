"""Nullain Agent SDK — Vision Provider Port (hexagonal boundary).

`VisionProvider` is the contract the core owns for visual understanding
(PLAN.md section 4: the Facade depends only on this `Protocol`, never on a
concrete adapter). `ModelRouterVisionProvider` (below) is the adapter that
ships in this SDK: it routes an image + prompt through the core's own
`ModelRouter` (model selection) and `LLMProvider` (request execution) as a
plain multimodal chat completion — no OCR/CV dependency, no client of its
own, so it carries none of the install weight the port's docstring
originally assumed.

That original assumption doesn't generalize, though: a *local* vision
adapter (OCR, CV, or an on-device VLM — e.g. the `nullain-agent` repo's
`vision/` package, PLAN.md's still-undecided extraction candidate) would
bring real OCR/CV dependency weight, and belongs outside this SDK's base
install as an optional extra for exactly that reason. This port stays
adapter-agnostic either way — `ModelRouterVisionProvider` merely happens to
be the first, lightest-weight implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nullain.errors import VisionError
from nullain.llm.provider import LLMProvider
from nullain.llm.types import ChatMessage, CompletionRequest, ImagePart, TextPart
from nullain.router.router import ModelRouter


@runtime_checkable
class VisionProvider(Protocol):
    """Port: a visual-understanding adapter (image description, OCR, screenshots)."""

    async def describe_image(self, image: bytes, *, mime_type: str, hint: str | None = None) -> str:
        """Return a natural-language description of `image`.

        Args:
            image: Raw image bytes.
            mime_type: The image's MIME type (e.g. ``"image/png"``).
            hint: Optional free-text guidance to prioritize what the
                description covers (e.g. the user's own question about the
                image). Passed through to the underlying model as-is — an
                adapter does not sanitize it, so a caller feeding
                user-supplied text must sanitize it first (the SDK has no
                way to know what "untrusted" means for a given deployment).

        Returns:
            A plain-text description of the image's visual content.
        """
        ...

    async def ocr(self, image: bytes, *, mime_type: str, hint: str | None = None) -> str:
        """Return the text transcribed from `image`.

        Args:
            image: Raw image bytes.
            mime_type: The image's MIME type (e.g. ``"image/png"``).
            hint: Optional free-text guidance (see `describe_image`). Not
                sanitized by the adapter.

        Returns:
            The image's text content, verbatim, or an empty string when
            the image contains no recognizable text.
        """
        ...

    async def analyze_screenshot(
        self, image: bytes, *, mime_type: str, hint: str | None = None
    ) -> str:
        """Return a UI-oriented analysis of a screenshot in `image`.

        Args:
            image: Raw screenshot bytes.
            mime_type: The image's MIME type (e.g. ``"image/png"``).
            hint: Optional free-text guidance (see `describe_image`). Not
                sanitized by the adapter.

        Returns:
            A plain-text description of the screenshot's UI state —
            visible elements, layout, and any actionable affordances —
            distinct from `describe_image`'s general scene description.
        """
        ...


_DESCRIBE_PROMPT = "Describe this image's visual content in a few plain sentences."
_OCR_PROMPT = (
    "Transcribe all text visible in this image, verbatim, in reading order. "
    "Return only the transcribed text, or an empty string if there is none."
)
_ANALYZE_SCREENSHOT_PROMPT = (
    "This is a UI screenshot. Describe its visible elements, layout, and any "
    "actionable affordances (buttons, inputs, links, notifications)."
)


class ModelRouterVisionProvider:
    """`VisionProvider` adapter backed by an injected `ModelRouter` + `LLMProvider`.

    `router.select_model(tier)` picks the model name; `llm.generate()` (an
    `LLMProvider` already configured for that model's endpoint) executes the
    request as a plain multimodal chat completion — the image travels as an
    `ImagePart` content block (`nullain.llm.types`), the same content-blocks
    shape `OpenAICompatibleProvider` already serializes for any OpenAI-style
    endpoint. `describe_image`, `ocr`, and `analyze_screenshot` share one
    code path (`_run`) and differ only in the prompt sent alongside the
    image.

    Neither `router` nor `llm` is instantiated here — both must already be
    configured (API keys, base URLs, tier→model mapping) by the caller.

    Args:
        router: Picks a model name for the vision `tier`.
        llm: Executes the completion request against that model's endpoint.
        tier: The `ModelRouter` tier to select a model from. Defaults to
            ``"vision"`` — the caller's `RouterConfig` must define it (or a
            fallback chain reaching a multimodal-capable model).

    Any failure surfaced by `llm.generate()` — including a model that
    rejects multimodal content outright — is translated to `VisionError`
    rather than propagating the underlying provider exception, so callers
    see one error type regardless of which model actually served the
    request. There is no separate capability pre-check: the underlying
    `LLMProvider`/`ModelRouter` layer has no way to introspect "does this
    model accept images" ahead of time, so the request is simply attempted
    and any rejection becomes a `VisionError`.
    """

    def __init__(self, router: ModelRouter, llm: LLMProvider, *, tier: str = "vision") -> None:
        self._router = router
        self._llm = llm
        self._tier = tier

    async def _run(self, prompt: str, image: bytes, mime_type: str, hint: str | None) -> str:
        if hint and hint.strip():
            prompt = (
                f"{prompt}\n\nUser context (use it to prioritize relevant details):\n{hint.strip()}"
            )
        model = self._router.select_model(self._tier)
        request = CompletionRequest(
            model=model,
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        TextPart(text=prompt),
                        ImagePart(data=image, mime_type=mime_type),
                    ],
                )
            ],
            stream=False,
        )
        try:
            chunk = await self._llm.generate(request)
        except Exception as err:
            raise VisionError(
                f"Vision request failed for model '{model}': {err}",
                details={"model": model, "mime_type": mime_type},
            ) from err
        return chunk.delta_text

    async def describe_image(self, image: bytes, *, mime_type: str, hint: str | None = None) -> str:
        return await self._run(_DESCRIBE_PROMPT, image, mime_type, hint)

    async def ocr(self, image: bytes, *, mime_type: str, hint: str | None = None) -> str:
        return await self._run(_OCR_PROMPT, image, mime_type, hint)

    async def analyze_screenshot(
        self, image: bytes, *, mime_type: str, hint: str | None = None
    ) -> str:
        return await self._run(_ANALYZE_SCREENSHOT_PROMPT, image, mime_type, hint)


__all__ = ["ModelRouterVisionProvider", "VisionProvider"]
