"""Nullain Agent SDK — Vision Provider Port (hexagonal boundary).

`VisionProvider` is the contract the core owns for visual understanding
(PLAN.md section 4: the Facade depends only on this `Protocol`, never on a
concrete adapter). No adapter ships in this SDK — implementations live in
the separate `nullain-vision` package (PLAN.md Fase 2), installed as an
optional extra so the base SDK install stays free of OCR/VLM dependency
weight (`docs/api-stability.md` applies once an adapter is public; this
port itself does not change that surface).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VisionProvider(Protocol):
    """Port: a visual-understanding adapter (image description, OCR, screenshots)."""

    async def describe_image(self, image: bytes, *, mime_type: str) -> str:
        """Return a natural-language description of `image`.

        Args:
            image: Raw image bytes.
            mime_type: The image's MIME type (e.g. ``"image/png"``).

        Returns:
            A plain-text description of the image's visual content.
        """
        ...

    async def ocr(self, image: bytes, *, mime_type: str) -> str:
        """Return the text transcribed from `image`.

        Args:
            image: Raw image bytes.
            mime_type: The image's MIME type (e.g. ``"image/png"``).

        Returns:
            The image's text content, verbatim, or an empty string when
            the image contains no recognizable text.
        """
        ...

    async def analyze_screenshot(self, image: bytes, *, mime_type: str) -> str:
        """Return a UI-oriented analysis of a screenshot in `image`.

        Args:
            image: Raw screenshot bytes.
            mime_type: The image's MIME type (e.g. ``"image/png"``).

        Returns:
            A plain-text description of the screenshot's UI state —
            visible elements, layout, and any actionable affordances —
            distinct from `describe_image`'s general scene description.
        """
        ...


__all__ = ["VisionProvider"]
