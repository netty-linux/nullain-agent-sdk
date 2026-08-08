"""Nullain Agent SDK — Embedding Provider Port and FastEmbed Adapter.

`EmbeddingProvider` mirrors the shape of `nullain.llm.LLMProvider`: a small
Protocol port so the RAG pipeline (Cuckoo Filter pre-filter, RAG Tree,
`VectorStore`) never depends on a concrete embedding backend.

`FastEmbedProvider` is the default adapter (docs/FUSION_PLAN.md ADR-1):
FastEmbed runs the embedding model in-process via ONNX Runtime — no
PyTorch, no separate service, no Ollama (bge-m3 is unavailable on Ollama
Cloud, only self-hosted). `fastembed` is an optional dependency
(`pip install nullain-sdk[rag]`); importing this module without it installed
raises a clear `ImportError` at adapter-construction time, not at
package-import time — the port itself has zero dependency on the package.
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any, Protocol, runtime_checkable

from nullain.errors import EmbeddingError


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Abstract protocol for text-embedding backends (Ports & Adapters)."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into dense vectors.

        Args:
            texts: Non-empty list of texts to embed.

        Returns:
            One vector per input text, same order, same length.
        """
        ...

    @property
    def dimension(self) -> int:
        """The dimensionality of vectors this provider produces."""
        ...


#: Known FastEmbed models by size/quality tradeoff (docs/FUSION_PLAN.md).
#: Values are (model_name, dimension) — dimension is fixed per model, needed
#: up front to size the vector-store collection without loading the model.
_FASTEMBED_MODELS: dict[str, tuple[str, int]] = {
    "bge-m3": ("BAAI/bge-m3", 1024),
    "embeddinggemma": ("google/embeddinggemma-300m", 768),
    "e5-small": ("intfloat/multilingual-e5-small", 384),
}


class FastEmbedProvider:
    """In-process embedding via FastEmbed (ONNX Runtime, CPU).

    Args:
        model: Either a key into `_FASTEMBED_MODELS` (`"bge-m3"` — the
            default, `"embeddinggemma"`, `"e5-small"`) or a raw FastEmbed
            model name (e.g. `"BAAI/bge-m3"`) for anything not in that
            shortlist. Model weights download and cache on first use.

    Raises:
        ImportError: `fastembed` is not installed
            (`pip install nullain-sdk[rag]`).
    """

    def __init__(self, model: str = "bge-m3") -> None:
        # importlib (not a static import) so the optional `fastembed` extra
        # is resolved at runtime and does not trip static analysis when absent.
        try:
            fastembed_mod = importlib.import_module("fastembed")
        except ImportError as exc:
            raise ImportError(
                "FastEmbedProvider requires the 'rag' extra: pip install nullain-sdk[rag]"
            ) from exc
        text_embedding_cls: Any = fastembed_mod.TextEmbedding

        resolved_name, dim = _FASTEMBED_MODELS.get(model, (model, 0))
        self._model_name = resolved_name
        self._model: Any = text_embedding_cls(model_name=resolved_name)
        self._dimension = dim or self._probe_dimension()

    def _probe_dimension(self) -> int:
        """Determine vector size for a model name not in the known shortlist
        by embedding a throwaway string once, at construction time."""
        sample = list(self._model.embed(["dimension probe"]))
        return len(sample[0].tolist())

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts. FastEmbed's `.embed()` is synchronous/CPU-bound, so
        this runs it in a worker thread to avoid blocking the event loop."""
        if not texts:
            return []
        try:
            return await asyncio.to_thread(self._embed_sync, texts)
        except Exception as exc:
            raise EmbeddingError(f"FastEmbed embedding failed: {exc}") from exc

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]


__all__ = ["EmbeddingProvider", "FastEmbedProvider"]
