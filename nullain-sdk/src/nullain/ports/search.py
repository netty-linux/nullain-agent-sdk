"""Nullain Agent SDK — Search Provider Port (hexagonal boundary).

`SearchProvider` is the contract the core owns for search capability
(PLAN.md section 4: "Como tudo se interliga" — the Facade depends only on
this `Protocol`, never on a concrete adapter). Two adapters implement it:
`WebSearchProvider` (`nullain_tools.web_search`, the always-available
default — SearXNG/DuckDuckGo web search, no local index) and
`RustSearchAdapter` (below — a local BM25 index backed by the
`nullain-search` wheel, PLAN.md Fase 1, `nullain-sdk-search`).

Three orthogonal operations, not a linear pipeline — an adapter is free to
make any of them a documented no-op when it doesn't apply (e.g. a web-only
adapter has nothing to `index`):

- `index`: ingest content into the adapter's searchable store, if it has
  one. A pure web-search adapter treats this as a no-op (nothing to
  persist); a local-index adapter (tantivy) performs real ingestion.
- `query`: search for content and return ranked results.
- `fetch`: retrieve the full content of one specific, already-known
  location (mirrors `web_fetch`'s contract: given a URL, return its
  content — not a search).
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any, Protocol, runtime_checkable

from nullain.errors import SearchError


@runtime_checkable
class SearchProvider(Protocol):
    """Port: a search adapter (web search, or a local content index)."""

    async def index(self, content: str, *, source: str) -> None:
        """Ingest `content` into the adapter's searchable store, if any.

        Args:
            content: Text to make searchable.
            source: Stable identifier for `content` (e.g. a file path or
                URL) — how a later `query`/`fetch` result refers back to
                this item.

        A no-op for adapters with no index of their own (e.g. a
        web-search-only adapter) — such adapters must document that
        explicitly rather than raising, since indexing is optional per
        the port's contract, not a required capability.
        """
        ...

    async def query(self, text: str, *, limit: int = 5) -> str:
        """Search for `text` and return up to `limit` formatted results.

        Args:
            text: The search query.
            limit: Maximum number of results to include.

        Returns:
            A human-readable string describing the results (title, source,
            snippet per result), or a message stating no results were
            found. Never raises for a zero-result search — only for a
            malformed request (e.g. an empty query).
        """
        ...

    async def fetch(self, source: str) -> str:
        """Retrieve the full content at `source`.

        Args:
            source: A location identifier previously seen from `index` or
                `query` (e.g. a URL).

        Returns:
            The content at `source` as plain text.
        """
        ...


class RustSearchAdapter:
    """`SearchProvider` adapter backed by the `nullain-search` wheel
    (PLAN.md Fase 1, `nullain-sdk-search` — tantivy BM25 core, PyO3
    bindings). `index`/`query` delegate to the Rust index; `fetch` delegates
    to an injected fallback `SearchProvider` instead of the Rust index's own
    `fetch` — the Rust side only retrieves content IT indexed, which is a
    different contract than this port's `fetch` (retrieve a URL's content,
    decided in PLAN.md Fase 0), so mixing them would silently break for any
    `source` that was never indexed locally.

    `nullain_search`'s `SearchIndex` is synchronous/blocking; every call is
    wrapped in `asyncio.to_thread` so it never blocks the event loop.

    Args:
        web_fallback: A `SearchProvider` used for `fetch` (typically a
            `WebSearchProvider` — this adapter never fetches URLs itself).
        index_dir: If set, indexes every readable text file under this path
            once, at construction time, via `SearchIndex.index_directory`.

    Raises:
        ImportError: the `nullain-search` wheel is not installed
            (`pip install nullain-sdk[search-rust]`).
    """

    def __init__(self, web_fallback: SearchProvider, *, index_dir: str | None = None) -> None:
        # importlib (not a static import) so the optional `search-rust`
        # extra is resolved at runtime and does not trip static analysis
        # when absent — same pattern as FastEmbedProvider (rag/embedding.py).
        try:
            nullain_search_mod = importlib.import_module("nullain_search")
        except ImportError as exc:
            raise ImportError(
                "RustSearchAdapter requires the 'search-rust' extra: "
                "pip install nullain-sdk[search-rust]"
            ) from exc
        search_index_cls: Any = nullain_search_mod.SearchIndex

        self._web_fallback = web_fallback
        self._index: Any = search_index_cls()
        if index_dir is not None:
            self._index.index_directory(index_dir)

    async def index(self, content: str, *, source: str) -> None:
        """Ingest `content` into the local BM25 index."""
        try:
            await asyncio.to_thread(self._index.index, content, source=source)
        except (ValueError, OSError, RuntimeError) as exc:
            raise SearchError(f"RustSearchAdapter.index failed for {source!r}: {exc}") from exc

    async def query(self, text: str, *, limit: int = 5) -> str:
        """BM25 search over the local index."""
        try:
            hits: list[Any] = await asyncio.to_thread(self._index.query, text, limit)
        except (ValueError, OSError, RuntimeError) as exc:
            raise SearchError(f"RustSearchAdapter.query failed: {exc}") from exc

        if not hits:
            return f"No results found for '{text}'."
        formatted = [
            f"{i}. {hit.source}\n   {hit.snippet} (score: {hit.score:.2f})"
            for i, hit in enumerate(hits, start=1)
        ]
        return "\n\n".join(formatted)

    async def fetch(self, source: str) -> str:
        """Delegate to the injected fallback `SearchProvider` — the local
        Rust index is not consulted, since its `fetch` only returns content
        it indexed itself (see class docstring)."""
        return await self._web_fallback.fetch(source)


__all__ = ["RustSearchAdapter", "SearchProvider"]
