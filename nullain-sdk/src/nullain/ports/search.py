"""Nullain Agent SDK — Search Provider Port (hexagonal boundary).

`SearchProvider` is the contract the core owns for search capability
(PLAN.md section 4: "Como tudo se interliga" — the Facade depends only on
this `Protocol`, never on a concrete adapter). Two adapters are expected to
implement it over time: `WebSearchProvider` (`nullain_tools.web_search`,
the always-available default — SearXNG/DuckDuckGo web search, no local
index) and a future Rust-backed local-index adapter (PLAN.md Fase 1,
`nullain-sdk-search`, tantivy-based).

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

from typing import Protocol, runtime_checkable


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


__all__ = ["SearchProvider"]
