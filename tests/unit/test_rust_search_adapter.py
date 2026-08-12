"""Unit tests for RustSearchAdapter — mocked `nullain_search` module (no
Rust wheel needed in CI; the tantivy index itself is out of scope here,
only the adapter's contract: to_thread wrapping, error translation, result
formatting, and fetch delegation to the injected fallback).

`tests/unit/test_search_provider_contract.py` covers RustSearchAdapter
against the real `nullain-search` wheel when it's installed (skipped
otherwise) — this file covers the adapter's own logic independent of
whether the wheel is present, mirroring `test_rag_embedding.py`'s
fake-module pattern for the same kind of optional-dependency adapter.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from nullain.errors import SearchError
from nullain.ports.search import RustSearchAdapter, SearchProvider


class _FakeSearchHit:
    def __init__(self, source: str, snippet: str, score: float) -> None:
        self.source = source
        self.snippet = snippet
        self.score = score


class _FakeWebFallback:
    """A minimal `SearchProvider` used only to verify `fetch` delegation —
    `index`/`query` are never expected to be called on it by RustSearchAdapter."""

    def __init__(self) -> None:
        self.fetch_calls: list[str] = []

    async def index(self, content: str, *, source: str) -> None:
        raise AssertionError("RustSearchAdapter must not delegate index() to the fallback")

    async def query(self, text: str, *, limit: int = 5) -> str:
        raise AssertionError("RustSearchAdapter must not delegate query() to the fallback")

    async def fetch(self, source: str) -> str:
        self.fetch_calls.append(source)
        return f"fetched:{source}"


@pytest.fixture
def fake_nullain_search(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Installs a fake `nullain_search` module so RustSearchAdapter's lazy
    `importlib.import_module("nullain_search")` resolves without the real
    (compiled) wheel installed."""
    fake_module = ModuleType("nullain_search")
    search_index_cls = MagicMock()
    instance = MagicMock()
    search_index_cls.return_value = instance
    fake_module.SearchIndex = search_index_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nullain_search", fake_module)
    return instance


def test_satisfies_search_provider_protocol(fake_nullain_search: MagicMock) -> None:
    adapter = RustSearchAdapter(web_fallback=_FakeWebFallback())
    assert isinstance(adapter, SearchProvider)


def test_index_dir_indexes_directory_once_at_construction(
    fake_nullain_search: MagicMock,
) -> None:
    RustSearchAdapter(web_fallback=_FakeWebFallback(), index_dir="/some/workspace")
    fake_nullain_search.index_directory.assert_called_once_with("/some/workspace")


def test_no_index_dir_never_calls_index_directory(fake_nullain_search: MagicMock) -> None:
    RustSearchAdapter(web_fallback=_FakeWebFallback())
    fake_nullain_search.index_directory.assert_not_called()


async def test_index_delegates_to_rust_index(fake_nullain_search: MagicMock) -> None:
    adapter = RustSearchAdapter(web_fallback=_FakeWebFallback())
    await adapter.index("some text", source="a.txt")
    fake_nullain_search.index.assert_called_once_with("some text", source="a.txt")


@pytest.mark.parametrize("exc_cls", [ValueError, OSError, RuntimeError])
async def test_index_translates_pyo3_errors_to_search_error(
    fake_nullain_search: MagicMock, exc_cls: type[Exception]
) -> None:
    fake_nullain_search.index.side_effect = exc_cls("boom")
    adapter = RustSearchAdapter(web_fallback=_FakeWebFallback())
    with pytest.raises(SearchError, match=r"RustSearchAdapter\.index failed"):
        await adapter.index("some text", source="a.txt")


async def test_query_formats_hits_into_text(fake_nullain_search: MagicMock) -> None:
    fake_nullain_search.query.return_value = [
        _FakeSearchHit(source="a.txt", snippet="the quick brown fox", score=1.5),
    ]
    adapter = RustSearchAdapter(web_fallback=_FakeWebFallback())
    result = await adapter.query("fox", limit=5)
    assert isinstance(result, str)
    assert "a.txt" in result
    assert "the quick brown fox" in result
    fake_nullain_search.query.assert_called_once_with("fox", 5)


async def test_query_reports_no_results_without_raising(fake_nullain_search: MagicMock) -> None:
    fake_nullain_search.query.return_value = []
    adapter = RustSearchAdapter(web_fallback=_FakeWebFallback())
    result = await adapter.query("nothing matches")
    assert "No results found" in result


@pytest.mark.parametrize("exc_cls", [ValueError, OSError, RuntimeError])
async def test_query_translates_pyo3_errors_to_search_error(
    fake_nullain_search: MagicMock, exc_cls: type[Exception]
) -> None:
    fake_nullain_search.query.side_effect = exc_cls("boom")
    adapter = RustSearchAdapter(web_fallback=_FakeWebFallback())
    with pytest.raises(SearchError, match=r"RustSearchAdapter\.query failed"):
        await adapter.query("fox")


async def test_fetch_delegates_to_web_fallback_not_rust_index(
    fake_nullain_search: MagicMock,
) -> None:
    fallback = _FakeWebFallback()
    adapter = RustSearchAdapter(web_fallback=fallback)
    result = await adapter.fetch("https://example.com/page")
    assert result == "fetched:https://example.com/page"
    assert fallback.fetch_calls == ["https://example.com/page"]
    fake_nullain_search.fetch.assert_not_called()


def test_missing_nullain_search_dependency_raises_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "nullain_search", None)  # simulates "not installed"
    with pytest.raises(ImportError, match=r"pip install nullain-sdk\[search-rust\]"):
        RustSearchAdapter(web_fallback=_FakeWebFallback())
