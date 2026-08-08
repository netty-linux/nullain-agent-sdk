"""Unit tests for FastEmbedProvider — mocked fastembed (no ~2GB model
download in CI; the ONNX model itself is out of scope here, only the
adapter's contract: dimension resolution, async wrapping, error handling)."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from nullain.errors import EmbeddingError
from nullain.rag import EmbeddingProvider, FastEmbedProvider


class _FakeVector:
    """Mimics a numpy array's `.tolist()` — FastEmbed yields numpy arrays."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


@pytest.fixture
def fake_fastembed(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Installs a fake `fastembed` module so FastEmbedProvider's lazy
    `from fastembed import TextEmbedding` import resolves without the real
    (heavy) dependency installed."""
    fake_module = ModuleType("fastembed")
    text_embedding_cls = MagicMock()
    instance = MagicMock()
    instance.embed.return_value = [_FakeVector([0.1, 0.2, 0.3])]
    text_embedding_cls.return_value = instance
    fake_module.TextEmbedding = text_embedding_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fake_module)
    return instance


def test_fastembed_provider_satisfies_embedding_provider_protocol(
    fake_fastembed: MagicMock,
) -> None:
    provider = FastEmbedProvider(model="bge-m3")
    assert isinstance(provider, EmbeddingProvider)


def test_known_model_uses_declared_dimension_without_probing(fake_fastembed: MagicMock) -> None:
    """bge-m3 is in the known-model shortlist (1024 dims) — construction
    must not need to embed a probe string to learn the dimension."""
    provider = FastEmbedProvider(model="bge-m3")
    assert provider.dimension == 1024
    fake_fastembed.embed.assert_not_called()


def test_unknown_model_probes_dimension_from_a_sample_embed(fake_fastembed: MagicMock) -> None:
    fake_fastembed.embed.return_value = [_FakeVector([0.1] * 512)]
    provider = FastEmbedProvider(model="some/custom-model")
    assert provider.dimension == 512
    fake_fastembed.embed.assert_called_once()


@pytest.mark.asyncio
async def test_embed_returns_lists_of_floats(fake_fastembed: MagicMock) -> None:
    fake_fastembed.embed.return_value = [_FakeVector([0.1, 0.2, 0.3])]
    provider = FastEmbedProvider(model="bge-m3")
    result = await provider.embed(["hello"])
    assert result == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_embed_empty_list_returns_empty_without_calling_model(
    fake_fastembed: MagicMock,
) -> None:
    provider = FastEmbedProvider(model="bge-m3")
    fake_fastembed.embed.reset_mock()
    result = await provider.embed([])
    assert result == []
    fake_fastembed.embed.assert_not_called()


@pytest.mark.asyncio
async def test_embed_wraps_backend_failure_in_embedding_error(fake_fastembed: MagicMock) -> None:
    fake_fastembed.embed.side_effect = RuntimeError("ONNX runtime crashed")
    provider = FastEmbedProvider(model="bge-m3")
    with pytest.raises(EmbeddingError, match="FastEmbed embedding failed"):
        await provider.embed(["hello"])


def test_missing_fastembed_dependency_raises_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "fastembed", None)  # simulates "not installed"
    with pytest.raises(ImportError, match=r"pip install nullain-sdk\[rag\]"):
        FastEmbedProvider(model="bge-m3")
