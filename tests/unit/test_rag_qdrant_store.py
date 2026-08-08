"""Unit tests for QdrantStore — mocked qdrant-client (no live Qdrant cluster
in CI; the adapter's own logic under test is tenant-id enforcement and
error wrapping, not Qdrant's server-side behavior)."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest
from nullain.errors import TenantIsolationError, VectorStoreError
from nullain.rag import QdrantStore, VectorRecord


def _new_magicmock(*_args: object, **_kwargs: object) -> MagicMock:
    """Typed stand-in for `qdrant_client.models`' constructor classes
    (FieldCondition, Filter, ...) — each just needs to be callable with
    arbitrary args and return something inert."""
    return MagicMock()


@pytest.fixture
def fake_qdrant_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Installs fake `qdrant_client` and `qdrant_client.models` modules so
    QdrantStore's lazy imports resolve without the real dependency."""
    client_module = ModuleType("qdrant_client")
    models_module = ModuleType("qdrant_client.models")

    async_client_cls = MagicMock()
    client_instance = MagicMock()
    client_instance.collection_exists = AsyncMock(return_value=False)
    client_instance.create_collection = AsyncMock()
    client_instance.upsert = AsyncMock()
    client_instance.delete = AsyncMock()
    client_instance.query_points = AsyncMock(return_value=MagicMock(points=[]))
    async_client_cls.return_value = client_instance
    client_module.AsyncQdrantClient = async_client_cls  # type: ignore[attr-defined]

    for name in (
        "Distance",
        "VectorParams",
        "PointStruct",
        "FieldCondition",
        "Filter",
        "MatchValue",
        "HasIdCondition",
    ):
        setattr(models_module, name, MagicMock(side_effect=_new_magicmock))
    models_module.Distance = MagicMock(COSINE="cosine")  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "qdrant_client", client_module)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models_module)
    return client_instance


def test_missing_qdrant_client_dependency_raises_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    with pytest.raises(ImportError, match=r"pip install nullain-sdk\[rag\]"):
        QdrantStore(url="http://localhost:6333", api_key="k")


@pytest.mark.asyncio
async def test_upsert_rejects_empty_tenant_id_before_any_network_call(
    fake_qdrant_client: MagicMock,
) -> None:
    """tenant_id validation must happen before the client is touched at
    all — a caller passing an empty tenant_id should never see a request
    reach Qdrant, even a malformed one."""
    store = QdrantStore(url="http://localhost:6333", api_key="k")
    with pytest.raises(TenantIsolationError):
        await store.upsert(
            [VectorRecord.model_construct(tenant_id="", vector=[0.1], text_preview="x", id="x")]
        )
    fake_qdrant_client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_scoped_search_rejects_empty_tenant_id_before_any_network_call(
    fake_qdrant_client: MagicMock,
) -> None:
    store = QdrantStore(url="http://localhost:6333", api_key="k")
    with pytest.raises(TenantIsolationError):
        await store.scoped_search("", [0.1, 0.2])
    fake_qdrant_client.query_points.assert_not_called()


@pytest.mark.asyncio
async def test_delete_rejects_empty_tenant_id_before_any_network_call(
    fake_qdrant_client: MagicMock,
) -> None:
    store = QdrantStore(url="http://localhost:6333", api_key="k")
    with pytest.raises(TenantIsolationError):
        await store.delete("", ["id-1"])
    fake_qdrant_client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_delete_with_empty_ids_is_a_noop(fake_qdrant_client: MagicMock) -> None:
    store = QdrantStore(url="http://localhost:6333", api_key="k")
    await store.delete("tenant-a", [])
    fake_qdrant_client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_wraps_client_failure_in_vector_store_error(
    fake_qdrant_client: MagicMock,
) -> None:
    fake_qdrant_client.upsert.side_effect = RuntimeError("connection refused")
    store = QdrantStore(url="http://localhost:6333", api_key="k")
    with pytest.raises(VectorStoreError, match="Qdrant upsert failed"):
        await store.upsert([VectorRecord(tenant_id="tenant-a", vector=[0.1], text_preview="x")])


@pytest.mark.asyncio
async def test_search_wraps_client_failure_in_vector_store_error(
    fake_qdrant_client: MagicMock,
) -> None:
    fake_qdrant_client.query_points.side_effect = RuntimeError("timeout")
    store = QdrantStore(url="http://localhost:6333", api_key="k")
    with pytest.raises(VectorStoreError, match="Qdrant search failed"):
        await store.scoped_search("tenant-a", [0.1, 0.2])


@pytest.mark.asyncio
async def test_ensure_collection_wraps_failure_in_vector_store_error(
    fake_qdrant_client: MagicMock,
) -> None:
    fake_qdrant_client.collection_exists.side_effect = RuntimeError("unreachable")
    store = QdrantStore(url="http://localhost:6333", api_key="k")
    with pytest.raises(VectorStoreError, match="Failed to ensure Qdrant collection"):
        await store.ensure_collection()


@pytest.mark.asyncio
async def test_ensure_collection_skips_create_when_already_exists(
    fake_qdrant_client: MagicMock,
) -> None:
    fake_qdrant_client.collection_exists.return_value = True
    store = QdrantStore(url="http://localhost:6333", api_key="k")
    await store.ensure_collection()
    fake_qdrant_client.create_collection.assert_not_called()
