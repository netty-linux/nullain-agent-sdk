"""Nullain Agent SDK — VectorStore Port and Qdrant Adapter.

`VectorStore` mirrors `LLMProvider`/`EmbeddingProvider`: a small Protocol
port isolating the RAG pipeline from a concrete vector database. The one
rule this whole module exists to enforce (docs/FUSION_PLAN.md's multi-tenant
isolation requirement, called out explicitly by the project's senior AI
advisor as the top risk of vector search — approximate nearest-neighbor
search can otherwise surface another tenant's data): **every read goes
through `scoped_search`, which always applies a `tenant_id` payload filter.
There is no unfiltered search method anywhere in this module.** A caller
cannot "forget" the filter because there is no method that omits it.

`QdrantStore` is the concrete adapter (docs/FUSION_PLAN.md §4: single
collection `nullain_rag`, Free Tier). `qdrant-client` is an optional
dependency (`pip install nullain-sdk[rag]`); the port itself has zero
dependency on it.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

from nullain.errors import TenantIsolationError, VectorStoreError
from nullain.rag.metadata import VectorRecord


@runtime_checkable
class VectorStore(Protocol):
    """Abstract protocol for tenant-scoped vector storage and search."""

    async def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or update records. Every record's `tenant_id` must be set
        (enforced by `VectorRecord` itself being a required field, but
        adapters should not assume — validate defensively at the boundary
        since records may come from untrusted deserialization)."""
        ...

    async def scoped_search(
        self,
        tenant_id: str,
        query_vector: list[float],
        *,
        limit: int = 10,
        cluster_id: str | None = None,
        level: int | None = None,
    ) -> list[VectorRecord]:
        """Search for the `limit` records closest to `query_vector`,
        restricted to `tenant_id` — this is the ONLY search entry point;
        there is no method that searches without a tenant filter.

        Args:
            tenant_id: Required. Every result is guaranteed to belong to
                this tenant.
            query_vector: The embedding to search against.
            limit: Maximum results to return.
            cluster_id: Optional — restrict to one RAG Tree node's leaves
                (the caller navigated `RagTreeIndex` first and wants to
                search only within the chosen cluster).
            level: Optional — restrict to one RAG Tree depth.

        Returns:
            Matching records, most similar first. Never includes another
            tenant's records, by construction.
        """
        ...

    async def delete(self, tenant_id: str, record_ids: list[str]) -> None:
        """Delete records by id, scoped to `tenant_id` — an id belonging to
        a different tenant is silently ignored, not an error (the caller
        has no way to have legitimately learned another tenant's record id
        through this API, so treating it as "not found" rather than
        raising avoids leaking existence information cross-tenant)."""
        ...


def _require_tenant_id(tenant_id: str) -> None:
    if not tenant_id or not tenant_id.strip():
        raise TenantIsolationError(
            "tenant_id is required and cannot be empty — every vector "
            "operation must be scoped to a tenant (docs/FUSION_PLAN.md)"
        )


class QdrantStore:
    """Qdrant-backed `VectorStore` — single collection, payload-filtered
    reads only.

    Args:
        url: Qdrant endpoint (e.g. a Qdrant Cloud Free Tier cluster URL).
        api_key: Qdrant API key.
        collection: Collection name — `nullain_rag` by default
            (docs/FUSION_PLAN.md: one collection with a rich payload, since
            the Free Tier caps collection count).
        vector_size: Dimensionality of stored vectors — must match the
            `EmbeddingProvider` in use. Required at collection-creation time
            (`ensure_collection`); mismatched dimensions fail loudly on
            upsert rather than silently corrupting the index.

    Raises:
        ImportError: `qdrant-client` is not installed
            (`pip install nullain-sdk[rag]`).
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        *,
        collection: str = "nullain_rag",
        vector_size: int = 1024,
    ) -> None:
        # importlib (not a static import) so the optional `qdrant-client`
        # extra is resolved at runtime and does not trip static analysis
        # when absent — same pattern as nullain.plugins.signing's optional
        # `cryptography` import.
        try:
            client_mod = importlib.import_module("qdrant_client")
        except ImportError as exc:
            raise ImportError(
                "QdrantStore requires the 'rag' extra: pip install nullain-sdk[rag]"
            ) from exc
        async_client_cls: Any = client_mod.AsyncQdrantClient

        self._client: Any = async_client_cls(url=url, api_key=api_key)
        self._collection = collection
        self._vector_size = vector_size

    async def ensure_collection(self) -> None:
        """Create the collection if it doesn't already exist. Idempotent —
        safe to call on every startup."""
        models_mod: Any = importlib.import_module("qdrant_client.models")

        try:
            exists = await self._client.collection_exists(self._collection)
            if not exists:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=models_mod.VectorParams(
                        size=self._vector_size, distance=models_mod.Distance.COSINE
                    ),
                )
        except Exception as exc:
            raise VectorStoreError(f"Failed to ensure Qdrant collection: {exc}") from exc

    async def upsert(self, records: list[VectorRecord]) -> None:
        models_mod: Any = importlib.import_module("qdrant_client.models")

        for record in records:
            _require_tenant_id(record.tenant_id)

        points = [
            models_mod.PointStruct(
                id=record.id,
                vector=record.vector,
                payload={
                    "tenant_id": record.tenant_id,
                    "text_preview": record.text_preview,
                    "source": record.source,
                    "session_id": record.session_id,
                    "level": record.level,
                    "cluster_id": record.cluster_id,
                    "parent_id": record.parent_id,
                    "created_at": record.created_at,
                    **record.extra,
                },
            )
            for record in records
        ]
        try:
            await self._client.upsert(collection_name=self._collection, points=points)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant upsert failed: {exc}") from exc

    async def scoped_search(
        self,
        tenant_id: str,
        query_vector: list[float],
        *,
        limit: int = 10,
        cluster_id: str | None = None,
        level: int | None = None,
    ) -> list[VectorRecord]:
        _require_tenant_id(tenant_id)
        models_mod: Any = importlib.import_module("qdrant_client.models")

        must: list[Any] = [
            models_mod.FieldCondition(key="tenant_id", match=models_mod.MatchValue(value=tenant_id))
        ]
        if cluster_id is not None:
            must.append(
                models_mod.FieldCondition(
                    key="cluster_id", match=models_mod.MatchValue(value=cluster_id)
                )
            )
        if level is not None:
            must.append(
                models_mod.FieldCondition(key="level", match=models_mod.MatchValue(value=level))
            )

        try:
            result = await self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                query_filter=models_mod.Filter(must=must),
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc

        return [_point_to_record(point) for point in result.points]

    async def delete(self, tenant_id: str, record_ids: list[str]) -> None:
        _require_tenant_id(tenant_id)
        if not record_ids:
            return
        models_mod: Any = importlib.import_module("qdrant_client.models")

        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=models_mod.Filter(
                    must=[
                        models_mod.FieldCondition(
                            key="tenant_id", match=models_mod.MatchValue(value=tenant_id)
                        ),
                        models_mod.HasIdCondition(has_id=record_ids),
                    ]
                ),
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant delete failed: {exc}") from exc


def _point_to_record(point: Any) -> VectorRecord:
    payload = dict(point.payload or {})
    known = {
        "tenant_id",
        "text_preview",
        "source",
        "session_id",
        "level",
        "cluster_id",
        "parent_id",
        "created_at",
    }
    return VectorRecord(
        id=str(point.id),
        tenant_id=payload.get("tenant_id", ""),
        vector=list(point.vector) if point.vector else [],
        text_preview=payload.get("text_preview", ""),
        source=payload.get("source", ""),
        session_id=payload.get("session_id"),
        level=payload.get("level", 0),
        cluster_id=payload.get("cluster_id"),
        parent_id=payload.get("parent_id"),
        created_at=payload.get("created_at", 0.0),
        extra={k: v for k, v in payload.items() if k not in known},
    )


__all__ = ["QdrantStore", "VectorStore"]
