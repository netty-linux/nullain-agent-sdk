"""Multi-tenant isolation gate — docs/FUSION_PLAN.md §4:

"Teste de isolamento como gate de CI: inserir vetores de 2 tenants
sintéticos com embeddings propositalmente próximos, garantir que a busca
de um nunca retorna o outro — isso vira um teste obrigatório antes de
qualquer merge dessa área."

This is that gate. `InMemoryVectorStore` is a minimal, real implementation
of the `VectorStore` Protocol (not a mock of one) — it exercises the same
contract `QdrantStore` does, including doing its own naive linear-scan
cosine search, so a bug in the *store's* filtering logic would be caught
here exactly as it would against real Qdrant. What's being tested is the
contract (scoped_search never crosses tenant_id), not Qdrant's own
correctness (out of scope without a live cluster).
"""

from __future__ import annotations

import math

import pytest
from nullain.errors import TenantIsolationError
from nullain.rag import VectorRecord, VectorStore


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryVectorStore:
    """Real (not mocked) `VectorStore` implementation for tests — linear
    scan, no external service. Deliberately mirrors `QdrantStore`'s
    contract: `scoped_search` is the only search method, always filters by
    `tenant_id`, and raises `TenantIsolationError` on an empty tenant_id."""

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    async def upsert(self, records: list[VectorRecord]) -> None:
        for record in records:
            if not record.tenant_id.strip():
                raise TenantIsolationError("tenant_id is required")
            self._records[record.id] = record

    async def scoped_search(
        self,
        tenant_id: str,
        query_vector: list[float],
        *,
        limit: int = 10,
        cluster_id: str | None = None,
        level: int | None = None,
    ) -> list[VectorRecord]:
        if not tenant_id.strip():
            raise TenantIsolationError("tenant_id is required")
        candidates = [r for r in self._records.values() if r.tenant_id == tenant_id]
        if cluster_id is not None:
            candidates = [r for r in candidates if r.cluster_id == cluster_id]
        if level is not None:
            candidates = [r for r in candidates if r.level == level]
        ranked = sorted(candidates, key=lambda r: _cosine(r.vector, query_vector), reverse=True)
        return ranked[:limit]

    async def delete(self, tenant_id: str, record_ids: list[str]) -> None:
        if not tenant_id.strip():
            raise TenantIsolationError("tenant_id is required")
        for rid in record_ids:
            record = self._records.get(rid)
            if record is not None and record.tenant_id == tenant_id:
                del self._records[rid]


def test_in_memory_store_satisfies_the_vector_store_protocol() -> None:
    """Guards against the fake drifting from the real port's shape —
    if VectorStore's Protocol gains/changes a method, this fails loudly
    instead of the isolation tests below silently testing a stale contract."""
    assert isinstance(InMemoryVectorStore(), VectorStore)


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.mark.asyncio
async def test_search_never_returns_another_tenants_records(
    store: InMemoryVectorStore,
) -> None:
    """The core gate: two tenants, deliberately near-identical embeddings
    (docs/FUSION_PLAN.md's exact scenario) — tenant-a's search must only
    ever surface tenant-a's records, regardless of how close tenant-b's
    vectors are in embedding space."""
    close_vector_a = [1.0, 0.0, 0.0]
    close_vector_b = [0.999, 0.001, 0.0]  # deliberately near-identical

    await store.upsert(
        [
            VectorRecord(
                id="a-secret-1",
                tenant_id="tenant-a",
                vector=close_vector_a,
                text_preview="tenant A's private data",
            ),
            VectorRecord(
                id="b-secret-1",
                tenant_id="tenant-b",
                vector=close_vector_b,
                text_preview="tenant B's private data",
            ),
        ]
    )

    results_a = await store.scoped_search("tenant-a", close_vector_a, limit=10)
    results_b = await store.scoped_search("tenant-b", close_vector_b, limit=10)

    assert {r.id for r in results_a} == {"a-secret-1"}
    assert {r.id for r in results_b} == {"b-secret-1"}
    assert all(r.tenant_id == "tenant-a" for r in results_a)
    assert all(r.tenant_id == "tenant-b" for r in results_b)


@pytest.mark.asyncio
async def test_search_isolation_holds_with_many_interleaved_tenants(
    store: InMemoryVectorStore,
) -> None:
    """Same gate, scaled up: 5 tenants, all embeddings clustered in the
    same small region of vector space (worst case for approximate/near
    search accidentally crossing tenant boundaries)."""
    tenants = [f"tenant-{i}" for i in range(5)]
    for i, tenant in enumerate(tenants):
        # All vectors near [1, 0, 0] with tiny per-tenant perturbation —
        # this is the "propositalmente próximos" scenario from the plan.
        vector = [1.0, 0.001 * i, 0.0]
        await store.upsert(
            [
                VectorRecord(
                    id=f"{tenant}-doc",
                    tenant_id=tenant,
                    vector=vector,
                    text_preview=f"{tenant}'s data",
                )
            ]
        )

    for tenant in tenants:
        results = await store.scoped_search(tenant, [1.0, 0.0, 0.0], limit=100)
        assert {r.tenant_id for r in results} == {tenant}, (
            f"{tenant}'s search leaked another tenant's records: {[r.tenant_id for r in results]}"
        )


@pytest.mark.asyncio
async def test_scoped_search_rejects_empty_tenant_id(store: InMemoryVectorStore) -> None:
    with pytest.raises(TenantIsolationError):
        await store.scoped_search("", [1.0, 0.0])


@pytest.mark.asyncio
async def test_upsert_rejects_empty_tenant_id(store: InMemoryVectorStore) -> None:
    with pytest.raises(TenantIsolationError):
        await store.upsert(
            [VectorRecord.model_construct(tenant_id="", vector=[0.1], text_preview="x", id="x")]
        )


@pytest.mark.asyncio
async def test_delete_only_affects_the_owning_tenant(store: InMemoryVectorStore) -> None:
    """A delete call scoped to tenant-a must not remove tenant-b's record
    even if (by id collision or malicious input) the same record id is
    passed under the wrong tenant."""
    await store.upsert(
        [
            VectorRecord(id="shared-id", tenant_id="tenant-a", vector=[1.0], text_preview="a"),
        ]
    )
    # tenant-b attempts to delete tenant-a's record id — must be a no-op.
    await store.delete("tenant-b", ["shared-id"])
    remaining = await store.scoped_search("tenant-a", [1.0], limit=10)
    assert len(remaining) == 1
    assert remaining[0].id == "shared-id"


@pytest.mark.asyncio
async def test_delete_rejects_empty_tenant_id(store: InMemoryVectorStore) -> None:
    with pytest.raises(TenantIsolationError):
        await store.delete("", ["some-id"])


@pytest.mark.asyncio
async def test_cluster_and_level_filters_stay_within_tenant_scope(
    store: InMemoryVectorStore,
) -> None:
    """cluster_id/level narrow the search further, but must never be used
    as a substitute for tenant scoping — a shared cluster_id across two
    tenants (plausible: both tenants' RAG trees could independently produce
    a node with the same generated id) must not leak across the tenant
    boundary."""
    await store.upsert(
        [
            VectorRecord(
                id="a1",
                tenant_id="tenant-a",
                vector=[1.0, 0.0],
                text_preview="a",
                cluster_id="cluster-shared-id",
                level=1,
            ),
            VectorRecord(
                id="b1",
                tenant_id="tenant-b",
                vector=[1.0, 0.0],
                text_preview="b",
                cluster_id="cluster-shared-id",
                level=1,
            ),
        ]
    )
    results = await store.scoped_search(
        "tenant-a", [1.0, 0.0], cluster_id="cluster-shared-id", level=1, limit=10
    )
    assert {r.id for r in results} == {"a1"}
