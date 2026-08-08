"""Nullain Agent SDK — RAG vector record schema and metadata contract.

Every vector written to or read from a `VectorStore` carries this shape.
`tenant_id` has no default — a caller must supply it explicitly, so an
accidental omission fails validation immediately rather than silently
writing (or querying) an unscoped vector. This is the type-level half of
the multi-tenant isolation guarantee (docs/FUSION_PLAN.md); the other half
is `VectorStore.scoped_search` being the only public search entry point.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class VectorRecord(BaseModel, frozen=True):
    """One embedded chunk plus its RAG Tree position and provenance.

    Attributes:
        id: Unique record id (a fresh UUID4 when not supplied).
        tenant_id: Owning tenant/user — required, no default. Every write
            and every read is scoped by this field.
        vector: The dense embedding, already produced by an
            EmbeddingProvider — VectorRecord itself does no embedding.
        text_preview: A bounded excerpt of the source text, stored in the
            vector database for quick display. The full text is expected
            to live in relational storage (docs/FUSION_PLAN.md's Supabase
            schema), not duplicated here.
        source: Free-form origin label (filename, URL, conversation id, ...).
        session_id: Optional session/thread this chunk was captured from.
        level: RAG Tree depth — 0 is the root/coarsest cluster level.
        cluster_id: Id of the RAG Tree node this record belongs to.
        parent_id: Id of `cluster_id`'s parent node, or None at the root.
        created_at: Unix timestamp (seconds) of when the record was built.
        extra: Free-form additional payload, merged into the store's
            payload alongside the named fields above — for adapter- or
            application-specific metadata that doesn't warrant a first-class
            field yet.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    vector: list[float]
    text_preview: str
    source: str = ""
    session_id: str | None = None
    level: int = 0
    cluster_id: str | None = None
    parent_id: str | None = None
    created_at: float = Field(default_factory=time.time)
    extra: dict[str, Any] = Field(default_factory=dict)


__all__ = ["VectorRecord"]
