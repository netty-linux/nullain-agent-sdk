"""Nullain Agent SDK — RAG Module (Cuckoo Filter + RAG Tree).

Replaces the app's AnythingLLM integration (docs/FUSION_PLAN.md §4) with a
SDK-owned retrieval pipeline built from three independently useful pieces:

- `CuckooFilter` — O(1) existence pre-check ("could this be indexed?")
  before paying for an embedding + vector search.
- `RagTreeIndex`/`RagTreeNode` — hierarchical cluster navigation, so a
  query descends from a broad cluster to a specific one instead of a flat
  k-NN over every vector a tenant owns.
- `EmbeddingProvider`/`FastEmbedProvider` and `VectorStore`/`QdrantStore` —
  the two Ports & Adapters pairs the pipeline is built on, mirroring
  `LLMProvider`'s shape elsewhere in the SDK.

All vector operations are tenant-scoped by construction: `VectorRecord`
requires `tenant_id`, and `VectorStore.scoped_search` is the only search
entry point in the store's public interface — see `qdrant_store.py`'s
module docstring for the full isolation rationale.
"""

from nullain.rag.cuckoo_filter import CuckooFilter
from nullain.rag.embedding import EmbeddingProvider, FastEmbedProvider
from nullain.rag.metadata import VectorRecord
from nullain.rag.qdrant_store import QdrantStore, VectorStore
from nullain.rag.rag_tree import RagTreeIndex, RagTreeNode

__all__ = [
    "CuckooFilter",
    "EmbeddingProvider",
    "FastEmbedProvider",
    "QdrantStore",
    "RagTreeIndex",
    "RagTreeNode",
    "VectorRecord",
    "VectorStore",
]
