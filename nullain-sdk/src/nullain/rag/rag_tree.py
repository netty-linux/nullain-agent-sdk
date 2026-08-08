"""Nullain Agent SDK — RAG Tree: hierarchical cluster navigation.

A RAG Tree organizes a tenant's embedded chunks into a shallow hierarchy of
clusters (`RagTreeNode`s, root at `level=0`) so a query can navigate from a
broad cluster down to a specific one before doing the expensive vector
search, instead of a flat k-NN over every vector the tenant owns
(docs/FUSION_PLAN.md §4). The tree is a routing index — it holds
per-cluster centroid vectors, never the leaf embeddings themselves, which
live in the `VectorStore` (`VectorRecord.cluster_id`/`level`/`parent_id`
tag which node each leaf belongs to).

This module is intentionally storage-agnostic: `RagTreeIndex` is an
in-memory structure the caller builds from whatever clustering process it
runs (k-means over a tenant's embeddings, a simple recursive bisection,
etc.) — building/updating the tree is out of scope here, this module is
the read-side navigation structure and the node data model.
"""

from __future__ import annotations

import math

from pydantic import BaseModel


class RagTreeNode(BaseModel, frozen=True):
    """One cluster in the tree.

    Attributes:
        id: Unique node id.
        tenant_id: Owning tenant — every node belongs to exactly one tenant,
            same discipline as `VectorRecord`.
        level: Depth in the tree; 0 is the root.
        parent_id: Parent node's id, or None for a root node.
        centroid: The cluster's representative vector — same dimensionality
            as the leaf embeddings it summarizes. Used to rank children
            during navigation, never searched against the full vector store
            directly.
        label: Optional human-readable summary of what this cluster covers
            (a short description, topic name, or similar) — useful for
            debugging/observability, not required for navigation.
    """

    id: str
    tenant_id: str
    level: int = 0
    parent_id: str | None = None
    centroid: list[float]
    label: str = ""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class RagTreeIndex:
    """In-memory navigable index over a tenant's `RagTreeNode`s.

    Nodes from every tenant may be added to the same index instance —
    `navigate`/`children_of`/`path_to_root` all take `tenant_id` explicitly
    and never cross tenant boundaries, matching the isolation discipline
    the rest of the RAG module enforces (a caller mixing tenants in one
    `RagTreeIndex` still gets correctly scoped results per call).
    """

    def __init__(self) -> None:
        self._nodes: dict[str, RagTreeNode] = {}
        self._children: dict[str, list[str]] = {}  # parent_id -> [child_id, ...]
        self._roots: dict[str, list[str]] = {}  # tenant_id -> [root_node_id, ...]

    def add_node(self, node: RagTreeNode) -> None:
        """Insert or replace a node. Re-adding an existing id updates it in
        place (its old parent/children linkage is not automatically
        migrated — remove and re-add children explicitly if restructuring)."""
        self._nodes[node.id] = node
        if node.parent_id is None:
            roots = self._roots.setdefault(node.tenant_id, [])
            if node.id not in roots:
                roots.append(node.id)
        else:
            siblings = self._children.setdefault(node.parent_id, [])
            if node.id not in siblings:
                siblings.append(node.id)

    def get_node(self, node_id: str) -> RagTreeNode | None:
        return self._nodes.get(node_id)

    def children_of(self, node_id: str) -> list[RagTreeNode]:
        return [self._nodes[cid] for cid in self._children.get(node_id, []) if cid in self._nodes]

    def roots_for(self, tenant_id: str) -> list[RagTreeNode]:
        return [self._nodes[rid] for rid in self._roots.get(tenant_id, []) if rid in self._nodes]

    def path_to_root(self, node_id: str) -> list[RagTreeNode]:
        """Return the chain from `node_id` up to (and including) its root,
        ordered root-first — useful for building a "cluster breadcrumb" for
        observability or for constructing a Qdrant payload filter that
        matches an entire subtree."""
        path: list[RagTreeNode] = []
        current = self._nodes.get(node_id)
        while current is not None:
            path.append(current)
            current = self._nodes.get(current.parent_id) if current.parent_id else None
        path.reverse()
        return path

    def navigate(
        self, tenant_id: str, query_vector: list[float], *, top_k: int = 1
    ) -> list[RagTreeNode]:
        """Greedily descend the tree for `tenant_id`, at each level keeping
        the `top_k` children closest (cosine similarity) to `query_vector`,
        until a level has no closer children than its parents already
        explored. Returns the leaf-most nodes reached, most-similar first.

        This is a beam search of width `top_k`, not an exhaustive tree
        search — it trades a small chance of missing the single best branch
        for O(depth * branching_factor) instead of O(tree size) navigation
        cost, which is the entire point of clustering the search space.
        """
        frontier = self.roots_for(tenant_id)
        if not frontier:
            return []

        best_leaves: list[RagTreeNode] = []
        while frontier:
            ranked = sorted(
                frontier,
                key=lambda n: _cosine_similarity(n.centroid, query_vector),
                reverse=True,
            )[:top_k]
            next_frontier: list[RagTreeNode] = []
            for node in ranked:
                children = self.children_of(node.id)
                if children:
                    next_frontier.extend(children)
                else:
                    best_leaves.append(node)
            if not next_frontier:
                # Every ranked node this round was a leaf — done descending.
                break
            frontier = next_frontier

        if not best_leaves:
            # No node in the tree is a leaf yet (single-level tree, or the
            # walk never reached one) — surface whatever the last ranked
            # frontier was, so navigate() never silently returns nothing
            # for a non-empty tenant tree.
            best_leaves = sorted(
                frontier,
                key=lambda n: _cosine_similarity(n.centroid, query_vector),
                reverse=True,
            )[:top_k]
        return sorted(
            best_leaves, key=lambda n: _cosine_similarity(n.centroid, query_vector), reverse=True
        )


__all__ = ["RagTreeIndex", "RagTreeNode"]
