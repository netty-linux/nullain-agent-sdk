"""Unit tests for RagTreeIndex — hierarchical cluster navigation."""

import pytest
from nullain.rag import RagTreeIndex, RagTreeNode


def _idx() -> RagTreeIndex:
    idx = RagTreeIndex()
    idx.add_node(RagTreeNode(id="root-a", tenant_id="t1", level=0, centroid=[1.0, 0.0]))
    idx.add_node(RagTreeNode(id="root-b", tenant_id="t1", level=0, centroid=[0.0, 1.0]))
    idx.add_node(
        RagTreeNode(id="leaf-a1", tenant_id="t1", level=1, parent_id="root-a", centroid=[0.9, 0.1])
    )
    idx.add_node(
        RagTreeNode(id="leaf-a2", tenant_id="t1", level=1, parent_id="root-a", centroid=[0.8, -0.1])
    )
    idx.add_node(
        RagTreeNode(id="leaf-b1", tenant_id="t1", level=1, parent_id="root-b", centroid=[0.1, 0.9])
    )
    return idx


def test_add_node_and_get_node() -> None:
    idx = _idx()
    node = idx.get_node("root-a")
    assert node is not None
    assert node.tenant_id == "t1"
    assert node.level == 0


def test_get_node_returns_none_for_unknown_id() -> None:
    idx = _idx()
    assert idx.get_node("does-not-exist") is None


def test_roots_for_tenant() -> None:
    idx = _idx()
    roots = idx.roots_for("t1")
    assert {n.id for n in roots} == {"root-a", "root-b"}


def test_roots_for_unknown_tenant_is_empty() -> None:
    idx = _idx()
    assert idx.roots_for("no-such-tenant") == []


def test_children_of() -> None:
    idx = _idx()
    children = idx.children_of("root-a")
    assert {n.id for n in children} == {"leaf-a1", "leaf-a2"}


def test_children_of_leaf_is_empty() -> None:
    idx = _idx()
    assert idx.children_of("leaf-a1") == []


def test_path_to_root_is_root_first() -> None:
    idx = _idx()
    path = idx.path_to_root("leaf-a1")
    assert [n.id for n in path] == ["root-a", "leaf-a1"]


def test_path_to_root_of_a_root_is_itself() -> None:
    idx = _idx()
    path = idx.path_to_root("root-a")
    assert [n.id for n in path] == ["root-a"]


def test_navigate_descends_toward_closest_root() -> None:
    idx = _idx()
    result = idx.navigate("t1", [1.0, 0.05], top_k=1)
    assert len(result) == 1
    assert result[0].id in {"leaf-a1", "leaf-a2"}


def test_navigate_most_similar_leaf_first() -> None:
    idx = _idx()
    result = idx.navigate("t1", [1.0, 0.05], top_k=2)
    assert [n.id for n in result] == ["leaf-a1", "leaf-a2"]


def test_navigate_toward_other_branch() -> None:
    idx = _idx()
    result = idx.navigate("t1", [0.0, 1.0], top_k=1)
    assert result[0].id == "leaf-b1"


def test_navigate_unknown_tenant_returns_empty() -> None:
    idx = _idx()
    assert idx.navigate("unknown-tenant", [1.0, 0.0]) == []


def test_navigate_single_level_tree_returns_roots() -> None:
    """A tree with no children at all — navigate() should still surface
    the (leaf) roots rather than returning nothing."""
    idx = RagTreeIndex()
    idx.add_node(RagTreeNode(id="only-root", tenant_id="t2", level=0, centroid=[1.0, 0.0]))
    result = idx.navigate("t2", [1.0, 0.0], top_k=1)
    assert [n.id for n in result] == ["only-root"]


def test_cosine_similarity_dimension_mismatch_raises() -> None:
    idx = RagTreeIndex()
    idx.add_node(RagTreeNode(id="r", tenant_id="t3", level=0, centroid=[1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="vector length mismatch"):
        idx.navigate("t3", [1.0, 0.0])
