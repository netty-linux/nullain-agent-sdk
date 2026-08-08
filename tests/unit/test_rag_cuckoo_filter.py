"""Unit tests for CuckooFilter — the RAG existence pre-filter."""

from nullain.rag import CuckooFilter


def test_add_and_might_contain() -> None:
    cf = CuckooFilter(capacity=1000)
    cf.add("tenant-a", "doc-1")
    assert cf.might_contain("tenant-a", "doc-1") is True


def test_might_contain_false_for_never_added_key() -> None:
    cf = CuckooFilter(capacity=1000)
    cf.add("tenant-a", "doc-1")
    assert cf.might_contain("tenant-a", "doc-2") is False


def test_tenant_scoping_same_key_different_tenant() -> None:
    """A key added under one tenant must not appear present under another
    — the fingerprint hashes tenant_id + key together, not just key."""
    cf = CuckooFilter(capacity=1000)
    cf.add("tenant-a", "doc-1")
    assert cf.might_contain("tenant-b", "doc-1") is False


def test_remove_returns_true_and_clears_membership() -> None:
    cf = CuckooFilter(capacity=1000)
    cf.add("tenant-a", "doc-1")
    assert cf.remove("tenant-a", "doc-1") is True
    assert cf.might_contain("tenant-a", "doc-1") is False


def test_remove_returns_false_for_absent_key() -> None:
    cf = CuckooFilter(capacity=1000)
    assert cf.remove("tenant-a", "never-added") is False


def test_remove_of_one_tenant_does_not_affect_another() -> None:
    """Same key, two tenants — removing tenant-a's copy must leave
    tenant-b's membership untouched (they are different fingerprints)."""
    cf = CuckooFilter(capacity=1000)
    cf.add("tenant-a", "doc-1")
    cf.add("tenant-b", "doc-1")
    cf.remove("tenant-a", "doc-1")
    assert cf.might_contain("tenant-a", "doc-1") is False
    assert cf.might_contain("tenant-b", "doc-1") is True


def test_len_tracks_insertions_and_removals() -> None:
    cf = CuckooFilter(capacity=1000)
    assert len(cf) == 0
    cf.add("t", "a")
    cf.add("t", "b")
    assert len(cf) == 2
    cf.remove("t", "a")
    assert len(cf) == 1


def test_many_insertions_stay_under_capacity_without_add_failing() -> None:
    """A filter sized well under its stated capacity should never report
    'full' — this is a load-factor sanity check, not an exhaustive proof."""
    cf = CuckooFilter(capacity=2000)
    ok = [cf.add("t", f"key-{i}") for i in range(1500)]
    assert all(ok), "add() reported full well under stated capacity"
    assert len(cf) == 1500


def test_false_positive_rate_is_bounded_for_absent_keys() -> None:
    """Not a proof of the theoretical FP bound — a coarse sanity check that
    membership queries for thousands of never-inserted keys don't come back
    mostly-true, which would indicate a hashing bug rather than the
    filter's expected (low, bounded) false-positive rate."""
    cf = CuckooFilter(capacity=5000, fingerprint_bits=16)
    for i in range(2000):
        cf.add("tenant", f"present-{i}")

    false_positives = sum(1 for i in range(5000) if cf.might_contain("tenant", f"absent-{i}"))
    # 16-bit fingerprints -> expected FP rate on the order of 2/2^16 per
    # bucket-pair check; generous 5% ceiling keeps this from being flaky
    # while still catching a broken hash (which would show ~100%).
    assert false_positives / 5000 < 0.05


def test_capacity_must_be_positive() -> None:
    import pytest

    with pytest.raises(ValueError, match="capacity must be positive"):
        CuckooFilter(capacity=0)
