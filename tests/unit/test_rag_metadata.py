"""Unit tests for VectorRecord — the mandatory tenant-scoped record schema."""

import pytest
from nullain.rag import VectorRecord
from pydantic import ValidationError


def test_tenant_id_is_required() -> None:
    """No default for tenant_id — omitting it must fail validation
    immediately (docs/FUSION_PLAN.md's fail-closed isolation requirement),
    not silently produce an unscoped record."""
    with pytest.raises(ValidationError):
        VectorRecord(vector=[0.1, 0.2], text_preview="x")  # type: ignore[call-arg]


def test_id_defaults_to_a_fresh_uuid() -> None:
    r1 = VectorRecord(tenant_id="t1", vector=[0.1], text_preview="x")
    r2 = VectorRecord(tenant_id="t1", vector=[0.1], text_preview="x")
    assert r1.id != r2.id


def test_record_is_frozen() -> None:
    record = VectorRecord(tenant_id="t1", vector=[0.1], text_preview="x")
    with pytest.raises(ValidationError):
        record.tenant_id = "t2"  # type: ignore[misc]


def test_optional_fields_have_sane_defaults() -> None:
    record = VectorRecord(tenant_id="t1", vector=[0.1], text_preview="x")
    assert record.source == ""
    assert record.session_id is None
    assert record.level == 0
    assert record.cluster_id is None
    assert record.parent_id is None
    assert record.extra == {}
