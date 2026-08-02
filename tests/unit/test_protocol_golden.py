"""Golden file tests for NDJSON protocol envelope validation."""

import json
from pathlib import Path

import pytest
from nullain.protocol import ProtocolEnvelope

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "schema" / "golden"


def _load_golden(filename: str) -> list[dict[str, object]]:
    """Load all NDJSON lines from a golden file."""
    path = GOLDEN_DIR / filename
    if not path.exists():
        pytest.skip(f"Golden file not found: {path}")
    lines: list[dict[str, object]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                lines.append(json.loads(stripped))
    return lines


class TestProtocolGoldenFiles:
    """Validate golden NDJSON session files against ProtocolEnvelope schema."""

    def test_session_basic_validates(self) -> None:
        """Every line in session_basic.ndjson must be a valid ProtocolEnvelope."""
        records = _load_golden("session_basic.ndjson")
        assert len(records) > 0, "Golden file is empty"
        for idx, record in enumerate(records):
            envelope = ProtocolEnvelope.model_validate(record)
            assert envelope.v == 1, f"Line {idx}: version must be 1"
            assert envelope.type, f"Line {idx}: type must be non-empty"

    def test_session_basic_has_start_and_end(self) -> None:
        """Golden session must contain session.start and session.end."""
        records = _load_golden("session_basic.ndjson")
        types = [r.get("type") for r in records]
        assert "session.start" in types
        assert "session.end" in types

    def test_session_basic_has_user_message(self) -> None:
        """Golden session must contain a user.message envelope."""
        records = _load_golden("session_basic.ndjson")
        types = [r.get("type") for r in records]
        assert "user.message" in types
