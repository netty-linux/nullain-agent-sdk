"""Nullain Agent SDK — JSON Schema Exporter for Go CLI Contract."""

import json
from pathlib import Path

from nullain.protocol.types import ProtocolEnvelope


def export_schema(output_path: str | Path = "schema/protocol_v1.json") -> Path:
    """Export ProtocolEnvelope JSON Schema to target filepath or directory."""
    target = Path(output_path)
    if target.is_dir() or str(output_path).endswith(("/", "\\")):
        target = target / "protocol_v1.json"
    target.parent.mkdir(parents=True, exist_ok=True)

    schema = ProtocolEnvelope.model_json_schema()
    target.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return target


if __name__ == "__main__":
    export_schema()


__all__ = ["export_schema"]
