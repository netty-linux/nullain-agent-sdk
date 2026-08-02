"""Integration tests for Stdio NDJSON Agent Daemon and Episodic Memory."""

import json
from pathlib import Path

import pytest
from nullain.memory import EpisodicMemory, TrajectoryRecord
from nullain.protocol import ProtocolEnvelope, SessionStartPayload, export_schema


def test_schema_export(tmp_path: Path) -> None:
    schema_file = tmp_path / "protocol_v1.json"
    exported = export_schema(schema_file)
    assert exported.exists()

    content = json.loads(exported.read_text())
    assert content.get("title") == "ProtocolEnvelope"
    assert "properties" in content


@pytest.mark.asyncio
async def test_episodic_memory_learning_loop(tmp_path: Path) -> None:
    db_file = tmp_path / "memory.db"
    memory = EpisodicMemory(db_file)
    await memory.initialize()

    # Record past successful trajectory
    rec1 = TrajectoryRecord(
        session_id="s1",
        intent="simple_edit",
        model="gpt-oss:20b",
        steps_count=2,
        success=True,
        objective="Create FACTS.txt",
    )
    await memory.record_trajectory(rec1)

    # Record failed trajectory (should be ignored by get_relevant_examples)
    rec2 = TrajectoryRecord(
        session_id="s2",
        intent="simple_edit",
        model="gpt-oss:20b",
        steps_count=5,
        success=False,
        objective="Failed attempt",
    )
    await memory.record_trajectory(rec2)

    # Retrieve relevant examples for simple_edit intent
    examples = await memory.get_relevant_examples("simple_edit", limit=2)
    assert len(examples) == 1
    assert examples[0].session_id == "s1"
    assert examples[0].objective == "Create FACTS.txt"

    await memory.close()


def test_protocol_ndjson_serialization(tmp_path: Path) -> None:
    payload = SessionStartPayload(session_id="sess_100", workspace_root=str(tmp_path))
    env = ProtocolEnvelope(v=1, type="session.start", payload=payload.model_dump())

    line = env.model_dump_json()
    assert '"v":1' in line
    assert '"type":"session.start"' in line

    deserialized = ProtocolEnvelope.model_validate_json(line)
    assert deserialized.type == "session.start"
    assert deserialized.payload["session_id"] == "sess_100"
