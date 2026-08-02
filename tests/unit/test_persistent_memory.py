"""Unit tests for persistent memory (P3.21)."""

from pathlib import Path

import pytest
from nullain.agent import AgentLoop
from nullain.events import CompactionEvent
from nullain.llm import CompletionChunk, CompletionRequest, LLMProvider
from nullain.memory import MAX_INDEX_BYTES, MemoryEntry, MemoryType, PersistentMemory
from nullain.tools import ToolRegistry
from nullain_tools import create_memory_tools, register_default_tools


class _FakeProvider(LLMProvider):
    def __init__(self, responses: list[CompletionChunk]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        if self.call_count < len(self.responses):
            chunk = self.responses[self.call_count]
            self.call_count += 1
            return chunk
        return CompletionChunk(delta_text="Done")

    async def stream(self, request: CompletionRequest):
        chunk = await self.generate(request)
        yield chunk

    async def health_check(self) -> bool:
        return True


def _entry(name: str, desc: str = "a fact", body: str = "details") -> MemoryEntry:
    return MemoryEntry(name=name, description=desc, body=body, type=MemoryType.PROJECT)


def test_persistent_memory_write_read_list_delete(tmp_path: Path) -> None:
    pm = PersistentMemory(workspace_root=tmp_path)

    pm.write(_entry("preferred-stack", desc="User likes uv", body="Use uv for python."))
    assert pm.read("preferred-stack") == "Use uv for python."
    assert pm.list_entries() == ["preferred-stack"]

    # Index file exists and references the topic file.
    index = (pm.memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert "preferred-stack" in index
    assert "User likes uv" in index

    # Update in place replaces the line, keeps one entry.
    pm.write(_entry("preferred-stack", desc="User loves uv", body="Use uv always."))
    assert pm.list_entries() == ["preferred-stack"]
    assert pm.read("preferred-stack") == "Use uv always."

    assert pm.delete("preferred-stack") is True
    assert pm.read("preferred-stack") is None
    assert pm.list_entries() == []
    assert not (pm.memory_dir / "preferred-stack.md").exists()


def test_persistent_memory_slug_validation(tmp_path: Path) -> None:
    pm = PersistentMemory(workspace_root=tmp_path)
    for bad in ["Bad-Name", "with space", "../escape", "UPPER", "a_b", ""]:
        with pytest.raises(ValueError, match="kebab-case"):
            pm.write(_entry(bad))


def test_persistent_memory_to_context_empty_and_filled(tmp_path: Path) -> None:
    pm = PersistentMemory(workspace_root=tmp_path)
    assert pm.to_context() == ""

    pm.write(_entry("alpha", desc="first", body="x"))
    ctx = pm.to_context()
    assert ctx.startswith("# PERSISTENT MEMORY INDEX")
    assert "- [alpha](alpha.md) — first" in ctx


def test_persistent_memory_index_cap_evicts_oldest(tmp_path: Path) -> None:
    pm = PersistentMemory(workspace_root=tmp_path, max_index_bytes=400)
    # Each entry adds an index line ~60 bytes; with a 400-byte cap only a
    # handful survive. Write enough to force eviction of the oldest.
    names = [f"item-{i:02d}" for i in range(20)]
    for n in names:
        pm.write(_entry(n, desc=f"description for {n}", body=f"body {n}"))

    remaining = pm.list_entries()
    # The oldest items must have been evicted; the most recent survive.
    assert "item-00" not in remaining, "oldest entry should be evicted by the cap"
    assert names[-1] in remaining
    # Evicted topic files are deleted.
    assert not (pm.memory_dir / "item-00.md").exists()
    # Index file is within cap.
    assert len((pm.memory_dir / "MEMORY.md").read_bytes()) <= 400


def test_default_cap_is_25kb() -> None:
    assert MAX_INDEX_BYTES == 25_000


@pytest.mark.asyncio
async def test_memory_tools_save_and_read(tmp_path: Path) -> None:
    pm = PersistentMemory(workspace_root=tmp_path)
    registry = ToolRegistry()
    for t in create_memory_tools(pm):
        registry.register(t)

    saved = await registry.execute(
        "save_memory",
        {
            "name": "team-convention",
            "description": "Trunk-based master",
            "body": "Always commit to master.",
            "memory_type": "project",
        },
    )
    assert "Saved memory 'team-convention'" in saved

    listed = await registry.execute("read_memory", {"name": ""})
    assert "team-convention" in listed

    body = await registry.execute("read_memory", {"name": "team-convention"})
    assert "Always commit to master." in body


@pytest.mark.asyncio
async def test_memory_tool_rejects_invalid_type(tmp_path: Path) -> None:
    pm = PersistentMemory(workspace_root=tmp_path)
    registry = ToolRegistry()
    for t in create_memory_tools(pm):
        registry.register(t)

    out = await registry.execute(
        "save_memory",
        {
            "name": "x",
            "description": "d",
            "body": "b",
            "memory_type": "bogus",
        },
    )
    assert "invalid memory type" in out


@pytest.mark.asyncio
async def test_agent_loop_reinjects_agents_md_and_memory_post_compaction(tmp_path: Path) -> None:
    """After a compaction event, _build_messages still leads the context with
    a system message containing AGENTS.md content and the persistent-memory
    index — they survive compaction rather than being replaced by the recap."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text(
        "# Project Conventions\nMAGIC_MARKER_AGENTS_42\n", encoding="utf-8"
    )

    pm = PersistentMemory(workspace_root=workspace)
    pm.write(_entry("golden-rule", desc="The golden rule", body="Be honest about limits."))

    registry = ToolRegistry()
    register_default_tools(registry, workspace, persistent_memory=pm)

    agent = AgentLoop(
        provider=_FakeProvider([CompletionChunk(delta_text="Done")]),
        tools=registry,
        workspace_root=workspace,
        persistent_memory=pm,
    )
    # Prime the few-shot cache path used by _assemble_system_prompt.
    agent._episodic_few_shot = None  # type: ignore[reportPrivateUsage]

    # Build messages with a compaction summary already in the trajectory: the
    # system prompt (index 0) must still carry AGENTS.md + memory markers.
    sess = "s-test"
    compaction = CompactionEvent(
        session_id=sess,
        summary="[Compacted] earlier trajectory recap",
        compacted_event_ids=(),
    )
    messages = agent._build_messages(  # type: ignore[reportPrivateUsage]
        sess, [compaction], agent._assemble_system_prompt(), step=1  # type: ignore[reportPrivateUsage]
    )

    system_text = messages[0].content
    assert system_text is not None
    assert "MAGIC_MARKER_AGENTS_42" in system_text, "AGENTS.md must be re-injected post-compaction"
    assert "golden-rule" in system_text, "persistent memory index must be re-injected"
    assert "The golden rule" in system_text
