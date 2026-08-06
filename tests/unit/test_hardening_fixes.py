"""Regression tests for the post-M11 hardening fixes.

Three defects found in the deep check-in, each proven here:

1. ``ContextManager.estimate_tokens`` was a fixed ``len // 4`` guess. The
   provider's real ``usage`` never reached it, so compaction fired on an
   uncalibrated estimate. Now ``calibrate`` folds the measured ratio in.
2. ``CheckpointStore.undo`` popped a single checkpoint, so a write touching
   several files needed several undos and left the workspace half-reverted in
   between. Undo is now per-operation.
3. Retention evicted checkpoints silently, so an undo whose snapshot had been
   dropped failed with no explanation. Eviction now drops whole operations,
   logs, and ``undo`` reports the truncated horizon.
"""

from pathlib import Path

import pytest
from nullain.context import ContextManager
from nullain.llm.types import ChatMessage, ToolCall
from nullain_tools.checkpoints import CheckpointStore

# --------------------------------------------------------------------------
# Fix 1 — token estimation calibrated from the provider's real usage
# --------------------------------------------------------------------------


def _messages(text: str) -> list[ChatMessage]:
    return [ChatMessage(role="user", content=text)]


def test_estimate_tokens_defaults_to_heuristic_before_calibration() -> None:
    """Before any real usage is seen, the ~4-chars-per-token bootstrap applies."""
    cm = ContextManager()
    assert cm.calibration_samples == 0
    assert cm.chars_per_token == pytest.approx(4.0)
    assert cm.estimate_tokens("x" * 400) == 100


def test_calibrate_shifts_ratio_towards_observed() -> None:
    """A real usage sample moves the ratio off the fixed 4.0 guess.

    Regression: previously ``estimate_tokens`` was hard-coded to ``len // 4``
    and no provider feedback could change it.
    """
    cm = ContextManager()
    # 1000 chars really cost 500 tokens => 2.0 chars/token, not the 4.0 guess.
    cm.calibrate(_messages("x" * 1000), prompt_tokens=500)

    assert cm.calibration_samples == 1
    assert cm.chars_per_token == pytest.approx(2.0)
    # The estimate now reflects the model's actual tokenizer: twice the tokens
    # the old heuristic would have reported.
    assert cm.estimate_tokens("x" * 1000) == 500


def test_calibration_converges_over_repeated_samples() -> None:
    """Repeated consistent samples converge on the observed ratio."""
    cm = ContextManager()
    for _ in range(10):
        cm.calibrate(_messages("x" * 1000), prompt_tokens=500)
    assert cm.chars_per_token == pytest.approx(2.0, abs=0.05)
    assert cm.calibration_samples == 10


def test_calibration_rejects_degenerate_samples() -> None:
    """Zero tokens, tiny contexts and implausible ratios never skew the ratio."""
    cm = ContextManager()

    cm.calibrate(_messages("x" * 1000), prompt_tokens=0)  # no usage reported
    cm.calibrate(_messages("x" * 10), prompt_tokens=5)  # context too small
    cm.calibrate(_messages("x" * 1000), prompt_tokens=1)  # ratio 1000, absurd
    cm.calibrate(_messages("x" * 1000), prompt_tokens=5000)  # ratio 0.2, absurd

    assert cm.calibration_samples == 0
    assert cm.chars_per_token == pytest.approx(4.0)


def test_calibration_is_resilient_to_one_outlier() -> None:
    """A single anomalous sample cannot swing an established ratio far."""
    cm = ContextManager()
    for _ in range(10):
        cm.calibrate(_messages("x" * 1000), prompt_tokens=500)  # settled at 2.0
    before = cm.chars_per_token

    # One outlier at the top of the plausible band.
    cm.calibrate(_messages("x" * 1200), prompt_tokens=100)  # ratio 12.0

    # Moved, but nowhere near the outlier — the moving average damps it.
    assert cm.chars_per_token > before
    assert cm.chars_per_token < 6.0


def test_calibration_counts_tool_call_arguments() -> None:
    """Calibration measures the same content estimate_context_tokens measures."""
    cm = ContextManager()
    messages = [
        ChatMessage(role="user", content="x" * 500),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="c1", name="grep", arguments={"pattern": "y" * 400})],
        ),
    ]
    # Both sides of the calibration see the tool-call arguments, so the derived
    # ratio applied back to the same messages reproduces the token count.
    cm.calibrate(messages, prompt_tokens=250)
    assert cm.estimate_context_tokens(messages) == pytest.approx(250, abs=2)


def test_should_compact_follows_calibrated_estimate() -> None:
    """Calibration changes when compaction fires — the point of the fix."""
    cm = ContextManager(max_window_tokens=1000, compaction_threshold=0.75)
    messages = _messages("x" * 3200)

    # Under the 4.0 bootstrap: 800 tokens, over the 750 threshold.
    assert cm.should_compact(cm.estimate_context_tokens(messages))

    # The model actually tokenizes far more densely (8 chars/token) => 400
    # tokens, comfortably under threshold. Compaction must no longer fire.
    cm.calibrate(_messages("x" * 8000), prompt_tokens=1000)
    assert not cm.should_compact(cm.estimate_context_tokens(messages))


# --------------------------------------------------------------------------
# Fix 2 — undo reverts a whole operation, not one file
# --------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> CheckpointStore:
    """A checkpoint store over a non-git temp workspace (copy storage)."""
    return CheckpointStore(tmp_path)


async def test_undo_reverts_every_file_of_one_operation(
    tmp_path: Path, store: CheckpointStore
) -> None:
    """A multi-file operation reverts in a single undo.

    Regression: undo popped one checkpoint, so reverting a 3-file write took 3
    undo calls and left the workspace half-reverted after the first.
    """
    files: list[Path] = []
    for i in range(3):
        p = tmp_path / f"f{i}.txt"
        p.write_text("original", encoding="utf-8")
        files.append(p)

    async with store.operation():
        for p in files:
            await store.snapshot(p)
            p.write_text("modified", encoding="utf-8")

    assert store.operation_count == 1
    assert store.checkpoint_count == 3

    result = await store.undo()

    assert all(p.read_text(encoding="utf-8") == "original" for p in files)
    assert "3 file(s)" in result
    assert store.checkpoint_count == 0


async def test_undo_without_operation_context_is_per_snapshot(
    tmp_path: Path, store: CheckpointStore
) -> None:
    """Snapshots outside an operation context stay independently undoable."""
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("a0", encoding="utf-8")
    b.write_text("b0", encoding="utf-8")

    await store.snapshot(a)
    a.write_text("a1", encoding="utf-8")
    await store.snapshot(b)
    b.write_text("b1", encoding="utf-8")

    assert store.operation_count == 2

    await store.undo()
    assert b.read_text(encoding="utf-8") == "b0"
    assert a.read_text(encoding="utf-8") == "a1"  # untouched by this undo

    await store.undo()
    assert a.read_text(encoding="utf-8") == "a0"


async def test_undo_deletes_files_that_did_not_exist_before(
    tmp_path: Path, store: CheckpointStore
) -> None:
    """Undoing a create removes the file rather than restoring empty content."""
    created = tmp_path / "new.txt"
    async with store.operation():
        await store.snapshot(created)
        created.write_text("fresh", encoding="utf-8")

    assert created.exists()
    await store.undo()
    assert not created.exists()


async def test_nested_operation_contexts_form_one_unit(
    tmp_path: Path, store: CheckpointStore
) -> None:
    """A nested operation joins the outer one instead of splitting it."""
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("a0", encoding="utf-8")
    b.write_text("b0", encoding="utf-8")

    async with store.operation():
        await store.snapshot(a)
        a.write_text("a1", encoding="utf-8")
        async with store.operation():
            await store.snapshot(b)
            b.write_text("b1", encoding="utf-8")

    assert store.operation_count == 1
    await store.undo()
    assert a.read_text(encoding="utf-8") == "a0"
    assert b.read_text(encoding="utf-8") == "b0"


async def test_undo_on_empty_store_reports_nothing_to_undo(store: CheckpointStore) -> None:
    """An untouched store says there is nothing to undo, without claiming loss."""
    result = await store.undo()
    assert result == "No checkpoints to undo."


# --------------------------------------------------------------------------
# Fix 3 — eviction drops whole operations and is visible to the caller
# --------------------------------------------------------------------------


async def test_eviction_drops_whole_operations_never_a_partial_one(tmp_path: Path) -> None:
    """Retention never leaves half an operation behind.

    Regression: eviction popped individual checkpoints, so an operation could
    lose some of its files and undo would half-revert it.
    """
    store = CheckpointStore(tmp_path, max_checkpoints=4)

    for op in range(3):
        async with store.operation():
            for i in range(2):
                p = tmp_path / f"op{op}_f{i}.txt"
                p.write_text("v0", encoding="utf-8")
                await store.snapshot(p)
                p.write_text("v1", encoding="utf-8")

    # 6 snapshots taken, cap is 4 => the oldest whole operation was dropped.
    assert store.evicted_operations == 1
    assert store.checkpoint_count == 4
    # Every surviving operation is intact (2 files each), never a partial one.
    assert store.operation_count == 2


async def test_eviction_by_byte_budget_drops_whole_operations(tmp_path: Path) -> None:
    """The byte bound evicts by operation too, not by individual snapshot."""
    store = CheckpointStore(tmp_path, max_bytes=100)

    for op in range(3):
        async with store.operation():
            p = tmp_path / f"big{op}.txt"
            p.write_text("x" * 80, encoding="utf-8")
            await store.snapshot(p)
            p.write_text("y" * 80, encoding="utf-8")

    assert store.evicted_operations >= 1
    assert sum(1 for _ in range(store.checkpoint_count)) <= 2


async def test_undo_reports_truncated_history_after_eviction(tmp_path: Path) -> None:
    """When retention dropped history, undo says so instead of staying silent.

    Regression: an evicted checkpoint left the caller with a bare "nothing to
    undo", indistinguishable from a workspace that was never written to.
    """
    store = CheckpointStore(tmp_path, max_checkpoints=1)

    for op in range(3):
        async with store.operation():
            p = tmp_path / f"f{op}.txt"
            p.write_text("v0", encoding="utf-8")
            await store.snapshot(p)
            p.write_text("v1", encoding="utf-8")

    assert store.evicted_operations == 2

    # The one surviving operation undoes, and warns no further undo remains.
    first = await store.undo()
    assert "No further undo available" in first
    assert "2 older" in first

    # Subsequent undo distinguishes "truncated" from "never had anything".
    second = await store.undo()
    assert "dropped by checkpoint retention" in second
    assert second != "No checkpoints to undo."


async def test_evicted_copy_files_are_removed_from_disk(tmp_path: Path) -> None:
    """Copy-storage eviction reclaims disk rather than leaking snapshot files."""
    store = CheckpointStore(tmp_path, max_checkpoints=1)
    checkpoints_dir = tmp_path / ".nullain" / "checkpoints"

    for op in range(3):
        async with store.operation():
            p = tmp_path / f"f{op}.txt"
            p.write_text("v0", encoding="utf-8")
            await store.snapshot(p)
            p.write_text("v1", encoding="utf-8")

    surviving = [p for p in checkpoints_dir.rglob("*") if p.is_file()]
    assert len(surviving) == store.checkpoint_count
