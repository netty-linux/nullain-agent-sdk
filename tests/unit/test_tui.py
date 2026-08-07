"""Unit tests for the Rich-based TUIRenderer (M15)."""

from pathlib import Path

from nullain.agent import RunResult
from nullain.events import (
    ErrorEvent,
    ModelResponseEvent,
    SpecCreatedEvent,
    SpecVerifiedEvent,
    StreamDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from nullain.llm import ToolCall
from nullain.tui import TUIRenderer
from rich.console import Console


def _renderer() -> tuple[TUIRenderer, Console]:
    """A renderer writing to an in-memory Console for output assertions."""
    console = Console(file=None, force_terminal=False, width=100, record=True)
    return TUIRenderer(console=console), console


def _text_out(console: Console) -> str:
    return console.export_text()


def test_stream_delta_accumulates_and_renders_text() -> None:
    renderer, console = _renderer()
    renderer.handle(StreamDeltaEvent(session_id="s", delta="Hello, ", model="m"))
    renderer.handle(StreamDeltaEvent(session_id="s", delta="world!", model="m"))
    renderer.finish()
    assert "Hello, world!" in _text_out(console)


def test_model_response_without_prior_stream_renders_full_text() -> None:
    """A non-streaming run delivers content in one ModelResponseEvent."""
    renderer, console = _renderer()
    renderer.handle(ModelResponseEvent(session_id="s", model="m", content="Done."))
    renderer.finish()
    assert "Done." in _text_out(console)


def test_tool_call_then_result_renders_success_panel() -> None:
    renderer, console = _renderer()
    renderer.handle(
        ToolCallEvent(
            session_id="s", call_id="c1", tool_name="read_file", arguments={"path": "a.txt"}
        )
    )
    renderer.handle(
        ToolResultEvent(
            session_id="s", call_id="c1", tool_name="read_file", output="1\thello", is_error=False
        )
    )
    renderer.finish()
    out = _text_out(console)
    assert "read_file" in out
    assert renderer._ok_marker in out  # type: ignore[reportPrivateUsage]


def test_tool_result_error_renders_error_panel() -> None:
    renderer, console = _renderer()
    renderer.handle(
        ToolCallEvent(
            session_id="s", call_id="c1", tool_name="bash", arguments={"command_args": ["x"]}
        )
    )
    renderer.handle(
        ToolResultEvent(
            session_id="s",
            call_id="c1",
            tool_name="bash",
            output="command not found",
            is_error=True,
        )
    )
    renderer.finish()
    out = _text_out(console)
    assert renderer._fail_marker in out  # type: ignore[reportPrivateUsage]
    assert "command not found" in out


def test_write_file_result_renders_diff(tmp_path: Path) -> None:
    """A write_file tool call/result pair renders a colored diff of the change."""
    target = tmp_path / "a.txt"
    target.write_text("line1\nline2\n")

    renderer, console = _renderer()
    # Snapshot is taken on the ToolCallEvent, before the write has happened.
    renderer.handle(
        ToolCallEvent(
            session_id="s",
            call_id="c1",
            tool_name="write_file",
            arguments={"path": str(target), "content": "line1\nCHANGED\n"},
        )
    )
    # Simulate the actual write happening between call and result.
    target.write_text("line1\nCHANGED\n")
    renderer.handle(
        ToolResultEvent(
            session_id="s",
            call_id="c1",
            tool_name="write_file",
            output="Successfully wrote 15 characters.",
            is_error=False,
        )
    )
    renderer.finish()
    out = _text_out(console)
    assert "CHANGED" in out
    assert "line2" in out  # removed line should still appear in the diff


def test_spec_created_renders_plan_panel() -> None:
    renderer, console = _renderer()
    renderer.handle(
        SpecCreatedEvent(
            session_id="s", spec_id="spec1", title="Do the thing", steps=("Step one", "Step two")
        )
    )
    renderer.finish()
    out = _text_out(console)
    assert "Do the thing" in out
    assert "Step one" in out


def test_spec_verified_success_and_failure_panels() -> None:
    renderer, console = _renderer()
    renderer.handle(
        SpecVerifiedEvent(session_id="s", spec_id="spec1", success=True, feedback="All good")
    )
    renderer.finish()
    assert "All good" in _text_out(console)

    renderer2, console2 = _renderer()
    renderer2.handle(
        SpecVerifiedEvent(session_id="s", spec_id="spec1", success=False, feedback="Missing file")
    )
    renderer2.finish()
    assert "Missing file" in _text_out(console2)


def test_error_event_renders_error_panel() -> None:
    renderer, console = _renderer()
    renderer.handle(ErrorEvent(session_id="s", error_type="TimeoutError", message="Loop timed out"))
    renderer.finish()
    out = _text_out(console)
    assert "TimeoutError" in out
    assert "Loop timed out" in out


def test_run_result_success_renders_nothing_extra() -> None:
    """A success RunResult is a no-op — the final text was already streamed."""
    renderer, console = _renderer()
    renderer.handle(StreamDeltaEvent(session_id="s", delta="Already shown", model="m"))
    renderer.handle(
        RunResult(session_id="s", status="success", success=True, final_text="Already shown")
    )
    renderer.finish()
    out = _text_out(console)
    assert out.count("Already shown") == 1  # not duplicated by the result handler


def test_run_result_failure_renders_status_panel() -> None:
    renderer, console = _renderer()
    renderer.handle(
        RunResult(session_id="s", status="max_steps", success=False, error="hit step cap")
    )
    renderer.finish()
    out = _text_out(console)
    assert "max_steps" in out
    assert "hit step cap" in out


def test_reset_clears_state_between_runs() -> None:
    renderer, console = _renderer()
    renderer.handle(StreamDeltaEvent(session_id="s", delta="First run text", model="m"))
    renderer.finish()
    renderer.reset()
    renderer.handle(StreamDeltaEvent(session_id="s", delta="Second run text", model="m"))
    renderer.finish()
    out = _text_out(console)
    assert "First run text" in out
    assert "Second run text" in out
    # After reset, the second run's live block should not have contained the
    # first run's already-flushed text (accumulation state must be cleared).
    assert "First run textSecond run text" not in out.replace("\n", "")


def test_unknown_event_type_does_not_raise() -> None:
    """Event types the renderer doesn't handle yet are silently ignored."""
    from nullain.events import TodoEvent, TodoItem

    renderer, _ = _renderer()
    renderer.handle(TodoEvent(session_id="s", items=(TodoItem(content="x", status="pending"),)))
    renderer.finish()  # no exception


def test_tool_call_with_command_args_describes_command() -> None:
    renderer, console = _renderer()
    renderer.handle(
        ToolCallEvent(
            session_id="s",
            call_id="c1",
            tool_name="bash",
            arguments={"command_args": ["pytest", "-q"]},
        )
    )
    renderer.finish()
    out = _text_out(console)
    assert "pytest -q" in out


def test_model_response_with_tool_calls_closes_live_block() -> None:
    """A ModelResponseEvent carrying tool_calls should not swallow streamed text
    it never rendered — verifies no crash when transitioning stream -> tool call."""
    renderer, console = _renderer()
    renderer.handle(StreamDeltaEvent(session_id="s", delta="thinking...", model="m"))
    renderer.handle(
        ModelResponseEvent(
            session_id="s",
            model="m",
            tool_calls=(ToolCall(id="c1", name="read_file", arguments={"path": "a.txt"}),),
        )
    )
    renderer.finish()
    out = _text_out(console)
    assert "thinking..." in out


def test_bracket_content_renders_literally_not_as_markup() -> None:
    """Regression: text containing literal "[...]" must render as-is, not be
    interpreted (and silently swallowed) as Rich console markup — this was a
    real bug where a RunResult status like "max_steps" rendered in a title
    as f"[{status}]" vanished entirely, since Rich parsed "[max_steps]" as an
    (invalid, so dropped) style tag rather than literal text."""
    renderer, console = _renderer()
    renderer.handle(
        RunResult(session_id="s", status="max_steps", success=False, error="[odd] bracket text")
    )
    renderer.finish()
    out = _text_out(console)
    assert "max_steps" in out
    assert "[odd] bracket text" in out

    renderer2, console2 = _renderer()
    renderer2.handle(
        ErrorEvent(session_id="s", error_type="ToolError", message="path [not found] here")
    )
    renderer2.finish()
    assert "path [not found] here" in _text_out(console2)

    renderer3, console3 = _renderer()
    renderer3.handle(
        SpecCreatedEvent(
            session_id="s", spec_id="spec1", title="Fix [bracketed] title", steps=("step [x]",)
        )
    )
    renderer3.finish()
    out3 = _text_out(console3)
    assert "Fix [bracketed] title" in out3
    assert "step [x]" in out3

    renderer4, console4 = _renderer()
    renderer4.handle(
        ToolCallEvent(
            session_id="s", call_id="c1", tool_name="read_file", arguments={"path": "[weird].txt"}
        )
    )
    renderer4.handle(
        ToolResultEvent(
            session_id="s",
            call_id="c1",
            tool_name="read_file",
            output="content with [brackets] in it",
            is_error=False,
        )
    )
    renderer4.finish()
    assert "content with [brackets] in it" in _text_out(console4)


def test_legacy_windows_console_uses_ascii_safe_markers() -> None:
    """Regression: on a legacy Windows console (cp1252 or similar), the Braille
    spinner frames and ✓/✗/⚠ glyphs cannot be encoded and previously crashed
    with UnicodeEncodeError instead of degrading gracefully. TUIRenderer must
    detect this (via Console.legacy_windows, the same signal Rich itself uses)
    and fall back to ASCII markers/spinner."""
    console = Console(legacy_windows=True, file=None, force_terminal=True, width=100, record=True)
    renderer = TUIRenderer(console=console)

    assert renderer._ok_marker == "OK"  # type: ignore[reportPrivateUsage]
    assert renderer._fail_marker == "X"  # type: ignore[reportPrivateUsage]
    assert renderer._warn_marker == "!"  # type: ignore[reportPrivateUsage]
    assert renderer._spinner_name == "line"  # type: ignore[reportPrivateUsage]

    renderer.handle(
        ToolCallEvent(
            session_id="s", call_id="c1", tool_name="read_file", arguments={"path": "a.txt"}
        )
    )
    renderer.handle(
        ToolResultEvent(
            session_id="s", call_id="c1", tool_name="read_file", output="1\thello", is_error=False
        )
    )
    renderer.finish()
    out = _text_out(console)
    assert "OK" in out
    assert "✓" not in out  # the Unicode checkmark must not appear
