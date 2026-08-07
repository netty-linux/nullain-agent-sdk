"""Nullain Agent SDK — Rich-based terminal renderer for interactive CLI use.

Consumes the event stream from ``Agent.stream()`` (already emitted by
``AgentLoop`` for every tool call, tool result, streamed token delta, spec
creation/verification, and error) and renders it live: streamed text grows in
place, tool calls show a spinner that resolves to a check or cross, and
write_file/edit_file results are rendered as colored diffs — the pieces most
visibly missing next to Claude Code / Gemini CLI's terminal UX compared to the
prior plain ``print()``-based CLI.

This module is presentation-only: it never touches ``AgentLoop`` internals,
only the public event stream, so it stays decoupled from engine changes.
"""

from __future__ import annotations

import contextlib
import difflib
import io
import sys
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from typing import TextIO

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from nullain.agent import RunResult
from nullain.events import (
    BaseEvent,
    ErrorEvent,
    ModelResponseEvent,
    SpecCreatedEvent,
    SpecVerifiedEvent,
    StreamDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)

#: Tool names whose arguments carry a file path worth diffing on result.
_FILE_WRITE_TOOLS = frozenset({"write_file", "edit_file", "multi_edit"})
#: Argument keys checked (in order) for a tool call's target file path,
#: mirroring the same heuristic ToolRegistry / AgentLoop already use.
_PATH_ARG_KEYS = ("file_path", "target_file", "path")


def _legacy_safe_stdout() -> TextIO:
    """Return stdout, reconfigured to tolerate unencodable characters.

    A legacy Windows console's codepage (e.g. cp1252) cannot encode most
    emoji or many Unicode symbols — and unlike the glyphs this module
    controls itself (spinner frames, ✓/✗/⚠, chosen based on
    ``Console.legacy_windows`` at call sites), the *model's own output* can
    contain any character. Rich's default stdout write would raise
    ``UnicodeEncodeError`` and crash the whole CLI on an emoji in an
    otherwise-successful response — an unacceptable failure mode for text
    this module has no control over.

    Reconfigures stdout's error handler to ``"replace"`` (unencodable
    characters become ``?``) when the encoding is not already
    Unicode-capable (UTF-8, or any UTF variant) and stdout supports
    reconfiguration (a plain ``io.TextIOWrapper`` — true for the standard
    interpreter's real stdout, not necessarily for one replaced by a test
    harness or an unusual embedding). A real UTF-8 terminal, or a
    non-reconfigurable stdout, is returned unchanged — Rich's own error
    handling is the fallback in that case.
    """
    stream = sys.stdout
    encoding = (stream.encoding or "").lower().replace("-", "")
    if not encoding.startswith("utf") and isinstance(stream, io.TextIOWrapper):
        with contextlib.suppress(AttributeError, ValueError, io.UnsupportedOperation):
            stream.reconfigure(errors="replace")
    return cast("TextIO", stream)


def _tool_call_arg_path(arguments: dict[str, Any] | str) -> str | None:
    if not isinstance(arguments, dict):
        return None
    for key in _PATH_ARG_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return None


class TUIRenderer:
    """Renders one ``Agent.stream()`` run live to the terminal via Rich.

    Usage::

        renderer = TUIRenderer()
        async for item in agent.stream(prompt):
            renderer.handle(item)
        renderer.finish()

    A fresh instance should be used per run (or call :meth:`reset` between
    runs in a chat loop) — it accumulates streamed text and per-call state
    that only makes sense within one run's event sequence.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console(file=_legacy_safe_stdout())
        # Legacy Windows consoles (cmd.exe / older PowerShell without UTF-8
        # mode) use a codepage like cp1252 that cannot encode Braille spinner
        # frames or the ✓/✗/⚠ glyphs — printing them crashes with a
        # UnicodeEncodeError rather than degrading gracefully. Detect via the
        # same signal Rich itself uses for legacy Windows rendering and fall
        # back to ASCII-safe equivalents for the glyphs this module controls.
        # _legacy_safe_stdout() (above) separately covers text this module
        # does NOT control — the model's own output, which can contain any
        # emoji or Unicode character — by replacing unencodable characters
        # instead of raising.
        ascii_only = self.console.legacy_windows
        self._spinner_name = "line" if ascii_only else "dots"
        self._ok_marker = "OK" if ascii_only else "✓"
        self._fail_marker = "X" if ascii_only else "✗"
        self._warn_marker = "!" if ascii_only else "⚠"
        # Status dot for the Claude-Code-style tool-call line: colored ● for
        # a resolved call (green/red), dim ● while pending — ASCII-safe "o"
        # on a legacy Windows console that can't encode "●".
        self._dot = "o" if ascii_only else "●"
        self._detail_prefix = "  - " if ascii_only else "  └ "
        self._reset_state()

    def _reset_state(self) -> None:
        self._streamed_text = ""
        self._pending_calls: dict[str, ToolCallEvent] = {}
        self._live: Live | None = None
        # Snapshot of file content right before a write/edit call resolves,
        # keyed by call_id, so the ToolResultEvent can render a diff against
        # what the file looked like before this call. Populated by the
        # ToolCallEvent handler (before the tool has actually run) — good
        # enough for CLI display purposes even though it re-reads from disk
        # rather than being fed the exact pre-write bytes.
        self._pre_write_snapshots: dict[str, str] = {}
        # Run-of-same-tool collapsing (Claude Code's style): consecutive
        # calls to the *same* tool name update one live line in place
        # instead of each printing its own permanent line — a loop of 20
        # list_directory calls exploring a project would otherwise dump 20
        # lines into the chat. Tracks the repeated tool's name and how many
        # times it has resolved in the current streak; reset (streak
        # finalized as "name xN") the moment a different tool starts or an
        # error breaks the streak.
        self._streak_tool: str | None = None
        self._streak_count: int = 0

    def reset(self) -> None:
        """Clear accumulated state between runs in a long-lived chat loop."""
        self._reset_state()

    def handle(self, item: BaseEvent | RunResult) -> None:
        """Render one item from ``Agent.stream()``."""
        if isinstance(item, RunResult):
            self._handle_result(item)
            return
        if isinstance(item, StreamDeltaEvent):
            self._handle_stream_delta(item)
        elif isinstance(item, ModelResponseEvent):
            self._handle_model_response(item)
        elif isinstance(item, ToolCallEvent):
            self._handle_tool_call(item)
        elif isinstance(item, ToolResultEvent):
            self._handle_tool_result(item)
        elif isinstance(item, SpecCreatedEvent):
            self._handle_spec_created(item)
        elif isinstance(item, SpecVerifiedEvent):
            self._handle_spec_verified(item)
        elif isinstance(item, ErrorEvent):
            self._handle_error(item)
        # Other event types (compaction, workflow, todo) are not yet
        # rendered — silently ignored rather than raising, since new event
        # types may be added to the engine without every renderer needing an
        # immediate update.

    def finish(self) -> None:
        """Flush any open live display. Call after the stream is exhausted."""
        self._finalize_streak()

    # -- streamed text -----------------------------------------------------

    def _ensure_live(self) -> Live:
        if self._live is None:
            self._live = Live(console=self.console, refresh_per_second=12, transient=False)
            self._live.start()
        return self._live

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _handle_stream_delta(self, ev: StreamDeltaEvent) -> None:
        if not self._streamed_text:
            # First delta of this turn: finalize any pending tool-call
            # streak first, so its summary line prints before the model's
            # text starts, rather than the streak's live line being
            # silently overwritten by the incoming Markdown.
            self._finalize_streak()
        self._streamed_text += ev.delta
        live = self._ensure_live()
        live.update(Markdown(self._streamed_text))

    def _handle_model_response(self, ev: ModelResponseEvent) -> None:
        # Non-streaming runs (or the final chunk of a streaming one) deliver
        # the full text in one ModelResponseEvent rather than deltas; render
        # it the same way if nothing was streamed yet.
        if ev.content and not self._streamed_text:
            self._finalize_streak()
            live = self._ensure_live()
            live.update(Markdown(ev.content))
        # A tool-call step means more output is coming (tool result, then
        # another model step) — close this live block so the tool call's own
        # panel renders below it rather than overwriting it.
        if ev.tool_calls:
            self._stop_live()
            self._streamed_text = ""

    # -- tool calls ----------------------------------------------------------

    def _handle_tool_call(self, ev: ToolCallEvent) -> None:
        self._pending_calls[ev.call_id] = ev
        if ev.tool_name in _FILE_WRITE_TOOLS:
            path = _tool_call_arg_path(ev.arguments)
            if path:
                self._pre_write_snapshots[ev.call_id] = _read_text_safe(path)
        if ev.tool_name != self._streak_tool:
            # A different tool than the current streak — finalize the
            # previous streak's line (if any) as "name xN" and start fresh.
            self._finalize_streak()
            self._streak_tool = ev.tool_name
            self._streak_count = 0
        # Claude Code's status-line style: a dim ● while pending, a bold
        # tool name, and a dim detail (path/command) beside it. Resolved in
        # place to a colored ●/✓/✗ line by _handle_tool_result — not a
        # bordered panel, which drowned the "what ran, did it work" signal
        # in a chat full of tool calls. The one thing worth keeping full
        # detail for is a file diff, kept below for write/edit tools.
        live = self._ensure_live()
        live.update(_tool_line(self._dot, "dim", ev.tool_name, _tool_call_detail(ev)))

    def _handle_tool_result(self, ev: ToolResultEvent) -> None:
        call = self._pending_calls.pop(ev.call_id, None)
        style = "red" if ev.is_error else "green"
        detail = _tool_call_detail(call) if call is not None else None
        line = _tool_line(self._dot, style, ev.tool_name, detail)
        self._streak_count += 1

        if not ev.is_error and call is not None and ev.tool_name in _FILE_WRITE_TOOLS:
            path = _tool_call_arg_path(call.arguments)
            if path:
                before = self._pre_write_snapshots.pop(call.call_id, "")
                after = _read_text_safe(path)
                diff_renderable = _render_diff(before, after, path)
                if diff_renderable is not None:
                    # A diff is always worth its own permanent line — never
                    # collapsed into a streak counter even if the same
                    # write/edit tool repeats.
                    self._finalize_streak()
                    self._stop_live()
                    self.console.print(line)
                    self.console.print(Panel(diff_renderable, border_style=style))
                    return

        if ev.is_error:
            # An error always ends the streak and gets its own permanent
            # line + detail — collapsing a failure into a "xN" count would
            # hide exactly the thing worth seeing.
            self._finalize_streak()
            live = self._ensure_live()
            live.update(line)
            self._stop_live()
            # A one-line summary of *why* it failed, indented under the
            # status line — full output remains available via
            # ToolResultEvent for anything consuming the raw event stream.
            detail_text = f"{self._detail_prefix}{_truncate(ev.output, limit=300)}"
            self.console.print(Text(detail_text, style="dim red"))
            return

        # Success, same tool as the current streak: update the one live
        # line in place (showing this call's own detail) rather than
        # printing a new permanent line — this is what collapses a run of
        # 20 list_directory calls into a single animated line instead of
        # 20 stacked ones. Finalized as "name xN" once a different tool
        # starts, an error breaks the streak, or the run ends.
        live = self._ensure_live()
        live.update(line)

    def _finalize_streak(self) -> None:
        """Print the current tool streak's permanent summary line, if any.

        A streak of exactly one call prints its own full detail (the
        common case — most tool calls aren't repeated); a longer streak
        collapses to "● name xN" since the per-call detail (which file,
        which path) has already scrolled past as the live line updated.
        """
        if self._streak_tool is None or self._streak_count == 0:
            self._stop_live()
            return
        if self._streak_count > 1:
            style = "green"
            line = Text()
            line.append(f"{self._dot} ", style=style)
            line.append(self._streak_tool, style="bold")
            line.append(f"  x{self._streak_count}", style="dim")
            self._stop_live()
            self.console.print(line)
        else:
            # Exactly one call: the live line already shows full detail —
            # stopping Live leaves that single frame printed as-is.
            self._stop_live()
        self._streak_tool = None
        self._streak_count = 0

    # -- spec / verify ---------------------------------------------------------

    def _handle_spec_created(self, ev: SpecCreatedEvent) -> None:
        self._finalize_streak()
        # ev.title is model-generated text, not trusted markup — build the
        # body with Text so any literal "[...]" in it renders as plain text
        # instead of being interpreted as (or breaking) Rich markup.
        body = Text()
        body.append(ev.title, style="bold")
        for i, s in enumerate(ev.steps):
            body.append(f"\n  {i + 1}. {s}")
        self.console.print(Panel(body, title="Plan", border_style="cyan"))

    def _handle_spec_verified(self, ev: SpecVerifiedEvent) -> None:
        self._finalize_streak()
        style = "green" if ev.success else "yellow"
        marker = self._ok_marker if ev.success else self._warn_marker
        self.console.print(Panel(Text(ev.feedback), title=f"{marker} Verify", border_style=style))

    def _handle_error(self, ev: ErrorEvent) -> None:
        self._finalize_streak()
        self.console.print(
            Panel(
                Text(ev.message), title=f"{self._fail_marker} {ev.error_type}", border_style="red"
            )
        )

    # -- terminal result -----------------------------------------------------

    def _handle_result(self, result: RunResult) -> None:
        self._finalize_streak()
        if result.status == "success":
            return  # final answer was already rendered via streaming above
        style = "yellow" if result.status == "verification_failed" else "red"
        message = result.error or result.feedback or f"Run ended: {result.status}"
        # Avoid literal brackets in the title — Rich's console markup parses
        # "[...]" as a style tag, so "[max_steps]" would silently vanish
        # instead of rendering as text.
        self.console.print(
            Panel(Text(message), title=f"Status: {result.status}", border_style=style)
        )


def _tool_call_detail(ev: ToolCallEvent) -> str | None:
    """A short, dim detail string beside the bold tool name (path or command).

    Returns None when the call carries nothing worth showing beyond its
    name (e.g. ``todo_write``, ``git_status``) — the caller omits the
    detail segment entirely rather than printing an empty one.
    """
    path = _tool_call_arg_path(ev.arguments)
    if path:
        return path
    if isinstance(ev.arguments, dict) and "command_args" in ev.arguments:
        raw: object = ev.arguments["command_args"]
        if isinstance(raw, list):
            parts = cast(list[Any], raw)
            return " ".join(str(x) for x in parts)
    return None


def _tool_line(dot: str, style: str, tool_name: str, detail: str | None) -> Text:
    """Build one Claude-Code-style status line: colored ● + bold name + dim detail.

    ``tool_name``/``detail`` may come from an MCP-declared tool or
    model-supplied arguments, not just trusted built-ins — appended as
    plain segments (never f-string-interpolated into markup) so neither
    can be misread as Rich console markup.
    """
    line = Text()
    line.append(f"{dot} ", style=style)
    line.append(tool_name, style="bold")
    if detail:
        line.append(f"  {detail}", style="dim")
    return line


def _read_text_safe(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _render_diff(before: str, after: str, path: str) -> Syntax | Group | None:
    """Render a unified-style diff between before/after file content.

    Returns None when there is no difference to show (defensive — the caller
    only reaches here after a successful write, so this should not normally
    happen).
    """
    if before == after:
        return None
    diff_lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            lineterm="",
            n=2,
        )
    )
    if not diff_lines:
        return None
    # Skip the two `---`/`+++` header lines from unified_diff; the panel
    # title already names the file.
    body_lines = [line for line in diff_lines if not line.startswith(("---", "+++"))]
    diff_text = "\n".join(body_lines)
    return Syntax(diff_text, "diff", theme="ansi_dark", word_wrap=True)


def _truncate(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


__all__ = ["TUIRenderer"]
