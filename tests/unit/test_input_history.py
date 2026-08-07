"""Tests for session-scoped chat command history (InputHistory)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from nullain import input_history


def test_prompt_returns_input_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    """On POSIX, prompt() delegates straight to input() — readline (imported
    at module load) handles line editing/history transparently, so
    InputHistory's own logic is only exercised for the platform switch and
    the internal entry tracking."""
    monkeypatch.setattr(input_history.sys, "platform", "linux")
    monkeypatch.setattr("builtins.input", lambda prompt="": "hello world")

    history = input_history.InputHistory()
    assert history.prompt("> ") == "hello world"


def test_blank_lines_are_not_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors shell behavior: hitting Enter on an empty line doesn't
    pollute history with empty entries."""
    monkeypatch.setattr(input_history.sys, "platform", "linux")
    answers: Iterator[str] = iter(["first", "", "   ", "second"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    history = input_history.InputHistory()
    for _ in range(4):
        history.prompt("> ")

    assert history._entries == ["first", "second"]  # type: ignore[reportPrivateUsage]


def test_windows_path_used_only_when_stdin_is_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """The raw msvcrt line editor is only engaged on Windows AND when stdin
    is a real interactive terminal — piped input (CI, non-interactive
    callers) must fall back to plain input(), the same non-TTY fallback
    prompt_select.select() already uses, rather than hanging on a raw-mode
    read that can never receive a real keypress."""
    monkeypatch.setattr(input_history.sys, "platform", "win32")
    monkeypatch.setattr(input_history.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "piped line")

    history = input_history.InputHistory()
    assert history.prompt("> ") == "piped line"


def test_windows_raw_editor_typing_and_enter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typing characters then Enter returns the typed line and records it."""
    monkeypatch.setattr(input_history.sys, "platform", "win32")
    monkeypatch.setattr(input_history.sys.stdin, "isatty", lambda: True)

    keys: Iterator[bytes] = iter([b"h", b"i", b"\r"])

    class _FakeMsvcrt:
        @staticmethod
        def getch() -> bytes:
            return next(keys)

    monkeypatch.setattr(input_history, "msvcrt", _FakeMsvcrt(), raising=False)

    history = input_history.InputHistory()
    assert history.prompt("> ") == "hi"
    assert history._entries == ["hi"]  # type: ignore[reportPrivateUsage]


def test_windows_raw_editor_backspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backspace removes the character before the cursor."""
    monkeypatch.setattr(input_history.sys, "platform", "win32")
    monkeypatch.setattr(input_history.sys.stdin, "isatty", lambda: True)

    keys: Iterator[bytes] = iter([b"h", b"i", b"x", b"\x08", b"\r"])

    class _FakeMsvcrt:
        @staticmethod
        def getch() -> bytes:
            return next(keys)

    monkeypatch.setattr(input_history, "msvcrt", _FakeMsvcrt(), raising=False)

    history = input_history.InputHistory()
    assert history.prompt("> ") == "hi"


def test_windows_raw_editor_up_recalls_previous_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Up arrow recalls the most recent history entry into the buffer."""
    monkeypatch.setattr(input_history.sys, "platform", "win32")
    monkeypatch.setattr(input_history.sys.stdin, "isatty", lambda: True)

    # First prompt: type "first", Enter.
    first_keys: Iterator[bytes] = iter([b"f", b"i", b"r", b"s", b"t", b"\r"])

    class _FakeMsvcrt1:
        @staticmethod
        def getch() -> bytes:
            return next(first_keys)

    monkeypatch.setattr(input_history, "msvcrt", _FakeMsvcrt1(), raising=False)
    history = input_history.InputHistory()
    assert history.prompt("> ") == "first"

    # Second prompt: Up (recalls "first"), Enter (accepts the recalled line).
    second_keys: Iterator[bytes] = iter([b"\x00", b"H", b"\r"])

    class _FakeMsvcrt2:
        @staticmethod
        def getch() -> bytes:
            return next(second_keys)

    monkeypatch.setattr(input_history, "msvcrt", _FakeMsvcrt2(), raising=False)
    assert history.prompt("> ") == "first"


def test_windows_raw_editor_ctrl_c_raises_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(input_history.sys, "platform", "win32")
    monkeypatch.setattr(input_history.sys.stdin, "isatty", lambda: True)

    class _FakeMsvcrt:
        @staticmethod
        def getch() -> bytes:
            return b"\x03"

    monkeypatch.setattr(input_history, "msvcrt", _FakeMsvcrt(), raising=False)

    history = input_history.InputHistory()
    with pytest.raises(KeyboardInterrupt):
        history.prompt("> ")


def test_windows_raw_editor_ctrl_d_on_empty_buffer_raises_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(input_history.sys, "platform", "win32")
    monkeypatch.setattr(input_history.sys.stdin, "isatty", lambda: True)

    class _FakeMsvcrt:
        @staticmethod
        def getch() -> bytes:
            return b"\x04"

    monkeypatch.setattr(input_history, "msvcrt", _FakeMsvcrt(), raising=False)

    history = input_history.InputHistory()
    with pytest.raises(EOFError):
        history.prompt("> ")
