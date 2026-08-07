"""Tests for the arrow-key interactive selection prompt."""

from __future__ import annotations

import io
from collections.abc import Iterator

import pytest
from nullain import prompt_select
from rich.console import Console


def test_select_fallback_picks_numbered_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-interactive stdin (piped input, CI) uses the numbered fallback."""
    monkeypatch.setattr(prompt_select.sys.stdin, "isatty", lambda: False)
    answers = iter(["2"])

    def _input(prompt: str = "") -> str:
        return next(answers)

    monkeypatch.setattr("builtins.input", _input)

    result = prompt_select.select("Allow bash?", ["Yes", "No", "Yes, always"])
    assert result == 1


def test_select_fallback_rejects_out_of_range_then_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prompt_select.sys.stdin, "isatty", lambda: False)
    answers = iter(["9", "abc", "3"])

    def _input(prompt: str = "") -> str:
        return next(answers)

    monkeypatch.setattr("builtins.input", _input)

    result = prompt_select.select("Allow bash?", ["Yes", "No", "Yes, always"])
    assert result == 2


def test_select_fallback_eof_returns_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompt_select.sys.stdin, "isatty", lambda: False)

    def _raise(*_: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise)
    result = prompt_select.select("Allow bash?", ["Yes", "No", "Yes, always"])
    assert result == 0


def test_select_interactive_navigates_and_confirms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Down, Down, Enter on a 3-option menu lands on index 2."""
    monkeypatch.setattr(prompt_select.sys.stdin, "isatty", lambda: True)
    keys: Iterator[str] = iter(["down", "down", "enter"])
    monkeypatch.setattr(prompt_select, "_read_key", lambda: next(keys))

    console = Console(file=io.StringIO(), force_terminal=True)
    result = prompt_select.select("Allow bash?", ["Yes", "No", "Yes, always"], console=console)
    assert result == 2


def test_select_interactive_wraps_around(monkeypatch: pytest.MonkeyPatch) -> None:
    """Up from the first option wraps to the last."""
    monkeypatch.setattr(prompt_select.sys.stdin, "isatty", lambda: True)
    keys: Iterator[str] = iter(["up", "enter"])
    monkeypatch.setattr(prompt_select, "_read_key", lambda: next(keys))

    console = Console(file=io.StringIO(), force_terminal=True)
    result = prompt_select.select("Allow bash?", ["Yes", "No", "Yes, always"], console=console)
    assert result == 2


def test_select_interactive_digit_jumps_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompt_select.sys.stdin, "isatty", lambda: True)
    keys: Iterator[str] = iter(["2"])
    monkeypatch.setattr(prompt_select, "_read_key", lambda: next(keys))

    console = Console(file=io.StringIO(), force_terminal=True)
    result = prompt_select.select("Allow bash?", ["Yes", "No", "Yes, always"], console=console)
    assert result == 1
