"""Nullain Agent SDK — Chat prompt input with session-scoped command history.

Plain ``input()`` has no history: every CLI coder worth comparing against
(Claude Code, Codex, Gemini CLI) lets Up/Down recall a previous prompt in
the same session, the same way a shell does. On POSIX, importing the
stdlib ``readline`` module before calling ``input()`` gets this for free —
``readline`` patches ``input()`` itself to add line editing and, once
entries are added via ``add_history``, Up/Down recall. Windows has no
stdlib ``readline`` (and ``pyreadline3`` is not a project dependency), so
this module also carries a minimal raw-mode line editor built on
``msvcrt``, matching :mod:`nullain.prompt_select`'s existing cross-platform
approach (stdlib-only, no new dependency) rather than special-casing one
platform to feel worse than the others.

History is session-scoped only (an in-memory list, never written to disk):
closing the chat process discards it. That's a deliberate choice, not a
missing feature — nothing here should risk persisting a prompt containing
something sensitive to a file the user didn't ask for.
"""

from __future__ import annotations

import sys

if sys.platform == "win32":
    import msvcrt
else:
    # Imported for its side effect on input(): readline patches input()
    # itself to add line editing and Up/Down history recall. Never
    # referenced directly, hence the lint/typecheck suppressions — ruff's
    # F401 (unused import) and pyright's reportUnusedImport both flag it
    # otherwise, with no way to express "imported for its side effect"
    # more directly for a stdlib module.
    import readline as _readline  # noqa: F401  # pyright: ignore[reportUnusedImport]


class InputHistory:
    """Session-scoped command history backing a ``prompt()`` call in a chat loop.

    Usage::

        history = InputHistory()
        while True:
            line = history.prompt("> ")
    """

    def __init__(self) -> None:
        self._entries: list[str] = []

    def prompt(self, prompt: str) -> str:
        """Read one line, with Up/Down recalling this session's prior entries.

        Blank lines are read (and returned) but never added to history —
        mirrors shell behavior, where repeatedly hitting Enter doesn't
        pollute history with empty entries. Raises ``EOFError`` on Ctrl-D /
        closed stdin, same as ``input()``, for the caller's existing
        handling to catch.
        """
        use_raw_editor = sys.platform == "win32" and sys.stdin.isatty()
        line = _prompt_windows(prompt, self._entries) if use_raw_editor else input(prompt)
        if line.strip():
            self._entries.append(line)
        return line


def _prompt_windows(prompt: str, history: list[str]) -> str:
    """Raw-mode line editor for Windows: typing, backspace, left/right, Up/Down history.

    pyright type-checks this repo against a single host platform (no
    ``pythonPlatform`` pin in ``pyproject.toml``, same situation documented
    in :mod:`nullain.prompt_select`) — this branch is dead code on
    Linux/macOS CI runners where ``msvcrt`` is unresolvable, and the
    Windows runner (and Windows dev machines) type-check it for real.
    """
    sys.stdout.write(prompt)
    sys.stdout.flush()
    buf: list[str] = []
    cursor = 0
    hist_idx = len(history)  # one past the end == "not currently browsing history"

    def redraw() -> None:
        # \r returns to column 0; \x1b[K clears from cursor to end of line —
        # avoids leftover characters when the new line is shorter than what
        # was there before (e.g. after Backspace or recalling a shorter
        # history entry).
        sys.stdout.write("\r" + prompt + "".join(buf) + "\x1b[K")
        if cursor < len(buf):
            sys.stdout.write(f"\x1b[{len(buf) - cursor}D")
        sys.stdout.flush()

    while True:
        ch = msvcrt.getch()  # pyright: ignore[reportUndefinedVariable, reportUnknownVariableType, reportUnknownMemberType]
        if ch in (b"\x00", b"\xe0"):  # arrow/function key prefix
            ch2 = msvcrt.getch()  # pyright: ignore[reportUndefinedVariable, reportUnknownVariableType, reportUnknownMemberType]
            if ch2 == b"H" and hist_idx > 0:  # up
                hist_idx -= 1
                buf = list(history[hist_idx])
                cursor = len(buf)
                redraw()
            elif ch2 == b"P":  # down
                if hist_idx < len(history) - 1:
                    hist_idx += 1
                    buf = list(history[hist_idx])
                elif hist_idx == len(history) - 1:
                    hist_idx += 1
                    buf = []
                else:
                    continue
                cursor = len(buf)
                redraw()
            elif ch2 == b"K" and cursor > 0:  # left
                cursor -= 1
                redraw()
            elif ch2 == b"M" and cursor < len(buf):  # right
                cursor += 1
                redraw()
            continue
        if ch == b"\r":
            sys.stdout.write("\n")
            return "".join(buf)
        if ch == b"\x03":  # Ctrl-C
            raise KeyboardInterrupt
        if ch in (b"\x04",):  # Ctrl-D (EOF)
            if not buf:
                raise EOFError
            continue
        if ch == b"\x08":  # Backspace
            if cursor > 0:
                del buf[cursor - 1]
                cursor -= 1
                redraw()
            continue
        try:
            decoded: str = ch.decode("utf-8")  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        except UnicodeDecodeError:
            continue
        if decoded.isprintable():
            buf.insert(cursor, decoded)
            cursor += 1
            redraw()


__all__ = ["InputHistory"]
