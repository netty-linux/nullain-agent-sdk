"""Nullain Agent SDK — Arrow-key interactive selection prompt for the CLI.

Terminals cannot dispatch a real mouse click on plain text (that's a GUI-only
affordance), so "clickable" here means what Claude Code and similar CLIs
actually ship: an arrow-key-navigable list with the current choice
highlighted and Enter to confirm — no typing "y"/"n"/1/2/3 required, though
number keys are accepted as a fast path.

Cross-platform raw keyboard reading with no new dependency: ``msvcrt`` on
Windows, ``termios``/``tty`` on POSIX. Falls back to numbered plain-text
input when stdin isn't a real interactive terminal (piped input, CI, tests)
so scripted/non-interactive callers still work.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.text import Text

if sys.platform == "win32":
    import msvcrt
else:
    import termios
    import tty


def _read_key_windows() -> str:
    """Read one keypress on Windows, returning a normalized key name.

    pyright type-checks this repo against a single host platform (no
    ``pythonPlatform`` pin in ``pyproject.toml``): on Linux/macOS CI runners
    this whole branch is dead code as far as static analysis is concerned
    and ``msvcrt`` is unresolvable — the Windows CI runner (and Windows dev
    machines) type-check it for real. Ignored rather than worked around so a
    real regression on that platform still fails CI.
    """
    # fmt: off
    ch = msvcrt.getch()  # pyright: ignore[reportUndefinedVariable, reportUnknownVariableType, reportUnknownMemberType]
    if ch in (b"\x00", b"\xe0"):  # arrow/function key prefix
        ch2 = msvcrt.getch()  # pyright: ignore[reportUndefinedVariable, reportUnknownVariableType, reportUnknownMemberType]
        return {b"H": "up", b"P": "down"}.get(ch2, "")  # pyright: ignore[reportUnknownArgumentType]
    if ch == b"\r":
        return "enter"
    if ch == b"\x03":  # Ctrl-C
        raise KeyboardInterrupt
    try:
        return ch.decode("utf-8", errors="ignore")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    except UnicodeDecodeError:
        return ""
    # fmt: on


def _read_key_posix() -> str:
    """Read one keypress on POSIX, returning a normalized key name.

    pyright type-checks this repo against a single host platform (no
    ``pythonPlatform`` pin in ``pyproject.toml``), so on a Windows dev
    machine this whole branch is dead code as far as static analysis is
    concerned and ``termios``/``tty`` are unresolvable — CI's Linux/macOS
    runners type-check it for real. Ignored rather than worked around so a
    real regression on those platforms still fails CI.
    """
    fd = sys.stdin.fileno()
    # fmt: off
    old_settings = termios.tcgetattr(fd)  # pyright: ignore[reportUndefinedVariable, reportUnknownVariableType, reportUnknownMemberType]
    try:
        tty.setraw(fd)  # pyright: ignore[reportUndefinedVariable, reportUnknownMemberType]
        ch = sys.stdin.read(1)
        if ch == "\x1b":  # escape sequence: arrow keys are ESC [ A/B/C/D
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                return {"A": "up", "B": "down"}.get(ch3, "")
            return ""
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":  # Ctrl-C
            raise KeyboardInterrupt
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)  # pyright: ignore[reportUndefinedVariable, reportUnknownMemberType]
    # fmt: on


def _read_key() -> str:
    return _read_key_windows() if sys.platform == "win32" else _read_key_posix()


def select(
    prompt: str,
    options: list[str],
    *,
    console: Console | None = None,
    default_index: int = 0,
) -> int:
    """Prompt the user to pick one of ``options`` via arrow keys + Enter.

    Renders ``prompt`` followed by each option on its own line, the
    currently-selected one highlighted. Up/Down (or ``k``/``j``, the vi
    convention) move the selection; Enter confirms; a digit key ``1``-``9``
    jumps directly to and confirms that option (a fast path some users
    prefer over arrow navigation, without requiring it). The rendered lines
    are erased and replaced in place on each keypress rather than
    scrolling the terminal.

    Falls back to numbered plain-text input (``Prompt: 1) ... \\nChoice: ``)
    when stdin is not an interactive terminal — piped input, CI, or a
    non-TTY test harness — so scripted callers still get a usable prompt
    instead of hanging on a raw-mode read that can never receive a real
    keypress.

    Returns:
        The 0-indexed position of the chosen option in ``options``.
    """
    con = console or Console()
    if not sys.stdin.isatty():
        return _select_fallback(prompt, options, con)

    selected = default_index
    lines_drawn = 0

    def render() -> None:
        nonlocal lines_drawn
        if lines_drawn:
            con.file.write(f"\x1b[{lines_drawn}A")  # cursor up
        out = Text(f"{prompt}\n")
        for i, opt in enumerate(options):
            marker = "> " if i == selected else "  "
            line = Text(f"{marker}{opt}\n")
            if i == selected:
                line.stylize("bold cyan")
            out.append(line)
        con.file.write("\x1b[J")  # clear from cursor to end of screen
        con.print(out, end="")
        lines_drawn = len(options) + 1

    render()
    while True:
        key = _read_key()
        if key in ("up", "k"):
            selected = (selected - 1) % len(options)
            render()
        elif key in ("down", "j"):
            selected = (selected + 1) % len(options)
            render()
        elif key == "enter":
            return selected
        elif key.isdigit() and 1 <= int(key) <= len(options):
            return int(key) - 1


def _select_fallback(prompt: str, options: list[str], console: Console) -> int:
    """Numbered plain-text fallback for non-interactive stdin."""
    console.print(prompt)
    for i, opt in enumerate(options, start=1):
        console.print(f"  {i}) {opt}")
    while True:
        try:
            raw = input("Choice: ").strip()
        except EOFError:
            return 0
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1


__all__ = ["select"]
