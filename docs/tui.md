# Terminal UI

`nullain chat` and `nullain run` (without `--json`) render live through
`TUIRenderer` (`nullain.tui`), built on [Rich](https://github.com/Textualize/rich).
This page documents what you'll actually see in the terminal — the rest of the
docs describe the engine; this one describes the screen.

## Streamed text

Model output streams token-by-token in place, rendered as Markdown as it
arrives — headings, code fences, and lists render live rather than only after
the full response has been generated.

## Tool calls

A tool call renders as a single status-dot line, not a bordered panel — the
Claude Code convention this project deliberately matches rather than the
verbose boxed-output style some other CLIs use:

```
● read_file  src/app.py
● bash  npm test
  - 2 failing
● git_commit
```

- **Dim `●`** while the call is in flight.
- **Green `●`** on success, **red `●`** on failure — color, not a text
  marker, carries the status.
- The tool name renders **bold**; a short detail (the file path, or the
  command for `bash`) follows in dim gray.
- A failed call prints one indented detail line underneath with a truncated
  summary of the error output. The full output is still available on the
  underlying `ToolResultEvent` for anything consuming the raw event stream
  (logs, `--json` output, a non-TUI caller).

On a legacy Windows console (`cmd.exe`, or PowerShell without UTF-8 mode,
where `Console.legacy_windows` is `True`) the `●` glyph and the `└` detail
prefix aren't encodable — the renderer falls back automatically to ASCII: `o`
for the dot, `-` for the detail prefix.

### Repeated tool calls collapse into one line

An agent exploring a project routinely calls the same tool many times in a
row — `list_directory` across a directory tree, `read_file` across several
files while gathering context. Printing one permanent line per call would
turn a normal turn into dozens of stacked lines. Instead, consecutive calls
to the *same* tool update a single live line in place (showing the current
call's own detail as it goes), and only finalize once something changes:

```
● list_directory  x20
● read_file  package.json
```

The streak finalizes to a compact `● name  xN` summary the moment a
different tool starts, an error breaks the streak, the model starts
responding, or the run ends. Two things are never collapsed into a streak,
even mid-run:

- **Errors** — always get their own full line plus the failure detail, never
  hidden inside a count.
- **A tool called exactly once** — the common case — still shows its own
  full detail (`● read_file  a.txt`), not a `x1` suffix.

### File-write diffs

`write_file`, `edit_file`, and `multi_edit` are never collapsed into a
streak, even when several writes happen back to back — each one prints its
status line followed by a colored unified diff of what changed:

```
● write_file  src/modules/courses/index.ts
╭──────────────────────────────────────────────────────────────╮
│ @@ -1,2 +1,2 @@                                               │
│ -export {};                                                   │
│ +export { courseRoutes } from "./routes";                     │
╰──────────────────────────────────────────────────────────────╯
```

## Permission prompts

An `ASK`-level tool call (see [tools.md](tools.md) for the permission model)
stops the run and shows an arrow-key menu — no typed `y`/`n`:

```
Allow bash? (rm -rf build/)
> Yes
  No
  Yes, always allow this tool
```

The currently-selected option renders in bold cyan. Navigate with `↑`/`↓`
(or `k`/`j`), confirm with `Enter`; a digit key jumps straight to that
option. **"Yes, always allow this tool"** approves that
tool name for the rest of the current chat process — later calls to the same
tool in the same session skip the prompt entirely. The approval is
session-scoped only: it is never written to disk, and a new `nullain chat`
process starts with a clean slate.

On a non-interactive terminal (piped input, CI, a scripted caller) the menu
falls back to a numbered prompt (`1) Yes`, `2) No`, ...) instead of hanging
on a raw keypress read that could never arrive.

## Command history

Inside `nullain chat`, `↑`/`↓` recall earlier prompts from the same session —
the same convention as a shell. History is in-memory only for the lifetime
of the chat process; nothing is written to disk, so a prompt containing
something sensitive is never persisted to a history file.

## Plan / Verify panels

Two structural events still render as full bordered panels — deliberately
different from the compact tool-call lines above, since a plan or a
verification result is meant to be read in full, not skimmed as a status
signal:

```
╭─────────────────────────────── Plan ────────────────────────────────╮
│ Add a login endpoint with JWT auth                                  │
│   1. Add /auth/login route                                          │
│   2. Verify credentials against the users table                     │
│   3. Sign and return a JWT                                          │
╰───────────────────────────────────────────────────────────────────────╯
```

```
╭────────────────────────────── ✓ Verify ──────────────────────────────╮
│ All acceptance criteria met.                                        │
╰───────────────────────────────────────────────────────────────────────╯
```

A failed verification renders the same way in yellow, and — when the agent
has verify-retry budget left — is followed by a `[VERIFY-CORRECTION]`
turn that attempts a fix automatically.

## Errors and run status

A run that doesn't end in plain success (hit `max_steps`, loop detection,
verification failed after exhausting retries, a provider error) prints a
final status panel naming the reason:

```
╭──────────────────────────── Status: max_steps ────────────────────────────╮
│ Agent loop reached maximum step count (100)                              │
╰──────────────────────────────────────────────────────────────────────────╯
```

## Non-interactive output

None of the above applies to `--json`: it emits one NDJSON object per event
on stdout with no rendering at all, for scripts and other tools to consume.
See [configuration.md](configuration.md) and the CLI's `--help` for the full
flag reference.
