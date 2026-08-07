# Contributing

Thanks for considering a contribution to the Nullain Agent SDK.

## Before you start

For anything beyond a small fix, open an issue first to discuss the
approach — it saves everyone time if a larger change turns out to be out
of scope or needs a different design than planned. For a bug fix or small
improvement, a PR directly is fine.

## Development setup

```bash
git clone https://github.com/netty-linux/nullain-agent-sdk.git
cd nullain-agent-sdk
uv sync --all-packages
```

This is a `uv` workspace monorepo (`nullain-sdk`, `nullain-tools`,
`nullain-agentd`) — `uv sync --all-packages` installs every member and
their dev dependencies, including `cryptography` (needed for the plugin
signing tests to run for real instead of skipping) and the tools each
platform's sandbox adapter tests need.

## Making a change

1. Create a branch off `master`.
2. Make your change, with tests. This project treats "found a real bug
   while building this" as worth its own regression test, not just a fix —
   see recent commit history for the pattern.
3. Run the full check suite before opening a PR:

   ```bash
   make check   # lint + typecheck + test
   ```

   Or individually:

   ```bash
   make lint        # ruff check + ruff format --check
   make format      # auto-fix what ruff can fix
   make typecheck   # pyright, strict mode
   make test        # pytest
   make cov         # pytest with coverage (HTML report in htmlcov/)
   make audit       # pip-audit for known CVEs in dependencies
   ```

4. Open a PR against `master`. CI runs the same checks across a 3-OS ×
   2-Python-version matrix (the sandbox adapters — Landlock, Seatbelt,
   Windows Job Object — are platform-specific, so the full matrix matters
   here more than it would in a typical pure-Python project).

## Code style

- Ruff for linting and formatting; Pyright in strict mode for types. Both
  run in CI and are non-negotiable — a PR won't merge with either failing.
- No commented-out code, no dead code paths "just in case." If something's
  not used, delete it.
- Prefer fixing the root cause over adding a workaround. If you find a real
  bug while working on something else, it's worth its own commit and test
  rather than being silently patched around.
- Comments explain *why*, not *what* — the code should already say what it
  does through naming; a comment earns its place by capturing something
  non-obvious (a constraint, an invariant, the reason a workaround exists).

## Security

Found a security issue? Please don't open a public issue — see
[SECURITY.md](SECURITY.md) for how to report it privately.

## License

By contributing, you agree that your contributions will be licensed under
the project's [MIT License](LICENSE).
