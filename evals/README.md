# Nullain Agent SDK — Evaluation Harness

Objective, reproducible measurement of what the agent actually gets right —
issue #45. Not published to PyPI; not part of `nullain-sdk`. This directory
is consumed only by `make evals` / `make evals-live` and its own test suite.

## Why this exists

The SDK has strong unit-level quality gates (hundreds of tests, ~83%
coverage, hypothesis property tests, sandbox escape tests) but before this
existed there was no objective measure of *agent competence* — whether a
change to the harness (prompt wording, retry policy, compaction, tool
batching) makes the agent better or worse at real tasks. This harness closes
that gap with a small suite of self-contained coding tasks, each graded
programmatically (never by string-similarity or vibes).

It also unblocks the multi-provider work (issue #40): without a way to run
the same task suite against different providers/models and compare pass
rates, there's no data to justify router defaults across providers.

## Two modes

- **`make evals`** — offline. Every task replays a pre-recorded response
  sequence from `evals/fixtures/<task_id>.json` through a `ReplayProvider`.
  No network access at all, fully deterministic, safe for CI (this is what
  the CI `evals` job runs, informationally — see below).
- **`make evals-live MODEL=glm-5.2:cloud`** — live. Runs the same task suite
  against a real Ollama Cloud model. Requires `OLLAMA_API_KEY` (or
  `NULLAIN_OLLAMA_API_KEY`). Add `SAVE_FIXTURES=1` to record every *passing*
  task's responses as its new offline fixture — this is also how you record
  the fixture for a brand-new task (see below). A failing live run's
  responses are never saved as a fixture; replaying a known-bad trajectory
  forever would defeat the point of the baseline.

Both modes drive the agent through the public `nullain.Agent` API — never
through internal harness pieces directly — so a change to any layer between
the facade and the model is exercised exactly as a real user would
experience it.

## Reading a report

Both modes print a summary table and write `evals/report.json` (gitignored —
a fresh artifact per run, not committed). Compare it against
`evals/baselines/offline-baseline.json` (the only baseline checked into git)
to see whether a harness change moved the pass rate — there's no automated
diff tool yet; `git diff` against the baseline file, or just eyeball the two
`results` arrays, task by task.

The report schema (`nullain_evals.report.EvalReport`) is a plain Pydantic
model with a `schema_version` field — bump it if a field is removed or its
meaning changes; additive fields don't require a bump.

## Adding a task

1. Create (or extend) a module under `evals/nullain_evals/tasks/` exposing
   `build() -> list[EvalTask]`. An `EvalTask` needs: `task_id` (unique,
   filesystem-safe — it's also the fixture filename), `description`,
   `prompt` (the exact text handed to `Agent.run`), `setup(workspace: Path)`
   (populates the fresh temp workspace before the agent runs), and `grader`
   (inspects the finished workspace + run outcome, returns a `GradeResult`).
   Prefer a **programmatic** grader — run pytest, check file contents,
   check a structural property — over an LLM judge; every task in this
   suite today uses one. `forbidden_paths` optionally asserts specific
   paths were never touched (useful for read-only tasks, or protecting a
   test file the agent shouldn't edit).
2. Register the module in `evals/nullain_evals/tasks/__init__.py`'s
   `_TASK_MODULES` tuple.
3. Add a known-good / known-bad test pair for the new grader in
   `evals/tests/test_graders.py` (see the existing `Test*` classes — each
   asserts the grader passes a correct solution and fails an incorrect one,
   built by hand, no Agent/provider involved).
4. Record the fixture. Two ways:
   - **Live** (preferred when you have an API key): `make evals-live
     MODEL=<model> SAVE_FIXTURES=1` — records fixtures for every task that
     currently passes live, including your new one.
   - **Hand-authored** (used to build this suite's initial 8 tasks, since no
     live API key was available at the time — see
     `evals/scripts/author_fixtures.py` for the exact pattern): construct
     the `CompletionChunk` sequence a correct model response would produce
     — a Plan-phase `emit_task_spec` tool call for any prompt that isn't a
     LOW-complexity keyword match (see `nullain.router.intent`), then the
     Act-phase tool calls, ending with a text-only response (no
     `tool_calls`) so the ReAct loop terminates. `nullain_evals.replay.
     dump_responses` writes the fixture in the format `ReplayProvider`
     reads. **Tool argument names must match the real tool signatures
     exactly** (e.g. `edit_file` takes `old_str`/`new_str`, not
     `old_text`/`new_text`) — a mismatch fails at tool-dispatch time with a
     permission or "unexpected argument" error, not a grader mismatch, so
     it's usually obvious when authoring goes wrong.
5. Run `make evals` — the new task must pass against its own fixture.
6. Update `evals/baselines/offline-baseline.json` if the new pass rate is
   the new expected baseline (copy `evals/report.json` after a green
   `make evals` run, then zero out every `wall_time_seconds` field — timing
   is expected to vary run to run and isn't part of what the baseline is
   for).

## Known limitation

The `Agent` facade's default permission policy denies `ASK`-level tool calls
(writes, edits) unless a `permission_callback` is configured. The eval
runner auto-approves everything (`nullain_evals.runner._run_one_task`'s
`_auto_approve` callback) — every task runs in a fresh, isolated
`tempfile.TemporaryDirectory` created just for that one task, so there is
nothing an auto-approved write could put at risk that the workspace
isolation doesn't already contain. This was found live: the harness's first
end-to-end run failed every write-based task with "no permission callback
configured — denied (fail-closed)" until this was wired in.
