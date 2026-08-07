"""Nullain Agent SDK — Conversation checkpoint/rewind over the event log (M14).

The engine is already event-sourced (:mod:`nullain.events.conversation`), so a
"checkpoint" is not separate state to maintain — it is simply a prefix of the
existing event log. Rewinding means truncating ``accumulated_events`` back to
a chosen point and resuming from there with a different instruction, the way
a user asking Claude Code to "try that again, but differently" gets a fresh
attempt from the same starting context rather than a doomed continuation of
the failed one.
"""

from collections.abc import Sequence

from pydantic import BaseModel

from nullain.events.types import BaseEvent, ModelResponseEvent


class RewindPoint(BaseModel):
    """One point in the trajectory a run can be rewound to.

    Each completed ReAct step ends with a :class:`ModelResponseEvent` (a
    final answer, or a batch of tool calls); a ``RewindPoint`` marks that
    boundary. ``step`` is 1-indexed in step order (the order
    ``ModelResponseEvent``s were emitted), not an event-array index, so it
    reads the same way ``AgentLoop``'s own step counter does.
    """

    step: int
    event_id: str
    model: str
    summary: str
    had_tool_calls: bool


def list_rewind_points(events: Sequence[BaseEvent]) -> list[RewindPoint]:
    """Enumerate the rewindable points in an event trajectory.

    One :class:`RewindPoint` per :class:`ModelResponseEvent` in the log, in
    emission order. ``summary`` is the response's text content when the step
    produced a final answer, or a description of the tool calls requested
    when it did not — enough for a caller to identify "the step right before
    it went wrong" without re-deriving the full conversation state.
    """
    points: list[RewindPoint] = []
    step = 0
    for ev in events:
        if not isinstance(ev, ModelResponseEvent):
            continue
        step += 1
        if ev.tool_calls:
            names = ", ".join(tc.name for tc in ev.tool_calls)
            summary = f"[requested tools: {names}]"
        else:
            summary = (ev.content or "").strip()[:200]
        points.append(
            RewindPoint(
                step=step,
                event_id=ev.id,
                model=ev.model,
                summary=summary,
                had_tool_calls=bool(ev.tool_calls),
            )
        )
    return points


def rewind_events(events: Sequence[BaseEvent], to_step: int) -> list[BaseEvent]:
    """Truncate a trajectory back to (and including) the given step.

    ``to_step`` is a 1-indexed step number as returned by
    :func:`list_rewind_points` — everything emitted after that step's
    :class:`ModelResponseEvent` (including it, since a rewind to step N means
    "undo what happened as a result of step N's response") is dropped, so the
    returned prefix ends right before step N's ``ModelResponseEvent``: the
    resumed run sees the same context step N did, and gets to answer fresh.

    Raises:
        ValueError: if ``to_step`` is not a valid step number (< 1, or beyond
            the number of completed steps in ``events``).
    """
    if to_step < 1:
        raise ValueError(f"to_step must be >= 1, got {to_step}")

    step = 0
    for idx, ev in enumerate(events):
        if not isinstance(ev, ModelResponseEvent):
            continue
        step += 1
        if step == to_step:
            return list(events[:idx])

    raise ValueError(f"to_step={to_step} exceeds the trajectory's {step} completed step(s)")


__all__ = ["RewindPoint", "list_rewind_points", "rewind_events"]
