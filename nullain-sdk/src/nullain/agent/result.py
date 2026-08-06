"""Nullain Agent SDK — RunResult structured termination contract.

A ``RunResult`` is the standardized outcome of one ``AgentLoop`` run. It
replaces the bare ``str`` return value (which conflated success with a final
answer that may have failed verification) with an explicit termination reason,
mirroring the result contracts of Claude Code / Gemini CLI.

The terminal-error paths (budget, timeout, context exhaustion) are represented
as statuses here so callers that opt into structured results can branch on
outcome without catching exceptions. The legacy ``run()`` / ``run_streaming()``
methods still raise the corresponding exceptions for backwards compatibility.
"""

from typing import Literal

from pydantic import BaseModel

RunStatus = Literal[
    "success",
    "max_steps",
    "budget",
    "timeout",
    "context_exhausted",
    "verification_failed",
    "loop_detected",
    "cancelled",
    "error",
]


class RunResult(BaseModel):
    """Structured outcome of a single agent run.

    Attributes:
        session_id: The run's session identifier.
        status: Why the run terminated. ``success`` means the model returned a
            final answer and (if a spec existed) verification passed.
            ``verification_failed`` means a final answer was produced but
            acceptance criteria were not met. ``max_steps`` means the loop
            exhausted its step budget without a final answer. ``budget``,
            ``timeout`` and ``context_exhausted`` are terminal-resource
            failures. ``loop_detected`` means the agent repeated the same
            tool calls for several consecutive steps and was stopped.
            ``cancelled`` means the run was cancelled (e.g. via a daemon
            ``session.cancel`` or closed stdin) and returned a structured
            outcome instead of a raw exception. ``error`` covers unexpected
            exceptions.
        success: Convenience boolean — True only for ``status == "success"``.
        final_text: The model's final answer text (may be empty on terminal
            failures before any final answer was produced).
        steps: Number of ReAct steps executed.
        model: The model identifier actually used for the run.
        intent: Classified intent type, if intent parsing ran.
        feedback: Verification feedback message, when a spec was verified.
        error: Human-readable error message for terminal-failure statuses.
    """

    session_id: str
    status: RunStatus
    success: bool = False
    final_text: str = ""
    steps: int = 0
    model: str = ""
    intent: str | None = None
    feedback: str | None = None
    error: str | None = None


__all__ = ["RunResult", "RunStatus"]
