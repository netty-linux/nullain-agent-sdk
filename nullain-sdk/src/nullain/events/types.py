"""Nullain Agent SDK — Immutable Event Sourcing Types."""

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from nullain.llm.types import TokenUsage, ToolCall


class BaseEvent(BaseModel, frozen=True):
    """Base immutable event for conversation trajectory sourcing."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp: float = Field(default_factory=time.time)
    event_type: str


class UserMessageEvent(BaseEvent, frozen=True):
    """Event emitted when a user submits a prompt message."""

    event_type: Literal["user_message"] = "user_message"
    content: str


class ModelResponseEvent(BaseEvent, frozen=True):
    """Event emitted when LLM generates text or tool calls."""

    event_type: Literal["model_response"] = "model_response"
    model: str
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] | None = None
    usage: TokenUsage | None = None


class ToolCallEvent(BaseEvent, frozen=True):
    """Event emitted prior to executing a tool."""

    event_type: Literal["tool_call"] = "tool_call"
    call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultEvent(BaseEvent, frozen=True):
    """Event emitted with execution output of a tool."""

    event_type: Literal["tool_result"] = "tool_result"
    call_id: str
    tool_name: str
    output: str
    is_error: bool = False


class CompactionEvent(BaseEvent, frozen=True):
    """Event emitted when context manager compacts earlier trajectory history."""

    event_type: Literal["compaction"] = "compaction"
    summary: str
    compacted_event_ids: tuple[str, ...]


class ErrorEvent(BaseEvent, frozen=True):
    """Event emitted when an error occurs during execution."""

    event_type: Literal["error"] = "error"
    error_type: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class SpecCreatedEvent(BaseEvent, frozen=True):
    """Event emitted when Plan phase creates a task specification."""

    event_type: Literal["spec_created"] = "spec_created"
    spec_id: str
    title: str
    steps: tuple[str, ...]


class SpecVerifiedEvent(BaseEvent, frozen=True):
    """Event emitted when Verify phase evaluates task criteria."""

    event_type: Literal["spec_verified"] = "spec_verified"
    spec_id: str
    success: bool
    feedback: str


class StreamDeltaEvent(BaseEvent, frozen=True):
    """Event emitted for each streamed token delta during streaming execution."""

    event_type: Literal["stream_delta"] = "stream_delta"
    delta: str
    model: str


class WorkflowPhaseEvent(BaseEvent, frozen=True):
    """Event emitted when a workflow enters a phase (P4.27)."""

    event_type: Literal["workflow_phase"] = "workflow_phase"
    workflow: str
    phase: str


class WorkflowLogEvent(BaseEvent, frozen=True):
    """Event emitted for a workflow progress message (P4.27)."""

    event_type: Literal["workflow_log"] = "workflow_log"
    workflow: str
    message: str


class WorkflowAgentEvent(BaseEvent, frozen=True):
    """Event emitted when a workflow subagent starts, completes, or fails (P4.27)."""

    event_type: Literal["workflow_agent"] = "workflow_agent"
    workflow: str
    label: str
    status: Literal["started", "completed", "failed"]
    output: str | None = None


class TodoItem(BaseModel, frozen=True):
    """A single task in the agent's todo list (M8)."""

    content: str
    status: Literal["pending", "in_progress", "completed"]


class TodoEvent(BaseEvent, frozen=True):
    """Event emitted when the agent updates its todo list (M8).

    Carries the full todo list after the update so the daemon can mirror the
    agent's progress to the client without tracking state itself.
    """

    event_type: Literal["todo"] = "todo"
    items: tuple[TodoItem, ...]


EventUnion = (
    UserMessageEvent
    | ModelResponseEvent
    | ToolCallEvent
    | ToolResultEvent
    | CompactionEvent
    | ErrorEvent
    | SpecCreatedEvent
    | SpecVerifiedEvent
    | StreamDeltaEvent
    | WorkflowPhaseEvent
    | WorkflowLogEvent
    | WorkflowAgentEvent
    | TodoEvent
)


__all__ = [
    "BaseEvent",
    "CompactionEvent",
    "ErrorEvent",
    "EventUnion",
    "ModelResponseEvent",
    "SpecCreatedEvent",
    "SpecVerifiedEvent",
    "StreamDeltaEvent",
    "TodoEvent",
    "TodoItem",
    "ToolCallEvent",
    "ToolResultEvent",
    "UserMessageEvent",
    "WorkflowAgentEvent",
    "WorkflowLogEvent",
    "WorkflowPhaseEvent",
]
