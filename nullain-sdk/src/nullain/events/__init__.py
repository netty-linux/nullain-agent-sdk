"""Nullain Agent SDK — Event Sourcing and Trajectory Module."""

from nullain.events.bus import EventBus, EventHandler
from nullain.events.conversation import Conversation, ConversationState, IncrementalFold
from nullain.events.port import EventStorePort
from nullain.events.postgres_store import EventStoreConnectionError, PostgresEventStore
from nullain.events.repair import find_orphaned_tool_results, repair_session_events
from nullain.events.store import EventStore, SQLiteEventStore
from nullain.events.types import (
    BaseEvent,
    CompactionEvent,
    ErrorEvent,
    EventUnion,
    ModelResponseEvent,
    SessionRepairedEvent,
    SpecCreatedEvent,
    SpecVerifiedEvent,
    StreamDeltaEvent,
    TodoEvent,
    TodoItem,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
    WorkflowAgentEvent,
    WorkflowLogEvent,
    WorkflowPhaseEvent,
)

__all__ = [
    "BaseEvent",
    "CompactionEvent",
    "Conversation",
    "ConversationState",
    "ErrorEvent",
    "EventBus",
    "EventHandler",
    "EventStore",
    "EventStoreConnectionError",
    "EventStorePort",
    "EventUnion",
    "IncrementalFold",
    "ModelResponseEvent",
    "PostgresEventStore",
    "SQLiteEventStore",
    "SessionRepairedEvent",
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
    "find_orphaned_tool_results",
    "repair_session_events",
]
