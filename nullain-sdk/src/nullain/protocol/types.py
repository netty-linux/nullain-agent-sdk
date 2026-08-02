"""Nullain Agent SDK — Stdio NDJSON Protocol Schema Definition."""

import uuid
from typing import Any

from pydantic import BaseModel, Field


class ProtocolEnvelope(BaseModel):
    """Envelope wrapper for NDJSON protocol messages exchanged between Go CLI and Agent Daemon."""

    v: int = Field(default=1, description="Protocol version number")
    type: str = Field(description="Message type string")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    payload: dict[str, Any] = Field(default_factory=dict)


class SessionStartPayload(BaseModel):
    """Payload for session.start message."""

    session_id: str
    workspace_root: str
    model: str | None = None


class UserMessagePayload(BaseModel):
    """Payload for user.message message."""

    session_id: str
    prompt: str


class AgentEventPayload(BaseModel):
    """Payload for agent.event message."""

    session_id: str
    event_type: str
    data: dict[str, Any]


class PermissionRequestPayload(BaseModel):
    """Payload for permission.request message."""

    request_id: str
    tool_name: str
    description: str


class PermissionResponsePayload(BaseModel):
    """Payload for permission.response message."""

    request_id: str
    granted: bool


class AskUserRequestPayload(BaseModel):
    """Payload for ask_user.request message.

    The daemon emits this when the agent invokes the ``ask_user`` tool; the
    client responds with an ``ask_user.response`` carrying the user's answer.
    """

    request_id: str
    question: str


class AskUserResponsePayload(BaseModel):
    """Payload for ask_user.response message."""

    request_id: str
    answer: str


class SessionEndPayload(BaseModel):
    """Payload for session.end message."""

    session_id: str
    status: str = "completed"


__all__ = [
    "AgentEventPayload",
    "AskUserRequestPayload",
    "AskUserResponsePayload",
    "PermissionRequestPayload",
    "PermissionResponsePayload",
    "ProtocolEnvelope",
    "SessionEndPayload",
    "SessionStartPayload",
    "UserMessagePayload",
]
