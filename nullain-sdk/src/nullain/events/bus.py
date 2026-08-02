"""Nullain Agent SDK — Async Event Bus Implementation."""

from collections.abc import Awaitable, Callable
from typing import TypeVar

from nullain.events.types import BaseEvent

EventHandler = Callable[[BaseEvent], Awaitable[None]]
E = TypeVar("E", bound=BaseEvent)


class EventBus:
    """Publish-Subscribe Async Event Bus for trajectory events."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._wildcard_handlers: list[EventHandler] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register a handler for a specific event_type or '*' for all events."""
        if event_type == "*":
            self._wildcard_handlers.append(handler)
        else:
            self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unregister a handler."""
        if event_type == "*":
            if handler in self._wildcard_handlers:
                self._wildcard_handlers.remove(handler)
        elif event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    async def publish(self, event: BaseEvent) -> None:
        """Publish an event to all subscribed handlers asynchronously."""
        handlers = list(self._handlers.get(event.event_type, [])) + list(self._wildcard_handlers)
        for handler in handlers:
            await handler(event)


__all__ = ["EventBus", "EventHandler"]
