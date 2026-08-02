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
        """Publish an event to all subscribed handlers concurrently."""
        import asyncio

        import structlog

        logger = structlog.get_logger("nullain.events.bus")

        handlers = list(self._handlers.get(event.event_type, [])) + list(self._wildcard_handlers)
        if handlers:
            results = await asyncio.gather(
                *(handler(event) for handler in handlers),
                return_exceptions=True,
            )
            for res in results:
                if isinstance(res, BaseException):
                    logger.error(
                        "event_handler_failed",
                        event_type=event.event_type,
                        error=str(res),
                        exc_info=res,
                    )


__all__ = ["EventBus", "EventHandler"]
