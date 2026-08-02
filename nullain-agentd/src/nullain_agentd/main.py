"""Nullain Agent Daemon — Stdio NDJSON Daemon process."""

import asyncio
import json
import sys
from typing import Any, cast

from nullain.agent import AgentLoop
from nullain.events import BaseEvent, EventBus
from nullain.llm import OllamaCloudProvider
from nullain.protocol import AgentEventPayload, ProtocolEnvelope
from nullain.tools import ToolRegistry
from nullain_tools import register_default_tools


async def run_agentd() -> None:
    """Async main loop reading NDJSON from stdin and writing responses to stdout."""
    registry: ToolRegistry | None = None
    event_bus = EventBus()

    async def emit_agent_event(ev: BaseEvent) -> None:
        payload = AgentEventPayload(
            session_id=ev.session_id,
            event_type=ev.event_type,
            data=json.loads(ev.model_dump_json()),
        )
        env = ProtocolEnvelope(
            v=1,
            type="agent.event",
            payload=json.loads(payload.model_dump_json()),
        )
        sys.stdout.write(env.model_dump_json() + "\n")
        sys.stdout.flush()

    event_bus.subscribe("*", emit_agent_event)

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line_bytes = await reader.readline()
        if not line_bytes:
            break

        line = line_bytes.decode("utf-8").strip()
        if not line:
            continue

        try:
            raw_dict = cast(dict[str, Any], json.loads(line))
            env = ProtocolEnvelope.model_validate(raw_dict)
        except Exception as err:
            err_env = ProtocolEnvelope(
                v=1,
                type="error",
                payload={"message": f"Invalid NDJSON envelope: {err}"},
            )
            sys.stdout.write(err_env.model_dump_json() + "\n")
            sys.stdout.flush()
            continue

        if env.type == "session.start":
            ws_root = str(env.payload.get("workspace_root", "."))
            registry = ToolRegistry()
            register_default_tools(registry, ws_root)

            resp_env = ProtocolEnvelope(
                v=1,
                type="session.started",
                id=env.id,
                payload={"session_id": env.payload.get("session_id", "s1"), "status": "ok"},
            )
            sys.stdout.write(resp_env.model_dump_json() + "\n")
            sys.stdout.flush()

        elif env.type == "user.message":
            prompt = str(env.payload.get("prompt", ""))
            sess_id = str(env.payload.get("session_id", "s1"))
            if registry is None:
                registry = ToolRegistry()
                register_default_tools(registry, ".")

            provider = OllamaCloudProvider()
            agent = AgentLoop(
                provider=provider,
                tools=registry,
                event_bus=event_bus,
            )

            try:
                res_text = await agent.run(prompt=prompt, session_id=sess_id)
                end_env = ProtocolEnvelope(
                    v=1,
                    type="session.end",
                    id=env.id,
                    payload={"session_id": sess_id, "status": "completed", "output": res_text},
                )
                sys.stdout.write(end_env.model_dump_json() + "\n")
                sys.stdout.flush()
            except Exception as err:
                err_env = ProtocolEnvelope(
                    v=1,
                    type="session.end",
                    id=env.id,
                    payload={"session_id": sess_id, "status": "error", "error": str(err)},
                )
                sys.stdout.write(err_env.model_dump_json() + "\n")
                sys.stdout.flush()


def main() -> None:
    asyncio.run(run_agentd())


if __name__ == "__main__":
    main()
