"""Nullain Agent SDK — OpenAI-Compatible Provider Smoke Example (issue #40).

Mirrors 00_llm_smoke.py, but against OpenAICompatibleProvider instead of
OllamaCloudProvider — proving the same LLMProvider port works unmodified
against any OpenAI-compatible endpoint. Set NULLAIN_SMOKE_BASE_URL to point
this at something other than OpenAI itself (e.g. OpenRouter, a local vLLM
server, LM Studio) without touching this file.
"""

import asyncio
import os

from nullain.llm import ChatMessage, CompletionRequest, OpenAICompatibleProvider


async def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("NULLAIN_SMOKE_BASE_URL", "https://api.openai.com")
    model = os.getenv("NULLAIN_SMOKE_MODEL", "gpt-4o-mini")
    provider = OpenAICompatibleProvider(api_key=api_key, base_url=base_url)

    request = CompletionRequest(
        model=model,
        messages=[
            ChatMessage(role="system", content="You are a helpful code assistant."),
            ChatMessage(role="user", content="Say 'Hello Nullain SDK!'"),
        ],
        stream=False,
    )

    print(f"Checking health of {base_url}...")
    healthy = await provider.health_check()
    print(f"Provider health: {healthy}")

    if healthy or os.getenv("NULLAIN_LIVE_TESTS") == "1":
        response = await provider.generate(request)
        print(f"Generated output: {response.delta_text}")
    else:
        print("Skipping live generation (OPENAI_API_KEY / endpoint not active).")


if __name__ == "__main__":
    asyncio.run(main())
