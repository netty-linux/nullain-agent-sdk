"""Nullain Agent SDK — LLM Provider Smoke Example."""

import asyncio
import os

from nullain.llm import ChatMessage, CompletionRequest, OllamaCloudProvider


async def main() -> None:
    api_key = os.getenv("OLLAMA_API_KEY", "")
    provider = OllamaCloudProvider(api_key=api_key)

    request = CompletionRequest(
        model="qwen3-coder:480b-cloud",
        messages=[
            ChatMessage(role="system", content="You are a helpful code assistant."),
            ChatMessage(role="user", content="Say 'Hello Nullain SDK!'"),
        ],
        stream=False,
    )

    print("Checking health...")
    healthy = await provider.health_check()
    print(f"Provider health: {healthy}")

    if healthy or os.getenv("NULLAIN_LIVE_TESTS") == "1":
        response = await provider.generate(request)
        print(f"Generated output: {response.delta_text}")
    else:
        print("Skipping live generation (OLLAMA_API_KEY / endpoint not active).")


if __name__ == "__main__":
    asyncio.run(main())
