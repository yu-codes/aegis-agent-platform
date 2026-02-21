"""
Stub Adapter

Mock adapter for testing and offline mode.
"""

from collections.abc import AsyncIterator
from typing import Any
import asyncio

from services.reasoning.model_adapters.base import (
    BaseAdapter,
    AdapterConfig,
    CompletionResult,
)


class StubAdapter(BaseAdapter):
    """
    Stub adapter for testing and offline development.

    Returns deterministic responses without API calls.
    """

    def __init__(self, config: AdapterConfig | None = None):
        super().__init__(config or AdapterConfig(model="stub-v1"))
        self._responses: dict[str, str] = {}
        self._default_response = "This is a stub response for testing purposes."

    def set_response(self, pattern: str, response: str) -> None:
        """Set a canned response for a pattern."""
        self._responses[pattern.lower()] = response

    def set_default_response(self, response: str) -> None:
        """Set the default response."""
        self._default_response = response

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Generate stub completion."""
        # Simulate processing time
        await asyncio.sleep(0.1)

        # Get last user message
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "").lower()
                break

        # Check for matching response
        response_text = self._default_response
        for pattern, response in self._responses.items():
            if pattern in last_user_msg:
                response_text = response
                break

        # Simulate tool calls if tools provided and query mentions tool
        tool_calls = []
        if tools and "calculate" in last_user_msg:
            tool_calls.append(
                {
                    "id": "stub_call_1",
                    "type": "function",
                    "function": {
                        "name": tools[0].get("function", {}).get("name", "calculator"),
                        "arguments": {"expression": "2 + 2"},
                    },
                }
            )

        return CompletionResult(
            content=response_text,
            tool_calls=tool_calls,
            input_tokens=len(str(messages)) // 4,
            output_tokens=len(response_text) // 4,
            model="stub-v1",
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream stub completion."""
        result = await self.complete(messages, tools, temperature, max_tokens, **kwargs)

        # Simulate streaming
        words = result.content.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.05)

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True
