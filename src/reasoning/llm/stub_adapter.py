"""
Stub LLM Adapter

A deterministic, offline-capable LLM adapter for testing and CI.

Design decisions:
- Implements full LLMAdapterProtocol interface
- Returns scripted, deterministic responses
- Supports streaming (simulated with delays)
- Recognizes tool calls via simple pattern matching
- Provides useful diagnostic output
- NEVER makes external network calls

Usage:
    # Explicitly
    from src.reasoning.llm.stub_adapter import StubLLMAdapter
    adapter = StubLLMAdapter()

    # Via environment
    AEGIS_OFFLINE_MODE=true  # Uses stub adapter automatically

This adapter is safe for:
- Unit tests
- Integration tests
- CI pipelines
- Demos without API keys
- Local development
"""

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from src.core.types import LLMResponse, Message, MessageRole, ToolCall


@dataclass
class StubResponse:
    """A scripted response for the stub adapter."""

    pattern: str | None = None  # Regex to match user message
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tokens: int = 100

    def matches(self, message: str) -> bool:
        if self.pattern is None:
            return True
        return bool(re.search(self.pattern, message, re.IGNORECASE))


# Default responses for common patterns
DEFAULT_RESPONSES: list[StubResponse] = [
    # Tool usage patterns
    StubResponse(
        pattern=r"search|find|look\s*up|query",
        content="I'll search for that information.",
        tool_calls=[
            {
                "id": "stub_call_001",
                "name": "search_knowledge_base",
                "arguments": {"query": "user query"},
            }
        ],
    ),
    StubResponse(
        pattern=r"calculate|compute|math|\d+\s*[\+\-\*/]\s*\d+",
        content="Let me calculate that for you.",
        tool_calls=[
            {"id": "stub_call_002", "name": "calculator", "arguments": {"expression": "1 + 1"}}
        ],
    ),
    StubResponse(
        pattern=r"weather|temperature|forecast",
        content="I'll check the weather for you.",
        tool_calls=[
            {
                "id": "stub_call_003",
                "name": "get_weather",
                "arguments": {"location": "San Francisco"},
            }
        ],
    ),
    # Domain-specific patterns
    StubResponse(
        pattern=r"ticket|support|issue|problem|error|bug",
        content="I understand you're experiencing an issue. Let me help troubleshoot this step by step.\n\n1. Can you describe when this issue first started?\n2. Have you tried restarting the application?\n3. Are there any error messages displayed?\n\n[STUB: Technical Support Domain Response]",
    ),
    StubResponse(
        pattern=r"stock|portfolio|invest|market|financial|trading",
        content="Based on the market data, here's my analysis:\n\n**Note**: This is a stub response for offline testing.\n\n⚠️ **Disclaimer**: This is for educational purposes only and does not constitute financial advice.\n\n[STUB: Financial Analysis Domain Response]",
    ),
    # Default fallback
    StubResponse(
        pattern=None,  # Matches anything
        content="Hello! I'm the Aegis AI Assistant running in **offline mode**.\n\nThis is a deterministic stub response for testing. In production, this would be a real LLM response.\n\nI can demonstrate:\n- ✅ API endpoint connectivity\n- ✅ Message routing\n- ✅ Domain resolution\n- ✅ Tool schema loading\n- ✅ Response streaming\n\n[STUB: General Response]",
    ),
]


class StubLLMAdapter:
    """
    A deterministic LLM adapter for offline testing.

    Features:
    - No external API calls
    - Deterministic responses
    - Pattern-based response selection
    - Simulated streaming
    - Tool call simulation
    - Configurable delays

    This adapter fully implements LLMAdapterProtocol.
    """

    def __init__(
        self,
        model: str = "stub-model-v1",
        responses: list[StubResponse] | None = None,
        stream_delay_ms: int = 20,
        default_response: str | None = None,
    ):
        """
        Initialize the stub adapter.

        Args:
            model: Model identifier (for logging/tracing)
            responses: Custom response patterns (uses defaults if None)
            stream_delay_ms: Delay between stream chunks
            default_response: Override default response content
        """
        self._model = model
        self._responses = responses or DEFAULT_RESPONSES.copy()
        self._stream_delay = stream_delay_ms / 1000.0
        self._call_count = 0
        self._total_tokens = 0

        if default_response:
            # Add as highest-priority fallback
            self._responses.insert(
                0,
                StubResponse(
                    pattern=None,
                    content=default_response,
                ),
            )

    @property
    def model(self) -> str:
        """Current model identifier."""
        return self._model

    @property
    def provider_name(self) -> str:
        """Provider identifier."""
        return "stub"

    @property
    def call_count(self) -> int:
        """Number of calls made to this adapter."""
        return self._call_count

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (simulated)."""
        return self._total_tokens

    def add_response(self, response: StubResponse) -> None:
        """Add a custom response pattern."""
        self._responses.insert(0, response)  # Higher priority

    def _find_response(self, messages: list[Message] | list[dict]) -> StubResponse:
        """Find the best matching response for the messages."""
        # Get the last user message
        user_message = ""
        for msg in reversed(messages):
            # Handle both Message objects and dicts
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                role = msg.role
                content = msg.content

            # Check if it's a user message
            is_user = role == MessageRole.USER if isinstance(role, MessageRole) else role == "user"
            if is_user:
                user_message = content
                break

        # Find matching response
        for response in self._responses:
            if response.matches(user_message):
                return response

        # Should never reach here due to None pattern fallback
        return self._responses[-1]

    def _make_tool_calls(self, response: StubResponse) -> list[ToolCall]:
        """Convert response tool call dicts to ToolCall objects."""
        return [
            ToolCall(
                id=tc.get("id", f"stub_{self._call_count}_{i}"),
                name=tc["name"],
                arguments=tc.get("arguments", {}),
            )
            for i, tc in enumerate(response.tool_calls)
        ]

    async def complete(
        self,
        messages: list[Message] | list[dict],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion (non-streaming).

        Returns a deterministic response based on message content.
        Accepts both Message objects and dicts for flexibility.
        """
        self._call_count += 1
        response = self._find_response(messages)

        # Simulate some processing time
        await asyncio.sleep(0.01)

        # Count tokens (rough simulation)
        tokens = response.tokens
        self._total_tokens += tokens

        tool_calls = self._make_tool_calls(response) if response.tool_calls else []

        return LLMResponse(
            content=response.content,
            model=self._model,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            prompt_tokens=tokens // 3,
            completion_tokens=tokens - tokens // 3,
            total_tokens=tokens,
        )

    async def stream(
        self,
        messages: list[Message] | list[dict],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponse]:
        """
        Generate a streaming completion.

        Simulates streaming by yielding content word by word.
        Accepts both Message objects and dicts for flexibility.
        """
        self._call_count += 1
        response = self._find_response(messages)

        tool_calls = self._make_tool_calls(response) if response.tool_calls else []
        tokens = response.tokens
        self._total_tokens += tokens

        # Stream content word by word
        words = response.content.split()
        content_so_far = ""

        for i, word in enumerate(words):
            content_so_far += word + (" " if i < len(words) - 1 else "")

            yield LLMResponse(
                content=word + (" " if i < len(words) - 1 else ""),
                model=self._model,
                tool_calls=[],  # Tool calls come at the end
                finish_reason=None,
                prompt_tokens=tokens // 3,
                completion_tokens=len(content_so_far.split()),
                total_tokens=tokens // 3 + len(content_so_far.split()),
            )

            await asyncio.sleep(self._stream_delay)

        # Final chunk with tool calls if any
        yield LLMResponse(
            content="",
            model=self._model,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            prompt_tokens=tokens // 3,
            completion_tokens=tokens - tokens // 3,
            total_tokens=tokens,
        )

    def reset_stats(self) -> None:
        """Reset call and token counters."""
        self._call_count = 0
        self._total_tokens = 0

    def __repr__(self) -> str:
        return f"StubLLMAdapter(model={self._model!r}, calls={self._call_count})"


# =============================================================================
# Factory function for easy creation
# =============================================================================


def create_stub_adapter(
    echo_input: bool = False,
    include_metadata: bool = True,
) -> StubLLMAdapter:
    """
    Create a stub adapter with sensible defaults.

    Args:
        echo_input: If True, echo user input in response
        include_metadata: If True, include offline mode metadata

    Returns:
        Configured StubLLMAdapter
    """
    adapter = StubLLMAdapter()

    if echo_input:
        adapter.add_response(
            StubResponse(
                pattern=r".*",
                content="[ECHO] Your message was received. Running in offline mode.",
            )
        )

    return adapter


# =============================================================================
# Scripted conversation support
# =============================================================================


class ScriptedLLMAdapter(StubLLMAdapter):
    """
    An adapter that follows a scripted conversation.

    Useful for testing specific multi-turn flows.
    """

    def __init__(
        self,
        script: list[str],
        model: str = "scripted-model-v1",
    ):
        """
        Initialize with a conversation script.

        Args:
            script: List of responses to return in order
            model: Model identifier
        """
        super().__init__(model=model)
        self._script = script
        self._script_index = 0

    async def complete(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> LLMResponse:
        """Return the next scripted response."""
        self._call_count += 1

        if self._script_index < len(self._script):
            content = self._script[self._script_index]
            self._script_index += 1
        else:
            content = "[SCRIPTED ADAPTER] Script exhausted. No more responses."

        return LLMResponse(
            content=content,
            model=self._model,
            tool_calls=[],
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=len(content.split()),
            total_tokens=50 + len(content.split()),
        )

    def reset_script(self) -> None:
        """Reset script to beginning."""
        self._script_index = 0
