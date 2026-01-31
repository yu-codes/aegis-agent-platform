"""
Base LLM Adapter

Defines the abstract interface for all LLM providers.
Implementations must handle retries, timeouts, and streaming.

Design decisions:
- Async-first: All methods are async for non-blocking I/O
- Streaming as first-class: AsyncGenerator for streaming responses
- Provider-agnostic: Common interface hides provider differences
- Retry logic: Built into base class with configurable backoff
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from src.config.settings import LLMSettings
from src.core.exceptions import (
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from src.core.types import LLMResponse, Message, ToolCall


class BaseLLMAdapter(ABC):
    """
    Abstract base class for LLM providers.

    All LLM interactions go through this interface, enabling:
    - Provider switching without code changes
    - Consistent error handling
    - Unified streaming interface
    - Built-in retry logic
    """

    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self._retry_count = settings.max_retries
        self._retry_delay = settings.retry_delay
        self._timeout = settings.request_timeout

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier."""
        pass

    @abstractmethod
    async def _do_complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Provider-specific implementation of completion.

        This is called by complete() after retry logic.
        Implementations should NOT handle retries.
        """
        pass

    @abstractmethod
    async def _do_stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str | ToolCall]:
        """
        Provider-specific streaming implementation.

        Yields either string chunks or complete ToolCall objects.
        """
        pass

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Send a completion request to the LLM.

        Includes automatic retry with exponential backoff for
        transient errors (rate limits, timeouts, connection issues).

        Args:
            messages: Conversation history
            model: Model identifier (uses default if not specified)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            tools: Tool definitions for function calling
            tool_choice: Tool selection strategy
            stop: Stop sequences
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse with content and/or tool calls

        Raises:
            LLMConnectionError: Cannot reach provider
            LLMRateLimitError: Rate limit exceeded (after retries)
            LLMTimeoutError: Request timed out (after retries)
        """
        last_error: Exception | None = None

        for attempt in range(self._retry_count + 1):
            try:
                return await asyncio.wait_for(
                    self._do_complete(
                        messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools,
                        tool_choice=tool_choice,
                        stop=stop,
                        **kwargs,
                    ),
                    timeout=self._timeout,
                )
            except TimeoutError:
                last_error = LLMTimeoutError(
                    f"Request timed out after {self._timeout}s",
                    context={"attempt": attempt + 1},
                )
            except LLMRateLimitError as e:
                last_error = e
                # Use retry-after if provided
                delay = e.retry_after or (self._retry_delay * (2**attempt))
                if attempt < self._retry_count:
                    await asyncio.sleep(delay)
            except LLMConnectionError as e:
                last_error = e
                if attempt < self._retry_count:
                    await asyncio.sleep(self._retry_delay * (2**attempt))
            except Exception:
                # Non-retryable errors
                raise

        # All retries exhausted
        if last_error:
            raise last_error
        raise LLMConnectionError("Request failed after all retries")

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str | ToolCall]:
        """
        Stream a completion response.

        Yields content chunks as they arrive. For tool calls,
        yields complete ToolCall objects after the full call
        is received.

        Note: Streaming is best-effort for retries. If the
        stream fails mid-way, an exception is raised.
        """
        async for chunk in self._do_stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            stop=stop,
            **kwargs,
        ):
            yield chunk

    @abstractmethod
    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Count tokens in text for the specified model.

        Used for context length management.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Clean up resources (connection pools, etc).

        Should be called when adapter is no longer needed.
        """
        pass

    async def __aenter__(self) -> "BaseLLMAdapter":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


def convert_messages_to_dicts(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert Message objects to provider-compatible dicts."""
    result = []
    for msg in messages:
        d = {"role": msg.role.value, "content": msg.content}
        if msg.name:
            d["name"] = msg.name
        if msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        result.append(d)
    return result
