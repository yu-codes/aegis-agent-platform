"""
Base Model Adapter

Abstract base class for LLM adapters.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdapterConfig:
    """Configuration for model adapters."""

    api_key: str = ""
    api_base: str | None = None
    timeout: float = 120.0
    max_retries: int = 3

    # Model defaults
    model: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7

    # Features
    enable_streaming: bool = True


@dataclass
class CompletionResult:
    """Result of a completion request."""

    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)

    # Usage
    input_tokens: int = 0
    output_tokens: int = 0

    # Metadata
    model: str = ""
    finish_reason: str = ""
    raw_response: dict = field(default_factory=dict)


class BaseAdapter(ABC):
    """
    Abstract base class for LLM adapters.

    All adapters must implement these methods.
    """

    def __init__(self, config: AdapterConfig):
        self._config = config

    @property
    def config(self) -> AdapterConfig:
        return self._config

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """
        Generate a completion.

        Args:
            messages: Conversation messages
            tools: Available tools (optional)
            temperature: Override temperature
            max_tokens: Override max tokens
            **kwargs: Additional parameters

        Returns:
            Completion result
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a completion.

        Args:
            messages: Conversation messages
            tools: Available tools (optional)
            temperature: Override temperature
            max_tokens: Override max tokens
            **kwargs: Additional parameters

        Yields:
            Content chunks
        """
        pass

    @abstractmethod
    def supports_tools(self) -> bool:
        """Check if adapter supports tools."""
        pass

    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if adapter supports streaming."""
        pass

    def _build_messages(self, messages: list[dict]) -> list[dict]:
        """Convert messages to provider format."""
        return messages  # Default pass-through
