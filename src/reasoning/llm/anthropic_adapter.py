"""
Anthropic LLM Adapter

Implementation for Anthropic's Claude API.

Design decisions:
- Uses official anthropic library
- Converts tool calling format to match Anthropic's schema
- Handles Claude's unique message structure
"""

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from src.config.settings import LLMSettings
from src.core.exceptions import (
    ContextLengthExceededError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
)
from src.core.types import LLMResponse, Message, MessageRole, ToolCall
from src.reasoning.llm.base import BaseLLMAdapter

# Lazy import
_anthropic_client = None


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic

            _anthropic_client = anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
    return _anthropic_client


class AnthropicAdapter(BaseLLMAdapter):
    """
    Anthropic Claude API adapter.

    Supports:
    - Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
    - Tool use
    - Streaming
    - Vision
    """

    def __init__(self, settings: LLMSettings):
        super().__init__(settings)

        anthropic = _get_anthropic()

        client_kwargs: dict[str, Any] = {
            "timeout": settings.request_timeout,
            "max_retries": 0,
        }

        if settings.anthropic_api_key:
            client_kwargs["api_key"] = settings.anthropic_api_key.get_secret_value()

        self._client = anthropic.AsyncAnthropic(**client_kwargs)
        self._default_model = settings.anthropic_default_model

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _convert_messages(
        self,
        messages: list[Message],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert messages to Anthropic format.

        Anthropic requires system message separate from conversation.
        Returns (system_prompt, messages).
        """
        system_prompt = None
        converted = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Anthropic takes system as separate parameter
                system_prompt = msg.content
            elif msg.role == MessageRole.TOOL:
                # Tool results in Anthropic format
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )
            elif msg.role == MessageRole.ASSISTANT:
                converted.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                    }
                )
            else:
                # User messages
                converted.append(
                    {
                        "role": "user",
                        "content": msg.content,
                    }
                )

        return system_prompt, converted

    def _convert_tools(
        self,
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """Convert OpenAI-style tools to Anthropic format."""
        if not tools:
            return None

        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                converted.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object"}),
                    }
                )
            else:
                # Already in Anthropic format
                converted.append(tool)

        return converted

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
        """Execute a message request."""
        anthropic = _get_anthropic()

        model = model or self._default_model
        temperature = temperature if temperature is not None else self.settings.default_temperature
        max_tokens = max_tokens or self.settings.default_max_tokens

        system_prompt, converted_messages = self._convert_messages(messages)

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": max_tokens,
        }

        if temperature is not None:
            request_kwargs["temperature"] = temperature

        if system_prompt:
            request_kwargs["system"] = system_prompt

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            request_kwargs["tools"] = converted_tools

        if stop:
            request_kwargs["stop_sequences"] = stop

        start_time = time.perf_counter()

        try:
            response = await self._client.messages.create(**request_kwargs)
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(str(e))
        except anthropic.APIConnectionError as e:
            raise LLMConnectionError(str(e), cause=e)
        except anthropic.BadRequestError as e:
            if "context length" in str(e).lower():
                raise ContextLengthExceededError(
                    str(e),
                    max_tokens=0,
                    actual_tokens=0,
                    cause=e,
                )
            raise LLMResponseError(str(e), cause=e)
        except anthropic.APIError as e:
            raise LLMResponseError(str(e), cause=e)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Parse response content blocks
        content_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        return LLMResponse(
            content="\n".join(content_parts) if content_parts else None,
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            finish_reason=response.stop_reason,
            latency_ms=latency_ms,
        )

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
        """Stream a message response."""
        anthropic = _get_anthropic()

        model = model or self._default_model
        temperature = temperature if temperature is not None else self.settings.default_temperature
        max_tokens = max_tokens or self.settings.default_max_tokens

        system_prompt, converted_messages = self._convert_messages(messages)

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": max_tokens,
        }

        if temperature is not None:
            request_kwargs["temperature"] = temperature

        if system_prompt:
            request_kwargs["system"] = system_prompt

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            request_kwargs["tools"] = converted_tools

        if stop:
            request_kwargs["stop_sequences"] = stop

        try:
            async with self._client.messages.stream(**request_kwargs) as stream:
                current_tool: dict[str, Any] | None = None

                async for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            current_tool = {
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                                "input": "",
                            }
                    elif event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            yield event.delta.text
                        elif hasattr(event.delta, "partial_json"):
                            if current_tool:
                                current_tool["input"] += event.delta.partial_json
                    elif event.type == "content_block_stop" and current_tool:
                        try:
                            args = json.loads(current_tool["input"])
                        except json.JSONDecodeError:
                            args = {}

                        yield ToolCall(
                            id=current_tool["id"],
                            name=current_tool["name"],
                            arguments=args,
                        )
                        current_tool = None
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(str(e))
        except anthropic.APIConnectionError as e:
            raise LLMConnectionError(str(e), cause=e)
        except anthropic.APIError as e:
            raise LLMResponseError(str(e), cause=e)

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Estimate token count for Claude.

        Anthropic doesn't provide a public tokenizer, so we estimate
        using a simple heuristic (approximately 4 characters per token).
        For production, use the API's token counting endpoint.
        """
        # Rough estimation: ~4 chars per token for English
        # This is a simplification; real implementation should use
        # Anthropic's token counting API
        return len(text) // 4

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.close()
