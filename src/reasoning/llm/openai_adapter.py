"""
OpenAI LLM Adapter

Implementation for OpenAI API (GPT-4, GPT-3.5, etc).
Also works with OpenAI-compatible APIs (Azure, local servers).

Design decisions:
- Uses official openai library for stability
- Handles both chat and tool calling
- Token counting via tiktoken for accuracy
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
from src.core.types import LLMResponse, Message, ToolCall
from src.reasoning.llm.base import BaseLLMAdapter, convert_messages_to_dicts

# Lazy imports to avoid requiring openai if not used
_openai_client = None
_tiktoken = None


def _get_openai():
    global _openai_client
    if _openai_client is None:
        try:
            import openai

            _openai_client = openai
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
    return _openai_client


def _get_tiktoken():
    global _tiktoken
    if _tiktoken is None:
        try:
            import tiktoken

            _tiktoken = tiktoken
        except ImportError:
            raise ImportError("tiktoken package required. Install with: pip install tiktoken")
    return _tiktoken


class OpenAIAdapter(BaseLLMAdapter):
    """
    OpenAI API adapter.

    Supports:
    - GPT-4o, GPT-4, GPT-3.5 series
    - Function/tool calling
    - Streaming responses
    - JSON mode
    - Vision (with supported models)
    """

    def __init__(self, settings: LLMSettings):
        super().__init__(settings)

        openai = _get_openai()

        # Build client configuration
        client_kwargs: dict[str, Any] = {
            "timeout": settings.request_timeout,
            "max_retries": 0,  # We handle retries ourselves
        }

        if settings.openai_api_key:
            client_kwargs["api_key"] = settings.openai_api_key.get_secret_value()

        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url

        if settings.openai_org_id:
            client_kwargs["organization"] = settings.openai_org_id

        self._client = openai.AsyncOpenAI(**client_kwargs)
        self._default_model = settings.openai_default_model
        self._encoding_cache: dict[str, Any] = {}

    @property
    def provider_name(self) -> str:
        return "openai"

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
        """Execute a chat completion request."""
        openai = _get_openai()

        model = model or self._default_model
        temperature = temperature if temperature is not None else self.settings.default_temperature
        max_tokens = max_tokens or self.settings.default_max_tokens

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": convert_messages_to_dicts(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            request_kwargs["tools"] = tools
            if tool_choice:
                request_kwargs["tool_choice"] = tool_choice

        if stop:
            request_kwargs["stop"] = stop

        # Handle extra kwargs (e.g., response_format for JSON mode)
        request_kwargs.update(kwargs)

        start_time = time.perf_counter()

        try:
            response = await self._client.chat.completions.create(**request_kwargs)
        except openai.RateLimitError as e:
            raise LLMRateLimitError(
                str(e),
                retry_after=float(e.response.headers.get("retry-after", 1.0)),
            )
        except openai.APIConnectionError as e:
            raise LLMConnectionError(str(e), cause=e)
        except openai.BadRequestError as e:
            # Check for context length error
            if "maximum context length" in str(e).lower():
                raise ContextLengthExceededError(
                    str(e),
                    max_tokens=0,  # OpenAI doesn't always tell us
                    actual_tokens=0,
                    cause=e,
                )
            raise LLMResponseError(str(e), cause=e)
        except openai.APIError as e:
            raise LLMResponseError(str(e), cause=e)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Parse response
        choice = response.choices[0]
        content = choice.message.content

        # Parse tool calls if present
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            model=response.model,
            finish_reason=choice.finish_reason,
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
        """Stream a chat completion response."""
        openai = _get_openai()

        model = model or self._default_model
        temperature = temperature if temperature is not None else self.settings.default_temperature
        max_tokens = max_tokens or self.settings.default_max_tokens

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": convert_messages_to_dicts(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if tools:
            request_kwargs["tools"] = tools

        if stop:
            request_kwargs["stop"] = stop

        request_kwargs.update(kwargs)

        try:
            stream = await self._client.chat.completions.create(**request_kwargs)
        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e))
        except openai.APIConnectionError as e:
            raise LLMConnectionError(str(e), cause=e)
        except openai.APIError as e:
            raise LLMResponseError(str(e), cause=e)

        # Accumulate tool calls across chunks
        tool_call_accumulators: dict[int, dict[str, Any]] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Yield content chunks
            if delta.content:
                yield delta.content

            # Accumulate tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_accumulators:
                        tool_call_accumulators[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }

                    if tc.id:
                        tool_call_accumulators[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_call_accumulators[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_call_accumulators[idx]["arguments"] += tc.function.arguments

        # Yield complete tool calls at the end
        for acc in tool_call_accumulators.values():
            try:
                args = json.loads(acc["arguments"])
            except json.JSONDecodeError:
                args = {"raw": acc["arguments"]}

            yield ToolCall(
                id=acc["id"],
                name=acc["name"],
                arguments=args,
            )

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens using tiktoken."""
        tiktoken = _get_tiktoken()
        model = model or self._default_model

        # Get or create encoding
        if model not in self._encoding_cache:
            try:
                self._encoding_cache[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback for unknown models
                self._encoding_cache[model] = tiktoken.get_encoding("cl100k_base")

        encoding = self._encoding_cache[model]
        return len(encoding.encode(text))

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.close()
