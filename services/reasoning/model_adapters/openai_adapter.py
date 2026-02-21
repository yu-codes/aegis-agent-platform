"""
OpenAI Adapter

Adapter for OpenAI GPT models.
"""

from collections.abc import AsyncIterator
from typing import Any
import json

from services.reasoning.model_adapters.base import (
    BaseAdapter,
    AdapterConfig,
    CompletionResult,
)


class OpenAIAdapter(BaseAdapter):
    """
    Adapter for OpenAI API.
    """

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazy initialization of client."""
        if self._client is None:
            try:
                import openai

                self._client = openai.AsyncOpenAI(
                    api_key=self._config.api_key,
                    base_url=self._config.api_base,
                    timeout=self._config.timeout,
                    max_retries=self._config.max_retries,
                )
            except ImportError:
                raise RuntimeError("openai package not installed")
        return self._client

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Generate completion using GPT."""
        client = self._get_client()

        # Build request
        request_params = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": max_tokens or self._config.max_tokens,
            "temperature": temperature if temperature is not None else self._config.temperature,
        }

        if tools:
            request_params["tools"] = tools

        # Additional params
        if "response_format" in kwargs:
            request_params["response_format"] = kwargs["response_format"]

        # Make request
        response = await client.chat.completions.create(**request_params)

        # Parse response
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": (
                                json.loads(tc.function.arguments) if tc.function.arguments else {}
                            ),
                        },
                    }
                )

        return CompletionResult(
            content=message.content or "",
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            model=response.model,
            finish_reason=choice.finish_reason or "",
            raw_response={"id": response.id},
        )

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion using GPT."""
        client = self._get_client()

        # Build request
        request_params = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": max_tokens or self._config.max_tokens,
            "temperature": temperature if temperature is not None else self._config.temperature,
            "stream": True,
        }

        # Stream response
        stream = await client.chat.completions.create(**request_params)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True
