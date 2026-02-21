"""
Anthropic Adapter

Adapter for Anthropic Claude models.
"""

from collections.abc import AsyncIterator
from typing import Any

from services.reasoning.model_adapters.base import (
    BaseAdapter,
    AdapterConfig,
    CompletionResult,
)


class AnthropicAdapter(BaseAdapter):
    """
    Adapter for Anthropic Claude API.
    """

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazy initialization of client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.AsyncAnthropic(
                    api_key=self._config.api_key,
                    timeout=self._config.timeout,
                    max_retries=self._config.max_retries,
                )
            except ImportError:
                raise RuntimeError("anthropic package not installed")
        return self._client

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Generate completion using Claude."""
        client = self._get_client()

        # Extract system message
        system_content = ""
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                filtered_messages.append(self._convert_message(msg))

        # Build request
        request_params = {
            "model": self._config.model,
            "messages": filtered_messages,
            "max_tokens": max_tokens or self._config.max_tokens,
            "temperature": temperature if temperature is not None else self._config.temperature,
        }

        if system_content:
            request_params["system"] = system_content

        if tools:
            request_params["tools"] = [self._convert_tool(t) for t in tools]

        # Make request
        response = await client.messages.create(**request_params)

        # Parse response
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": block.input,
                        },
                    }
                )

        return CompletionResult(
            content=content,
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            finish_reason=response.stop_reason or "",
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
        """Stream completion using Claude."""
        client = self._get_client()

        # Extract system message
        system_content = ""
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                filtered_messages.append(self._convert_message(msg))

        # Build request
        request_params = {
            "model": self._config.model,
            "messages": filtered_messages,
            "max_tokens": max_tokens or self._config.max_tokens,
            "temperature": temperature if temperature is not None else self._config.temperature,
        }

        if system_content:
            request_params["system"] = system_content

        # Stream response
        async with client.messages.stream(**request_params) as stream:
            async for text in stream.text_stream:
                yield text

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def _convert_message(self, msg: dict) -> dict:
        """Convert message to Anthropic format."""
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Map assistant to assistant, user to user
        if role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": content,
                    }
                ],
            }

        return {"role": role, "content": content}

    def _convert_tool(self, tool: dict) -> dict:
        """Convert tool to Anthropic format."""
        if tool.get("type") == "function":
            func = tool.get("function", {})
            return {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {}),
            }
        return tool
