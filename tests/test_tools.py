"""
Tests for Tool Registry and Execution
"""

import pytest

from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry, tool


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()

        @registry.register
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert "add" in registry.list_tools()

    def test_get_tool(self):
        """Test getting a registered tool."""
        registry = ToolRegistry()

        @registry.register
        def multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        tool_def = registry.get("multiply")
        assert tool_def is not None
        assert tool_def.name == "multiply"

    def test_tool_decorator(self):
        """Test the @tool decorator."""
        @tool(name="custom_name", description="Custom tool")
        def my_func(value: str) -> str:
            return value.upper()

        assert my_func._tool_metadata["name"] == "custom_name"
        assert my_func._tool_metadata["description"] == "Custom tool"

    def test_list_tools(self):
        """Test listing all tools."""
        registry = ToolRegistry()

        @registry.register
        def tool1():
            pass

        @registry.register
        def tool2():
            pass

        tools = registry.list_tools()
        assert "tool1" in tools
        assert "tool2" in tools

    def test_openai_format(self):
        """Test OpenAI tool format export."""
        registry = ToolRegistry()

        @registry.register
        def greet(name: str) -> str:
            """Greet a person."""
            return f"Hello, {name}!"

        openai_tools = registry.to_openai_format()
        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "greet"


class TestToolExecutor:
    """Tests for ToolExecutor."""

    @pytest.fixture
    def registry_with_tools(self):
        """Create a registry with test tools."""
        registry = ToolRegistry()

        @registry.register
        def echo(message: str) -> str:
            """Echo a message."""
            return message

        @registry.register
        def divide(a: float, b: float) -> float:
            """Divide a by b."""
            if b == 0:
                raise ValueError("Cannot divide by zero")
            return a / b

        return registry

    @pytest.mark.asyncio
    async def test_execute_tool(self, registry_with_tools):
        """Test executing a tool."""
        executor = ToolExecutor(registry_with_tools)
        result = await executor.execute("echo", {"message": "hello"})

        assert result.success is True
        assert result.output == "hello"

    @pytest.mark.asyncio
    async def test_execute_with_error(self, registry_with_tools):
        """Test executing a tool that raises an error."""
        executor = ToolExecutor(registry_with_tools)
        result = await executor.execute("divide", {"a": 1, "b": 0})

        assert result.success is False
        assert "divide by zero" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, registry_with_tools):
        """Test executing an unknown tool."""
        executor = ToolExecutor(registry_with_tools)
        result = await executor.execute("unknown_tool", {})

        assert result.success is False
        assert "not found" in result.error.lower()
