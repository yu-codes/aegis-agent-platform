"""
Unit Tests - Tools Service

Tests for tool components.
"""

import pytest


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_registry_initialization(self):
        """Test registry can be initialized."""
        from services.tools import ToolRegistry

        registry = ToolRegistry()
        assert registry is not None

    def test_tool_registration(self):
        """Test tool registration."""
        from services.tools import ToolRegistry

        registry = ToolRegistry()

        @registry.tool(
            name="test_tool",
            description="A test tool",
        )
        def test_func(x: int) -> int:
            return x * 2

        tools = registry.list_tools()
        tool_names = [t.name for t in tools]
        assert "test_tool" in tool_names

    def test_openai_schema_generation(self):
        """Test OpenAI schema generation."""
        from services.tools import ToolRegistry

        registry = ToolRegistry()

        @registry.tool(
            name="multiply",
            description="Multiply two numbers",
        )
        def multiply(a: int, b: int) -> int:
            return a * b

        schema = registry.get_openai_schema("multiply")

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "multiply"
        assert "parameters" in schema["function"]


class TestToolExecutor:
    """Tests for ToolExecutor."""

    def test_executor_initialization(self):
        """Test executor can be initialized."""
        from services.tools import ToolExecutor, ToolRegistry

        registry = ToolRegistry()
        executor = ToolExecutor(registry=registry)
        assert executor is not None

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test tool execution."""
        from services.tools import ToolExecutor, ToolRegistry

        registry = ToolRegistry()

        @registry.tool(name="add", description="Add numbers")
        def add(a: int, b: int) -> int:
            return a + b

        executor = ToolExecutor(registry=registry)
        result = await executor.execute("add", {"a": 2, "b": 3})

        assert result.success
        assert result.result == 5

    @pytest.mark.asyncio
    async def test_tool_timeout(self):
        """Test tool execution timeout."""
        import asyncio
        from services.tools import ToolExecutor, ToolRegistry

        registry = ToolRegistry()

        @registry.tool(name="slow_tool", description="Slow tool")
        async def slow_tool() -> str:
            await asyncio.sleep(10)
            return "done"

        executor = ToolExecutor(registry=registry, default_timeout=0.1)
        result = await executor.execute("slow_tool", {})

        assert not result.success
        assert "timeout" in result.error.lower()


class TestToolValidator:
    """Tests for ToolValidator."""

    def test_validator_initialization(self):
        """Test validator can be initialized."""
        from services.tools import ToolValidator

        validator = ToolValidator()
        assert validator is not None

    def test_input_validation(self):
        """Test input validation."""
        from services.tools import ToolValidator

        validator = ToolValidator()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
            },
            "required": ["name"],
        }

        # Valid input
        result = validator.validate_input("test", {"name": "John", "age": 30}, schema)
        assert result.valid

        # Missing required field
        result = validator.validate_input("test", {"age": 30}, schema)
        assert not result.valid

    def test_injection_detection(self):
        """Test injection detection."""
        from services.tools import ToolValidator

        validator = ToolValidator()

        # SQL injection
        result = validator.check_injection("SELECT * FROM users; DROP TABLE users;")
        assert result.detected

        # Command injection
        result = validator.check_injection("file.txt; rm -rf /")
        assert result.detected

        # Safe input
        result = validator.check_injection("Hello, world!")
        assert not result.detected


class TestBuiltinTools:
    """Tests for built-in tools."""

    def test_calculator_tool(self):
        """Test calculator tool."""
        from services.tools.builtins import CalculatorTool

        calc = CalculatorTool()

        # Basic math
        result = calc.execute("2 + 2")
        assert result == 4

        # Functions
        result = calc.execute("sqrt(16)")
        assert result == 4.0

    def test_code_analyzer_basic(self):
        """Test code analyzer."""
        from services.tools.builtins import CodeAnalyzerTool

        analyzer = CodeAnalyzerTool()

        code = """
def hello():
    print("Hello, World!")
"""

        result = analyzer.execute(code, "python")
        assert "functions" in result
        assert "hello" in result["functions"]
