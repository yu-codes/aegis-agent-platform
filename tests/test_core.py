"""
Tests for Core Types and Data Models
"""

from uuid import uuid4

from src.core.types import (
    ExecutionContext,
    LLMResponse,
    Message,
    Session,
    ToolCall,
    ToolResult,
)


class TestMessage:
    """Tests for Message type."""

    def test_create_message(self):
        """Test creating a basic message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls is None

    def test_message_with_tool_calls(self):
        """Test message with tool calls."""
        tool_call = ToolCall(
            id="call_123",
            name="search",
            arguments={"query": "test"},
        )
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[tool_call],
        )
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"

    def test_message_dict_conversion(self):
        """Test converting message to dict."""
        msg = Message(role="user", content="Hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"


class TestSession:
    """Tests for Session type."""

    def test_create_session(self):
        """Test creating a session."""
        session = Session()
        assert session.id is not None
        assert session.messages == []

    def test_session_with_user(self):
        """Test session with user ID."""
        session = Session(user_id="user123")
        assert session.user_id == "user123"

    def test_add_message_to_session(self):
        """Test adding messages to session."""
        session = Session()
        msg = Message(role="user", content="Hello")
        session.messages.append(msg)
        assert len(session.messages) == 1

    def test_session_metadata(self):
        """Test session metadata."""
        session = Session(metadata={"key": "value"})
        assert session.metadata["key"] == "value"


class TestToolCall:
    """Tests for ToolCall type."""

    def test_create_tool_call(self):
        """Test creating a tool call."""
        tc = ToolCall(
            id="call_1",
            name="calculator",
            arguments={"expression": "2+2"},
        )
        assert tc.name == "calculator"
        assert tc.arguments["expression"] == "2+2"

    def test_tool_call_auto_id(self):
        """Test auto-generated ID."""
        tc = ToolCall(name="test", arguments={})
        assert tc.id is not None


class TestToolResult:
    """Tests for ToolResult type."""

    def test_success_result(self):
        """Test successful tool result."""
        result = ToolResult(
            tool_call_id="call_1",
            output="4",
            success=True,
        )
        assert result.success is True
        assert result.output == "4"

    def test_error_result(self):
        """Test error tool result."""
        result = ToolResult(
            tool_call_id="call_1",
            output="",
            error="Division by zero",
            success=False,
        )
        assert result.success is False
        assert result.error == "Division by zero"


class TestLLMResponse:
    """Tests for LLMResponse type."""

    def test_create_response(self):
        """Test creating LLM response."""
        response = LLMResponse(
            content="Hello!",
            model="gpt-4",
        )
        assert response.content == "Hello!"
        assert response.model == "gpt-4"

    def test_response_with_usage(self):
        """Test response with usage stats."""
        response = LLMResponse(
            content="Test",
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        assert response.usage["prompt_tokens"] == 10


class TestExecutionContext:
    """Tests for ExecutionContext type."""

    def test_create_context(self):
        """Test creating execution context."""
        ctx = ExecutionContext(
            session_id=uuid4(),
            request_id="req_123",
        )
        assert ctx.request_id == "req_123"

    def test_context_with_user(self):
        """Test context with user info."""
        ctx = ExecutionContext(
            session_id=uuid4(),
            user_id="user123",
        )
        assert ctx.user_id == "user123"

    def test_context_metadata(self):
        """Test context metadata."""
        ctx = ExecutionContext(
            session_id=uuid4(),
            metadata={"trace_id": "abc"},
        )
        assert ctx.metadata["trace_id"] == "abc"
