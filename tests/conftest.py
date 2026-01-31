"""
Pytest Configuration

Shared fixtures and configuration for tests.
"""

from uuid import uuid4

import pytest

from src.core.types import ExecutionContext, Message, Session


@pytest.fixture
def sample_session() -> Session:
    """Create a sample session for testing."""
    return Session(
        id=uuid4(),
        user_id="test-user",
    )


@pytest.fixture
def sample_context(sample_session: Session) -> ExecutionContext:
    """Create a sample execution context."""
    return ExecutionContext(
        session_id=sample_session.id,
        request_id=str(uuid4()),
        user_id="test-user",
    )


@pytest.fixture
def sample_messages() -> list[Message]:
    """Create sample messages for testing."""
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello, how are you?"),
    ]
