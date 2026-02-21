"""
Test Configuration

Shared fixtures and test utilities for the monorepo.
"""

import asyncio
import sys
from typing import AsyncGenerator
from uuid import uuid4

import pytest
from httpx import AsyncClient, ASGITransport


# Configure event loop for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the test session."""
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def app():
    """Create test application."""
    from apps.api_server.app import create_app

    app = create_app(enable_docs=True, debug=True)
    yield app


@pytest.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_chat_request():
    """Sample chat request data."""
    return {
        "query": "Hello, how are you?",
        "session_id": f"test-session-{uuid4()}",
        "domain": "general_chat",
    }


@pytest.fixture
def sample_tool_request():
    """Sample tool execution request."""
    return {
        "tool_name": "calculator",
        "arguments": {"expression": "2 + 2"},
    }


@pytest.fixture
def sample_session():
    """Sample session data."""
    return {
        "user_id": "test-user",
        "domain": "general_chat",
        "metadata": {"source": "test"},
    }


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]


# Mark all tests as async by default
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests")
