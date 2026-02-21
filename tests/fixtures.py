"""
Test Fixtures

Shared test fixtures and configuration.
"""

import pytest
import asyncio
from typing import AsyncGenerator


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""

    class MockLLM:
        async def complete(self, prompt: str, **kwargs) -> str:
            return f"Mock response to: {prompt[:50]}..."

        async def stream(self, prompt: str, **kwargs):
            for word in "This is a mock streaming response.".split():
                yield word + " "

    return MockLLM()


@pytest.fixture
def mock_embeddings():
    """Mock embedding function."""
    import random

    def embed(texts: list[str]) -> list[list[float]]:
        return [[random.random() for _ in range(384)] for _ in texts]

    return embed


@pytest.fixture
async def session_memory():
    """Create session memory for testing."""
    from services.memory import SessionMemory

    memory = SessionMemory()
    yield memory
    # Cleanup


@pytest.fixture
async def tool_registry():
    """Create tool registry with test tools."""
    from services.tools import ToolRegistry

    registry = ToolRegistry()

    @registry.tool(name="test_echo", description="Echo input")
    def echo(message: str) -> str:
        return f"Echo: {message}"

    @registry.tool(name="test_add", description="Add numbers")
    def add(a: int, b: int) -> int:
        return a + b

    yield registry


@pytest.fixture
def sample_documents():
    """Sample documents for RAG testing."""
    return [
        {
            "id": "doc1",
            "content": "Python is a programming language with clear syntax and readability.",
            "metadata": {"topic": "programming", "language": "python"},
        },
        {
            "id": "doc2",
            "content": "Machine learning uses algorithms to learn patterns from data.",
            "metadata": {"topic": "ml", "category": "ai"},
        },
        {
            "id": "doc3",
            "content": "FastAPI is a modern web framework for building APIs with Python.",
            "metadata": {"topic": "web", "language": "python"},
        },
    ]


# Pytest markers
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "e2e: end-to-end tests")
    config.addinivalue_line("markers", "slow: slow tests")
    config.addinivalue_line("markers", "requires_llm: tests requiring LLM")
