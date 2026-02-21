"""
Unit Tests - Memory Service

Tests for memory components.
"""

import pytest
from datetime import datetime


class TestSessionMemory:
    """Tests for SessionMemory."""

    def test_session_memory_initialization(self):
        """Test session memory can be initialized."""
        from services.memory import SessionMemory

        memory = SessionMemory()
        assert memory is not None

    @pytest.mark.asyncio
    async def test_session_creation(self):
        """Test session creation."""
        from services.memory import SessionMemory

        memory = SessionMemory()
        session_id = await memory.create_session()

        assert session_id is not None
        assert len(session_id) > 0

    @pytest.mark.asyncio
    async def test_message_operations(self):
        """Test adding and retrieving messages."""
        from services.memory import SessionMemory

        memory = SessionMemory()
        session_id = await memory.create_session()

        await memory.add_message(session_id, "user", "Hello!")
        await memory.add_message(session_id, "assistant", "Hi there!")

        history = await memory.get_history(session_id)
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"


class TestLongTermMemory:
    """Tests for LongTermMemory."""

    def test_long_term_memory_initialization(self):
        """Test long term memory can be initialized."""
        from services.memory import LongTermMemory

        memory = LongTermMemory()
        assert memory is not None

    @pytest.mark.asyncio
    async def test_memory_storage(self):
        """Test storing memories."""
        from services.memory import LongTermMemory

        memory = LongTermMemory()

        memory_id = await memory.store(
            content="User prefers Python over JavaScript",
            user_id="user123",
            importance=0.8,
        )

        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_memory_retrieval(self):
        """Test retrieving memories."""
        from services.memory import LongTermMemory

        memory = LongTermMemory()

        await memory.store(
            content="User works at a tech company",
            user_id="user123",
            importance=0.7,
        )

        results = await memory.retrieve(
            query="What is the user's job?",
            user_id="user123",
            limit=5,
        )

        assert isinstance(results, list)


class TestSummarizer:
    """Tests for Summarizer."""

    def test_summarizer_initialization(self):
        """Test summarizer can be initialized."""
        from services.memory import Summarizer

        summarizer = Summarizer()
        assert summarizer is not None

    @pytest.mark.asyncio
    async def test_extractive_summary(self):
        """Test extractive summarization."""
        from services.memory import Summarizer

        summarizer = Summarizer()

        text = """
        User: Hi, I need help with Python.
        Assistant: Of course! What do you need help with?
        User: I'm trying to read a CSV file.
        Assistant: You can use pandas.read_csv() for that.
        User: Thanks, that worked!
        """

        summary = await summarizer.summarize_extractive(text, max_sentences=2)
        assert len(summary) > 0
        assert len(summary) < len(text)


class TestVectorMemory:
    """Tests for VectorMemory."""

    def test_vector_memory_initialization(self):
        """Test vector memory can be initialized."""
        from services.memory import VectorMemory

        memory = VectorMemory()
        assert memory is not None
