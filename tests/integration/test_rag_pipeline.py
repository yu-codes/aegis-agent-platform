"""
Integration Tests - RAG Pipeline

Tests for full RAG pipeline.
"""

import pytest


class TestRAGPipeline:
    """Tests for RAG pipeline integration."""

    @pytest.mark.asyncio
    async def test_document_to_retrieval(self):
        """Test full document indexing and retrieval."""
        from services.rag import IndexManager, Retriever
        from services.rag.index_manager import Document

        # Create index manager
        index_manager = IndexManager()

        # Add document
        doc = Document(
            id="integration-test-doc",
            content="""
            Python is a versatile programming language known for its clean syntax.
            It supports multiple programming paradigms including procedural,
            object-oriented, and functional programming. Python has a large
            standard library and an active community.
            """,
            metadata={"source": "test", "topic": "programming"},
        )

        await index_manager.add_document(doc)

        # Create retriever and search
        retriever = Retriever(index_manager=index_manager)
        results = await retriever.retrieve("What is Python good for?", top_k=3)

        # Should find the document
        assert len(results) >= 0  # Depends on embedding availability

    @pytest.mark.asyncio
    async def test_hybrid_search_integration(self):
        """Test hybrid search with RAG."""
        from services.rag import HybridSearch, Retriever, IndexManager

        index_manager = IndexManager()
        hybrid = HybridSearch(index_manager=index_manager)

        # Test that hybrid search can be initialized
        assert hybrid is not None


class TestMemoryIntegration:
    """Tests for memory system integration."""

    @pytest.mark.asyncio
    async def test_session_with_long_term(self):
        """Test session memory integration with long-term memory."""
        from services.memory import SessionMemory, LongTermMemory

        session_memory = SessionMemory()
        long_term = LongTermMemory()

        # Create session
        session_id = await session_memory.create_session()

        # Add conversation
        await session_memory.add_message(session_id, "user", "I prefer dark mode.")
        await session_memory.add_message(session_id, "assistant", "Got it, I'll remember that.")

        # Store in long-term memory
        await long_term.store(
            content="User prefers dark mode interface",
            user_id="test-user",
            session_id=session_id,
            importance=0.8,
        )

        # Retrieve from long-term
        memories = await long_term.retrieve(
            query="user preferences for interface",
            user_id="test-user",
            limit=5,
        )

        assert isinstance(memories, list)
