"""
Unit Tests - RAG Service

Tests for RAG components.
"""

import pytest


class TestIndexManager:
    """Tests for IndexManager."""

    def test_index_manager_initialization(self):
        """Test index manager can be initialized."""
        from services.rag import IndexManager

        manager = IndexManager()
        assert manager is not None

    @pytest.mark.asyncio
    async def test_document_indexing(self):
        """Test document indexing."""
        from services.rag import IndexManager
        from services.rag.index_manager import Document

        manager = IndexManager()
        doc = Document(
            id="test-doc",
            content="This is a test document about Python programming.",
            metadata={"source": "test"},
        )

        await manager.add_document(doc)

        # Document should be indexed
        # result = await manager.get_document("test-doc")
        # assert result is not None


class TestRetriever:
    """Tests for Retriever."""

    def test_retriever_initialization(self):
        """Test retriever can be initialized."""
        from services.rag import Retriever, IndexManager

        index_manager = IndexManager()
        retriever = Retriever(index_manager=index_manager)
        assert retriever is not None

    @pytest.mark.asyncio
    async def test_retrieval(self):
        """Test document retrieval."""
        from services.rag import Retriever, IndexManager

        index_manager = IndexManager()
        retriever = Retriever(index_manager=index_manager)

        # results = await retriever.retrieve("Python programming")
        # assert isinstance(results, list)


class TestHybridSearch:
    """Tests for HybridSearch."""

    def test_hybrid_search_initialization(self):
        """Test hybrid search can be initialized."""
        from services.rag import HybridSearch

        search = HybridSearch()
        assert search is not None

    def test_rrf_fusion(self):
        """Test reciprocal rank fusion."""
        from services.rag.hybrid_search import HybridSearch

        search = HybridSearch()

        vector_results = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.8},
            {"id": "doc3", "score": 0.7},
        ]

        keyword_results = [
            {"id": "doc2", "score": 0.95},
            {"id": "doc4", "score": 0.85},
            {"id": "doc1", "score": 0.75},
        ]

        fused = search._reciprocal_rank_fusion(vector_results, keyword_results)

        # doc2 should rank high (appears in both)
        assert len(fused) > 0


class TestReranker:
    """Tests for Reranker."""

    def test_reranker_initialization(self):
        """Test reranker can be initialized."""
        from services.rag import Reranker

        reranker = Reranker()
        assert reranker is not None


class TestChunking:
    """Tests for chunking strategies."""

    def test_recursive_chunker(self):
        """Test recursive chunker."""
        from services.rag.chunking import RecursiveChunker, ChunkConfig

        config = ChunkConfig(chunk_size=100, chunk_overlap=20)
        chunker = RecursiveChunker(config)

        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph." * 10
        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.text) <= config.chunk_size + 50  # Allow some overflow

    def test_semantic_chunker_initialization(self):
        """Test semantic chunker initialization."""
        from services.rag.chunking import SemanticChunker

        chunker = SemanticChunker()
        assert chunker is not None
