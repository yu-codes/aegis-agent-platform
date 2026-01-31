"""
Knowledge / RAG Module

Document ingestion, chunking, embedding, and retrieval.
Provides the knowledge base for retrieval-augmented generation.
"""

from src.knowledge.ingestion import DocumentIngester, Document, DocumentChunk
from src.knowledge.chunking import ChunkingStrategy, RecursiveChunker, SemanticChunker
from src.knowledge.embeddings import EmbeddingService, OpenAIEmbeddings
from src.knowledge.vector_store import VectorStore, FAISSVectorStore, MilvusVectorStore
from src.knowledge.retriever import Retriever, ContextAssembler

__all__ = [
    # Ingestion
    "DocumentIngester",
    "Document",
    "DocumentChunk",
    # Chunking
    "ChunkingStrategy",
    "RecursiveChunker",
    "SemanticChunker",
    # Embeddings
    "EmbeddingService",
    "OpenAIEmbeddings",
    # Vector Store
    "VectorStore",
    "FAISSVectorStore",
    "MilvusVectorStore",
    # Retrieval
    "Retriever",
    "ContextAssembler",
]
