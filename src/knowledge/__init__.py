"""
Knowledge / RAG Module

Document ingestion, chunking, embedding, and retrieval.
Provides the knowledge base for retrieval-augmented generation.
"""

from src.knowledge.chunking import ChunkingStrategy, RecursiveChunker, SemanticChunker
from src.knowledge.embeddings import EmbeddingService, OpenAIEmbeddings
from src.knowledge.ingestion import Document, DocumentChunk, DocumentIngester
from src.knowledge.retriever import ContextAssembler, Retriever
from src.knowledge.vector_store import FAISSVectorStore, MilvusVectorStore, VectorStore

__all__ = [
    # Chunking
    "ChunkingStrategy",
    "ContextAssembler",
    "Document",
    "DocumentChunk",
    # Ingestion
    "DocumentIngester",
    # Embeddings
    "EmbeddingService",
    "FAISSVectorStore",
    "MilvusVectorStore",
    "OpenAIEmbeddings",
    "RecursiveChunker",
    # Retrieval
    "Retriever",
    "SemanticChunker",
    # Vector Store
    "VectorStore",
]
