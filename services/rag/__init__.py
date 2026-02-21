"""
RAG Service

Retrieval-augmented generation service.

Components:
- IndexManager: Document indexing
- Retriever: Semantic retrieval
- HybridSearch: Combined vector + keyword search
- Reranker: Cross-encoder reranking
- DomainRegistry: Domain-specific configurations
"""

from services.rag.index_manager import IndexManager
from services.rag.retriever import Retriever, RetrievalConfig
from services.rag.hybrid_search import HybridSearch
from services.rag.reranker import Reranker
from services.rag.domain_registry import DomainRegistry, DomainConfig

__all__ = [
    "IndexManager",
    "Retriever",
    "RetrievalConfig",
    "HybridSearch",
    "Reranker",
    "DomainRegistry",
    "DomainConfig",
]
