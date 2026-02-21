"""
Retriever

Semantic retrieval from vector store.

Based on: src/knowledge/retriever.py
"""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""

    top_k: int = 5
    min_score: float = 0.0
    include_metadata: bool = True
    rerank: bool = False
    rerank_top_k: int = 3


@dataclass
class RetrievedDocument:
    """A retrieved document."""

    id: str = ""
    content: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    chunk_index: int | None = None


class EmbeddingProtocol(Protocol):
    """Protocol for embedding services."""

    async def embed(self, text: str) -> list[float]: ...


class VectorStoreProtocol(Protocol):
    """Protocol for vector stores."""

    async def search(
        self, query_embedding: list[float], top_k: int, filter: dict | None = None
    ) -> list[dict]: ...


class Retriever:
    """
    Semantic document retriever.

    Retrieves relevant documents using vector similarity.
    """

    def __init__(
        self,
        embedding_service: EmbeddingProtocol | None = None,
        vector_store: VectorStoreProtocol | None = None,
        config: RetrievalConfig | None = None,
    ):
        self._embeddings = embedding_service
        self._store = vector_store
        self._config = config or RetrievalConfig()

        # In-memory fallback
        self._documents: list[RetrievedDocument] = []

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve relevant documents.

        Args:
            query: Search query
            top_k: Number of results (overrides config)
            filter: Metadata filter

        Returns:
            List of retrieved documents
        """
        k = top_k or self._config.top_k

        if self._embeddings and self._store:
            # Vector search
            query_embedding = await self._embeddings.embed(query)
            results = await self._store.search(query_embedding, top_k=k, filter=filter)

            documents = [
                RetrievedDocument(
                    id=r.get("id", ""),
                    content=r.get("content", ""),
                    score=r.get("score", 0.0),
                    metadata=r.get("metadata", {}),
                    source=r.get("source") or r.get("metadata", {}).get("source"),
                    chunk_index=r.get("chunk_index") or r.get("metadata", {}).get("chunk_index"),
                )
                for r in results
            ]
        else:
            # Fallback: simple keyword matching
            documents = self._keyword_search(query, k)

        # Filter by minimum score
        documents = [d for d in documents if d.score >= self._config.min_score]

        # Rerank if enabled
        if self._config.rerank:
            documents = await self._rerank(query, documents)

        return documents

    async def retrieve_with_context(
        self,
        query: str,
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> tuple[list[RetrievedDocument], str]:
        """
        Retrieve and format as context string.

        Returns:
            Tuple of (documents, formatted_context)
        """
        documents = await self.retrieve(query, top_k, filter)
        context = self._format_context(documents)
        return documents, context

    def _format_context(self, documents: list[RetrievedDocument]) -> str:
        """Format documents as context string."""
        if not documents:
            return ""

        lines = []
        for i, doc in enumerate(documents, 1):
            source = doc.source or "Unknown"
            lines.append(f"[Document {i}] (Source: {source}, Score: {doc.score:.2f})")
            lines.append(doc.content)
            lines.append("")

        return "\n".join(lines)

    def _keyword_search(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedDocument]:
        """Simple keyword-based search fallback."""
        query_terms = set(query.lower().split())

        scored = []
        for doc in self._documents:
            doc_terms = set(doc.content.lower().split())
            overlap = len(query_terms & doc_terms)
            if overlap > 0:
                score = overlap / len(query_terms)
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scored[:top_k]:
            doc.score = score
            results.append(doc)

        return results

    async def _rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Rerank documents using cross-encoder."""
        if not documents:
            return documents

        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(query, doc.content) for doc in documents]
            scores = model.predict(pairs)

            for doc, score in zip(documents, scores):
                doc.score = float(score)

            documents.sort(key=lambda x: x.score, reverse=True)
            return documents[: self._config.rerank_top_k]

        except ImportError:
            # Fall back to original ranking
            return documents

    def add_document(self, doc: RetrievedDocument) -> None:
        """Add document to in-memory store (for testing)."""
        self._documents.append(doc)

    def clear(self) -> None:
        """Clear in-memory documents."""
        self._documents.clear()
