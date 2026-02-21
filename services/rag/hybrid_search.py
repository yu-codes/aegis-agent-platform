"""
Hybrid Search

Combined vector and keyword search.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""

    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    top_k: int = 10
    min_score: float = 0.0


@dataclass
class SearchResult:
    """A search result."""

    id: str = ""
    content: str = ""
    score: float = 0.0
    vector_score: float = 0.0
    keyword_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class EmbeddingProtocol(Protocol):
    """Protocol for embedding services."""

    async def embed(self, text: str) -> list[float]: ...


class VectorStoreProtocol(Protocol):
    """Protocol for vector stores."""

    async def search(self, query_embedding: list[float], top_k: int) -> list[dict]: ...


class HybridSearch:
    """
    Hybrid search combining vector and keyword search.

    Uses reciprocal rank fusion for score combination.
    """

    def __init__(
        self,
        embedding_service: EmbeddingProtocol | None = None,
        vector_store: VectorStoreProtocol | None = None,
        config: HybridSearchConfig | None = None,
    ):
        self._embeddings = embedding_service
        self._store = vector_store
        self._config = config or HybridSearchConfig()

        # In-memory document store for keyword search
        self._documents: dict[str, dict] = {}

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            top_k: Number of results
            filter: Metadata filter

        Returns:
            Combined search results
        """
        k = top_k or self._config.top_k

        # Get vector results
        vector_results = await self._vector_search(query, k * 2)

        # Get keyword results
        keyword_results = self._keyword_search(query, k * 2)

        # Combine using reciprocal rank fusion
        combined = self._reciprocal_rank_fusion(
            vector_results,
            keyword_results,
            self._config.vector_weight,
            self._config.keyword_weight,
        )

        # Filter and limit
        results = [r for r in combined if r.score >= self._config.min_score]
        return results[:k]

    async def _vector_search(
        self,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        """Perform vector search."""
        if not self._embeddings or not self._store:
            return []

        query_embedding = await self._embeddings.embed(query)
        results = await self._store.search(query_embedding, top_k=top_k)

        return [
            SearchResult(
                id=r.get("id", ""),
                content=r.get("content", ""),
                vector_score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

    def _keyword_search(
        self,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        """Perform keyword search."""
        query_terms = set(query.lower().split())

        scored = []
        for doc_id, doc in self._documents.items():
            content = doc.get("content", "").lower()
            doc_terms = set(content.split())

            # Calculate BM25-like score
            overlap = query_terms & doc_terms
            if overlap:
                tf = len(overlap)
                idf = len(overlap) / len(query_terms)
                score = tf * idf

                scored.append(
                    SearchResult(
                        id=doc_id,
                        content=doc.get("content", ""),
                        keyword_score=score,
                        metadata=doc.get("metadata", {}),
                    )
                )

        scored.sort(key=lambda x: x.keyword_score, reverse=True)
        return scored[:top_k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[SearchResult],
        keyword_results: list[SearchResult],
        vector_weight: float,
        keyword_weight: float,
        k: int = 60,
    ) -> list[SearchResult]:
        """
        Combine results using reciprocal rank fusion.

        RRF score = sum(1 / (k + rank))
        """
        scores: dict[str, SearchResult] = {}

        # Score vector results
        for rank, result in enumerate(vector_results):
            if result.id not in scores:
                scores[result.id] = result
            scores[result.id].score += vector_weight / (k + rank + 1)
            scores[result.id].vector_score = result.vector_score

        # Score keyword results
        for rank, result in enumerate(keyword_results):
            if result.id not in scores:
                scores[result.id] = result
            scores[result.id].score += keyword_weight / (k + rank + 1)
            scores[result.id].keyword_score = result.keyword_score

        # Sort by combined score
        results = list(scores.values())
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add document for keyword search."""
        self._documents[doc_id] = {
            "content": content,
            "metadata": metadata or {},
        }

    def remove_document(self, doc_id: str) -> None:
        """Remove document from keyword index."""
        self._documents.pop(doc_id, None)

    def clear(self) -> None:
        """Clear all documents."""
        self._documents.clear()
