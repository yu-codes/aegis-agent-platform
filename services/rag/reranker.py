"""
Reranker

Cross-encoder reranking for improved relevance.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class RerankerConfig:
    """Configuration for reranker."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5
    min_score: float = 0.0
    batch_size: int = 32


@dataclass
class RankedResult:
    """A reranked result."""

    id: str = ""
    content: str = ""
    original_score: float = 0.0
    rerank_score: float = 0.0
    metadata: dict[str, Any] | None = None


class Reranker:
    """
    Cross-encoder reranking.

    Reranks initial retrieval results for better relevance.
    """

    def __init__(self, config: RerankerConfig | None = None):
        self._config = config or RerankerConfig()
        self._model = None

    def _get_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self._config.model_name)
            except ImportError:
                return None
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[RankedResult]:
        """
        Rerank documents by relevance to query.

        Args:
            query: The search query
            documents: List of documents with 'content' and optional 'id', 'score', 'metadata'
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        if not documents:
            return []

        k = top_k or self._config.top_k
        model = self._get_model()

        if model is None:
            # Fallback: return original order
            return [
                RankedResult(
                    id=doc.get("id", str(i)),
                    content=doc.get("content", ""),
                    original_score=doc.get("score", 0.0),
                    rerank_score=doc.get("score", 0.0),
                    metadata=doc.get("metadata"),
                )
                for i, doc in enumerate(documents[:k])
            ]

        # Prepare pairs
        pairs = [(query, doc.get("content", "")) for doc in documents]

        # Score in batches
        all_scores = []
        for i in range(0, len(pairs), self._config.batch_size):
            batch = pairs[i : i + self._config.batch_size]
            scores = model.predict(batch)
            all_scores.extend(scores)

        # Create results
        results = []
        for i, (doc, score) in enumerate(zip(documents, all_scores)):
            results.append(
                RankedResult(
                    id=doc.get("id", str(i)),
                    content=doc.get("content", ""),
                    original_score=doc.get("score", 0.0),
                    rerank_score=float(score),
                    metadata=doc.get("metadata"),
                )
            )

        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score, reverse=True)

        # Filter and limit
        results = [r for r in results if r.rerank_score >= self._config.min_score]
        return results[:k]

    async def rerank_async(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[RankedResult]:
        """Async wrapper for reranking."""
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None, self.rerank, query, documents, top_k
        )
