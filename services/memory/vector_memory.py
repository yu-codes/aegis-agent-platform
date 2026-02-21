"""
Vector Memory

Vector-based memory retrieval.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID, uuid4


@dataclass
class MemoryRecord:
    """A memory record with vector embedding."""

    id: UUID = field(default_factory=uuid4)
    content: str = ""
    embedding: list[float] = field(default_factory=list)

    # Context
    context_type: str = "general"  # general, conversation, task, fact
    source_session: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Scoring
    relevance_score: float = 0.0
    recency_score: float = 1.0


class EmbeddingProtocol(Protocol):
    """Protocol for embedding services."""

    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class VectorMemory:
    """
    Vector-based memory system.

    Uses embeddings for semantic memory retrieval.
    """

    def __init__(
        self,
        embedding_service: EmbeddingProtocol | None = None,
        recency_decay: float = 0.99,  # Daily decay rate
    ):
        self._embeddings = embedding_service
        self._recency_decay = recency_decay

        # In-memory storage
        self._records: dict[UUID, MemoryRecord] = {}

    async def add(
        self,
        content: str,
        context_type: str = "general",
        source_session: str | None = None,
        metadata: dict | None = None,
    ) -> MemoryRecord:
        """
        Add a memory record.

        Args:
            content: Memory content
            context_type: Type of context
            source_session: Source session ID
            metadata: Additional metadata

        Returns:
            Created record
        """
        record = MemoryRecord(
            content=content,
            context_type=context_type,
            source_session=source_session,
            metadata=metadata or {},
        )

        if self._embeddings:
            record.embedding = await self._embeddings.embed(content)

        self._records[record.id] = record
        return record

    async def add_batch(
        self,
        contents: list[str],
        context_type: str = "general",
        source_session: str | None = None,
    ) -> list[MemoryRecord]:
        """Add multiple memory records."""
        records = []

        if self._embeddings:
            embeddings = await self._embeddings.embed_batch(contents)
            for content, embedding in zip(contents, embeddings):
                record = MemoryRecord(
                    content=content,
                    embedding=embedding,
                    context_type=context_type,
                    source_session=source_session,
                )
                self._records[record.id] = record
                records.append(record)
        else:
            for content in contents:
                record = await self.add(content, context_type, source_session)
                records.append(record)

        return records

    async def search(
        self,
        query: str,
        top_k: int = 5,
        context_type: str | None = None,
        min_score: float = 0.0,
        include_recency: bool = True,
    ) -> list[MemoryRecord]:
        """
        Search for relevant memories.

        Args:
            query: Search query
            top_k: Number of results
            context_type: Filter by context type
            min_score: Minimum similarity score
            include_recency: Factor in recency

        Returns:
            Relevant memory records
        """
        if not self._records:
            return []

        # Get query embedding
        if self._embeddings:
            query_embedding = await self._embeddings.embed(query)
        else:
            return self._keyword_search(query, top_k, context_type)

        # Score all records
        scored = []
        for record in self._records.values():
            # Apply type filter
            if context_type and record.context_type != context_type:
                continue

            if not record.embedding:
                continue

            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding, record.embedding)

            if similarity < min_score:
                continue

            # Calculate recency score
            if include_recency:
                days_old = (datetime.utcnow() - record.created_at).days
                recency = self._recency_decay**days_old
            else:
                recency = 1.0

            # Combined score
            record.relevance_score = similarity
            record.recency_score = recency
            final_score = similarity * 0.8 + recency * 0.2

            scored.append((final_score, record))

        # Sort and return top k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        context_type: str | None,
    ) -> list[MemoryRecord]:
        """Fallback keyword search."""
        query_terms = set(query.lower().split())

        scored = []
        for record in self._records.values():
            if context_type and record.context_type != context_type:
                continue

            record_terms = set(record.content.lower().split())
            overlap = len(query_terms & record_terms)

            if overlap > 0:
                score = overlap / len(query_terms)
                record.relevance_score = score
                scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def delete(self, record_id: UUID) -> bool:
        """Delete a memory record."""
        if record_id in self._records:
            del self._records[record_id]
            return True
        return False

    def clear(self, context_type: str | None = None) -> int:
        """
        Clear memory records.

        Args:
            context_type: Only clear this type, or all if None

        Returns:
            Number of records cleared
        """
        if context_type is None:
            count = len(self._records)
            self._records.clear()
            return count

        to_delete = [
            rid for rid, record in self._records.items() if record.context_type == context_type
        ]

        for rid in to_delete:
            del self._records[rid]

        return len(to_delete)

    def get_stats(self) -> dict:
        """Get memory statistics."""
        type_counts = {}
        for record in self._records.values():
            type_counts[record.context_type] = type_counts.get(record.context_type, 0) + 1

        return {
            "total_records": len(self._records),
            "by_type": type_counts,
            "has_embeddings": sum(1 for r in self._records.values() if r.embedding),
        }
