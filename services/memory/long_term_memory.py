"""
Long-Term Memory

Persistent memory storage.

Based on: src/memory/long_term.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID, uuid4


@dataclass
class MemoryEntry:
    """A memory entry."""

    id: UUID = field(default_factory=uuid4)
    content: str = ""
    memory_type: str = "fact"  # fact, episode, concept, preference

    # Context
    session_id: str | None = None
    user_id: str | None = None

    # Importance
    importance: float = 0.5  # 0-1 scale
    access_count: int = 0
    last_accessed: datetime | None = None

    # Embedding
    embedding: list[float] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "content": self.content,
            "memory_type": self.memory_type,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if data.get("id") else uuid4(),
            content=data.get("content", ""),
            memory_type=data.get("memory_type", "fact"),
            session_id=data.get("session_id"),
            user_id=data.get("user_id"),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=(
                datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None
            ),
            embedding=data.get("embedding", []),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.utcnow()
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data.get("updated_at")
                else datetime.utcnow()
            ),
            expires_at=(
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            ),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )


class EmbeddingProtocol(Protocol):
    """Protocol for embedding services."""

    async def embed(self, text: str) -> list[float]: ...


class VectorStoreProtocol(Protocol):
    """Protocol for vector stores."""

    async def add(self, embeddings: list[list[float]], metadata: list[dict]) -> list[str]: ...
    async def search(
        self, query_embedding: list[float], top_k: int, filter: dict | None = None
    ) -> list[dict]: ...
    async def delete(self, ids: list[str]) -> None: ...


class LongTermMemory:
    """
    Long-term persistent memory.

    Stores facts, episodes, and preferences.
    """

    def __init__(
        self,
        embedding_service: EmbeddingProtocol | None = None,
        vector_store: VectorStoreProtocol | None = None,
        decay_factor: float = 0.995,  # Daily decay rate
    ):
        self._embeddings = embedding_service
        self._store = vector_store
        self._decay_factor = decay_factor

        # In-memory storage fallback
        self._memories: dict[UUID, MemoryEntry] = {}

    async def store(
        self,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.5,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        """
        Store a new memory.

        Args:
            content: Memory content
            memory_type: Type (fact, episode, concept, preference)
            importance: Importance score (0-1)
            session_id: Associated session
            user_id: Associated user
            metadata: Additional metadata
            tags: Tags for filtering

        Returns:
            Stored memory entry
        """
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            importance=importance,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
            tags=tags or [],
        )

        # Generate embedding
        if self._embeddings:
            entry.embedding = await self._embeddings.embed(content)

        # Store in vector store
        if self._store and entry.embedding:
            await self._store.add(
                [entry.embedding],
                [{"id": str(entry.id), **entry.to_dict()}],
            )

        # Store in memory
        self._memories[entry.id] = entry

        return entry

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        memory_type: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
    ) -> list[MemoryEntry]:
        """
        Retrieve relevant memories.

        Args:
            query: Search query
            top_k: Number of results
            memory_type: Filter by type
            user_id: Filter by user
            tags: Filter by tags

        Returns:
            Retrieved memories
        """
        if self._embeddings and self._store:
            query_embedding = await self._embeddings.embed(query)

            filter_dict: dict[str, Any] = {}
            if memory_type:
                filter_dict["memory_type"] = memory_type
            if user_id:
                filter_dict["user_id"] = user_id

            results = await self._store.search(
                query_embedding,
                top_k=top_k,
                filter=filter_dict if filter_dict else None,
            )

            entries = []
            for result in results:
                if "id" in result:
                    entry_id = UUID(result["id"])
                    if entry_id in self._memories:
                        entry = self._memories[entry_id]
                        entry.access_count += 1
                        entry.last_accessed = datetime.utcnow()
                        entries.append(entry)

            return entries

        # Fallback: simple search
        return self._simple_search(query, top_k, memory_type, user_id, tags)

    async def update(
        self,
        memory_id: UUID,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> MemoryEntry | None:
        """Update an existing memory."""
        if memory_id not in self._memories:
            return None

        entry = self._memories[memory_id]

        if content is not None:
            entry.content = content
            if self._embeddings:
                entry.embedding = await self._embeddings.embed(content)

        if importance is not None:
            entry.importance = importance

        if metadata is not None:
            entry.metadata.update(metadata)

        if tags is not None:
            entry.tags = tags

        entry.updated_at = datetime.utcnow()

        return entry

    async def delete(self, memory_id: UUID) -> bool:
        """Delete a memory."""
        if memory_id not in self._memories:
            return False

        if self._store:
            await self._store.delete([str(memory_id)])

        del self._memories[memory_id]
        return True

    async def decay(self) -> int:
        """
        Apply memory decay.

        Returns number of memories affected.
        """
        count = 0
        to_delete = []

        for entry in self._memories.values():
            # Calculate decay based on access recency
            if entry.last_accessed:
                days_since_access = (datetime.utcnow() - entry.last_accessed).days
                decay_amount = self._decay_factor**days_since_access
                entry.importance *= decay_amount

            # Boost based on access count
            boost = min(1.0 + (entry.access_count * 0.01), 1.2)
            entry.importance *= boost

            # Cap importance
            entry.importance = min(entry.importance, 1.0)

            # Mark for deletion if expired or very low importance
            if entry.expires_at and entry.expires_at < datetime.utcnow():
                to_delete.append(entry.id)
            elif entry.importance < 0.01:
                to_delete.append(entry.id)
            else:
                count += 1

        # Delete decayed memories
        for memory_id in to_delete:
            await self.delete(memory_id)

        return count

    def _simple_search(
        self,
        query: str,
        top_k: int,
        memory_type: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
    ) -> list[MemoryEntry]:
        """Simple keyword-based search fallback."""
        query_terms = set(query.lower().split())

        scored = []
        for entry in self._memories.values():
            # Apply filters
            if memory_type and entry.memory_type != memory_type:
                continue
            if user_id and entry.user_id != user_id:
                continue
            if tags and not any(t in entry.tags for t in tags):
                continue

            # Score by keyword overlap
            entry_terms = set(entry.content.lower().split())
            overlap = len(query_terms & entry_terms)
            if overlap > 0:
                score = (overlap / len(query_terms)) * entry.importance
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, entry in scored[:top_k]:
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            results.append(entry)

        return results

    def get_stats(self) -> dict:
        """Get memory statistics."""
        type_counts = {}
        for entry in self._memories.values():
            type_counts[entry.memory_type] = type_counts.get(entry.memory_type, 0) + 1

        return {
            "total_memories": len(self._memories),
            "by_type": type_counts,
            "avg_importance": (
                sum(e.importance for e in self._memories.values()) / len(self._memories)
                if self._memories
                else 0
            ),
        }
