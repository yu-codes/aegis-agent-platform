"""
Long-Term Memory

Persistent memory that survives across sessions.
Used for user preferences, learned facts, and important context.

Design decisions:
- Vector-based storage for semantic retrieval
- Structured entries with metadata
- Decay/importance scoring
- Integration with knowledge base (but separate namespace)
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4
import json

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """
    A single entry in long-term memory.
    
    Entries are semantically searchable and include
    metadata for filtering and relevance scoring.
    """
    
    id: UUID = Field(default_factory=uuid4)
    
    # Content
    content: str
    summary: str | None = None  # Short version for quick retrieval
    
    # Classification
    memory_type: str = "general"  # fact, preference, context, task, etc.
    topics: list[str] = Field(default_factory=list)
    
    # Ownership
    user_id: str | None = None
    session_id: UUID | None = None
    
    # Importance and recency
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = 0
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Embedding (populated by memory store)
    embedding: list[float] | None = None
    
    # Custom metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    def record_access(self) -> None:
        """Record that this memory was accessed."""
        self.access_count += 1
        self.last_accessed_at = datetime.utcnow()
    
    def calculate_relevance(
        self,
        similarity_score: float,
        recency_weight: float = 0.2,
        importance_weight: float = 0.3,
        frequency_weight: float = 0.1,
    ) -> float:
        """
        Calculate overall relevance score.
        
        Combines:
        - Semantic similarity
        - Recency (how recently accessed)
        - Importance (explicit weight)
        - Frequency (access count)
        """
        # Similarity is the base
        similarity_weight = 1.0 - recency_weight - importance_weight - frequency_weight
        
        # Recency decay (exponential decay over days)
        days_since_access = (datetime.utcnow() - self.last_accessed_at).days
        recency_score = 1.0 / (1.0 + days_since_access * 0.1)
        
        # Frequency score (logarithmic scaling)
        import math
        frequency_score = min(1.0, math.log1p(self.access_count) / 5.0)
        
        return (
            similarity_score * similarity_weight +
            recency_score * recency_weight +
            self.importance * importance_weight +
            frequency_score * frequency_weight
        )


class LongTermMemoryBackend(ABC):
    """Abstract backend for long-term memory storage."""
    
    @abstractmethod
    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry."""
        pass
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        user_id: str | None = None,
        memory_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[tuple[MemoryEntry, float]]:
        """Retrieve memories by semantic similarity."""
        pass
    
    @abstractmethod
    async def get(self, memory_id: UUID) -> MemoryEntry | None:
        """Get a specific memory by ID."""
        pass
    
    @abstractmethod
    async def update(self, entry: MemoryEntry) -> None:
        """Update an existing memory."""
        pass
    
    @abstractmethod
    async def delete(self, memory_id: UUID) -> bool:
        """Delete a memory."""
        pass
    
    @abstractmethod
    async def list_by_user(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        """List memories for a user."""
        pass


class RedisLongTermMemory(LongTermMemoryBackend):
    """
    Redis-based long-term memory.
    
    Uses Redis for storage and a simple in-memory index for
    similarity search. For production, consider using Redis
    with vector search (Redis Stack) or a dedicated vector DB.
    """
    
    def __init__(
        self,
        redis_url: str,
        embedding_service: "EmbeddingService",  # Forward reference
        key_prefix: str = "aegis:ltm:",
    ):
        self._redis_url = redis_url
        self._embedding_service = embedding_service
        self._key_prefix = key_prefix
        self._client = None
        
        # Simple in-memory index (for demo; use proper vector DB in production)
        self._embeddings_index: dict[str, list[float]] = {}
    
    async def _get_client(self):
        if self._client is None:
            import redis.asyncio as redis
            self._client = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client
    
    def _entry_key(self, memory_id: UUID) -> str:
        return f"{self._key_prefix}entry:{memory_id}"
    
    def _user_index_key(self, user_id: str) -> str:
        return f"{self._key_prefix}user:{user_id}"
    
    async def store(self, entry: MemoryEntry) -> None:
        """Store memory with embedding."""
        # Generate embedding if not present
        if entry.embedding is None:
            entry.embedding = await self._embedding_service.embed(entry.content)
        
        client = await self._get_client()
        
        # Store entry as JSON
        key = self._entry_key(entry.id)
        await client.set(key, entry.model_dump_json())
        
        # Add to user index
        if entry.user_id:
            await client.sadd(self._user_index_key(entry.user_id), str(entry.id))
        
        # Update in-memory embedding index
        self._embeddings_index[str(entry.id)] = entry.embedding
    
    async def retrieve(
        self,
        query: str,
        user_id: str | None = None,
        memory_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[tuple[MemoryEntry, float]]:
        """Retrieve by semantic similarity."""
        # Generate query embedding
        query_embedding = await self._embedding_service.embed(query)
        
        # Calculate similarities
        similarities: list[tuple[str, float]] = []
        for memory_id, embedding in self._embeddings_index.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((memory_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Fetch and filter entries
        results: list[tuple[MemoryEntry, float]] = []
        client = await self._get_client()
        
        for memory_id, similarity in similarities:
            if len(results) >= limit:
                break
            
            entry = await self.get(UUID(memory_id))
            if entry is None:
                continue
            
            # Apply filters
            if user_id and entry.user_id != user_id:
                continue
            if memory_types and entry.memory_type not in memory_types:
                continue
            
            # Record access
            entry.record_access()
            await self.update(entry)
            
            results.append((entry, similarity))
        
        return results
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between vectors."""
        import math
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    async def get(self, memory_id: UUID) -> MemoryEntry | None:
        client = await self._get_client()
        key = self._entry_key(memory_id)
        data = await client.get(key)
        
        if data is None:
            return None
        
        return MemoryEntry.model_validate_json(data)
    
    async def update(self, entry: MemoryEntry) -> None:
        entry.updated_at = datetime.utcnow()
        client = await self._get_client()
        key = self._entry_key(entry.id)
        await client.set(key, entry.model_dump_json())
    
    async def delete(self, memory_id: UUID) -> bool:
        client = await self._get_client()
        key = self._entry_key(memory_id)
        
        # Get entry to find user_id
        entry = await self.get(memory_id)
        if entry is None:
            return False
        
        # Remove from user index
        if entry.user_id:
            await client.srem(self._user_index_key(entry.user_id), str(memory_id))
        
        # Remove from embedding index
        self._embeddings_index.pop(str(memory_id), None)
        
        # Delete entry
        result = await client.delete(key)
        return result > 0
    
    async def list_by_user(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        client = await self._get_client()
        index_key = self._user_index_key(user_id)
        
        # Get all memory IDs for user
        memory_ids = await client.smembers(index_key)
        
        # Fetch entries
        entries = []
        for memory_id in list(memory_ids)[offset:offset + limit]:
            entry = await self.get(UUID(memory_id))
            if entry:
                entries.append(entry)
        
        return entries


class LongTermMemory:
    """
    High-level interface for long-term memory.
    
    Provides convenient methods for common memory operations.
    """
    
    def __init__(self, backend: LongTermMemoryBackend):
        self._backend = backend
    
    async def remember(
        self,
        content: str,
        memory_type: str = "general",
        user_id: str | None = None,
        importance: float = 0.5,
        topics: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Store a new memory."""
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            importance=importance,
            topics=topics or [],
            metadata=metadata or {},
        )
        await self._backend.store(entry)
        return entry
    
    async def recall(
        self,
        query: str,
        user_id: str | None = None,
        memory_types: list[str] | None = None,
        limit: int = 5,
        min_relevance: float = 0.0,
    ) -> list[MemoryEntry]:
        """Recall relevant memories."""
        results = await self._backend.retrieve(
            query,
            user_id=user_id,
            memory_types=memory_types,
            limit=limit,
        )
        
        # Filter by minimum relevance and calculate final scores
        filtered = []
        for entry, similarity in results:
            relevance = entry.calculate_relevance(similarity)
            if relevance >= min_relevance:
                entry.metadata["relevance_score"] = relevance
                filtered.append(entry)
        
        return filtered
    
    async def forget(self, memory_id: UUID) -> bool:
        """Delete a memory."""
        return await self._backend.delete(memory_id)
    
    async def get_user_memories(
        self,
        user_id: str,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Get all memories for a user."""
        return await self._backend.list_by_user(user_id, limit=limit)
    
    async def update_importance(
        self,
        memory_id: UUID,
        importance: float,
    ) -> MemoryEntry | None:
        """Update the importance of a memory."""
        entry = await self._backend.get(memory_id)
        if entry is None:
            return None
        
        entry.importance = max(0.0, min(1.0, importance))
        await self._backend.update(entry)
        return entry


# Forward reference resolution
class EmbeddingService(ABC):
    """Abstract embedding service interface."""
    
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        pass
