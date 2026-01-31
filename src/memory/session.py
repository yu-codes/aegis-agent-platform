"""
Session Manager

Handles conversation sessions and their lifecycle.
Sessions are the primary unit of state in the system.

Design decisions:
- Sessions are stored in Redis for persistence and sharing
- Session data is JSON-serializable for portability
- Configurable TTL for automatic cleanup
- Support for session metadata (user, tenant, etc.)
"""

import json
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.core.types import Message


class Session(BaseModel):
    """
    A conversation session.
    
    Contains all state related to a single conversation,
    including messages, metadata, and custom data.
    """
    
    id: UUID = Field(default_factory=uuid4)
    
    # Ownership
    user_id: str | None = None
    tenant_id: str | None = None
    
    # Messages
    messages: list[Message] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    
    # Custom data for extensions
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # Agent state
    agent_state: dict[str, Any] = Field(default_factory=dict)
    
    @property
    def message_count(self) -> int:
        return len(self.messages)
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def add_message(self, message: Message) -> None:
        """Add a message and update timestamp."""
        # Create mutable copy
        self.messages = list(self.messages)
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
    
    def get_recent_messages(self, count: int) -> list[Message]:
        """Get the N most recent messages."""
        return self.messages[-count:] if count > 0 else []
    
    def clear_messages(self) -> None:
        """Clear all messages."""
        self.messages = []
        self.updated_at = datetime.utcnow()


class RedisSessionBackend:
    """
    Redis-based session storage.
    
    Sessions are stored as JSON with configurable TTL.
    Uses Redis SETEX for atomic set-with-expiry.
    """
    
    def __init__(
        self,
        redis_url: str,
        key_prefix: str = "aegis:session:",
        default_ttl_seconds: int = 86400,  # 24 hours
    ):
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._default_ttl = default_ttl_seconds
        self._client = None
    
    async def _get_client(self):
        """Lazy initialization of Redis client."""
        if self._client is None:
            import redis.asyncio as redis
            self._client = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client
    
    def _make_key(self, session_id: UUID) -> str:
        """Generate Redis key for session."""
        return f"{self._key_prefix}{session_id}"
    
    async def save(self, session: Session, ttl_seconds: int | None = None) -> None:
        """Save session to Redis."""
        client = await self._get_client()
        key = self._make_key(session.id)
        data = session.model_dump_json()
        ttl = ttl_seconds or self._default_ttl
        
        await client.setex(key, ttl, data)
    
    async def load(self, session_id: UUID) -> Session | None:
        """Load session from Redis."""
        client = await self._get_client()
        key = self._make_key(session_id)
        data = await client.get(key)
        
        if data is None:
            return None
        
        return Session.model_validate_json(data)
    
    async def delete(self, session_id: UUID) -> bool:
        """Delete session from Redis."""
        client = await self._get_client()
        key = self._make_key(session_id)
        result = await client.delete(key)
        return result > 0
    
    async def exists(self, session_id: UUID) -> bool:
        """Check if session exists."""
        client = await self._get_client()
        key = self._make_key(session_id)
        return await client.exists(key) > 0
    
    async def refresh_ttl(self, session_id: UUID, ttl_seconds: int | None = None) -> bool:
        """Refresh session TTL."""
        client = await self._get_client()
        key = self._make_key(session_id)
        ttl = ttl_seconds or self._default_ttl
        return await client.expire(key, ttl)
    
    async def list_sessions(
        self,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[UUID]:
        """
        List session IDs.
        
        Note: This scans Redis keys, which can be expensive.
        For production, consider maintaining a separate index.
        """
        client = await self._get_client()
        pattern = f"{self._key_prefix}*"
        session_ids = []
        
        async for key in client.scan_iter(match=pattern, count=100):
            if len(session_ids) >= limit:
                break
            
            # Extract session ID from key
            session_id_str = key.replace(self._key_prefix, "")
            try:
                session_id = UUID(session_id_str)
                
                # If filtering by user, load and check
                if user_id:
                    session = await self.load(session_id)
                    if session and session.user_id == user_id:
                        session_ids.append(session_id)
                else:
                    session_ids.append(session_id)
            except ValueError:
                continue
        
        return session_ids
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None


class InMemorySessionBackend:
    """
    In-memory session storage for development/testing.
    
    Not suitable for production as sessions are lost on restart.
    """
    
    def __init__(self, default_ttl_seconds: int = 86400):
        self._sessions: dict[UUID, tuple[Session, datetime]] = {}
        self._default_ttl = default_ttl_seconds
    
    async def save(self, session: Session, ttl_seconds: int | None = None) -> None:
        ttl = ttl_seconds or self._default_ttl
        expires = datetime.utcnow() + timedelta(seconds=ttl)
        self._sessions[session.id] = (session, expires)
    
    async def load(self, session_id: UUID) -> Session | None:
        if session_id not in self._sessions:
            return None
        
        session, expires = self._sessions[session_id]
        if datetime.utcnow() > expires:
            del self._sessions[session_id]
            return None
        
        return session
    
    async def delete(self, session_id: UUID) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    async def exists(self, session_id: UUID) -> bool:
        return await self.load(session_id) is not None
    
    async def refresh_ttl(self, session_id: UUID, ttl_seconds: int | None = None) -> bool:
        if session_id not in self._sessions:
            return False
        
        session, _ = self._sessions[session_id]
        ttl = ttl_seconds or self._default_ttl
        expires = datetime.utcnow() + timedelta(seconds=ttl)
        self._sessions[session_id] = (session, expires)
        return True
    
    async def list_sessions(
        self,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[UUID]:
        now = datetime.utcnow()
        result = []
        
        for session_id, (session, expires) in self._sessions.items():
            if len(result) >= limit:
                break
            if now > expires:
                continue
            if user_id and session.user_id != user_id:
                continue
            result.append(session_id)
        
        return result
    
    async def close(self) -> None:
        self._sessions.clear()


class SessionManager:
    """
    High-level session management interface.
    
    Provides a simple API for session operations while
    abstracting the underlying storage backend.
    """
    
    def __init__(
        self,
        backend: RedisSessionBackend | InMemorySessionBackend,
        default_ttl_seconds: int = 86400,
    ):
        self._backend = backend
        self._default_ttl = default_ttl_seconds
    
    @classmethod
    def create_redis(
        cls,
        redis_url: str,
        default_ttl_seconds: int = 86400,
    ) -> "SessionManager":
        """Create a session manager with Redis backend."""
        backend = RedisSessionBackend(redis_url, default_ttl_seconds=default_ttl_seconds)
        return cls(backend, default_ttl_seconds)
    
    @classmethod
    def create_memory(cls, default_ttl_seconds: int = 86400) -> "SessionManager":
        """Create a session manager with in-memory backend."""
        backend = InMemorySessionBackend(default_ttl_seconds)
        return cls(backend, default_ttl_seconds)
    
    async def create_session(
        self,
        user_id: str | None = None,
        tenant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> Session:
        """Create and persist a new session."""
        session = Session(
            user_id=user_id,
            tenant_id=tenant_id,
            metadata=metadata or {},
        )
        await self._backend.save(session, ttl_seconds)
        return session
    
    async def get_session(self, session_id: UUID) -> Session | None:
        """Get a session by ID."""
        return await self._backend.load(session_id)
    
    async def update_session(self, session: Session, ttl_seconds: int | None = None) -> None:
        """Update a session."""
        session.updated_at = datetime.utcnow()
        await self._backend.save(session, ttl_seconds)
    
    async def delete_session(self, session_id: UUID) -> bool:
        """Delete a session."""
        return await self._backend.delete(session_id)
    
    async def add_message(self, session_id: UUID, message: Message) -> Session | None:
        """Add a message to a session."""
        session = await self.get_session(session_id)
        if session is None:
            return None
        
        session.add_message(message)
        await self.update_session(session)
        return session
    
    async def get_messages(
        self,
        session_id: UUID,
        limit: int | None = None,
    ) -> list[Message]:
        """Get messages from a session."""
        session = await self.get_session(session_id)
        if session is None:
            return []
        
        if limit:
            return session.get_recent_messages(limit)
        return session.messages
    
    async def list_sessions(
        self,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[UUID]:
        """List session IDs."""
        return await self._backend.list_sessions(user_id, limit)
    
    async def close(self) -> None:
        """Close the session manager."""
        await self._backend.close()
