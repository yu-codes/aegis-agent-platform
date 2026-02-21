"""
Session Memory

Short-term conversation memory management.

Based on: src/memory/session.py, src/memory/short_term.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Protocol
from uuid import UUID, uuid4
import json


@dataclass
class ConversationTurn:
    """A turn in the conversation."""

    id: UUID = field(default_factory=uuid4)
    role: str = "user"  # user, assistant, system, tool
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Tool call info
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

    # Token counts
    token_count: int | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tool_calls": self.tool_calls,
            "tool_call_id": self.tool_call_id,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTurn":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if data.get("id") else uuid4(),
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if data.get("timestamp")
                else datetime.utcnow()
            ),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            token_count=data.get("token_count"),
            metadata=data.get("metadata", {}),
        )


class StorageBackendProtocol(Protocol):
    """Protocol for session storage backends."""

    async def save(self, session_id: str, data: dict) -> None: ...
    async def load(self, session_id: str) -> dict | None: ...
    async def delete(self, session_id: str) -> None: ...
    async def list_sessions(self, prefix: str | None = None) -> list[str]: ...


class SessionMemory:
    """
    Short-term conversation memory.

    Manages conversation history within a session.
    """

    def __init__(
        self,
        session_id: str | None = None,
        max_turns: int = 100,
        max_tokens: int = 16000,
        backend: StorageBackendProtocol | None = None,
    ):
        self._session_id = session_id or str(uuid4())
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._backend = backend

        self._turns: list[ConversationTurn] = []
        self._total_tokens = 0
        self._metadata: dict[str, Any] = {}
        self._created_at = datetime.utcnow()
        self._updated_at = datetime.utcnow()

    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self._session_id

    @property
    def turns(self) -> list[ConversationTurn]:
        """Get all turns."""
        return list(self._turns)

    def add_turn(
        self,
        role: str,
        content: str,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        """
        Add a conversation turn.

        Args:
            role: Role (user, assistant, system, tool)
            content: Message content
            tool_calls: Tool calls (for assistant)
            tool_call_id: Tool call ID (for tool response)
            metadata: Additional metadata

        Returns:
            The added turn
        """
        turn = ConversationTurn(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            token_count=self._estimate_tokens(content),
            metadata=metadata or {},
        )

        self._turns.append(turn)
        self._total_tokens += turn.token_count or 0
        self._updated_at = datetime.utcnow()

        # Trim if needed
        self._trim_if_needed()

        return turn

    def add_user_message(self, content: str, metadata: dict | None = None) -> ConversationTurn:
        """Add a user message."""
        return self.add_turn("user", content, metadata=metadata)

    def add_assistant_message(
        self,
        content: str,
        tool_calls: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> ConversationTurn:
        """Add an assistant message."""
        return self.add_turn("assistant", content, tool_calls=tool_calls, metadata=metadata)

    def add_tool_result(
        self,
        tool_call_id: str,
        content: str,
        metadata: dict | None = None,
    ) -> ConversationTurn:
        """Add a tool result."""
        return self.add_turn("tool", content, tool_call_id=tool_call_id, metadata=metadata)

    def get_messages_for_llm(
        self,
        include_system: bool = True,
        max_tokens: int | None = None,
    ) -> list[dict]:
        """
        Get messages formatted for LLM.

        Returns messages in OpenAI/Anthropic format.
        """
        messages = []
        total_tokens = 0
        limit = max_tokens or self._max_tokens

        # Start from most recent
        for turn in reversed(self._turns):
            if turn.token_count:
                total_tokens += turn.token_count
                if total_tokens > limit:
                    break

            if turn.role == "system" and not include_system:
                continue

            msg: dict = {"role": turn.role, "content": turn.content}

            if turn.tool_calls:
                msg["tool_calls"] = turn.tool_calls

            if turn.tool_call_id:
                msg["tool_call_id"] = turn.tool_call_id

            messages.insert(0, msg)

        return messages

    def get_last_n_turns(self, n: int) -> list[ConversationTurn]:
        """Get last N turns."""
        return self._turns[-n:]

    def clear(self) -> None:
        """Clear all turns."""
        self._turns.clear()
        self._total_tokens = 0
        self._updated_at = datetime.utcnow()

    async def save(self) -> None:
        """Save session to backend."""
        if self._backend:
            data = self._serialize()
            await self._backend.save(self._session_id, data)

    async def load(self) -> bool:
        """Load session from backend."""
        if self._backend:
            data = await self._backend.load(self._session_id)
            if data:
                self._deserialize(data)
                return True
        return False

    def _serialize(self) -> dict:
        """Serialize session data."""
        return {
            "session_id": self._session_id,
            "turns": [t.to_dict() for t in self._turns],
            "total_tokens": self._total_tokens,
            "metadata": self._metadata,
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
        }

    def _deserialize(self, data: dict) -> None:
        """Deserialize session data."""
        self._session_id = data.get("session_id", self._session_id)
        self._turns = [ConversationTurn.from_dict(t) for t in data.get("turns", [])]
        self._total_tokens = data.get("total_tokens", 0)
        self._metadata = data.get("metadata", {})
        self._created_at = (
            datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.utcnow()
        )
        self._updated_at = (
            datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else datetime.utcnow()
        )

    def _trim_if_needed(self) -> None:
        """Trim old turns if limits exceeded."""
        # Trim by turn count
        while len(self._turns) > self._max_turns:
            removed = self._turns.pop(0)
            self._total_tokens -= removed.token_count or 0

        # Trim by token count
        while self._total_tokens > self._max_tokens and self._turns:
            removed = self._turns.pop(0)
            self._total_tokens -= removed.token_count or 0

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimate: ~4 chars per token
        return len(text) // 4


class RedisSessionBackend:
    """Redis-based session storage."""

    def __init__(
        self,
        redis_client: Any = None,
        prefix: str = "session:",
        ttl_seconds: int = 86400,  # 24 hours
    ):
        self._redis = redis_client
        self._prefix = prefix
        self._ttl = ttl_seconds

    async def save(self, session_id: str, data: dict) -> None:
        """Save session to Redis."""
        if self._redis:
            key = f"{self._prefix}{session_id}"
            await self._redis.setex(key, self._ttl, json.dumps(data))

    async def load(self, session_id: str) -> dict | None:
        """Load session from Redis."""
        if self._redis:
            key = f"{self._prefix}{session_id}"
            data = await self._redis.get(key)
            if data:
                return json.loads(data)
        return None

    async def delete(self, session_id: str) -> None:
        """Delete session from Redis."""
        if self._redis:
            key = f"{self._prefix}{session_id}"
            await self._redis.delete(key)

    async def list_sessions(self, prefix: str | None = None) -> list[str]:
        """List all sessions."""
        if self._redis:
            pattern = f"{self._prefix}{prefix or ''}*"
            keys = await self._redis.keys(pattern)
            return [k.decode().replace(self._prefix, "") for k in keys]
        return []


class InMemorySessionBackend:
    """In-memory session storage for offline/testing mode."""

    def __init__(self):
        self._sessions: dict[str, dict] = {}

    async def save(self, session_id: str, data: dict) -> None:
        """Save session to memory."""
        self._sessions[session_id] = data

    async def load(self, session_id: str) -> dict | None:
        """Load session from memory."""
        return self._sessions.get(session_id)

    async def delete(self, session_id: str) -> None:
        """Delete session from memory."""
        self._sessions.pop(session_id, None)

    async def list_sessions(self, prefix: str | None = None) -> list[str]:
        """List all sessions."""
        if prefix:
            return [sid for sid in self._sessions if sid.startswith(prefix)]
        return list(self._sessions.keys())


class SessionManager:
    """
    Manages multiple conversation sessions.

    Provides a high-level interface for creating, retrieving,
    and managing SessionMemory instances.
    """

    def __init__(
        self,
        backend: StorageBackendProtocol | None = None,
        max_turns: int = 100,
        max_tokens: int = 16000,
    ):
        self._backend = backend or InMemorySessionBackend()
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._sessions: dict[str, SessionMemory] = {}

    async def create_session(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a new session.

        Args:
            session_id: Optional session ID (auto-generated if not provided)
            metadata: Optional session metadata

        Returns:
            Session ID
        """
        sid = session_id or str(uuid4())
        session = SessionMemory(
            session_id=sid,
            max_turns=self._max_turns,
            max_tokens=self._max_tokens,
            backend=self._backend,
        )
        if metadata:
            session._metadata = metadata
        self._sessions[sid] = session
        return sid

    async def get_session(self, session_id: str) -> SessionMemory | None:
        """
        Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            SessionMemory instance or None
        """
        # Check cache first
        if session_id in self._sessions:
            return self._sessions[session_id]

        # Try loading from backend
        data = await self._backend.load(session_id)
        if data:
            session = SessionMemory(
                session_id=session_id,
                max_turns=self._max_turns,
                max_tokens=self._max_tokens,
                backend=self._backend,
            )
            session._deserialize(data)
            self._sessions[session_id] = session
            return session

        return None

    async def get_or_create_session(
        self,
        session_id: str | None = None,
    ) -> tuple[str, SessionMemory]:
        """
        Get existing session or create new one.

        Args:
            session_id: Optional session ID

        Returns:
            Tuple of (session_id, SessionMemory)
        """
        if session_id:
            session = await self.get_session(session_id)
            if session:
                return session_id, session

        # Create new session
        sid = await self.create_session(session_id)
        return sid, self._sessions[sid]

    async def add_message(
        self,
        session_id: str | UUID,
        role: str,
        content: str,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        """
        Add a message to a session.

        Args:
            session_id: Session ID
            role: Message role
            content: Message content
            tool_calls: Tool calls (for assistant)
            tool_call_id: Tool call ID (for tool response)
            metadata: Additional metadata

        Returns:
            The added turn
        """
        sid = str(session_id) if isinstance(session_id, UUID) else session_id
        _, session = await self.get_or_create_session(sid)
        turn = session.add_turn(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            metadata=metadata,
        )
        # Persist to backend
        await self._backend.save(session_id, session._serialize())
        return turn

    async def get_messages(
        self,
        session_id: str,
        max_tokens: int | None = None,
    ) -> list[dict]:
        """
        Get messages from a session formatted for LLM.

        Args:
            session_id: Session ID
            max_tokens: Optional token limit

        Returns:
            List of message dicts
        """
        session = await self.get_session(session_id)
        if session:
            return session.get_messages_for_llm(max_tokens=max_tokens)
        return []

    async def get_context(
        self,
        session_id: str | UUID,
    ) -> list[dict]:
        """
        Get conversation context for a session.

        Args:
            session_id: Session ID

        Returns:
            List of message dicts for LLM context
        """
        sid = str(session_id) if isinstance(session_id, UUID) else session_id
        return await self.get_messages(sid)

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session ID

        Returns:
            True if deleted
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
        await self._backend.delete(session_id)
        return True

    async def list_sessions(
        self,
        prefix: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        List all sessions with pagination.

        Args:
            prefix: Optional prefix filter
            limit: Maximum number of sessions
            offset: Pagination offset

        Returns:
            List of session info dicts
        """
        session_ids = await self._backend.list_sessions(prefix)
        # Apply pagination
        paginated = session_ids[offset : offset + limit]

        result = []
        for sid in paginated:
            session = await self.get_session(sid)
            if session:
                result.append(
                    {
                        "session_id": sid,
                        "created_at": session._created_at,
                        "message_count": len(session._turns),
                        "last_activity": session._updated_at,
                    }
                )
        return result

    async def get_history(self, session_id: str) -> list[ConversationTurn]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session ID

        Returns:
            List of conversation turns

        Raises:
            KeyError: If session not found
        """
        session = await self.get_session(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found")
        return list(session._turns)

    async def get_metadata(self, session_id: str) -> dict[str, Any] | None:
        """
        Get session metadata.

        Args:
            session_id: Session ID

        Returns:
            Metadata dict or None
        """
        session = await self.get_session(session_id)
        if session:
            return session._metadata
        return None

    async def clear_history(self, session_id: str) -> bool:
        """
        Clear session history but keep session.

        Args:
            session_id: Session ID

        Returns:
            True if cleared

        Raises:
            KeyError: If session not found
        """
        session = await self.get_session(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found")
        session.clear()
        await self._backend.save(session_id, session._serialize())
        return True

    async def clear_session(self, session_id: str) -> bool:
        """
        Clear a session's history.

        Args:
            session_id: Session ID

        Returns:
            True if cleared
        """
        session = await self.get_session(session_id)
        if session:
            session.clear()
            await self._backend.save(session_id, session._serialize())
            return True
        return False
