"""
Memory Retrieval

Policies and strategies for retrieving relevant memories.
Combines short-term and long-term memory retrieval.

Design decisions:
- Policy pattern for configurable retrieval behavior
- Unified interface across memory types
- Scoring and ranking for relevance
- Integration with RAG system
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.core.types import ExecutionContext, Message
from src.memory.long_term import LongTermMemory, MemoryEntry
from src.memory.session import Session


class RetrievalSource(str, Enum):
    """Sources of retrieved context."""

    SESSION = "session"
    LONG_TERM = "long_term"
    KNOWLEDGE = "knowledge"


@dataclass
class RetrievedMemory:
    """
    A piece of retrieved memory.

    Unified representation regardless of source.
    """

    content: str
    source: RetrievalSource
    relevance: float
    metadata: dict[str, Any]

    # Original references
    message: Message | None = None
    memory_entry: MemoryEntry | None = None


class RetrievalPolicy(ABC):
    """
    Abstract policy for memory retrieval.

    Policies determine:
    - What to retrieve
    - How much to retrieve
    - How to rank results
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        session: Session | None,
        context: ExecutionContext,
    ) -> list[RetrievedMemory]:
        """
        Retrieve relevant memories.

        Args:
            query: The query to match against
            session: Current session (for short-term memory)
            context: Execution context

        Returns:
            List of retrieved memories, ranked by relevance
        """
        pass


class DefaultRetrievalPolicy(RetrievalPolicy):
    """
    Default retrieval policy.

    Retrieves from both session history and long-term memory,
    then merges and ranks results.
    """

    def __init__(
        self,
        long_term_memory: LongTermMemory | None = None,
        session_lookback: int = 10,
        long_term_limit: int = 5,
        min_relevance: float = 0.3,
    ):
        self._ltm = long_term_memory
        self._session_lookback = session_lookback
        self._ltm_limit = long_term_limit
        self._min_relevance = min_relevance

    async def retrieve(
        self,
        query: str,
        session: Session | None,
        context: ExecutionContext,
    ) -> list[RetrievedMemory]:
        """Retrieve from session and long-term memory."""
        results: list[RetrievedMemory] = []

        # Retrieve from session
        if session and context.enable_memory:
            session_memories = self._retrieve_from_session(session, query)
            results.extend(session_memories)

        # Retrieve from long-term memory
        if self._ltm and context.user_id:
            ltm_memories = await self._retrieve_from_ltm(query, context.user_id)
            results.extend(ltm_memories)

        # Sort by relevance
        results.sort(key=lambda m: m.relevance, reverse=True)

        # Filter by minimum relevance
        results = [m for m in results if m.relevance >= self._min_relevance]

        return results

    def _retrieve_from_session(
        self,
        session: Session,
        query: str,
    ) -> list[RetrievedMemory]:
        """
        Retrieve relevant messages from session.

        Uses simple keyword matching for now.
        Could be enhanced with semantic similarity.
        """
        results = []
        query_words = set(query.lower().split())

        # Get recent messages
        recent = session.get_recent_messages(self._session_lookback)

        for i, msg in enumerate(recent):
            # Simple relevance: keyword overlap
            msg_words = set(msg.content.lower().split())
            overlap = len(query_words & msg_words)

            if overlap > 0:
                # Relevance based on overlap and recency
                overlap_score = overlap / max(len(query_words), 1)
                recency_score = (i + 1) / len(recent)
                relevance = 0.6 * overlap_score + 0.4 * recency_score

                results.append(
                    RetrievedMemory(
                        content=msg.content,
                        source=RetrievalSource.SESSION,
                        relevance=relevance,
                        metadata={"role": msg.role.value},
                        message=msg,
                    )
                )

        return results

    async def _retrieve_from_ltm(
        self,
        query: str,
        user_id: str,
    ) -> list[RetrievedMemory]:
        """Retrieve from long-term memory."""
        if not self._ltm:
            return []

        entries = await self._ltm.recall(
            query,
            user_id=user_id,
            limit=self._ltm_limit,
        )

        return [
            RetrievedMemory(
                content=entry.content,
                source=RetrievalSource.LONG_TERM,
                relevance=entry.metadata.get("relevance_score", 0.5),
                metadata={
                    "memory_type": entry.memory_type,
                    "topics": entry.topics,
                },
                memory_entry=entry,
            )
            for entry in entries
        ]


class AggressiveRetrievalPolicy(RetrievalPolicy):
    """
    Aggressive retrieval for complex tasks.

    Retrieves more context from all available sources.
    Use when the task requires extensive background knowledge.
    """

    def __init__(
        self,
        long_term_memory: LongTermMemory | None = None,
    ):
        self._ltm = long_term_memory

    async def retrieve(
        self,
        query: str,
        session: Session | None,
        context: ExecutionContext,
    ) -> list[RetrievedMemory]:
        """Retrieve aggressively from all sources."""
        results: list[RetrievedMemory] = []

        # Get more session history
        if session:
            for msg in session.messages:
                results.append(
                    RetrievedMemory(
                        content=msg.content,
                        source=RetrievalSource.SESSION,
                        relevance=0.5,  # All included, equal weight
                        metadata={"role": msg.role.value},
                        message=msg,
                    )
                )

        # Get more from LTM
        if self._ltm and context.user_id:
            entries = await self._ltm.recall(
                query,
                user_id=context.user_id,
                limit=20,  # More results
            )

            for entry in entries:
                results.append(
                    RetrievedMemory(
                        content=entry.content,
                        source=RetrievalSource.LONG_TERM,
                        relevance=entry.metadata.get("relevance_score", 0.5),
                        metadata={"memory_type": entry.memory_type},
                        memory_entry=entry,
                    )
                )

        return results


class MemoryRetriever:
    """
    Unified memory retrieval interface.

    Orchestrates retrieval across memory systems
    with configurable policies.
    """

    def __init__(
        self,
        policy: RetrievalPolicy | None = None,
        long_term_memory: LongTermMemory | None = None,
    ):
        self._policy = policy or DefaultRetrievalPolicy(long_term_memory)
        self._ltm = long_term_memory

    def set_policy(self, policy: RetrievalPolicy) -> None:
        """Change the retrieval policy."""
        self._policy = policy

    async def retrieve(
        self,
        query: str,
        session: Session | None = None,
        context: ExecutionContext | None = None,
    ) -> list[RetrievedMemory]:
        """
        Retrieve relevant memories.

        Args:
            query: Search query
            session: Current session
            context: Execution context

        Returns:
            Ranked list of relevant memories
        """
        # Create default context if not provided
        if context is None:
            from uuid import uuid4

            context = ExecutionContext(session_id=uuid4())

        return await self._policy.retrieve(query, session, context)

    async def retrieve_as_context(
        self,
        query: str,
        session: Session | None = None,
        context: ExecutionContext | None = None,
        max_tokens: int = 2000,
    ) -> str:
        """
        Retrieve memories formatted as context string.

        Useful for injecting into prompts.
        """
        memories = await self.retrieve(query, session, context)

        if not memories:
            return ""

        # Format as context
        lines = ["Relevant context from memory:"]

        for i, mem in enumerate(memories, 1):
            source_label = mem.source.value.replace("_", " ").title()
            lines.append(f"\n[{i}] ({source_label}, relevance: {mem.relevance:.2f})")
            lines.append(mem.content)

        context_str = "\n".join(lines)

        # Truncate if too long (rough estimate)
        if len(context_str) > max_tokens * 4:
            context_str = context_str[: max_tokens * 4] + "\n[Truncated...]"

        return context_str
