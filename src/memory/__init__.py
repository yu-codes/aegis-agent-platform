"""
State & Memory Module

Manages conversation state, short-term memory, and long-term memory.
Uses Redis as the primary backend for persistence.
"""

from src.memory.session import (
    SessionManager,
    Session,
    InMemorySessionBackend,
    RedisSessionBackend,
)
from src.memory.short_term import ShortTermMemory, WindowMemory, SummarizingMemory
from src.memory.long_term import LongTermMemory, MemoryEntry
from src.memory.retrieval import MemoryRetriever, RetrievalPolicy

__all__ = [
    # Session
    "SessionManager",
    "Session",
    "InMemorySessionBackend",
    "RedisSessionBackend",
    # Short-term
    "ShortTermMemory",
    "WindowMemory",
    "SummarizingMemory",
    # Long-term
    "LongTermMemory",
    "MemoryEntry",
    # Retrieval
    "MemoryRetriever",
    "RetrievalPolicy",
]
