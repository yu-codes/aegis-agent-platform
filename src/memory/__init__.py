"""
State & Memory Module

Manages conversation state, short-term memory, and long-term memory.
Uses Redis as the primary backend for persistence.
"""

from src.memory.long_term import LongTermMemory, MemoryEntry
from src.memory.retrieval import MemoryRetriever, RetrievalPolicy
from src.memory.session import (
    InMemorySessionBackend,
    RedisSessionBackend,
    Session,
    SessionManager,
)
from src.memory.short_term import ShortTermMemory, SummarizingMemory, WindowMemory

__all__ = [
    "InMemorySessionBackend",
    # Long-term
    "LongTermMemory",
    "MemoryEntry",
    # Retrieval
    "MemoryRetriever",
    "RedisSessionBackend",
    "RetrievalPolicy",
    "Session",
    # Session
    "SessionManager",
    # Short-term
    "ShortTermMemory",
    "SummarizingMemory",
    "WindowMemory",
]
