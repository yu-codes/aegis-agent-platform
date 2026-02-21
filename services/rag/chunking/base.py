"""
Base Chunker

Abstract base for document chunking.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkConfig:
    """Configuration for chunking."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 50
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " "])


@dataclass
class TextChunk:
    """A chunk of text."""

    content: str = ""
    index: int = 0
    start: int = 0
    end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseChunker(ABC):
    """
    Abstract base class for chunking strategies.
    """

    def __init__(self, config: ChunkConfig | None = None):
        self._config = config or ChunkConfig()

    @abstractmethod
    def chunk(self, text: str) -> list[TextChunk]:
        """Chunk text into smaller pieces."""
        pass

    def chunk_documents(
        self,
        documents: list[dict[str, Any]],
        content_key: str = "content",
    ) -> list[TextChunk]:
        """Chunk multiple documents."""
        all_chunks = []

        for doc in documents:
            content = doc.get(content_key, "")
            if not content:
                continue

            chunks = self.chunk(content)

            # Add document metadata to chunks
            for chunk in chunks:
                chunk.metadata.update({k: v for k, v in doc.items() if k != content_key})

            all_chunks.extend(chunks)

        return all_chunks

    @property
    def config(self) -> ChunkConfig:
        """Get chunking configuration."""
        return self._config
