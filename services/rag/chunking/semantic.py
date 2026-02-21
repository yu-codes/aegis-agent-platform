"""
Semantic Chunker

Splits text based on semantic similarity.
"""

from dataclasses import dataclass
from typing import Protocol

from services.rag.chunking.base import BaseChunker, ChunkConfig, TextChunk


@dataclass
class SemanticChunkerConfig(ChunkConfig):
    """Configuration for semantic chunking."""

    similarity_threshold: float = 0.75
    sentence_split_regex: str = r"(?<=[.!?])\s+"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingProtocol(Protocol):
    """Protocol for embedding services."""

    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class SemanticChunker(BaseChunker):
    """
    Splits text based on semantic boundaries.

    Uses embeddings to detect topic changes.
    """

    def __init__(
        self,
        config: SemanticChunkerConfig | None = None,
        embedding_service: EmbeddingProtocol | None = None,
    ):
        super().__init__(config or SemanticChunkerConfig())
        self._embeddings = embedding_service
        self._similarity_threshold = getattr(self._config, "similarity_threshold", 0.75)

    def chunk(self, text: str) -> list[TextChunk]:
        """Chunk text by semantic boundaries."""
        import re

        # Split into sentences
        pattern = getattr(self._config, "sentence_split_regex", r"(?<=[.!?])\s+")
        sentences = re.split(pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        if len(sentences) == 1:
            return [TextChunk(content=sentences[0], index=0, start=0, end=len(text))]

        # Get embeddings
        if self._embeddings:
            embeddings = self._embeddings.embed_batch(sentences)
        else:
            # Fallback: use simple hashing
            embeddings = [self._simple_hash(s) for s in sentences]

        # Find semantic boundaries
        boundaries = self._find_boundaries(sentences, embeddings)

        # Create chunks
        chunks = []
        position = 0
        chunk_start_idx = 0

        for i, sentence in enumerate(sentences):
            if i in boundaries or i == len(sentences) - 1:
                # End of chunk
                chunk_sentences = sentences[chunk_start_idx : i + 1]
                chunk_text = " ".join(chunk_sentences)

                start = text.find(chunk_sentences[0], position)
                if start == -1:
                    start = position

                chunk = TextChunk(
                    content=chunk_text,
                    index=len(chunks),
                    start=start,
                    end=start + len(chunk_text),
                )
                chunks.append(chunk)

                position = start + len(chunk_text)
                chunk_start_idx = i + 1

        return chunks

    def _find_boundaries(
        self,
        sentences: list[str],
        embeddings: list[list[float]],
    ) -> set[int]:
        """Find semantic boundaries between sentences."""
        if len(sentences) <= 1:
            return set()

        boundaries = set()
        current_size = 0

        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            current_size += len(sentences[i])

            # Add boundary if:
            # 1. Similarity drops below threshold
            # 2. Current chunk is too large
            if similarity < self._similarity_threshold:
                boundaries.add(i - 1)
                current_size = len(sentences[i])
            elif current_size > self._config.chunk_size:
                boundaries.add(i - 1)
                current_size = len(sentences[i])

        return boundaries

    def _cosine_similarity(
        self,
        vec1: list[float],
        vec2: list[float],
    ) -> float:
        """Calculate cosine similarity between vectors."""
        if not vec1 or not vec2:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _simple_hash(self, text: str) -> list[float]:
        """Simple hash-based pseudo-embedding for fallback."""
        words = text.lower().split()
        vec = [0.0] * 64

        for word in words:
            for i, char in enumerate(word):
                idx = (ord(char) + i) % 64
                vec[idx] += 1.0

        # Normalize
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec
