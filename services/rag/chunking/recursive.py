"""
Recursive Chunker

Recursively splits text using separators.

Based on: src/knowledge/chunking.py
"""

from dataclasses import dataclass, field

from services.rag.chunking.base import BaseChunker, ChunkConfig, TextChunk


@dataclass
class RecursiveChunkerConfig(ChunkConfig):
    """Configuration for recursive chunking."""

    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])


class RecursiveChunker(BaseChunker):
    """
    Recursively splits text using a hierarchy of separators.

    Tries to split on larger separators first, then smaller ones.
    """

    def __init__(self, config: RecursiveChunkerConfig | None = None):
        super().__init__(config or RecursiveChunkerConfig())

    def chunk(self, text: str) -> list[TextChunk]:
        """Chunk text recursively."""
        chunks_text = self._split_recursive(
            text,
            self._config.separators,
        )

        # Create chunks with positions
        chunks = []
        position = 0

        for i, chunk_text in enumerate(chunks_text):
            start = text.find(chunk_text, position)
            if start == -1:
                start = position

            chunk = TextChunk(
                content=chunk_text,
                index=i,
                start=start,
                end=start + len(chunk_text),
            )
            chunks.append(chunk)
            position = start + len(chunk_text) - self._config.chunk_overlap

        return chunks

    def _split_recursive(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """Recursively split text."""
        if not text:
            return []

        if len(text) <= self._config.chunk_size:
            return [text]

        if not separators:
            # Hard split at chunk_size
            return self._hard_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator:
            splits = text.split(separator)
        else:
            # Empty separator means character-level split
            return self._hard_split(text)

        chunks = []
        current_chunk = []
        current_size = 0

        for split in splits:
            split_size = len(split) + len(separator)

            if current_size + split_size > self._config.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = separator.join(current_chunk)
                if len(chunk_text) >= self._config.min_chunk_size:
                    chunks.append(chunk_text)

                # Start new chunk with overlap
                overlap_chunks = []
                overlap_size = 0
                for c in reversed(current_chunk):
                    if overlap_size + len(c) > self._config.chunk_overlap:
                        break
                    overlap_chunks.insert(0, c)
                    overlap_size += len(c) + len(separator)

                current_chunk = overlap_chunks
                current_size = sum(len(c) + len(separator) for c in current_chunk)

            current_chunk.append(split)
            current_size += split_size

        # Save last chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if len(chunk_text) >= self._config.min_chunk_size:
                chunks.append(chunk_text)
            elif chunks:
                # Merge with previous chunk if too small
                chunks[-1] = chunks[-1] + separator + chunk_text

        # Recursively split any chunks that are still too large
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self._config.chunk_size:
                sub_chunks = self._split_recursive(chunk, remaining_separators)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _hard_split(self, text: str) -> list[str]:
        """Hard split at chunk_size."""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self._config.chunk_size, len(text))
            chunk = text[start:end]

            if len(chunk) >= self._config.min_chunk_size:
                chunks.append(chunk)
            elif chunks:
                chunks[-1] += chunk

            start = end - self._config.chunk_overlap
            if start <= 0 or end >= len(text):
                break

        return chunks
