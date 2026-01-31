"""
Chunking Strategies

Split documents into retrievable chunks.
Different strategies optimize for different use cases.

Design decisions:
- Strategy pattern for flexibility
- Configurable chunk size and overlap
- Semantic chunking for better coherence
- Metadata preservation across chunks
"""

from abc import ABC, abstractmethod
from typing import Any
import re

from src.knowledge.ingestion import Document, DocumentChunk


class ChunkingStrategy(ABC):
    """
    Abstract chunking strategy.
    
    Chunking determines how documents are split for retrieval.
    Good chunking balances:
    - Chunk size (too small = missing context, too large = noise)
    - Semantic coherence (chunks should be meaningful units)
    - Overlap (prevent information loss at boundaries)
    """
    
    @abstractmethod
    async def chunk(self, document: Document) -> list[DocumentChunk]:
        """Split document into chunks."""
        pass


class RecursiveChunker(ChunkingStrategy):
    """
    Recursive character-based chunking.
    
    Splits on a hierarchy of separators, trying to keep
    semantically related content together.
    
    Separator hierarchy:
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Sentences
    4. Words
    
    This is the most common and robust strategy.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        self._chunk_size = chunk_size
        self._overlap = chunk_overlap
        self._separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    async def chunk(self, document: Document) -> list[DocumentChunk]:
        """Split document recursively."""
        text = document.content
        chunks = self._split_text(text, self._separators)
        
        return [
            DocumentChunk(
                content=chunk_text,
                document_id=document.id,
                chunk_index=i,
                source=document.source,
                metadata={**document.metadata, "chunk_method": "recursive"},
            )
            for i, chunk_text in enumerate(chunks)
        ]
    
    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separator hierarchy."""
        if not text:
            return []
        
        if len(text) <= self._chunk_size:
            return [text]
        
        # Try each separator
        for sep in separators:
            if sep:
                parts = text.split(sep)
            else:
                # Last resort: character split
                parts = list(text)
            
            if len(parts) > 1:
                return self._merge_splits(parts, sep, separators)
        
        # Fallback: force split
        return self._force_split(text)
    
    def _merge_splits(
        self,
        parts: list[str],
        separator: str,
        remaining_separators: list[str],
    ) -> list[str]:
        """Merge small splits and recursively handle large ones."""
        chunks = []
        current_chunk = ""
        
        for part in parts:
            # Would adding this part exceed the limit?
            potential = current_chunk + (separator if current_chunk else "") + part
            
            if len(potential) <= self._chunk_size:
                current_chunk = potential
            else:
                # Save current chunk if non-empty
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Handle the new part
                if len(part) <= self._chunk_size:
                    current_chunk = part
                else:
                    # Part is too large, recurse with remaining separators
                    next_seps = remaining_separators[remaining_separators.index(separator) + 1:]
                    sub_chunks = self._split_text(part, next_seps)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Add overlap
        return self._add_overlap(chunks)
    
    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between chunks."""
        if len(chunks) <= 1 or self._overlap <= 0:
            return chunks
        
        result = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Add end of previous chunk as prefix
                prev_end = chunks[i - 1][-self._overlap:]
                chunk = prev_end + chunk
            result.append(chunk)
        
        return result
    
    def _force_split(self, text: str) -> list[str]:
        """Force split text when no separators work."""
        chunks = []
        for i in range(0, len(text), self._chunk_size - self._overlap):
            chunk = text[i:i + self._chunk_size]
            chunks.append(chunk)
        return chunks


class SemanticChunker(ChunkingStrategy):
    """
    Semantic chunking based on content structure.
    
    Identifies semantic boundaries (headings, sections, paragraphs)
    and keeps related content together.
    
    Better for structured documents like documentation, articles.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 100,
        heading_patterns: list[str] | None = None,
    ):
        self._max_size = max_chunk_size
        self._min_size = min_chunk_size
        self._heading_patterns = heading_patterns or [
            r"^#{1,6}\s+.+$",  # Markdown headings
            r"^[A-Z][A-Za-z\s]+:$",  # Title case with colon
            r"^\d+\.\s+.+$",  # Numbered sections
        ]
    
    async def chunk(self, document: Document) -> list[DocumentChunk]:
        """Split document by semantic boundaries."""
        text = document.content
        sections = self._identify_sections(text)
        chunks = self._sections_to_chunks(sections)
        
        return [
            DocumentChunk(
                content=chunk_text,
                document_id=document.id,
                chunk_index=i,
                source=document.source,
                metadata={
                    **document.metadata,
                    "chunk_method": "semantic",
                    "section_heading": chunk_meta.get("heading"),
                },
            )
            for i, (chunk_text, chunk_meta) in enumerate(chunks)
        ]
    
    def _identify_sections(self, text: str) -> list[tuple[str | None, str]]:
        """
        Identify sections in text.
        
        Returns list of (heading, content) tuples.
        """
        lines = text.split("\n")
        sections = []
        current_heading = None
        current_content = []
        
        combined_pattern = "|".join(f"({p})" for p in self._heading_patterns)
        
        for line in lines:
            if re.match(combined_pattern, line, re.MULTILINE):
                # Save previous section
                if current_content:
                    sections.append((current_heading, "\n".join(current_content)))
                
                current_heading = line
                current_content = []
            else:
                current_content.append(line)
        
        # Don't forget last section
        if current_content:
            sections.append((current_heading, "\n".join(current_content)))
        
        return sections
    
    def _sections_to_chunks(
        self,
        sections: list[tuple[str | None, str]],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Convert sections to appropriately sized chunks."""
        chunks = []
        
        for heading, content in sections:
            section_text = f"{heading}\n{content}" if heading else content
            
            if len(section_text) <= self._max_size:
                chunks.append((section_text, {"heading": heading}))
            else:
                # Section too large, split by paragraphs
                paragraphs = content.split("\n\n")
                current_chunk = heading + "\n" if heading else ""
                
                for para in paragraphs:
                    if len(current_chunk) + len(para) <= self._max_size:
                        current_chunk += para + "\n\n"
                    else:
                        if len(current_chunk) >= self._min_size:
                            chunks.append((current_chunk.strip(), {"heading": heading}))
                        current_chunk = para + "\n\n"
                
                if len(current_chunk) >= self._min_size:
                    chunks.append((current_chunk.strip(), {"heading": heading}))
        
        return chunks


class SentenceChunker(ChunkingStrategy):
    """
    Sentence-based chunking.
    
    Groups sentences into chunks, respecting sentence boundaries.
    Good for Q&A and conversational content.
    """
    
    def __init__(
        self,
        sentences_per_chunk: int = 5,
        overlap_sentences: int = 1,
    ):
        self._sentences_per_chunk = sentences_per_chunk
        self._overlap = overlap_sentences
    
    async def chunk(self, document: Document) -> list[DocumentChunk]:
        """Split document by sentences."""
        sentences = self._split_sentences(document.content)
        chunks = self._group_sentences(sentences)
        
        return [
            DocumentChunk(
                content=chunk_text,
                document_id=document.id,
                chunk_index=i,
                source=document.source,
                metadata={**document.metadata, "chunk_method": "sentence"},
            )
            for i, chunk_text in enumerate(chunks)
        ]
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting (could use nltk for better results)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _group_sentences(self, sentences: list[str]) -> list[str]:
        """Group sentences into chunks with overlap."""
        if not sentences:
            return []
        
        chunks = []
        step = self._sentences_per_chunk - self._overlap
        
        for i in range(0, len(sentences), step):
            chunk_sentences = sentences[i:i + self._sentences_per_chunk]
            chunks.append(" ".join(chunk_sentences))
        
        return chunks
