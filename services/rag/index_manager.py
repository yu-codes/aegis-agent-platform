"""
Index Manager

Document indexing and management service.

Based on: src/knowledge/vector_store.py, src/knowledge/ingestion.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID, uuid4
import hashlib


@dataclass
class Document:
    """A document to be indexed."""

    id: UUID = field(default_factory=uuid4)
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Source info
    source: str | None = None
    source_type: str = "text"

    # Processing
    content_hash: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def compute_hash(self) -> str:
        """Compute content hash."""
        self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        return self.content_hash


@dataclass
class Chunk:
    """A chunk of a document."""

    id: UUID = field(default_factory=uuid4)
    document_id: UUID = field(default_factory=uuid4)
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Position
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0

    # Embedding
    embedding: list[float] = field(default_factory=list)


@dataclass
class IndexStats:
    """Statistics for an index."""

    document_count: int = 0
    chunk_count: int = 0
    total_tokens: int = 0
    last_updated: datetime | None = None


class EmbeddingProtocol(Protocol):
    """Protocol for embedding services."""

    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class VectorStoreProtocol(Protocol):
    """Protocol for vector stores."""

    async def add(self, embeddings: list[list[float]], metadata: list[dict]) -> list[str]: ...
    async def search(self, query_embedding: list[float], top_k: int) -> list[dict]: ...
    async def delete(self, ids: list[str]) -> None: ...


class IndexManager:
    """
    Manages document indexing.

    Handles chunking, embedding, and storage.
    """

    def __init__(
        self,
        embedding_service: EmbeddingProtocol | None = None,
        vector_store: VectorStoreProtocol | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self._embeddings = embedding_service
        self._store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # In-memory fallback
        self._documents: dict[UUID, Document] = {}
        self._chunks: dict[UUID, Chunk] = {}
        self._index: list[Chunk] = []

    async def add_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Document:
        """
        Add a document to the index.

        Args:
            content: Document content
            metadata: Optional metadata
            source: Source identifier

        Returns:
            Indexed document
        """
        doc = Document(
            content=content,
            metadata=metadata or {},
            source=source,
        )
        doc.compute_hash()

        # Chunk the document
        chunks = self._chunk_document(doc)

        # Generate embeddings
        if self._embeddings:
            texts = [c.content for c in chunks]
            embeddings = await self._embeddings.embed_batch(texts)
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding

        # Store chunks
        if self._store:
            metadata_list = [
                {
                    "document_id": str(chunk.document_id),
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    **chunk.metadata,
                }
                for chunk in chunks
            ]
            embeddings_list = [c.embedding for c in chunks]
            await self._store.add(embeddings_list, metadata_list)

        # Store in memory
        self._documents[doc.id] = doc
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            self._index.append(chunk)

        return doc

    async def delete_document(self, document_id: UUID) -> bool:
        """Delete a document and its chunks."""
        if document_id not in self._documents:
            return False

        # Find and remove chunks
        chunk_ids = [str(c.id) for c in self._chunks.values() if c.document_id == document_id]

        if self._store:
            await self._store.delete(chunk_ids)

        # Remove from memory
        del self._documents[document_id]
        self._chunks = {cid: c for cid, c in self._chunks.items() if c.document_id != document_id}
        self._index = [c for c in self._index if c.document_id != document_id]

        return True

    def _chunk_document(self, doc: Document) -> list[Chunk]:
        """Chunk a document into smaller pieces."""
        chunks = []
        content = doc.content

        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + self._chunk_size, len(content))

            # Try to find a good break point
            if end < len(content):
                # Look for paragraph break
                para_break = content.rfind("\n\n", start, end)
                if para_break > start + self._chunk_size // 2:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    for sep in [". ", "! ", "? ", "\n"]:
                        sent_break = content.rfind(sep, start, end)
                        if sent_break > start + self._chunk_size // 2:
                            end = sent_break + len(sep)
                            break

            chunk_content = content[start:end].strip()

            if chunk_content:
                chunk = Chunk(
                    document_id=doc.id,
                    content=chunk_content,
                    metadata={
                        "source": doc.source,
                        **doc.metadata,
                    },
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move start with overlap
            start = end - self._chunk_overlap
            if start <= 0 or end >= len(content):
                break
            start = max(start, end - self._chunk_overlap)

        return chunks

    def get_stats(self) -> IndexStats:
        """Get index statistics."""
        return IndexStats(
            document_count=len(self._documents),
            chunk_count=len(self._chunks),
            total_tokens=sum(len(c.content.split()) for c in self._chunks.values()),
            last_updated=max(
                (d.created_at for d in self._documents.values()),
                default=None,
            ),
        )

    def get_document(self, document_id: UUID) -> Document | None:
        """Get a document by ID."""
        return self._documents.get(document_id)

    def list_documents(self, limit: int = 100) -> list[Document]:
        """List all documents."""
        return list(self._documents.values())[:limit]
