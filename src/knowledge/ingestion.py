"""
Document Ingestion Pipeline

Handles loading, parsing, and preprocessing documents.
Supports multiple document formats and sources.

Design decisions:
- Pipeline pattern for composable preprocessing
- Async for I/O-bound operations
- Metadata extraction for filtering
- Source tracking for attribution
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document types."""
    
    TEXT = "text"
    MARKDOWN = "markdown"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    CSV = "csv"


class Document(BaseModel):
    """
    A source document before chunking.
    
    Represents the original document with metadata.
    """
    
    id: UUID = Field(default_factory=uuid4)
    content: str
    doc_type: DocumentType = DocumentType.TEXT
    
    # Source information
    source: str  # File path, URL, or identifier
    title: str | None = None
    
    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Processing state
    is_processed: bool = False
    chunk_count: int = 0


class DocumentChunk(BaseModel):
    """
    A chunk of a document after splitting.
    
    Chunks are the unit of storage and retrieval.
    """
    
    id: UUID = Field(default_factory=uuid4)
    content: str
    
    # Parent document reference
    document_id: UUID
    chunk_index: int  # Order within document
    
    # Position in original
    start_char: int | None = None
    end_char: int | None = None
    
    # Inherited metadata
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # Embedding (populated during indexing)
    embedding: list[float] | None = None


class DocumentLoader(ABC):
    """Abstract document loader."""
    
    @abstractmethod
    async def load(self, source: str) -> Document:
        """Load a document from source."""
        pass
    
    @abstractmethod
    def supports(self, source: str) -> bool:
        """Check if this loader supports the source."""
        pass


class TextFileLoader(DocumentLoader):
    """Load plain text files."""
    
    def supports(self, source: str) -> bool:
        path = Path(source)
        return path.suffix.lower() in {".txt", ".text", ""}
    
    async def load(self, source: str) -> Document:
        path = Path(source)
        content = path.read_text(encoding="utf-8")
        
        return Document(
            content=content,
            doc_type=DocumentType.TEXT,
            source=str(path.absolute()),
            title=path.stem,
            metadata={"file_size": path.stat().st_size},
        )


class MarkdownLoader(DocumentLoader):
    """Load Markdown files."""
    
    def supports(self, source: str) -> bool:
        path = Path(source)
        return path.suffix.lower() in {".md", ".markdown"}
    
    async def load(self, source: str) -> Document:
        path = Path(source)
        content = path.read_text(encoding="utf-8")
        
        # Extract title from first heading
        title = path.stem
        for line in content.split("\n"):
            if line.startswith("# "):
                title = line[2:].strip()
                break
        
        return Document(
            content=content,
            doc_type=DocumentType.MARKDOWN,
            source=str(path.absolute()),
            title=title,
            metadata={"file_size": path.stat().st_size},
        )


class JSONLoader(DocumentLoader):
    """Load JSON files."""
    
    def __init__(self, content_key: str | None = None):
        self._content_key = content_key
    
    def supports(self, source: str) -> bool:
        return Path(source).suffix.lower() == ".json"
    
    async def load(self, source: str) -> Document:
        import json
        
        path = Path(source)
        data = json.loads(path.read_text(encoding="utf-8"))
        
        # Extract content
        if self._content_key and isinstance(data, dict):
            content = str(data.get(self._content_key, data))
        else:
            content = json.dumps(data, indent=2)
        
        return Document(
            content=content,
            doc_type=DocumentType.JSON,
            source=str(path.absolute()),
            title=path.stem,
            metadata={"original_data": data} if isinstance(data, dict) else {},
        )


class PDFLoader(DocumentLoader):
    """
    Load PDF files.
    
    Requires pypdf package.
    """
    
    def supports(self, source: str) -> bool:
        return Path(source).suffix.lower() == ".pdf"
    
    async def load(self, source: str) -> Document:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf required for PDF loading. Install with: pip install pypdf")
        
        path = Path(source)
        reader = PdfReader(path)
        
        # Extract text from all pages
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(f"[Page {i + 1}]\n{text}")
        
        content = "\n\n".join(pages)
        
        # Extract metadata
        metadata = {}
        if reader.metadata:
            metadata = {
                "author": reader.metadata.author,
                "title": reader.metadata.title,
                "subject": reader.metadata.subject,
                "page_count": len(reader.pages),
            }
        
        return Document(
            content=content,
            doc_type=DocumentType.PDF,
            source=str(path.absolute()),
            title=metadata.get("title") or path.stem,
            metadata=metadata,
        )


class DocumentPreprocessor(ABC):
    """Abstract document preprocessor."""
    
    @abstractmethod
    async def process(self, document: Document) -> Document:
        """Process a document."""
        pass


class WhitespaceNormalizer(DocumentPreprocessor):
    """Normalize whitespace in documents."""
    
    async def process(self, document: Document) -> Document:
        import re
        
        # Normalize whitespace
        content = re.sub(r"\r\n", "\n", document.content)
        content = re.sub(r"\t", "    ", content)
        content = re.sub(r" +", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = content.strip()
        
        return document.model_copy(update={"content": content})


class HTMLStripper(DocumentPreprocessor):
    """Remove HTML tags from content."""
    
    async def process(self, document: Document) -> Document:
        if document.doc_type != DocumentType.HTML:
            return document
        
        import re
        
        # Remove HTML tags
        content = re.sub(r"<script[^>]*>.*?</script>", "", document.content, flags=re.DOTALL)
        content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL)
        content = re.sub(r"<[^>]+>", " ", content)
        content = re.sub(r"&nbsp;", " ", content)
        content = re.sub(r"&[a-z]+;", "", content)
        
        return document.model_copy(update={"content": content.strip()})


class DocumentIngester:
    """
    Document ingestion pipeline.
    
    Orchestrates loading, preprocessing, and chunking of documents.
    """
    
    def __init__(
        self,
        loaders: list[DocumentLoader] | None = None,
        preprocessors: list[DocumentPreprocessor] | None = None,
        chunking_strategy: "ChunkingStrategy | None" = None,
    ):
        # Default loaders
        self._loaders = loaders or [
            TextFileLoader(),
            MarkdownLoader(),
            JSONLoader(),
            PDFLoader(),
        ]
        
        # Default preprocessors
        self._preprocessors = preprocessors or [
            WhitespaceNormalizer(),
            HTMLStripper(),
        ]
        
        self._chunker = chunking_strategy
    
    def set_chunking_strategy(self, strategy: "ChunkingStrategy") -> None:
        """Set the chunking strategy."""
        self._chunker = strategy
    
    def _find_loader(self, source: str) -> DocumentLoader | None:
        """Find a loader that supports the source."""
        for loader in self._loaders:
            if loader.supports(source):
                return loader
        return None
    
    async def ingest(self, source: str) -> list[DocumentChunk]:
        """
        Ingest a document and return chunks.
        
        Args:
            source: Path or URL to document
            
        Returns:
            List of document chunks
        """
        # Find loader
        loader = self._find_loader(source)
        if loader is None:
            raise ValueError(f"No loader available for: {source}")
        
        # Load document
        document = await loader.load(source)
        
        # Apply preprocessors
        for preprocessor in self._preprocessors:
            document = await preprocessor.process(document)
        
        # Chunk if strategy is set
        if self._chunker:
            chunks = await self._chunker.chunk(document)
        else:
            # No chunking, single chunk per document
            chunks = [
                DocumentChunk(
                    content=document.content,
                    document_id=document.id,
                    chunk_index=0,
                    source=document.source,
                    metadata=document.metadata,
                )
            ]
        
        document.is_processed = True
        document.chunk_count = len(chunks)
        
        return chunks
    
    async def ingest_batch(
        self,
        sources: list[str],
        on_error: str = "skip",  # skip, raise
    ) -> list[DocumentChunk]:
        """
        Ingest multiple documents.
        
        Args:
            sources: List of sources to ingest
            on_error: How to handle errors
            
        Returns:
            All chunks from all documents
        """
        all_chunks = []
        
        for source in sources:
            try:
                chunks = await self.ingest(source)
                all_chunks.extend(chunks)
            except Exception as e:
                if on_error == "raise":
                    raise
                # Log and continue
                print(f"Warning: Failed to ingest {source}: {e}")
        
        return all_chunks
    
    async def ingest_text(
        self,
        content: str,
        source: str = "direct_input",
        doc_type: DocumentType = DocumentType.TEXT,
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """
        Ingest text content directly.
        
        Useful for ingesting content not from files.
        """
        document = Document(
            content=content,
            doc_type=doc_type,
            source=source,
            metadata=metadata or {},
        )
        
        # Apply preprocessors
        for preprocessor in self._preprocessors:
            document = await preprocessor.process(document)
        
        # Chunk
        if self._chunker:
            chunks = await self._chunker.chunk(document)
        else:
            chunks = [
                DocumentChunk(
                    content=document.content,
                    document_id=document.id,
                    chunk_index=0,
                    source=document.source,
                    metadata=document.metadata,
                )
            ]
        
        return chunks


# Type hint for circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.knowledge.chunking import ChunkingStrategy
