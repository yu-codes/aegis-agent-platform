"""
Retriever and Context Assembler

Retrieves relevant documents and assembles context for the LLM.

Design decisions:
- Separation of retrieval and assembly
- Multiple retrieval strategies (simple, hybrid, reranking)
- Context budget management
- Source attribution
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.core.types import RetrievedDocument
from src.knowledge.vector_store import VectorStore, VectorSearchResult
from src.knowledge.embeddings import EmbeddingService


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    
    top_k: int = 5
    min_score: float = 0.0
    include_metadata: bool = True
    rerank: bool = False


class Retriever:
    """
    Document retriever.
    
    Retrieves relevant documents from the vector store
    based on semantic similarity to the query.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        config: RetrievalConfig | None = None,
    ):
        self._store = vector_store
        self._embeddings = embedding_service
        self._config = config or RetrievalConfig()
    
    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve relevant documents.
        
        Args:
            query: Search query
            top_k: Override default top_k
            filter: Metadata filter
            
        Returns:
            List of retrieved documents, ranked by relevance
        """
        k = top_k or self._config.top_k
        
        # Generate query embedding
        query_embedding = await self._embeddings.embed(query)
        
        # Search vector store
        results = await self._store.search(
            query_embedding,
            top_k=k,
            filter=filter,
        )
        
        # Filter by minimum score
        results = [r for r in results if r.score >= self._config.min_score]
        
        # Convert to RetrievedDocument
        documents = [
            RetrievedDocument(
                id=r.id,
                content=r.content,
                score=r.score,
                metadata=r.metadata if self._config.include_metadata else {},
                source=r.metadata.get("source"),
                chunk_index=r.metadata.get("chunk_index"),
            )
            for r in results
        ]
        
        # Optional reranking
        if self._config.rerank:
            documents = await self._rerank(query, documents)
        
        return documents
    
    async def retrieve_with_context(
        self,
        query: str,
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> tuple[list[RetrievedDocument], str]:
        """
        Retrieve and format as context string.
        
        Returns both the documents and a formatted context.
        """
        documents = await self.retrieve(query, top_k, filter)
        context = self._format_context(documents)
        return documents, context
    
    def _format_context(self, documents: list[RetrievedDocument]) -> str:
        """Format documents as context string."""
        if not documents:
            return ""
        
        lines = []
        for i, doc in enumerate(documents, 1):
            source = doc.source or "Unknown"
            lines.append(f"[Document {i}] (Source: {source}, Score: {doc.score:.2f})")
            lines.append(doc.content)
            lines.append("")
        
        return "\n".join(lines)
    
    async def _rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Rerank documents using cross-encoder.
        
        More accurate but slower than embedding similarity.
        """
        if not documents:
            return documents
        
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            # Fall back to original ranking
            return documents
        
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        pairs = [(query, doc.content) for doc in documents]
        scores = model.predict(pairs)
        
        # Update scores and re-sort
        for doc, score in zip(documents, scores):
            doc.score = float(score)
        
        documents.sort(key=lambda d: d.score, reverse=True)
        return documents


class HybridRetriever(Retriever):
    """
    Hybrid retriever combining dense and sparse search.
    
    Uses both semantic embeddings and keyword matching
    for better recall.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        config: RetrievalConfig | None = None,
        alpha: float = 0.5,  # Weight for dense vs sparse
    ):
        super().__init__(vector_store, embedding_service, config)
        self._alpha = alpha
        self._keyword_index: dict[str, set[str]] = {}  # token -> doc_ids
    
    def add_to_keyword_index(self, doc_id: str, content: str) -> None:
        """Add document to keyword index."""
        tokens = self._tokenize(content)
        for token in tokens:
            if token not in self._keyword_index:
                self._keyword_index[token] = set()
            self._keyword_index[token].add(doc_id)
    
    def _tokenize(self, text: str) -> set[str]:
        """Simple tokenization."""
        import re
        tokens = re.findall(r"\b\w+\b", text.lower())
        return set(tokens)
    
    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """Hybrid retrieval combining dense and sparse."""
        k = top_k or self._config.top_k
        
        # Dense retrieval (more candidates for fusion)
        dense_docs = await super().retrieve(query, top_k=k * 2, filter=filter)
        
        # Sparse retrieval (keyword matching)
        query_tokens = self._tokenize(query)
        sparse_doc_ids: dict[str, int] = {}  # doc_id -> match_count
        
        for token in query_tokens:
            for doc_id in self._keyword_index.get(token, set()):
                sparse_doc_ids[doc_id] = sparse_doc_ids.get(doc_id, 0) + 1
        
        # Combine scores
        combined_scores: dict[str, float] = {}
        
        for doc in dense_docs:
            combined_scores[doc.id] = self._alpha * doc.score
        
        max_sparse = max(sparse_doc_ids.values()) if sparse_doc_ids else 1
        for doc_id, count in sparse_doc_ids.items():
            sparse_score = count / max_sparse
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - self._alpha) * sparse_score
            else:
                combined_scores[doc_id] = (1 - self._alpha) * sparse_score
        
        # Merge and sort
        doc_map = {doc.id: doc for doc in dense_docs}
        
        results = []
        for doc_id, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]:
            if doc_id in doc_map:
                doc = doc_map[doc_id]
                doc.score = score
                results.append(doc)
        
        return results


class ContextAssembler:
    """
    Assembles retrieved documents into LLM context.
    
    Handles:
    - Context budget management
    - Deduplication
    - Ordering and prioritization
    - Template application
    """
    
    def __init__(
        self,
        max_tokens: int = 4000,
        chars_per_token: float = 4.0,
    ):
        self._max_tokens = max_tokens
        self._chars_per_token = chars_per_token
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from character count."""
        return int(len(text) / self._chars_per_token)
    
    def assemble(
        self,
        documents: list[RetrievedDocument],
        query: str | None = None,
        template: str | None = None,
    ) -> str:
        """
        Assemble documents into context.
        
        Args:
            documents: Retrieved documents
            query: Original query (for template)
            template: Optional template to use
            
        Returns:
            Assembled context string
        """
        if not documents:
            return ""
        
        # Deduplicate by content
        seen_content = set()
        unique_docs = []
        
        for doc in documents:
            content_hash = hash(doc.content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        # Build context within budget
        context_parts = []
        current_tokens = 0
        
        for doc in unique_docs:
            doc_tokens = self._estimate_tokens(doc.content)
            
            if current_tokens + doc_tokens > self._max_tokens:
                # Truncate this document
                remaining_tokens = self._max_tokens - current_tokens
                remaining_chars = int(remaining_tokens * self._chars_per_token)
                truncated = doc.content[:remaining_chars] + "..."
                context_parts.append(self._format_document(doc, truncated))
                break
            
            context_parts.append(self._format_document(doc, doc.content))
            current_tokens += doc_tokens
        
        context = "\n\n".join(context_parts)
        
        # Apply template if provided
        if template:
            try:
                from jinja2 import Environment, BaseLoader
                env = Environment(loader=BaseLoader())
                jinja_template = env.from_string(template)
                context = jinja_template.render(
                    documents=unique_docs,
                    context=context,
                    query=query,
                )
            except Exception:
                pass  # Use plain context on template error
        
        return context
    
    def _format_document(self, doc: RetrievedDocument, content: str) -> str:
        """Format a single document for context."""
        source = doc.source or "Unknown"
        return f"[Source: {source} | Relevance: {doc.score:.2f}]\n{content}"
    
    def assemble_with_metadata(
        self,
        documents: list[RetrievedDocument],
    ) -> dict[str, Any]:
        """
        Assemble context with metadata for tracing.
        
        Returns dict with context and source information.
        """
        context = self.assemble(documents)
        
        return {
            "context": context,
            "sources": [
                {
                    "id": doc.id,
                    "source": doc.source,
                    "score": doc.score,
                    "preview": doc.content[:100],
                }
                for doc in documents
            ],
            "document_count": len(documents),
            "total_chars": len(context),
            "estimated_tokens": self._estimate_tokens(context),
        }
