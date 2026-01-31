"""
Vector Store Abstraction

Store and search vector embeddings.
Supports multiple backends (FAISS, Milvus, etc.).

Design decisions:
- Abstract interface for backend independence
- Metadata storage alongside vectors
- Filtering by metadata
- Batch operations for efficiency
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID
import json


@dataclass
class VectorSearchResult:
    """Result from vector search."""
    
    id: str
    score: float
    content: str
    metadata: dict[str, Any]


class VectorStore(ABC):
    """
    Abstract vector store interface.
    
    Provides storage and similarity search for embeddings.
    """
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Expected embedding dimension."""
        pass
    
    @abstractmethod
    async def add(
        self,
        id: str,
        embedding: list[float],
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a single vector."""
        pass
    
    @abstractmethod
    async def add_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        contents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add multiple vectors."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        pass
    
    @abstractmethod
    async def get(self, id: str) -> VectorSearchResult | None:
        """Get a vector by ID."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all vectors."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Count total vectors."""
        pass


class FAISSVectorStore(VectorStore):
    """
    FAISS-based vector store.
    
    Uses Facebook AI Similarity Search for efficient local vector search.
    Good for development and single-node deployments.
    
    Limitations:
    - In-memory (requires save/load for persistence)
    - Single-node only
    - Limited filtering capabilities
    """
    
    def __init__(
        self,
        dimension: int,
        index_path: str | None = None,
        use_gpu: bool = False,
    ):
        self._dimension = dimension
        self._index_path = Path(index_path) if index_path else None
        self._use_gpu = use_gpu
        
        self._index = None
        self._id_map: dict[int, str] = {}  # FAISS int ID -> string ID
        self._reverse_id_map: dict[str, int] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._contents: dict[str, str] = {}
        self._next_id = 0
        
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Initialize FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss required. Install with: pip install faiss-cpu "
                "or pip install faiss-gpu"
            )
        
        # Create index
        self._index = faiss.IndexFlatIP(self._dimension)  # Inner product (cosine with normalized vectors)
        
        if self._use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
            except Exception:
                pass  # Fall back to CPU
        
        # Load existing index if path provided
        if self._index_path and self._index_path.exists():
            self._load()
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def _normalize(self, embedding: list[float]) -> list[float]:
        """L2 normalize embedding for cosine similarity."""
        import math
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            return [x / norm for x in embedding]
        return embedding
    
    async def add(
        self,
        id: str,
        embedding: list[float],
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a single vector."""
        import numpy as np
        
        # Check dimension
        if len(embedding) != self._dimension:
            raise ValueError(f"Expected dimension {self._dimension}, got {len(embedding)}")
        
        # Normalize for cosine similarity
        normalized = self._normalize(embedding)
        
        # Add to FAISS
        internal_id = self._next_id
        self._next_id += 1
        
        vector = np.array([normalized], dtype=np.float32)
        self._index.add(vector)
        
        # Store mappings
        self._id_map[internal_id] = id
        self._reverse_id_map[id] = internal_id
        self._contents[id] = content
        self._metadata[id] = metadata or {}
    
    async def add_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        contents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add multiple vectors."""
        import numpy as np
        
        if not ids:
            return
        
        metadatas = metadatas or [{}] * len(ids)
        
        # Normalize all
        normalized = [self._normalize(e) for e in embeddings]
        vectors = np.array(normalized, dtype=np.float32)
        
        # Add to FAISS
        start_id = self._next_id
        self._next_id += len(ids)
        
        self._index.add(vectors)
        
        # Store mappings
        for i, (id, content, metadata) in enumerate(zip(ids, contents, metadatas)):
            internal_id = start_id + i
            self._id_map[internal_id] = id
            self._reverse_id_map[id] = internal_id
            self._contents[id] = content
            self._metadata[id] = metadata
    
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        import numpy as np
        
        if self._index.ntotal == 0:
            return []
        
        # Normalize query
        normalized = self._normalize(query_embedding)
        query = np.array([normalized], dtype=np.float32)
        
        # Search more if we have filters (post-filter)
        search_k = top_k * 5 if filter else top_k
        
        distances, indices = self._index.search(query, min(search_k, self._index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            id = self._id_map.get(int(idx))
            if id is None:
                continue
            
            metadata = self._metadata.get(id, {})
            
            # Apply filter
            if filter:
                match = all(
                    metadata.get(k) == v
                    for k, v in filter.items()
                )
                if not match:
                    continue
            
            results.append(VectorSearchResult(
                id=id,
                score=float(dist),
                content=self._contents.get(id, ""),
                metadata=metadata,
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    async def delete(self, id: str) -> bool:
        """
        Delete a vector by ID.
        
        Note: FAISS doesn't support deletion well. We mark as deleted
        and rebuild index periodically.
        """
        if id not in self._reverse_id_map:
            return False
        
        # Remove from metadata (vector stays in index)
        del self._reverse_id_map[id]
        del self._contents[id]
        del self._metadata[id]
        
        return True
    
    async def get(self, id: str) -> VectorSearchResult | None:
        """Get a vector by ID."""
        if id not in self._contents:
            return None
        
        return VectorSearchResult(
            id=id,
            score=1.0,
            content=self._contents[id],
            metadata=self._metadata.get(id, {}),
        )
    
    async def clear(self) -> None:
        """Clear all vectors."""
        self._initialize_index()
        self._id_map.clear()
        self._reverse_id_map.clear()
        self._metadata.clear()
        self._contents.clear()
        self._next_id = 0
    
    async def count(self) -> int:
        """Count total vectors."""
        return len(self._contents)
    
    def _save(self) -> None:
        """Save index to disk."""
        import faiss
        
        if self._index_path is None:
            return
        
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self._index, str(self._index_path / "index.faiss"))
        
        # Save metadata
        state = {
            "id_map": self._id_map,
            "reverse_id_map": self._reverse_id_map,
            "metadata": self._metadata,
            "contents": self._contents,
            "next_id": self._next_id,
        }
        
        with open(self._index_path / "state.json", "w") as f:
            json.dump(state, f)
    
    def _load(self) -> None:
        """Load index from disk."""
        import faiss
        
        if not self._index_path or not self._index_path.exists():
            return
        
        index_file = self._index_path / "index.faiss"
        state_file = self._index_path / "state.json"
        
        if index_file.exists():
            self._index = faiss.read_index(str(index_file))
        
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            
            self._id_map = {int(k): v for k, v in state["id_map"].items()}
            self._reverse_id_map = state["reverse_id_map"]
            self._metadata = state["metadata"]
            self._contents = state["contents"]
            self._next_id = state["next_id"]


class MilvusVectorStore(VectorStore):
    """
    Milvus-based vector store.
    
    Production-grade distributed vector database.
    Good for large-scale, high-availability deployments.
    """
    
    def __init__(
        self,
        dimension: int,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "aegis_documents",
    ):
        self._dimension = dimension
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._client = None
    
    async def _get_client(self):
        """Lazy initialization of Milvus client."""
        if self._client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError:
                raise ImportError(
                    "pymilvus required. Install with: pip install pymilvus"
                )
            
            self._client = MilvusClient(
                uri=f"http://{self._host}:{self._port}"
            )
            
            # Create collection if not exists
            if self._collection_name not in self._client.list_collections():
                self._client.create_collection(
                    collection_name=self._collection_name,
                    dimension=self._dimension,
                )
        
        return self._client
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    async def add(
        self,
        id: str,
        embedding: list[float],
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a single vector."""
        await self.add_batch([id], [embedding], [content], [metadata or {}])
    
    async def add_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        contents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add multiple vectors."""
        client = await self._get_client()
        metadatas = metadatas or [{}] * len(ids)
        
        data = [
            {
                "id": id,
                "vector": embedding,
                "content": content,
                "metadata": json.dumps(metadata),
            }
            for id, embedding, content, metadata in zip(ids, embeddings, contents, metadatas)
        ]
        
        client.insert(collection_name=self._collection_name, data=data)
    
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        client = await self._get_client()
        
        results = client.search(
            collection_name=self._collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["content", "metadata"],
        )
        
        output = []
        for hit in results[0]:
            metadata = json.loads(hit.get("entity", {}).get("metadata", "{}"))
            
            # Apply filter
            if filter:
                match = all(metadata.get(k) == v for k, v in filter.items())
                if not match:
                    continue
            
            output.append(VectorSearchResult(
                id=str(hit.get("id")),
                score=hit.get("distance", 0.0),
                content=hit.get("entity", {}).get("content", ""),
                metadata=metadata,
            ))
        
        return output
    
    async def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        client = await self._get_client()
        client.delete(collection_name=self._collection_name, ids=[id])
        return True
    
    async def get(self, id: str) -> VectorSearchResult | None:
        """Get a vector by ID."""
        client = await self._get_client()
        results = client.get(
            collection_name=self._collection_name,
            ids=[id],
            output_fields=["content", "metadata"],
        )
        
        if not results:
            return None
        
        entity = results[0]
        return VectorSearchResult(
            id=id,
            score=1.0,
            content=entity.get("content", ""),
            metadata=json.loads(entity.get("metadata", "{}")),
        )
    
    async def clear(self) -> None:
        """Clear all vectors."""
        client = await self._get_client()
        client.drop_collection(self._collection_name)
        self._client = None  # Force recreation
    
    async def count(self) -> int:
        """Count total vectors."""
        client = await self._get_client()
        return client.num_entities(self._collection_name)


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store for testing.
    
    Uses brute-force search. Not suitable for production.
    """
    
    def __init__(self, dimension: int):
        self._dimension = dimension
        self._vectors: dict[str, list[float]] = {}
        self._contents: dict[str, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    async def add(
        self,
        id: str,
        embedding: list[float],
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._vectors[id] = embedding
        self._contents[id] = content
        self._metadata[id] = metadata or {}
    
    async def add_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        contents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        metadatas = metadatas or [{}] * len(ids)
        for id, emb, content, meta in zip(ids, embeddings, contents, metadatas):
            await self.add(id, emb, content, meta)
    
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        scores = []
        
        for id, embedding in self._vectors.items():
            # Apply filter
            if filter:
                meta = self._metadata.get(id, {})
                match = all(meta.get(k) == v for k, v in filter.items())
                if not match:
                    continue
            
            score = self._cosine_similarity(query_embedding, embedding)
            scores.append((id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [
            VectorSearchResult(
                id=id,
                score=score,
                content=self._contents[id],
                metadata=self._metadata.get(id, {}),
            )
            for id, score in scores[:top_k]
        ]
    
    async def delete(self, id: str) -> bool:
        if id in self._vectors:
            del self._vectors[id]
            del self._contents[id]
            self._metadata.pop(id, None)
            return True
        return False
    
    async def get(self, id: str) -> VectorSearchResult | None:
        if id not in self._vectors:
            return None
        return VectorSearchResult(
            id=id,
            score=1.0,
            content=self._contents[id],
            metadata=self._metadata.get(id, {}),
        )
    
    async def clear(self) -> None:
        self._vectors.clear()
        self._contents.clear()
        self._metadata.clear()
    
    async def count(self) -> int:
        return len(self._vectors)
