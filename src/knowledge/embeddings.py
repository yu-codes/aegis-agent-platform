"""
Embedding Service

Generate vector embeddings for text.
Abstracts different embedding providers.

Design decisions:
- Provider-agnostic interface
- Batch embedding for efficiency
- Caching for repeated texts
- Dimension normalization
"""

import hashlib
from abc import ABC, abstractmethod
from typing import Any


class EmbeddingService(ABC):
    """
    Abstract embedding service.

    Generates dense vector representations of text
    for semantic similarity search.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        pass


class OpenAIEmbeddings(EmbeddingService):
    """
    OpenAI embedding service.

    Uses text-embedding-3-small/large models.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,  # Optional dimension reduction
    ):
        self._model = model
        self._dimensions = dimensions
        self._client = None
        self._api_key = api_key

        # Cache for repeated embeddings
        self._cache: dict[str, list[float]] = {}
        self._cache_enabled = True

    async def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key

            self._client = AsyncOpenAI(**kwargs)
        return self._client

    @property
    def dimension(self) -> int:
        if self._dimensions:
            return self._dimensions

        # Default dimensions by model
        if "3-small" in self._model:
            return 1536
        elif "3-large" in self._model:
            return 3072
        else:
            return 1536

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        # Check cache
        if self._cache_enabled:
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                return self._cache[cache_key]

        client = await self._get_client()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": text,
        }
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions

        response = await client.embeddings.create(**kwargs)
        embedding = response.data[0].embedding

        # Cache result
        if self._cache_enabled:
            self._cache[cache_key] = embedding

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for batch."""
        if not texts:
            return []

        # Check cache for each text
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        if self._cache_enabled:
            for i, text in enumerate(texts):
                cache_key = self._cache_key(text)
                if cache_key in self._cache:
                    results[i] = self._cache[cache_key]
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        # Embed uncached texts
        if uncached_texts:
            client = await self._get_client()

            kwargs: dict[str, Any] = {
                "model": self._model,
                "input": uncached_texts,
            }
            if self._dimensions:
                kwargs["dimensions"] = self._dimensions

            response = await client.embeddings.create(**kwargs)

            for i, embedding_data in enumerate(response.data):
                original_idx = uncached_indices[i]
                embedding = embedding_data.embedding
                results[original_idx] = embedding

                # Cache
                if self._cache_enabled:
                    cache_key = self._cache_key(uncached_texts[i])
                    self._cache[cache_key] = embedding

        return [r for r in results if r is not None]


class LocalEmbeddings(EmbeddingService):
    """
    Local embedding service using sentence-transformers.

    Runs on CPU/GPU locally, no API calls needed.
    Good for privacy-sensitive applications.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self._model_name = model_name
        self._device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install with: "
                    "pip install sentence-transformers"
                )

            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    @property
    def dimension(self) -> int:
        model = self._get_model()
        return model.get_sentence_embedding_dimension()

    async def embed(self, text: str) -> list[float]:
        """Generate embedding locally."""
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for batch."""
        if not texts:
            return []

        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class CachedEmbeddings(EmbeddingService):
    """
    Embedding service wrapper with persistent caching.

    Uses Redis or file system to cache embeddings across restarts.
    """

    def __init__(
        self,
        base_service: EmbeddingService,
        cache_backend: str = "memory",  # memory, redis, file
        redis_url: str | None = None,
        cache_path: str | None = None,
    ):
        self._base = base_service
        self._cache_backend = cache_backend
        self._redis_url = redis_url
        self._cache_path = cache_path
        self._memory_cache: dict[str, list[float]] = {}

    @property
    def dimension(self) -> int:
        return self._base.dimension

    def _cache_key(self, text: str) -> str:
        return f"emb:{hashlib.md5(text.encode()).hexdigest()}"

    async def _get_from_cache(self, key: str) -> list[float] | None:
        if self._cache_backend == "memory":
            return self._memory_cache.get(key)
        elif self._cache_backend == "redis":
            import json

            import redis.asyncio as redis

            client = redis.from_url(self._redis_url)
            data = await client.get(key)
            await client.close()
            return json.loads(data) if data else None
        return None

    async def _set_in_cache(self, key: str, value: list[float]) -> None:
        if self._cache_backend == "memory":
            self._memory_cache[key] = value
        elif self._cache_backend == "redis":
            import json

            import redis.asyncio as redis

            client = redis.from_url(self._redis_url)
            await client.set(key, json.dumps(value))
            await client.close()

    async def embed(self, text: str) -> list[float]:
        key = self._cache_key(text)

        # Try cache
        cached = await self._get_from_cache(key)
        if cached:
            return cached

        # Generate
        embedding = await self._base.embed(text)

        # Cache
        await self._set_in_cache(key, embedding)

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # For batch, check cache for each
        results: list[list[float] | None] = [None] * len(texts)
        uncached = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            cached = await self._get_from_cache(key)
            if cached:
                results[i] = cached
            else:
                uncached.append((i, text))

        # Embed uncached
        if uncached:
            uncached_texts = [t for _, t in uncached]
            embeddings = await self._base.embed_batch(uncached_texts)

            for (original_idx, text), embedding in zip(uncached, embeddings, strict=False):
                results[original_idx] = embedding
                await self._set_in_cache(self._cache_key(text), embedding)

        return [r for r in results if r is not None]
