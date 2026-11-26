"""OpenAI embeddings with caching and batch processing for semantic RAG."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from openai import OpenAI

# Cache directory for embeddings
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / ".embeddings_cache"


class EmbeddingClient:
    """OpenAI embedding client with caching and batch support."""

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        cache_enabled: bool = True,
    ) -> None:
        self.model = model
        self.cache_enabled = cache_enabled
        self._client: Optional[OpenAI] = None
        self._dimension = 1536  # ada-002 dimension

        if cache_enabled:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            self._client = OpenAI(api_key=api_key)
        return self._client

    @property
    def dimension(self) -> int:
        """Return embedding dimension for the model."""
        return self._dimension

    def _cache_key(self, text: str) -> str:
        """Generate cache key from text content."""
        content = f"{self.model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cached(self, text: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache if exists."""
        if not self.cache_enabled:
            return None
        cache_file = CACHE_DIR / f"{self._cache_key(text)}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        return None

    def _save_to_cache(self, text: str, embedding: np.ndarray) -> None:
        """Save embedding to cache."""
        if not self.cache_enabled:
            return
        cache_file = CACHE_DIR / f"{self._cache_key(text)}.npy"
        np.save(cache_file, embedding)

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        cached = self._get_cached(text)
        if cached is not None:
            return cached

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        self._save_to_cache(text, embedding)
        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings for multiple texts with batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Maximum texts per API call (OpenAI limit is 2048)
            
        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)

        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cached = self._get_cached(text)
            if cached is not None:
                embeddings.append((i, cached))
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        # Batch embed remaining texts
        for start in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[start : start + batch_size]
            batch_indices = indices_to_embed[start : start + batch_size]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )

            for j, data in enumerate(response.data):
                embedding = np.array(data.embedding, dtype=np.float32)
                self._save_to_cache(batch[j], embedding)
                embeddings.append((batch_indices[j], embedding))

        # Sort by original index and stack
        embeddings.sort(key=lambda x: x[0])
        return np.stack([e[1] for e in embeddings])

    def clear_cache(self) -> int:
        """Clear all cached embeddings. Returns number of files deleted."""
        if not CACHE_DIR.exists():
            return 0
        count = 0
        for cache_file in CACHE_DIR.glob("*.npy"):
            cache_file.unlink()
            count += 1
        return count


# Singleton instance for convenience
_default_client: Optional[EmbeddingClient] = None


def get_embedding_client() -> EmbeddingClient:
    """Get the default embedding client instance."""
    global _default_client
    if _default_client is None:
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        _default_client = EmbeddingClient(model=model)
    return _default_client


__all__ = ["EmbeddingClient", "get_embedding_client"]
