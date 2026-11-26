"""FAISS-based vector store for semantic similarity search."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore

from .embeddings import EmbeddingClient, get_embedding_client


@dataclass
class VectorDocument:
    """A document stored in the vector store."""
    id: str
    content: str
    metadata: dict
    embedding: Optional[np.ndarray] = None


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    id: str
    content: str
    metadata: dict
    score: float  # Cosine similarity score (higher is better)


class VectorStore:
    """FAISS-based vector store with persistence support."""

    def __init__(
        self,
        index_path: Optional[Path] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ) -> None:
        """Initialize vector store.
        
        Args:
            index_path: Path to save/load the index
            embedding_client: Custom embedding client (uses default if None)
        """
        if faiss is None:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")

        self.index_path = index_path
        self.embedding_client = embedding_client or get_embedding_client()
        self.dimension = self.embedding_client.dimension

        # FAISS index for cosine similarity (normalized inner product)
        self._index: Optional[faiss.IndexFlatIP] = None
        self._documents: List[VectorDocument] = []
        self._id_to_idx: dict[str, int] = {}

    def _ensure_index(self) -> faiss.IndexFlatIP:
        """Ensure FAISS index is initialized."""
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.dimension)
        return self._index

    def add_documents(self, documents: List[VectorDocument]) -> None:
        """Add documents to the vector store.
        
        If documents have embeddings, they will be used directly.
        Otherwise, embeddings will be generated.
        """
        if not documents:
            return

        index = self._ensure_index()

        # Separate documents with and without embeddings
        docs_needing_embedding = []
        for doc in documents:
            if doc.embedding is None:
                docs_needing_embedding.append(doc)

        # Generate missing embeddings
        if docs_needing_embedding:
            texts = [doc.content for doc in docs_needing_embedding]
            embeddings = self.embedding_client.embed_batch(texts)
            for i, doc in enumerate(docs_needing_embedding):
                doc.embedding = embeddings[i]

        # Normalize and add to index
        for doc in documents:
            if doc.id in self._id_to_idx:
                continue  # Skip duplicates

            # Normalize for cosine similarity
            embedding = doc.embedding.copy()
            faiss.normalize_L2(embedding.reshape(1, -1))

            idx = len(self._documents)
            self._documents.append(doc)
            self._id_to_idx[doc.id] = idx
            index.add(embedding.reshape(1, -1))

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """Search for similar documents.
        
        Args:
            query: Search query text
            top_k: Maximum number of results
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of SearchResult sorted by score (highest first)
        """
        if not self._documents or self._index is None:
            return []

        # Generate query embedding
        query_embedding = self.embedding_client.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Search
        k = min(top_k, len(self._documents))
        scores, indices = self._index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < min_score:
                continue
            doc = self._documents[idx]
            results.append(SearchResult(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                score=float(score),
            ))

        return results

    def hybrid_search(
        self,
        query: str,
        documents_with_keyword_scores: List[Tuple[str, float]],
        top_k: int = 5,
        semantic_weight: float = 0.7,
    ) -> List[SearchResult]:
        """Combine semantic and keyword search results.
        
        Args:
            query: Search query text
            documents_with_keyword_scores: List of (doc_id, keyword_score) tuples
            top_k: Maximum number of results
            semantic_weight: Weight for semantic scores (keyword weight = 1 - semantic_weight)
            
        Returns:
            List of SearchResult with combined scores
        """
        # Get semantic results
        semantic_results = self.search(query, top_k=len(self._documents))
        semantic_scores = {r.id: r.score for r in semantic_results}

        # Normalize keyword scores to 0-1
        keyword_scores = dict(documents_with_keyword_scores)
        max_keyword = max(keyword_scores.values()) if keyword_scores else 1.0
        if max_keyword > 0:
            keyword_scores = {k: v / max_keyword for k, v in keyword_scores.items()}

        # Combine scores
        all_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        combined = []

        for doc_id in all_ids:
            sem_score = semantic_scores.get(doc_id, 0.0)
            kw_score = keyword_scores.get(doc_id, 0.0)
            combined_score = (semantic_weight * sem_score) + ((1 - semantic_weight) * kw_score)

            if doc_id in self._id_to_idx:
                doc = self._documents[self._id_to_idx[doc_id]]
                combined.append(SearchResult(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=combined_score,
                ))

        # Sort by score and limit
        combined.sort(key=lambda x: x.score, reverse=True)
        return combined[:top_k]

    def save(self, path: Optional[Path] = None) -> None:
        """Save the vector store to disk."""
        save_path = path or self.index_path
        if save_path is None:
            raise ValueError("No save path specified")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self._index is not None:
            faiss.write_index(self._index, str(save_path.with_suffix(".faiss")))

        # Save documents (without embeddings to save space)
        docs_to_save = [
            {"id": d.id, "content": d.content, "metadata": d.metadata}
            for d in self._documents
        ]
        with open(save_path.with_suffix(".pkl"), "wb") as f:
            pickle.dump({
                "documents": docs_to_save,
                "id_to_idx": self._id_to_idx,
            }, f)

    def load(self, path: Optional[Path] = None) -> bool:
        """Load the vector store from disk. Returns True if successful."""
        load_path = path or self.index_path
        if load_path is None:
            return False

        faiss_path = load_path.with_suffix(".faiss")
        pkl_path = load_path.with_suffix(".pkl")

        if not faiss_path.exists() or not pkl_path.exists():
            return False

        try:
            self._index = faiss.read_index(str(faiss_path))

            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            self._documents = [
                VectorDocument(id=d["id"], content=d["content"], metadata=d["metadata"])
                for d in data["documents"]
            ]
            self._id_to_idx = data["id_to_idx"]
            return True
        except Exception:
            return False

    def clear(self) -> None:
        """Clear all documents from the store."""
        self._index = None
        self._documents = []
        self._id_to_idx = {}

    def __len__(self) -> int:
        return len(self._documents)


__all__ = ["VectorStore", "VectorDocument", "SearchResult"]
