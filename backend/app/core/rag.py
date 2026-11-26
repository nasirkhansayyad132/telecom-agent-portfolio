"""Advanced RAG engine with semantic embeddings and hybrid retrieval."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import advanced components, fall back gracefully
try:
    from .vector_store import VectorStore, VectorDocument, SearchResult
    from .embeddings import get_embedding_client
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    VectorStore = None  # type: ignore
    VectorDocument = None  # type: ignore
    SearchResult = None  # type: ignore


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with source and relevance info."""
    source: str
    snippet: str
    score: float
    retrieval_method: str = "hybrid"  # "semantic", "keyword", or "hybrid"


class RAGEngine:
    """Production-grade RAG engine with semantic and keyword hybrid retrieval.
    
    Features:
    - Semantic search using OpenAI embeddings + FAISS
    - Keyword fallback using TF-IDF (works offline)
    - Hybrid mode combining both for best results
    - Automatic chunking with overlap
    - Persistent index caching
    """

    def __init__(
        self,
        data_dir: Path,
        chunk_size_words: int = 160,
        overlap_words: int = 30,
        use_semantic: bool = True,
        semantic_weight: float = 0.7,
    ) -> None:
        """Initialize the RAG engine.
        
        Args:
            data_dir: Directory containing markdown FAQ files
            chunk_size_words: Target words per chunk
            overlap_words: Overlap between consecutive chunks
            use_semantic: Enable semantic search (requires OPENAI_API_KEY)
            semantic_weight: Weight for semantic scores in hybrid mode (0-1)
        """
        self.data_dir = data_dir
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words
        self.use_semantic = use_semantic and SEMANTIC_AVAILABLE
        self.semantic_weight = semantic_weight

        # TF-IDF components (always available as fallback)
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )
        self._tfidf_matrix = None
        
        # Document storage
        self._documents: List[str] = []
        self._metadata: List[dict] = []
        self._doc_ids: List[str] = []
        
        # Semantic components
        self._vector_store: Optional[VectorStore] = None
        self._index_path = data_dir / ".index" / "rag_index"
        
        # Track content hash to detect changes
        self._content_hash: Optional[str] = None

    def _compute_content_hash(self) -> str:
        """Compute hash of all source files to detect changes."""
        hasher = hashlib.sha256()
        if self.data_dir.exists():
            for path in sorted(self.data_dir.glob("*.md")):
                hasher.update(path.read_bytes())
        return hasher.hexdigest()

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks while preserving paragraph structure."""
        if not text:
            return []

        # Split by double newline to preserve paragraphs/headers
        blocks = text.split('\n\n')
        chunks = []
        current_chunk_blocks = []
        current_word_count = 0
        
        for block in blocks:
            # Count words in this block
            words = block.split()
            word_count = len(words)
            
            if not word_count:
                continue
                
            # If a single block is huge (larger than chunk size), split it by words
            if word_count > self.chunk_size_words:
                # First, flush any accumulated blocks
                if current_chunk_blocks:
                    chunks.append("\n\n".join(current_chunk_blocks))
                    current_chunk_blocks = []
                    current_word_count = 0
                
                # Then split the huge block
                step = self.chunk_size_words - self.overlap_words
                for i in range(0, len(words), step):
                    chunk_words = words[i : i + self.chunk_size_words]
                    chunks.append(" ".join(chunk_words))
                continue

            # Check if adding this block exceeds chunk size
            if current_word_count + word_count > self.chunk_size_words:
                # Save current chunk
                chunks.append("\n\n".join(current_chunk_blocks))
                
                # Start new chunk with this block
                # (Simple approach: no overlap for blocks to avoid duplication, 
                # unless we implement a sliding window of blocks which is complex)
                current_chunk_blocks = [block]
                current_word_count = word_count
            else:
                current_chunk_blocks.append(block)
                current_word_count += word_count
                
        # Flush remaining
        if current_chunk_blocks:
            chunks.append("\n\n".join(current_chunk_blocks))
            
        return chunks

    def load(self, force_rebuild: bool = False) -> None:
        """Load documents and build/load indices.
        
        Args:
            force_rebuild: Force rebuilding indices even if cached
        """
        new_hash = self._compute_content_hash()
        
        # Check if we can use cached index
        if not force_rebuild and new_hash == self._content_hash:
            return  # Already loaded and up to date
            
        self._documents = []
        self._metadata = []
        self._doc_ids = []

        if not self.data_dir.exists():
            return

        # Load and chunk documents
        for path in sorted(self.data_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            
            for i, chunk in enumerate(self._chunk_text(text)):
                doc_id = f"{path.stem}_{i}"
                self._documents.append(chunk)
                self._metadata.append({
                    "source": path.name,
                    "chunk_index": i,
                })
                self._doc_ids.append(doc_id)

        if not self._documents:
            return

        # Build TF-IDF index (always available)
        self._tfidf_matrix = self.tfidf_vectorizer.fit_transform(self._documents)

        # Build semantic index if available and API key is set
        if self.use_semantic and os.getenv("OPENAI_API_KEY"):
            self._build_semantic_index(force_rebuild, new_hash)

        self._content_hash = new_hash

    def _build_semantic_index(self, force_rebuild: bool, content_hash: str) -> None:
        """Build or load the semantic vector index."""
        if not SEMANTIC_AVAILABLE:
            return

        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        hash_file = self._index_path.with_suffix(".hash")

        # Check if cached index matches current content
        cached_hash = None
        if hash_file.exists():
            cached_hash = hash_file.read_text().strip()

        if not force_rebuild and cached_hash == content_hash:
            # Try to load cached index
            self._vector_store = VectorStore(index_path=self._index_path)
            if self._vector_store.load():
                return

        # Build new index
        try:
            self._vector_store = VectorStore(index_path=self._index_path)
            
            # Create vector documents
            vector_docs = [
                VectorDocument(
                    id=self._doc_ids[i],
                    content=self._documents[i],
                    metadata=self._metadata[i],
                )
                for i in range(len(self._documents))
            ]
            
            self._vector_store.add_documents(vector_docs)
            self._vector_store.save()
            
            # Save content hash
            hash_file.write_text(content_hash)
            
        except Exception as e:
            # Fall back to TF-IDF only
            self._vector_store = None
            print(f"Warning: Semantic index failed, using TF-IDF only: {e}")

    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Perform TF-IDF keyword search. Returns (index, score) pairs."""
        if not self._documents or self._tfidf_matrix is None:
            return []

        query_vec = self.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> Tuple[List[RetrievedChunk], float]:
        """Retrieve relevant document chunks.
        
        Uses hybrid retrieval when semantic search is available,
        falls back to keyword-only otherwise.
        
        Args:
            query: Search query
            top_k: Maximum results to return
            min_score: Minimum relevance score threshold
            
        Returns:
            Tuple of (list of chunks, best score)
        """
        if not query.strip() or not self._documents:
            return [], 0.0

        # Get keyword results
        keyword_results = self._keyword_search(query, top_k=len(self._documents))
        
        # If semantic search is available, use hybrid mode
        if self._vector_store is not None and len(self._vector_store) > 0:
            keyword_scores = [
                (self._doc_ids[idx], score) for idx, score in keyword_results
            ]
            
            hybrid_results = self._vector_store.hybrid_search(
                query=query,
                documents_with_keyword_scores=keyword_scores,
                top_k=top_k,
                semantic_weight=self.semantic_weight,
            )
            
            chunks = [
                RetrievedChunk(
                    source=r.metadata["source"],
                    snippet=r.content[:400],
                    score=r.score,
                    retrieval_method="hybrid",
                )
                for r in hybrid_results
                if r.score >= min_score
            ]
            
            best_score = chunks[0].score if chunks else 0.0
            return chunks, best_score

        # Fallback to keyword-only
        chunks = []
        for idx, score in keyword_results[:top_k]:
            if score < min_score:
                continue
            chunks.append(RetrievedChunk(
                source=self._metadata[idx]["source"],
                snippet=self._documents[idx][:400],
                score=score,
                retrieval_method="keyword",
            ))

        best_score = chunks[0].score if chunks else 0.0
        return chunks, best_score

    def get_stats(self) -> dict:
        """Get statistics about the loaded knowledge base."""
        return {
            "total_chunks": len(self._documents),
            "total_sources": len(set(m["source"] for m in self._metadata)),
            "semantic_enabled": self._vector_store is not None,
            "content_hash": self._content_hash,
        }


__all__ = ["RAGEngine", "RetrievedChunk"]
