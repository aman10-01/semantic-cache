"""In-memory vector store with cosine similarity search.

This is the Phase-1 backend.  It keeps all embeddings in a NumPy matrix
for fast brute-force search.  Later phases can swap this out for
Redis+RedisVL or Qdrant without touching the rest of the code.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from .models import CacheEntry

logger = logging.getLogger(__name__)


class SimilarityMatch:
    """A single search result from the vector store."""

    __slots__ = ("entry", "score")

    def __init__(self, entry: CacheEntry, score: float) -> None:
        self.entry = entry
        self.score = score

    def __repr__(self) -> str:  # pragma: no cover
        return f"SimilarityMatch(id={self.entry.id!r}, score={self.score:.4f})"


class InMemoryVectorStore:
    """Thread-safe, brute-force vector store backed by NumPy.

    Stores embeddings in a (N × D) float32 matrix and performs cosine
    similarity via a single matrix–vector dot product (O(N·D)).
    Because all vectors are pre-normalised to unit length, dot product
    **equals** cosine similarity.

    Parameters
    ----------
    dimension:
        Embedding dimensionality (e.g. 384 for all-MiniLM-L6-v2).
    max_entries:
        Hard cap.  When reached, the oldest entry is evicted (FIFO).
    """

    def __init__(self, dimension: int, max_entries: int = 10_000) -> None:
        self._dim = dimension
        self._max = max_entries
        self._lock = threading.Lock()

        # Parallel arrays — same index in each structure refers to the
        # same cache entry.
        self._embeddings: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self._entries: list[CacheEntry] = []

    # ── Public API ──────────────────────────────────────────────────
    @property
    def size(self) -> int:
        """Current number of entries in the store."""
        return len(self._entries)

    def add(self, entry: CacheEntry) -> None:
        """Insert a cache entry.  Evicts the oldest entry if full."""
        vec = np.array(entry.prompt_embedding, dtype=np.float32).reshape(1, -1)
        with self._lock:
            if len(self._entries) >= self._max:
                self._evict_oldest()
            self._embeddings = (
                np.vstack([self._embeddings, vec])
                if self._embeddings.size
                else vec
            )
            self._entries.append(entry)
        logger.debug("Stored entry %s  (store size=%d)", entry.id, self.size)

    def search(
        self,
        query_embedding: list[float],
        cache_key: str,
        threshold: float = 0.95,
        top_k: int = 1,
    ) -> list[SimilarityMatch]:
        """Find the most similar entries **within the same cache-key partition**.

        Only entries whose ``cache_key`` matches are considered — this
        prevents cross-contamination between different system prompts,
        models, or temperature settings.

        Returns at most *top_k* matches whose similarity ≥ *threshold*,
        sorted by descending similarity.
        """
        query_vec = np.array(query_embedding, dtype=np.float32)

        with self._lock:
            if self._embeddings.size == 0:
                return []

            # Build a mask for entries sharing the same cache_key
            key_mask = np.array(
                [e.cache_key == cache_key for e in self._entries], dtype=bool
            )
            if not key_mask.any():
                return []

            # Filter out expired entries
            now = time.time()
            expiry_mask = np.array(
                [not e.metadata.is_expired for e in self._entries], dtype=bool
            )
            combined_mask = key_mask & expiry_mask
            if not combined_mask.any():
                return []

            # Cosine similarity = dot product (vectors are unit-normalised)
            scores = self._embeddings[combined_mask] @ query_vec
            masked_indices = np.where(combined_mask)[0]

            # Gather results above threshold
            matches: list[SimilarityMatch] = []
            for rank_idx in np.argsort(scores)[::-1][:top_k]:
                sim = float(scores[rank_idx])
                if sim < threshold:
                    break
                original_idx = int(masked_indices[rank_idx])
                matches.append(SimilarityMatch(self._entries[original_idx], sim))
            return matches

    def delete(self, entry_id: str) -> bool:
        """Remove a single entry by ID.  Returns True if found."""
        with self._lock:
            for i, e in enumerate(self._entries):
                if e.id == entry_id:
                    self._entries.pop(i)
                    self._embeddings = np.delete(self._embeddings, i, axis=0)
                    logger.debug("Deleted entry %s", entry_id)
                    return True
        return False

    def clear(self) -> None:
        """Wipe the entire store."""
        with self._lock:
            self._embeddings = np.empty((0, self._dim), dtype=np.float32)
            self._entries.clear()

    def get_all_entries(self) -> list[CacheEntry]:
        """Return a shallow copy of all entries (for debugging / stats)."""
        with self._lock:
            return list(self._entries)

    # ── Internal ────────────────────────────────────────────────────
    def _evict_oldest(self) -> None:
        """Remove the oldest entry (FIFO).  Caller must hold ``_lock``."""
        if not self._entries:
            return
        evicted = self._entries.pop(0)
        self._embeddings = self._embeddings[1:]
        logger.info("Evicted oldest entry %s (hit_count=%d)", evicted.id, evicted.metadata.hit_count)
