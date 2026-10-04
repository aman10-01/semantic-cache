"""Abstract base class for vector store backends.

Allows swapping between in-memory (NumPy) and distributed (Redis/Qdrant)
backends via configuration.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

from .models import CacheEntry


@dataclass
class VectorSearchResult:
    """Standard search result across all store implementations."""
    entry: CacheEntry
    score: float


class BaseVectorStore(abc.ABC):
    """Abstract contract that any cache storage backend must implement."""

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """Total number of entries in the store."""

    @abc.abstractmethod
    def add(self, entry: CacheEntry) -> None:
        """Add or update an entry."""

    @abc.abstractmethod
    def search(
        self,
        query_embedding: list[float],
        cache_key: str,
        threshold: float = 0.85,
        top_k: int = 1,
    ) -> list[VectorSearchResult]:
        """Search for entries with cosine similarity >= threshold."""

    @abc.abstractmethod
    def search_best_score(
        self,
        query_embedding: list[float],
        cache_key: str,
    ) -> float:
        """Return the highest similarity score regardless of threshold."""

    @abc.abstractmethod
    def get_all_entries(self) -> list[CacheEntry]:
        """Retrieve all active entries."""

    @abc.abstractmethod
    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID."""

    @abc.abstractmethod
    def clear(self) -> None:
        """Wipe all entries."""