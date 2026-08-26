"""Unit tests for the Phase 1 Semantic Cache engine."""

import time
import unittest

from src.semantic_cache.cache_engine import SemanticCache
from src.semantic_cache.cache_store import InMemoryVectorStore
from src.semantic_cache.config import CacheConfig
from src.semantic_cache.embeddings import EmbeddingEngine
from src.semantic_cache.models import CacheEntry, CacheEntryMetadata, build_cache_key


class TestBuildCacheKey(unittest.TestCase):
    """Tests for the deterministic cache-key hash."""

    def test_same_inputs_same_key(self):
        k1 = build_cache_key(system_prompt="You are helpful", model="gpt-4o")
        k2 = build_cache_key(system_prompt="You are helpful", model="gpt-4o")
        self.assertEqual(k1, k2)

    def test_different_system_prompts_different_keys(self):
        k1 = build_cache_key(system_prompt="You are a tutor")
        k2 = build_cache_key(system_prompt="You are a pirate")
        self.assertNotEqual(k1, k2)

    def test_different_models_different_keys(self):
        k1 = build_cache_key(model="gpt-4o")
        k2 = build_cache_key(model="claude-3")
        self.assertNotEqual(k1, k2)

    def test_different_temperatures_different_keys(self):
        k1 = build_cache_key(temperature=0.0)
        k2 = build_cache_key(temperature=1.0)
        self.assertNotEqual(k1, k2)

    def test_no_params_gives_default_key(self):
        k = build_cache_key()
        self.assertIsInstance(k, str)
        self.assertEqual(len(k), 16)

    def test_extra_params_affect_key(self):
        k1 = build_cache_key(max_tokens=100)
        k2 = build_cache_key(max_tokens=200)
        self.assertNotEqual(k1, k2)


class TestEmbeddingEngine(unittest.TestCase):
    """Tests for the local embedding model."""

    @classmethod
    def setUpClass(cls):
        cls.engine = EmbeddingEngine("all-MiniLM-L6-v2")

    def test_single_encode_returns_list(self):
        vec = self.engine.encode("hello world")
        self.assertIsInstance(vec, list)
        self.assertEqual(len(vec), self.engine.dimension)

    def test_dimension_is_384(self):
        self.assertEqual(self.engine.dimension, 384)

    def test_batch_encode(self):
        vecs = self.engine.encode_batch(["hello", "world"])
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), self.engine.dimension)

    def test_similar_prompts_have_high_similarity(self):
        import numpy as np

        v1 = np.array(self.engine.encode("What is Python?"))
        v2 = np.array(self.engine.encode("Explain Python to me"))
        sim = float(np.dot(v1, v2))
        self.assertGreater(sim, 0.7)

    def test_different_prompts_have_low_similarity(self):
        import numpy as np

        v1 = np.array(self.engine.encode("What is Python?"))
        v2 = np.array(self.engine.encode("Recipe for chocolate cake"))
        sim = float(np.dot(v1, v2))
        self.assertLess(sim, 0.5)


class TestInMemoryVectorStore(unittest.TestCase):
    """Tests for the in-memory vector store."""

    @classmethod
    def setUpClass(cls):
        cls.engine = EmbeddingEngine("all-MiniLM-L6-v2")

    def _make_entry(self, prompt: str, cache_key: str = "default", ttl: int = 86400) -> CacheEntry:
        return CacheEntry(
            prompt_text=prompt,
            prompt_embedding=self.engine.encode(prompt),
            response={"content": f"Response to: {prompt}"},
            cache_key=cache_key,
            metadata=CacheEntryMetadata(ttl_seconds=ttl),
        )

    def test_add_and_search(self):
        store = InMemoryVectorStore(dimension=384)
        entry = self._make_entry("What is Python?")
        store.add(entry)
        self.assertEqual(store.size, 1)

        query_vec = self.engine.encode("Tell me about Python")
        results = store.search(query_vec, cache_key="default", threshold=0.7)
        self.assertTrue(len(results) > 0)
        self.assertGreater(results[0].score, 0.7)

    def test_cache_key_isolation(self):
        store = InMemoryVectorStore(dimension=384)
        store.add(self._make_entry("What is Python?", cache_key="key_a"))
        store.add(self._make_entry("What is Python?", cache_key="key_b"))

        query_vec = self.engine.encode("What is Python?")

        results_a = store.search(query_vec, cache_key="key_a", threshold=0.9)
        results_b = store.search(query_vec, cache_key="key_b", threshold=0.9)
        results_c = store.search(query_vec, cache_key="key_c", threshold=0.9)

        self.assertEqual(len(results_a), 1)
        self.assertEqual(len(results_b), 1)
        self.assertEqual(len(results_c), 0)  # no entries under key_c

    def test_expired_entries_not_returned(self):
        store = InMemoryVectorStore(dimension=384)
        # TTL of 0 seconds → already expired
        entry = self._make_entry("What is Python?", ttl=0)
        entry.metadata.created_at = time.time() - 1  # ensure it's expired
        store.add(entry)

        query_vec = self.engine.encode("What is Python?")
        results = store.search(query_vec, cache_key="default", threshold=0.5)
        self.assertEqual(len(results), 0)

    def test_eviction_on_max_entries(self):
        store = InMemoryVectorStore(dimension=384, max_entries=2)
        store.add(self._make_entry("First prompt"))
        store.add(self._make_entry("Second prompt"))
        store.add(self._make_entry("Third prompt"))  # should evict first
        self.assertEqual(store.size, 2)

    def test_delete(self):
        store = InMemoryVectorStore(dimension=384)
        entry = self._make_entry("What is Python?")
        store.add(entry)
        self.assertEqual(store.size, 1)

        deleted = store.delete(entry.id)
        self.assertTrue(deleted)
        self.assertEqual(store.size, 0)

    def test_clear(self):
        store = InMemoryVectorStore(dimension=384)
        store.add(self._make_entry("A"))
        store.add(self._make_entry("B"))
        store.clear()
        self.assertEqual(store.size, 0)


class TestSemanticCache(unittest.TestCase):
    """Integration tests for the SemanticCache engine."""

    @classmethod
    def setUpClass(cls):
        cls.config = CacheConfig(similarity_threshold=0.75 , enable_adaptive_threshold = False)
        cls.cache = SemanticCache(cls.config)

    def setUp(self):
        self.cache.clear()

    def test_miss_then_hit(self):
        # First lookup → miss
        r1 = self.cache.lookup("What is Python?")
        self.assertFalse(r1.hit)

        # Store
        self.cache.store(
            prompt="What is Python?",
            response={"content": "Python is a programming language."},
        )

        # Second lookup with exact same prompt → hit
        r2 = self.cache.lookup("What is Python?")
        self.assertTrue(r2.hit)
        self.assertEqual(r2.response["content"], "Python is a programming language.")

    def test_semantic_hit(self):
        self.cache.store(
            prompt="What is Python?",
            response={"content": "Python is a programming language."},
        )

        # Semantically similar but not identical
        result = self.cache.lookup("Explain Python to me")
        self.assertTrue(result.hit)
        self.assertGreater(result.similarity_score, 0.75)

    def test_unrelated_miss(self):
        self.cache.store(
            prompt="What is Python?",
            response={"content": "Python is a programming language."},
        )

        result = self.cache.lookup("How to bake a cake?")
        self.assertFalse(result.hit)

    def test_system_prompt_isolation(self):
        # Same prompt, different system prompts
        self.cache.store(
            prompt="What is Python?",
            response={"content": "A programming language."},
            system_prompt="You are a coding tutor",
        )
        self.cache.store(
            prompt="What is Python?",
            response={"content": "A species of snake."},
            system_prompt="You are a biology expert",
        )

        r1 = self.cache.lookup("What is Python?", system_prompt="You are a coding tutor")
        r2 = self.cache.lookup("What is Python?", system_prompt="You are a biology expert")

        self.assertTrue(r1.hit)
        self.assertTrue(r2.hit)
        self.assertIn("programming", r1.response["content"])
        self.assertIn("snake", r2.response["content"])

    def test_stats(self):
        self.cache.store(prompt="A", response={"content": "a"})
        self.cache.store(prompt="B", response={"content": "b"})
        self.cache.lookup("A")

        stats = self.cache.stats()
        self.assertEqual(stats["total_entries"], 2)
        self.assertGreaterEqual(stats["total_hits"], 1)

    def test_custom_threshold_override(self):
        self.cache.store(
            prompt="What is Python?",
            response={"content": "Python is a programming language."},
        )

        # Very high threshold → likely miss even for similar prompts
        result = self.cache.lookup("Tell me about Python", threshold=0.99)
        self.assertFalse(result.hit)

        # Lower threshold → hit
        result = self.cache.lookup("Tell me about Python", threshold=0.70)
        self.assertTrue(result.hit)


if __name__ == "__main__":
    unittest.main()
