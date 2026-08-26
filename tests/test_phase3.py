"""Phase 3 tests -- TTL classifier, invalidation, threshold tuner, adaptive thresholds."""

import time
import pytest
from src.semantic_cache.config import CacheConfig
from src.semantic_cache.cache_engine import SemanticCache
from src.semantic_cache.ttl_classifier import classify_prompt, TTLTier
from src.semantic_cache.threshold_tuner import ThresholdTuner, LookupRecord


# ── TTL Classifier Tests ───────────────────────────────────────────

class TestTTLClassifier:
    """Test the keyword-based TTL classifier."""

    def test_factual_what_is(self):
        tier, ttl = classify_prompt("What is the capital of France?")
        assert tier == TTLTier.FACTUAL
        assert ttl == 86_400

    def test_factual_how_does(self):
        tier, ttl = classify_prompt("How does photosynthesis work?")
        assert tier == TTLTier.FACTUAL

    def test_factual_define(self):
        tier, ttl = classify_prompt("Define the word 'entropy'")
        assert tier == TTLTier.FACTUAL

    def test_volatile_today(self):
        tier, ttl = classify_prompt("What is the weather today?")
        assert tier == TTLTier.VOLATILE
        assert ttl == 3_600

    def test_volatile_stock_price(self):
        tier, ttl = classify_prompt("What is the stock price of AAPL?")
        assert tier == TTLTier.VOLATILE

    def test_volatile_latest_news(self):
        tier, ttl = classify_prompt("Tell me the latest news about AI")
        assert tier == TTLTier.VOLATILE

    def test_skip_write_poem(self):
        tier, ttl = classify_prompt("Write me a poem about the ocean")
        assert tier == TTLTier.SKIP
        assert ttl == 0

    def test_skip_creative(self):
        tier, ttl = classify_prompt("Be creative and imagine a new planet")
        assert tier == TTLTier.SKIP

    def test_standard_default(self):
        tier, ttl = classify_prompt("Summarize this document for me")
        assert tier == TTLTier.STANDARD
        assert ttl == 21_600


# ── Threshold Tuner Tests ──────────────────────────────────────────

class TestThresholdTuner:
    """Test the threshold analysis engine."""

    def test_empty_history(self):
        tuner = ThresholdTuner()
        result = tuner.analyse()
        assert "error" in result

    def test_records_lookups(self):
        tuner = ThresholdTuner()
        tuner.record(LookupRecord(
            prompt="test",
            best_similarity=0.92,
            was_hit=False,
            active_threshold=0.95,
            cache_key="abc",
        ))
        assert tuner.history_size == 1

    def test_analysis_output_structure(self):
        tuner = ThresholdTuner()
        # Add some fake history
        for score in [0.85, 0.90, 0.93, 0.96, 0.97, 0.99]:
            tuner.record(LookupRecord(
                prompt=f"test {score}",
                best_similarity=score,
                was_hit=score >= 0.95,
                active_threshold=0.95,
                cache_key="abc",
            ))

        result = tuner.analyse()
        assert "threshold_analysis" in result
        assert "total_lookups" in result
        assert "score_distribution" in result
        assert "recommendation" in result
        assert len(result["threshold_analysis"]) > 0

    def test_hit_rate_calculation(self):
        tuner = ThresholdTuner()
        # 5 lookups: scores = [0.80, 0.85, 0.90, 0.95, 0.99]
        for score in [0.80, 0.85, 0.90, 0.95, 0.99]:
            tuner.record(LookupRecord(
                prompt=f"test",
                best_similarity=score,
                was_hit=score >= 0.95,
                active_threshold=0.95,
                cache_key="abc",
            ))

        result = tuner.analyse(thresholds=[0.90, 0.95])
        analysis = {r["threshold"]: r for r in result["threshold_analysis"]}

        # At 0.90: scores 0.90, 0.95, 0.99 would hit = 3/5 = 60%
        assert analysis[0.90]["would_hit"] == 3
        assert analysis[0.90]["hit_rate"] == 0.6

        # At 0.95: scores 0.95, 0.99 would hit = 2/5 = 40%
        assert analysis[0.95]["would_hit"] == 2
        assert analysis[0.95]["hit_rate"] == 0.4

    def test_near_misses(self):
        tuner = ThresholdTuner()
        tuner.record(LookupRecord(
            prompt="near miss prompt",
            best_similarity=0.93,
            was_hit=False,
            active_threshold=0.95,
            cache_key="abc",
        ))
        misses = tuner.get_near_misses(threshold=0.95)
        assert len(misses) == 1
        assert misses[0]["prompt"] == "near miss prompt"
        assert misses[0]["gap"] == pytest.approx(0.02, abs=0.001)

    def test_max_history_trimming(self):
        tuner = ThresholdTuner(max_history=5)
        for i in range(10):
            tuner.record(LookupRecord(
                prompt=f"test {i}",
                best_similarity=0.90,
                was_hit=False,
                active_threshold=0.95,
                cache_key="abc",
            ))
        assert tuner.history_size == 5


# ── Cache Invalidation Tests ───────────────────────────────────────

class TestCacheInvalidation:
    """Test cache invalidation by model, system prompt, and prefix."""

    @pytest.fixture
    def cache(self):
        config = CacheConfig(similarity_threshold=0.75)
        return SemanticCache(config)

    def test_invalidate_by_model(self, cache):
        # Store 2 entries with different models
        cache.store("What is Python?", {"content": "A language"}, model_id="gpt-4o", ttl_seconds=3600)
        cache.store("What is Java?", {"content": "Another language"}, model_id="llama3.2", ttl_seconds=3600)
        assert cache.stats()["total_entries"] == 2

        # Invalidate gpt-4o entries
        removed = cache.invalidate_by_model("gpt-4o")
        assert removed == 1
        assert cache.stats()["total_entries"] == 1

    def test_invalidate_by_prefix(self, cache):
        cache.store("What is Python?", {"content": "A"}, ttl_seconds=3600)
        cache.store("What is Java?", {"content": "B"}, ttl_seconds=3600)
        cache.store("How does DNA work?", {"content": "C"}, ttl_seconds=3600)

        # Invalidate all "What is" entries
        removed = cache.invalidate_by_prefix("What is")
        assert removed == 2
        assert cache.stats()["total_entries"] == 1

    def test_invalidate_by_system_prompt(self, cache):
        cache.store("Hello", {"content": "Hi"}, system_prompt="You are a tutor", ttl_seconds=3600)
        cache.store("Hello", {"content": "Yo"}, system_prompt="You are a pirate", ttl_seconds=3600)

        removed = cache.invalidate_by_system_prompt("You are a tutor")
        assert removed == 1
        assert cache.stats()["total_entries"] == 1

    def test_purge_expired(self, cache):
        # Store with 0-second TTL (immediately expired)
        cache.store("Expire me", {"content": "gone"}, ttl_seconds=1)
        time.sleep(1.1)

        purged = cache.purge_expired()
        assert purged == 1
        assert cache.stats()["total_entries"] == 0


# ── TTL Auto-Classification in Store ───────────────────────────────

class TestTTLAutoClassification:
    """Test that store() auto-classifies TTL when not provided."""

    @pytest.fixture
    def cache(self):
        config = CacheConfig(similarity_threshold=0.75)
        return SemanticCache(config)

    def test_factual_gets_long_ttl(self, cache):
        entry_id = cache.store("What is the speed of light?", {"content": "fast"})
        assert entry_id != ""  # should be stored (not skipped)

    def test_creative_gets_skipped(self, cache):
        entry_id = cache.store("Write me a poem about stars", {"content": "twinkle"})
        assert entry_id == ""  # SKIP tier = not cached
        assert cache.stats()["total_entries"] == 0

    def test_explicit_ttl_overrides_classifier(self, cache):
        # Even a "creative" prompt gets cached if TTL is explicit
        entry_id = cache.store(
            "Write me a poem about stars",
            {"content": "twinkle"},
            ttl_seconds=600,
        )
        assert entry_id != ""
        assert cache.stats()["total_entries"] == 1