"""Interactive demo of the Semantic Cache — Phase 1.

Run:
    python demo.py

The demo:
1. Initialises the cache with the free local embedding model.
2. Stores a few sample prompts with mock LLM responses.
3. Shows cache HITs for semantically similar (but not identical) prompts.
4. Shows cache MISSes for unrelated or different-context prompts.
5. Prints cache stats.
"""

import logging
import time

from src.semantic_cache.cache_engine import SemanticCache
from src.semantic_cache.config import CacheConfig

# Pretty logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)

DIVIDER = "─" * 60


def main() -> None:
    print(f"\n{'═' * 60}")
    print("  Semantic Cache — Phase 1 Demo")
    print(f"{'═' * 60}\n")

    # ── 1. Initialise ───────────────────────────────────────────
    config = CacheConfig(similarity_threshold=0.90)
    cache = SemanticCache(config)
    print(f"\n{DIVIDER}")
    print("  Cache is ready!\n")

    # ── 2. Seed the cache with sample Q&A ───────────────────────
    seed_data = [
        {
            "prompt": "What is Python?",
            "response": {
                "content": "Python is a high-level, interpreted programming language known for its readability and versatility.",
                "model": "gpt-4o",
            },
        },
        {
            "prompt": "Explain machine learning in simple terms",
            "response": {
                "content": "Machine learning is a subset of AI where computers learn patterns from data without being explicitly programmed.",
                "model": "gpt-4o",
            },
        },
        {
            "prompt": "How does a REST API work?",
            "response": {
                "content": "A REST API uses HTTP methods (GET, POST, PUT, DELETE) to perform CRUD operations on resources identified by URLs.",
                "model": "gpt-4o",
            },
        },
    ]

    print("  Seeding cache with sample data …\n")
    for item in seed_data:
        entry_id = cache.store(
            prompt=item["prompt"],
            response=item["response"],
            model_id="gpt-4o",
            token_usage={"prompt": 15, "completion": 40, "total": 55},
            finish_reason="stop",
        )
        print(f"    ✓ Stored: \"{item['prompt']}\"  (id={entry_id[:8]}…)")

    # ── 3. Test semantically similar prompts (expect HITs) ──────
    print(f"\n{DIVIDER}")
    print("  Testing SIMILAR prompts (should be cache HITs):\n")

    similar_queries = [
        "What is Python programming language?",
        "Explain Python to me",
        "Tell me about machine learning simply",
        "What is ML in easy words?",
        "How do REST APIs function?",
    ]

    for query in similar_queries:
        t0 = time.perf_counter()
        result = cache.lookup(query)
        elapsed = (time.perf_counter() - t0) * 1000
        status = "✅ HIT " if result.hit else "❌ MISS"
        score = f"(score={result.similarity_score:.4f})" if result.hit else ""
        print(f"    {status} │ {elapsed:6.1f}ms │ \"{query}\" {score}")
        if result.hit:
            print(f"           │         │ → {result.response['content'][:70]}…")

    # ── 4. Test unrelated prompts (expect MISSes) ───────────────
    print(f"\n{DIVIDER}")
    print("  Testing UNRELATED prompts (should be cache MISSes):\n")

    unrelated_queries = [
        "What is the capital of France?",
        "How do I bake chocolate cookies?",
        "Explain quantum entanglement",
    ]

    for query in unrelated_queries:
        t0 = time.perf_counter()
        result = cache.lookup(query)
        elapsed = (time.perf_counter() - t0) * 1000
        status = "✅ HIT " if result.hit else "❌ MISS"
        print(f"    {status} │ {elapsed:6.1f}ms │ \"{query}\"")

    # ── 5. Test context isolation ───────────────────────────────
    print(f"\n{DIVIDER}")
    print("  Testing CONTEXT ISOLATION (same prompt, different system prompt):\n")

    # Store with a specific system prompt
    cache.store(
        prompt="What is Python?",
        response={"content": "Python is a snake species found in Asia and Africa."},
        system_prompt="You are a biology expert",
    )
    print('    ✓ Stored: "What is Python?" with system_prompt="You are a biology expert"')

    # Lookup without system prompt → should hit the ORIGINAL entry
    r1 = cache.lookup("What is Python?")
    print(f"\n    Lookup (no system prompt):     {'HIT' if r1.hit else 'MISS'} → {r1.response['content'][:60] if r1.hit else 'N/A'}…")

    # Lookup with biology system prompt → should hit the BIOLOGY entry
    r2 = cache.lookup("What is Python?", system_prompt="You are a biology expert")
    print(f"    Lookup (biology system prompt): {'HIT' if r2.hit else 'MISS'} → {r2.response['content'][:60] if r2.hit else 'N/A'}…")

    # ── 6. Stats ────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("  Cache Statistics:\n")
    stats = cache.stats()
    for key, value in stats.items():
        print(f"    {key:25s}: {value}")

    print(f"\n{'═' * 60}")
    print("  Phase 1 demo complete!")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
