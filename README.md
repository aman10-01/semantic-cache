# Semantic Caching Layer for LLM APIs

> **Phase 1 — Cache Index & Similarity Engine**

A middleware caching service that detects semantically similar LLM prompts and serves cached responses instantly — cutting latency to near-zero and eliminating redundant API costs.

## 🏗️ Project Structure

```
semantic-cache/
├── src/
│   └── semantic_cache/
│       ├── __init__.py          # Package init
│       ├── config.py            # Tunable config (env-var overridable)
│       ├── models.py            # CacheEntry, CacheLookupResult, cache key builder
│       ├── embeddings.py        # Local embedding engine (sentence-transformers)
│       ├── cache_store.py       # In-memory vector store with NumPy similarity
│       └── cache_engine.py      # Main SemanticCache class — the public API
├── tests/
│   └── test_cache.py            # Unit & integration tests
├── demo.py                      # Interactive demo script
├── requirements.txt
└── README.md
```

## ⚡ Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo
python demo.py

# 4. Run the tests
python -m pytest tests/ -v
```

> **Note:** The embedding model (`all-MiniLM-L6-v2`, ~80 MB) is downloaded automatically on first run. No API key needed — everything runs locally.

## 🔧 Configuration

All settings can be overridden via environment variables:

| Env Variable | Default | Description |
|---|---|---|
| `SEMCACHE_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `SEMCACHE_SIMILARITY_THRESHOLD` | `0.95` | Min cosine similarity for a cache hit |
| `SEMCACHE_DEFAULT_TTL_SECONDS` | `86400` | Default entry TTL (24h) |
| `SEMCACHE_MAX_CACHE_ENTRIES` | `10000` | Max entries before FIFO eviction |

## 🧠 How It Works

1. **Embed** every incoming prompt using `all-MiniLM-L6-v2` (free, local, sub-millisecond)
2. **Search** the vector store for the nearest neighbour within the same context partition
3. If cosine similarity ≥ threshold → **cache hit** → return stored response instantly
4. If below threshold → **cache miss** → forward to LLM, store the response for future hits
5. **Context isolation** — same prompt with different system prompts / models / temperatures are kept separate

## 📐 Phase 1 Scope

- ✅ Embedding engine with free local model
- ✅ In-memory vector store with cosine similarity search
- ✅ Cache-key partitioning by system prompt, model, temperature, and custom params
- ✅ TTL-based expiration and FIFO eviction
- ✅ Hit counting and stats
- ✅ Thread-safe operations
- ✅ Comprehensive test suite

## 🗺️ Upcoming Phases

| Phase | Focus | Status |
|---|---|---|
| Phase 1 | Cache Index & Similarity Engine | ✅ Complete |
| Phase 2 | Drop-In Proxy API (FastAPI, OpenAI-compatible) | ⬜ |
| Phase 3 | Cache Policies & Eviction | ⬜ |
| Phase 4 | Monitoring & Analytics (Prometheus + Grafana) | ⬜ |
| Phase 5 | Containerize & Load Test (Docker Compose) | ⬜ |
| Phase 6 | Polish for Portfolio | ⬜ |
