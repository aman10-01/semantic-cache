"""Configuration for the Semantic Cache."""

from pydantic_settings import BaseSettings
from pydantic import Field


class CacheConfig(BaseSettings):
    """All tunable parameters for the semantic cache engine.

    Values can be overridden via environment variables prefixed with
    ``SEMCACHE_`` (e.g. ``SEMCACHE_SIMILARITY_THRESHOLD=0.92``).
    """

    # -- Embedding Model ---------------------------------------------
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformers model name.",
    )

    # -- Similarity --------------------------------------------------
    similarity_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity for a cache hit. "
            "Start high (0.95) and tune down with the threshold tuner later."
        ),
    )
    enable_adaptive_threshold: bool = Field(
        default=True,
        description=(
            "When True, the cache auto-adjusts the similarity threshold "
            "based on detected task type (classification, creative, etc.)."
        ),
    )

    # -- Cache Limits ------------------------------------------------
    default_ttl_seconds: int = Field(
        default=86_400,  # 24 hours
        description="Default time-to-live for a cache entry in seconds.",
    )
    max_cache_entries: int = Field(
        default=10_000,
        description="Maximum number of entries before eviction kicks in.",
    )

    # -- Provider Settings (Phase 2) ---------------------------------
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key. Leave empty to skip OpenAI provider.",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL.",
    )
    default_provider: str = Field(
        default="ollama",
        description="Default LLM provider: 'openai' or 'ollama'.",
    )

    # -- Proxy Server ------------------------------------------------
    proxy_host: str = Field(default="0.0.0.0", description="Proxy listen host.")
    proxy_port: int = Field(default=8000, description="Proxy listen port.")

    model_config = {"env_prefix": "SEMCACHE_"}