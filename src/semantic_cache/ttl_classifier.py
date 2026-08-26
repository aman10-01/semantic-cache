"""TTL Classifier — auto-assigns cache TTL tiers based on prompt content.

Analyses the prompt text with simple keyword rules to decide:
  - FACTUAL  → 24 hours (stable answers, safe to cache long)
  - STANDARD → 6 hours  (general queries)
  - VOLATILE → 1 hour   (time-sensitive, current events)
  - SKIP     → 0        (creative / unique — don't cache at all)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TTLTier(str, Enum):
    FACTUAL = "factual"
    STANDARD = "standard"
    VOLATILE = "volatile"
    SKIP = "skip"


@dataclass(frozen=True)
class TTLTierConfig:
    """Maps a tier to its TTL in seconds."""
    tier: TTLTier
    ttl_seconds: int
    description: str


# Default tier configurations
DEFAULT_TIERS: dict[TTLTier, TTLTierConfig] = {
    TTLTier.FACTUAL: TTLTierConfig(
        tier=TTLTier.FACTUAL,
        ttl_seconds=86_400,  # 24 hours
        description="Factual/stable queries (definitions, math, history)",
    ),
    TTLTier.STANDARD: TTLTierConfig(
        tier=TTLTier.STANDARD,
        ttl_seconds=21_600,  # 6 hours
        description="General-purpose queries",
    ),
    TTLTier.VOLATILE: TTLTierConfig(
        tier=TTLTier.VOLATILE,
        ttl_seconds=3_600,  # 1 hour
        description="Time-sensitive queries (news, weather, prices)",
    ),
    TTLTier.SKIP: TTLTierConfig(
        tier=TTLTier.SKIP,
        ttl_seconds=0,
        description="Creative/unique — do not cache",
    ),
}

# ── Keyword patterns for classification ─────────────────────────────
_VOLATILE_PATTERNS = [
    r"\b(today|tonight|yesterday|right now|currently|latest|recent)\b",
    r"\b(weather|forecast|stock|price|score|news|breaking)\b",
    r"\b(this (week|month|year))\b",
    r"\b(2024|2025|2026)\b",
]

_FACTUAL_PATTERNS = [
    r"\b(what is|what are|define|definition|explain|meaning of)\b",
    r"\b(how does|how do|how to)\b",
    r"\b(capital of|population of|formula for)\b",
    r"\b(history of|who (is|was|invented|discovered))\b",
    r"\b(difference between|compare)\b",
    r"\b(math|calcul|equation|theorem|algorithm)\b",
]

_SKIP_PATTERNS = [
    r"\b(write me a (poem|story|song|essay|letter))\b",
    r"\b(creative|imagine|pretend|roleplay|fiction)\b",
    r"\b(brainstorm|generate ideas|come up with)\b",
    r"\b(unique|original|novel|inventive)\b",
]

# Pre-compile for performance
_VOLATILE_RE = [re.compile(p, re.IGNORECASE) for p in _VOLATILE_PATTERNS]
_FACTUAL_RE = [re.compile(p, re.IGNORECASE) for p in _FACTUAL_PATTERNS]
_SKIP_RE = [re.compile(p, re.IGNORECASE) for p in _SKIP_PATTERNS]


def classify_prompt(prompt: str) -> tuple[TTLTier, int]:
    """Classify a prompt into a TTL tier and return (tier, ttl_seconds).

    The classifier checks keyword patterns in priority order:
    1. SKIP patterns (creative) → return immediately, don't cache
    2. VOLATILE patterns (time-sensitive) → short TTL
    3. FACTUAL patterns (stable) → long TTL
    4. Default → STANDARD tier
    """
    # Priority 1: Skip (don't cache creative content)
    skip_score = sum(1 for r in _SKIP_RE if r.search(prompt))
    if skip_score >= 1:
        logger.debug("TTL classified as SKIP: %r", prompt[:50])
        tier = TTLTier.SKIP
        return tier, DEFAULT_TIERS[tier].ttl_seconds

    # Priority 2: Volatile (time-sensitive)
    volatile_score = sum(1 for r in _VOLATILE_RE if r.search(prompt))
    if volatile_score >= 1:
        logger.debug("TTL classified as VOLATILE: %r", prompt[:50])
        tier = TTLTier.VOLATILE
        return tier, DEFAULT_TIERS[tier].ttl_seconds

    # Priority 3: Factual (stable, long-lived)
    factual_score = sum(1 for r in _FACTUAL_RE if r.search(prompt))
    if factual_score >= 1:
        logger.debug("TTL classified as FACTUAL: %r", prompt[:50])
        tier = TTLTier.FACTUAL
        return tier, DEFAULT_TIERS[tier].ttl_seconds

    # Default: Standard
    logger.debug("TTL classified as STANDARD (default): %r", prompt[:50])
    tier = TTLTier.STANDARD
    return tier, DEFAULT_TIERS[tier].ttl_seconds