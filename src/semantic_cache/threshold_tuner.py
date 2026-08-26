"""Threshold Tuner — analyse how different thresholds affect hit rate.

This is the core interview talking point. It lets you visualise the
tradeoff between cache hit rate and accuracy at different thresholds.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LookupRecord:
    """Records a single cache lookup for historical analysis."""
    prompt: str
    best_similarity: float  # highest similarity score found (0.0 if no match)
    was_hit: bool           # did it pass the active threshold?
    active_threshold: float # what threshold was active at lookup time
    cache_key: str
    timestamp: float = field(default_factory=time.time)


class ThresholdTuner:
    """Collects lookup history and analyses threshold tradeoffs.

    Every lookup (hit or miss) is recorded. The tune() method then
    replays the history at different thresholds to show the tradeoff.
    """

    def __init__(self, max_history: int = 10_000) -> None:
        self._history: list[LookupRecord] = []
        self._max_history = max_history

    def record(self, record: LookupRecord) -> None:
        """Record a lookup event."""
        self._history.append(record)
        # Trim old records if over limit
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    @property
    def history_size(self) -> int:
        return len(self._history)

    def analyse(
        self,
        thresholds: list[float] | None = None,
    ) -> dict:
        """Analyse hit rates at different thresholds.

        Returns a dict with results for each test threshold showing
        what the hit rate would have been if that threshold was active.
        """
        if not self._history:
            return {
                "error": "No lookup history yet. Send some requests first.",
                "history_size": 0,
            }

        if thresholds is None:
            thresholds = [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99]

        # Only consider lookups that found at least one candidate
        lookups_with_candidates = [
            r for r in self._history if r.best_similarity > 0.0
        ]
        total_lookups = len(self._history)
        lookups_with_match = len(lookups_with_candidates)

        results = []
        for threshold in sorted(thresholds):
            # How many lookups would have been hits at this threshold?
            would_hit = sum(
                1 for r in self._history
                if r.best_similarity >= threshold
            )
            hit_rate = would_hit / total_lookups if total_lookups > 0 else 0.0

            # Near misses: lookups that fell within 0.05 of the threshold
            near_misses = sum(
                1 for r in self._history
                if (threshold - 0.05) <= r.best_similarity < threshold
            )

            results.append({
                "threshold": threshold,
                "would_hit": would_hit,
                "would_miss": total_lookups - would_hit,
                "hit_rate": round(hit_rate, 4),
                "hit_rate_pct": f"{hit_rate * 100:.1f}%",
                "near_misses": near_misses,
            })

        # Find the similarity score distribution
        all_scores = [r.best_similarity for r in self._history if r.best_similarity > 0]
        score_stats = {}
        if all_scores:
            all_scores.sort()
            score_stats = {
                "min": round(min(all_scores), 4),
                "max": round(max(all_scores), 4),
                "mean": round(sum(all_scores) / len(all_scores), 4),
                "median": round(all_scores[len(all_scores) // 2], 4),
                "p25": round(all_scores[len(all_scores) // 4], 4),
                "p75": round(all_scores[3 * len(all_scores) // 4], 4),
            }

        return {
            "total_lookups": total_lookups,
            "lookups_with_candidates": lookups_with_match,
            "lookups_no_candidates": total_lookups - lookups_with_match,
            "score_distribution": score_stats,
            "threshold_analysis": results,
            "recommendation": self._recommend(results),
        }

    def _recommend(self, results: list[dict]) -> str:
        """Generate a simple recommendation based on the analysis."""
        if not results:
            return "Not enough data for a recommendation."

        # Find the sweet spot: highest hit rate with threshold >= 0.90
        safe_results = [r for r in results if r["threshold"] >= 0.90]
        if safe_results:
            best = max(safe_results, key=lambda r: r["hit_rate"])
            return (
                f"Recommended threshold: {best['threshold']:.2f} "
                f"(hit rate: {best['hit_rate_pct']}, "
                f"near misses: {best['near_misses']})"
            )
        return "Not enough data for a recommendation."

    def get_near_misses(
        self, threshold: float, limit: int = 20
    ) -> list[dict]:
        """Return lookups that narrowly missed the threshold.

        These are opportunities: if you lowered the threshold slightly,
        or normalised the prompts, these would become cache hits.
        """
        near = []
        for r in reversed(self._history):
            if (threshold - 0.05) <= r.best_similarity < threshold:
                near.append({
                    "prompt": r.prompt,
                    "best_similarity": round(r.best_similarity, 4),
                    "gap": round(threshold - r.best_similarity, 4),
                    "timestamp": r.timestamp,
                })
            if len(near) >= limit:
                break
        return near