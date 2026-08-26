from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingEngine:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        logger.info("Loading embedding model '%s' …", model_name)
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_embedding_dimension()
        logger.info(
            "Model loaded — dimension=%d, max_seq_length=%d",
            self._dimension,
            self._model.max_seq_length,
        )

    # ── Public API ──────────────────────────────────────────────────
    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, text: str) -> list[float]:
        vec: np.ndarray = self._model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )
        return vec.tolist()

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        vecs: np.ndarray = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return vecs.tolist()
