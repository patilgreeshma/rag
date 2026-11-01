"""
app/embeddings/embedding_model.py – Phase 2.

Reusable Sentence Transformers embedding service.

Design principles
─────────────────
* Model is loaded once via a singleton; never re-loaded per chunk.
* Batch encoding is the primary interface.
* Embedding dimension is determined programmatically from the loaded model.
* Empty / whitespace-only inputs are handled gracefully (zero vector returned).
* The service is importable without triggering model download (lazy load).
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wraps a Sentence Transformer model with a clean, reusable API.

    Usage
    -----
    model = EmbeddingModel()          # uses settings.embedding_model
    dim   = model.get_embedding_dim() # 384 for all-MiniLM-L6-v2
    vec   = model.encode_text("What is CVE-2021-44228?")
    vecs  = model.encode_batch(["text 1", "text 2", ...])
    """

    def __init__(self, model_name: str | None = None) -> None:
        from app.config import settings

        self._model_name: str = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None
        self._dim: int | None = None

    # ── Lazy load ─────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> "SentenceTransformer":
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> "SentenceTransformer":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        logger.info("Loading embedding model: %s", self._model_name)
        t0 = time.perf_counter()
        model = SentenceTransformer(self._model_name)
        elapsed = time.perf_counter() - t0

        # Determine dimension from a probe embedding
        probe = model.encode(["probe"], convert_to_numpy=True)
        self._dim = int(probe.shape[1])
        logger.info(
            "Embedding model loaded in %.1fs  |  dimension: %d",
            elapsed,
            self._dim,
        )
        return model

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Eagerly load the model (useful for pre-warming at startup)."""
        self._ensure_loaded()

    def get_embedding_dim(self) -> int:
        """Return the vector dimension of this model."""
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def encode_text(self, text: str) -> list[float]:
        """
        Encode a single text string into a float list.

        Returns a zero vector for empty/whitespace-only input.
        """
        if not text or not text.strip():
            logger.warning("encode_text received empty string – returning zero vector.")
            dim = self.get_embedding_dim()
            return [0.0] * dim

        model = self._ensure_loaded()
        vec: np.ndarray = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.tolist()

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Encode a list of texts into a list of float vectors.

        Empty/whitespace texts within the batch are replaced with zero vectors.

        Parameters
        ----------
        texts:
            Input strings.
        batch_size:
            Override the per-call batch size (defaults to settings.embedding_batch_size).
        show_progress:
            Show a tqdm progress bar (useful for large batches in scripts).
        """
        from app.config import settings

        if not texts:
            return []

        bs = batch_size or settings.embedding_batch_size

        # Separate valid and empty indices
        valid_mask = [bool(t and t.strip()) for t in texts]
        valid_texts = [t for t, ok in zip(texts, valid_mask) if ok]

        model = self._ensure_loaded()
        dim = self._dim
        assert dim is not None

        result: list[list[float]] = [[0.0] * dim] * len(texts)

        if valid_texts:
            encoded: np.ndarray = model.encode(
                valid_texts,
                batch_size=bs,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )
            vi = 0
            for i, ok in enumerate(valid_mask):
                if ok:
                    result[i] = encoded[vi].tolist()
                    vi += 1

        return result

    @property
    def model_name(self) -> str:
        return self._model_name


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this for reuse across the application.
_default_model: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    """Return the application-wide singleton EmbeddingModel (lazy-loaded)."""
    global _default_model
    if _default_model is None:
        _default_model = EmbeddingModel()
    return _default_model
