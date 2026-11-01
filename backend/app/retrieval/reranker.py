"""
app/retrieval/reranker.py – Cross-Encoder reranker for FinRule AI.

Takes the top-N candidates from hybrid search and reranks them using a
Cross-Encoder model that reads query + chunk text jointly, capturing
fine-grained relevance signals that bi-encoder models miss.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Scores a (query, passage) pair with a single relevance logit.
  - Accurate but slow → only applied to top-N candidates (default 20).
  - CPU-friendly (< 1s for 20 pairs on modern hardware).

Pipeline
--------
  HybridSearcher → top 20 candidates
      ↓ CrossEncoderReranker
  top 5 (or top_k) → Gemini
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class RerankedChunk:
    chunk_id: str
    text: str
    rerank_score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    hybrid_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Convenience accessors (mirrors rag.py RetrievedChunk interface)
    @property
    def title(self) -> str | None:
        return self.metadata.get("title")

    @property
    def section_number(self) -> str | None:
        return self.metadata.get("section_number")

    @property
    def section(self) -> str | None:
        return self.metadata.get("section")

    @property
    def source_url(self) -> str | None:
        return self.metadata.get("source_url")

    @property
    def document_type(self) -> str | None:
        return self.metadata.get("document_type")

    @property
    def in_force(self) -> bool:
        return bool(self.metadata.get("in_force", True))

    @property
    def score(self) -> float:
        """Alias so the RAG pipeline can use .score uniformly."""
        return self.rerank_score


# ── Reranker ──────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Cross-Encoder reranker that refines a candidate list.

    Usage
    -----
    reranker = CrossEncoderReranker()
    top5 = reranker.rerank(query="…", candidates=hybrid_results, top_k=5)
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or _DEFAULT_MODEL
        self._model = None  # lazy init

    # ── Lazy load ─────────────────────────────────────────────────────────────

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the Cross-Encoder reranker. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading Cross-Encoder model: %s", self._model_name)
        self._model = CrossEncoder(self._model_name)
        logger.info("Cross-Encoder model loaded.")
        return self._model

    # ── Public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list,
        top_k: int,
    ) -> list[RerankedChunk]:
        """
        Score each candidate with the Cross-Encoder and return top_k.

        Parameters
        ----------
        query:
            The user's question.
        candidates:
            List of HybridResult (or any object with .text, .chunk_id, .metadata).
        top_k:
            Number of top reranked results to return.

        Returns
        -------
        list[RerankedChunk] sorted by rerank_score descending.
        """
        if not candidates:
            return []

        model = self._ensure_model()

        # Build (query, passage) pairs
        pairs = [(query, c.text) for c in candidates]

        logger.debug("Cross-Encoder scoring %d pairs …", len(pairs))
        scores = model.predict(pairs)

        reranked: list[RerankedChunk] = []
        for candidate, score in zip(candidates, scores):
            reranked.append(
                RerankedChunk(
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    rerank_score=float(score),
                    vector_score=getattr(candidate, "vector_score", None),
                    bm25_score=getattr(candidate, "bm25_score", None),
                    hybrid_score=getattr(candidate, "hybrid_score", None),
                    metadata=candidate.metadata,
                )
            )

        reranked.sort(key=lambda r: r.rerank_score, reverse=True)
        logger.debug("Reranking done. Top score: %.4f", reranked[0].rerank_score if reranked else 0)

        return reranked[:top_k]
