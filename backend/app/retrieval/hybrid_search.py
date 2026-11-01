"""
app/retrieval/hybrid_search.py – Hybrid (vector + BM25) score fusion.

Implements Reciprocal Rank Fusion (RRF) to combine ranked results from the
vector search and BM25 search into a single merged ranking.

Why RRF instead of linear interpolation?
  RRF is score-scale independent. Vector cosine scores (0–1) and BM25 scores
  (0–∞) live on completely different scales. RRF uses only ranks, making
  it robust without needing per-query normalisation.

  hybrid_score_i = Σ  1 / (k + rank_in_list_j)
                   j

  where k=60 is the RRF constant (standard default).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_RRF_K = 60  # RRF constant – higher = less weight to top ranks


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class HybridResult:
    chunk_id: str
    text: str
    hybrid_score: float
    vector_score: float | None
    bm25_score: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Hybrid Searcher ───────────────────────────────────────────────────────────

class HybridSearcher:
    """
    Combines VectorSearcher and BM25Searcher via Reciprocal Rank Fusion.

    Usage
    -----
    searcher = HybridSearcher()
    results  = searcher.search("LODR disclosure requirements", top_k=20)
    """

    def __init__(
        self,
        candidates: int = 50,
        rrf_k: int = _RRF_K,
    ) -> None:
        """
        Parameters
        ----------
        candidates:
            Number of results to fetch from each sub-searcher before fusion.
            Should be >= top_k passed to .search().
        rrf_k:
            RRF constant (default 60). Increase to widen the scoring spread.
        """
        self._candidates = candidates
        self._rrf_k = rrf_k
        self._vector_searcher = None
        self._bm25_searcher = None

    # ── Lazy init ─────────────────────────────────────────────────────────────

    def _get_searchers(self):
        if self._vector_searcher is None:
            from app.retrieval.vector_search import VectorSearcher
            self._vector_searcher = VectorSearcher()

        if self._bm25_searcher is None:
            from app.retrieval.bm25_search import BM25Searcher
            # Share the already-loaded VectorSearcher metadata to avoid
            # loading metadata.json a second time.
            self._bm25_searcher = BM25Searcher(
                vector_searcher=self._vector_searcher
            )

        return self._vector_searcher, self._bm25_searcher

    # ── RRF fusion ────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_score(rank: int, k: int) -> float:
        return 1.0 / (k + rank + 1)  # rank is 0-indexed

    def _fuse(
        self,
        vector_results,
        bm25_results,
    ) -> list[HybridResult]:
        """Merge two ranked lists with RRF and return combined results."""

        # Build lookup dicts: chunk_id → raw score
        vec_scores: dict[str, float] = {r.chunk_id: r.score for r in vector_results}
        bm25_scores: dict[str, float] = {r.chunk_id: r.score for r in bm25_results}
        metadata_map: dict[str, dict] = {}

        for r in vector_results:
            metadata_map[r.chunk_id] = r.metadata
        for r in bm25_results:
            metadata_map.setdefault(r.chunk_id, r.metadata)

        # Compute RRF scores from each list
        rrf: dict[str, float] = {}

        for rank, result in enumerate(vector_results):
            cid = result.chunk_id
            rrf[cid] = rrf.get(cid, 0.0) + self._rrf_score(rank, self._rrf_k)

        for rank, result in enumerate(bm25_results):
            cid = result.chunk_id
            rrf[cid] = rrf.get(cid, 0.0) + self._rrf_score(rank, self._rrf_k)

        # Build output list sorted by RRF score
        fused: list[HybridResult] = []
        for chunk_id, hybrid_score in rrf.items():
            meta = metadata_map.get(chunk_id, {})
            fused.append(
                HybridResult(
                    chunk_id=chunk_id,
                    text=meta.get("text", ""),
                    hybrid_score=hybrid_score,
                    vector_score=vec_scores.get(chunk_id),
                    bm25_score=bm25_scores.get(chunk_id),
                    metadata=meta,
                )
            )

        return sorted(fused, key=lambda r: r.hybrid_score, reverse=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int) -> list[HybridResult]:
        """
        Run vector + BM25 search in sequence and return RRF-fused results.

        Parameters
        ----------
        query:
            Natural-language question.
        top_k:
            Number of final results to return.

        Returns
        -------
        list[HybridResult] sorted by hybrid_score descending.
        """
        vector_searcher, bm25_searcher = self._get_searchers()

        n = max(top_k, self._candidates)

        logger.debug("Running vector search (top %d) …", n)
        vector_results = vector_searcher.search(query, top_k=n)

        logger.debug("Running BM25 search (top %d) …", n)
        bm25_results = bm25_searcher.search(query, top_k=n)

        logger.debug(
            "Fusing %d vector + %d BM25 results …",
            len(vector_results),
            len(bm25_results),
        )
        fused = self._fuse(vector_results, bm25_results)

        return fused[:top_k]

    @property
    def total_vectors(self) -> int:
        vec, _ = self._get_searchers()
        return vec.total_vectors
