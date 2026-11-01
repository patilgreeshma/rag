"""
app/retrieval/bm25_search.py – BM25 keyword search over the SEBI corpus.

Uses rank_bm25 (BM25Okapi) to score chunks by keyword overlap.
Critical for exact-identifier queries like regulation numbers, section IDs,
and specific legal terms that semantic search may miss.

The BM25 index is built lazily from the FAISS metadata (no separate store needed).
"""
from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ── Tokenizer ─────────────────────────────────────────────────────────────────

_PUNCT_TABLE = str.maketrans("", "", string.punctuation.replace("-", ""))


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + lowercase tokeniser.

    Keeps hyphens (important for regulation numbers like 'sub-section').
    Removes other punctuation and splits on whitespace.
    """
    text = text.lower().translate(_PUNCT_TABLE)
    return [tok for tok in re.split(r"\s+", text) if tok]


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class BM25Result:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


# ── BM25 Searcher ─────────────────────────────────────────────────────────────

class BM25Searcher:
    """
    BM25Okapi search over the entire SEBI corpus.

    The BM25 index is built once from the FAISS metadata (which already holds
    all chunk texts and metadata). No additional file I/O is required.

    Usage
    -----
    searcher = BM25Searcher()
    results  = searcher.search("LODR disclosure requirements", top_k=20)
    """

    def __init__(self, vector_searcher=None) -> None:
        """
        Parameters
        ----------
        vector_searcher:
            An already-loaded VectorSearcher whose .metadata list is reused.
            If None, a new VectorSearcher is created and its metadata loaded.
        """
        self._vector_searcher = vector_searcher
        self._bm25 = None
        self._corpus_meta: list[dict[str, Any]] = []

    # ── Lazy build ────────────────────────────────────────────────────────────

    def _build(self) -> None:
        if self._bm25 is not None:
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 is required for BM25 search. "
                "Install it with: pip install rank-bm25"
            ) from exc

        # Re-use metadata from VectorSearcher to avoid loading index.faiss again
        if self._vector_searcher is None:
            from app.retrieval.vector_search import VectorSearcher
            self._vector_searcher = VectorSearcher()

        self._corpus_meta = self._vector_searcher.metadata
        logger.info("Building BM25 index over %d chunks …", len(self._corpus_meta))

        tokenized_corpus = [
            _tokenize(meta.get("text", "")) for meta in self._corpus_meta
        ]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int) -> list[BM25Result]:
        """
        Return the top-k chunks by BM25 score for the given query.

        Parameters
        ----------
        query:
            Natural-language or keyword query.
        top_k:
            Number of results to return.

        Returns
        -------
        list[BM25Result] sorted by BM25 score descending.
        """
        self._build()

        import numpy as np

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Get indices of top-k scores
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[BM25Result] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            meta = self._corpus_meta[idx]
            results.append(
                BM25Result(
                    chunk_id=meta.get("chunk_id", f"idx_{idx}"),
                    text=meta.get("text", ""),
                    score=score,
                    metadata=meta,
                )
            )

        return results
