"""
app/retrieval/vector_search.py – FAISS vector search wrapper.

Wraps the FAISSRetriever from rag.py as a standalone, importable module
so it can be composed with BM25 in the hybrid pipeline.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VectorSearchResult:
    """A single hit from the FAISS index."""

    __slots__ = ("chunk_id", "text", "score", "metadata")

    def __init__(
        self,
        chunk_id: str,
        text: str,
        score: float,
        metadata: dict[str, Any],
    ) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.score = score
        self.metadata = metadata


class VectorSearcher:
    """
    Loads a FAISS index and retrieves the top-k most similar chunks.

    The index is built by scripts/build_index.py.
    """

    def __init__(self, index_dir: Optional[str] = None) -> None:
        from app.config import settings

        self._index_dir = index_dir or str(settings.faiss_index_path)
        self._index = None
        self._metadata: List[dict] = []
        self._embed_model = None

    # ── Lazy load ─────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._index is not None:
            return

        import json
        from pathlib import Path
        import faiss

        index_dir = Path(self._index_dir)
        index_file = index_dir / "index.faiss"
        meta_file = index_dir / "metadata.json"

        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_file}. "
                "Build it with: python scripts/build_index.py"
            )

        logger.info("Loading FAISS index from %s", index_file)
        self._index = faiss.read_index(str(index_file))

        logger.info("Loading metadata from %s", meta_file)
        with open(meta_file, encoding="utf-8") as f:
            self._metadata = json.load(f)

        logger.info(
            "FAISS index ready: %d vectors, %d metadata entries.",
            self._index.ntotal,
            len(self._metadata),
        )

    def _get_embed_model(self):
        if self._embed_model is None:
            from app.embeddings.embedding_model import get_embedding_model
            self._embed_model = get_embedding_model()
        return self._embed_model

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int) -> List[VectorSearchResult]:
        """
        Embed the query and return the top-k most similar chunks.

        Returns
        -------
        list[VectorSearchResult] sorted by score descending.
        """
        self._load()
        embed_model = self._get_embed_model()
        q_vec = embed_model.encode_text(query)
        q_arr = np.array([q_vec], dtype="float32")

        distances, indices = self._index.search(q_arr, top_k)

        results: List[VectorSearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            results.append(
                VectorSearchResult(
                    chunk_id=meta.get("chunk_id", f"idx_{idx}"),
                    text=meta.get("text", ""),
                    score=float(dist),
                    metadata=meta,
                )
            )

        return sorted(results, key=lambda r: r.score, reverse=True)

    @property
    def total_vectors(self) -> int:
        self._load()
        return self._index.ntotal if self._index else 0

    @property
    def metadata(self) -> List[dict]:
        """All metadata records (loaded lazily)."""
        self._load()
        return self._metadata
