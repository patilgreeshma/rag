"""
app/database/connection.py – FinRule AI FAISS Index Manager.

Manages the lifecycle of the FAISS flat-file index and its companion
metadata.json. This is the "database layer" for FinRule AI — instead of
a relational database, all chunk storage lives in two files on disk:

    data/processed/sebi/faiss_index/index.faiss    – FAISS IndexFlatIP
    data/processed/sebi/faiss_index/metadata.json  – parallel chunk metadata

Design
──────
* FAISSIndexManager is a singleton that loads the index once and keeps it
  in memory for the lifetime of the process.
* All retrieval components (VectorSearcher, BM25Searcher) read from the
  same loaded metadata to avoid redundant I/O.
* The manager exposes health-check helpers used by GET /health.

Usage
─────
    from app.database.connection import get_index_manager
    mgr = get_index_manager()
    print(mgr.total_vectors)   # e.g. 88310
    print(mgr.is_ready)        # True once index is loaded
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import faiss as _faiss_t

logger = logging.getLogger(__name__)


class FAISSIndexManager:
    """
    Singleton manager for the FAISS index and chunk metadata.

    The index is loaded lazily on first access to avoid slowing down
    application start-up. All sub-components that need the index should
    obtain it via ``get_index_manager()`` rather than loading it themselves.
    """

    def __init__(self, index_dir: str | Path | None = None) -> None:
        from app.config import settings

        self._index_dir = Path(index_dir or settings.faiss_index_path)
        self._index: "_faiss_t.Index | None" = None
        self._metadata: list[dict[str, Any]] = []
        self._loaded = False

    # ── Private ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load the FAISS index and metadata from disk (called once)."""
        if self._loaded:
            return

        import faiss

        index_file = self._index_dir / "index.faiss"
        meta_file = self._index_dir / "metadata.json"

        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_file}.\n"
                "Build it first:\n"
                "  python scripts/ingest_open_india_law.py\n"
                "  python scripts/build_index.py"
            )

        logger.info("Loading FAISS index from %s …", index_file)
        self._index = faiss.read_index(str(index_file))

        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                self._metadata = json.load(f)
        else:
            logger.warning("metadata.json not found at %s — metadata will be empty.", meta_file)
            self._metadata = []

        self._loaded = True
        logger.info(
            "FAISS index ready: %d vectors | %d metadata entries | dim=%d",
            self._index.ntotal,
            len(self._metadata),
            self._index.d,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def index(self) -> "_faiss_t.Index":
        """The loaded FAISS index. Triggers load on first access."""
        self._load()
        assert self._index is not None
        return self._index

    @property
    def metadata(self) -> list[dict[str, Any]]:
        """Parallel metadata list. Triggers load on first access."""
        self._load()
        return self._metadata

    @property
    def total_vectors(self) -> int:
        """Number of vectors in the index (0 if not yet loaded)."""
        if not self._loaded:
            try:
                self._load()
            except FileNotFoundError:
                return 0
        return self._index.ntotal if self._index else 0

    @property
    def embedding_dim(self) -> int:
        """Vector dimension of the loaded index."""
        self._load()
        return self._index.d if self._index else 0

    @property
    def is_ready(self) -> bool:
        """True if the index has been loaded successfully."""
        return self._loaded and self._index is not None

    def health(self) -> dict[str, Any]:
        """
        Return a dict suitable for the GET /health response.

        Safe to call even if the index is not yet built — returns
        status='unavailable' instead of raising.
        """
        try:
            return {
                "status": "ready",
                "index_dir": str(self._index_dir),
                "total_vectors": self.total_vectors,
                "embedding_dim": self.embedding_dim,
                "metadata_entries": len(self._metadata),
            }
        except FileNotFoundError as exc:
            return {
                "status": "unavailable",
                "error": str(exc),
                "index_dir": str(self._index_dir),
                "total_vectors": 0,
            }


# ── Module-level singleton ────────────────────────────────────────────────────

_manager: FAISSIndexManager | None = None


def get_index_manager(index_dir: str | Path | None = None) -> FAISSIndexManager:
    """
    Return the application-wide FAISSIndexManager singleton.

    Parameters
    ----------
    index_dir:
        Override the default index directory (useful in tests).
        If None, uses ``settings.faiss_index_path``.
    """
    global _manager
    if _manager is None or index_dir is not None:
        _manager = FAISSIndexManager(index_dir=index_dir)
    return _manager
