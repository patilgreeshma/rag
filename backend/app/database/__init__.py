"""
app/database – FAISS-based storage layer for FinRule AI.

Storage backend: FAISS flat-file index (IndexFlatIP) + JSON metadata.
No SQL database required.

Public exports
──────────────
get_index_manager()   – returns the singleton FAISSIndexManager
FAISSIndexManager     – manages index lifecycle (load, health-check)
ChunkRecord           – canonical chunk dataclass (serialise ↔ deserialise)
IndexStats            – lightweight health-check snapshot
"""
from app.database.connection import FAISSIndexManager, get_index_manager
from app.database.models import ChunkRecord, IndexStats

__all__ = [
    "FAISSIndexManager",
    "get_index_manager",
    "ChunkRecord",
    "IndexStats",
]
