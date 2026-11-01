"""
app/database/models.py – FinRule AI data models for FAISS-based storage.

FinRule AI uses FAISS (flat-file vector store) rather than a relational
database. This module defines the canonical Python dataclasses that represent
the two persisted artefacts on disk:

  ┌─────────────────────────────────────────────────────────────────┐
  │  chunks.jsonl          one line per provision chunk             │
  │  faiss_index/          FAISS index + parallel metadata.json     │
  └─────────────────────────────────────────────────────────────────┘

These models are used by:
  * scripts/ingest_open_india_law.py  – writes chunks.jsonl
  * scripts/build_index.py            – reads chunks.jsonl → FAISS
  * app/retrieval/vector_search.py    – reads FAISS + metadata.json
  * app/retrieval/bm25_search.py      – reads metadata.json
  * app/database/connection.py        – manages the index lifecycle

No ORM, no migrations, no SQL — the "schema" is defined here as dataclasses
with to_dict() / from_dict() helpers so serialisation stays consistent.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# ── Chunk record ──────────────────────────────────────────────────────────────

@dataclass
class ChunkRecord:
    """
    Canonical representation of one SEBI provision chunk as stored in
    chunks.jsonl and FAISS metadata.json.

    Every field that ends up in metadata.json is also available at retrieval
    time, so retrieval results can be fully hydrated without hitting any
    external store.

    Field contract
    ──────────────
    chunk_id        – globally unique identifier (from Open India Law)
    document_id     – the act / instrument this chunk belongs to
    regulator       – always "SEBI" for this corpus
    title           – instrument title (e.g. "SEBI LODR Regulations 2015")
    section_number  – provision number (e.g. "33", "Schedule II - Clause 4")
    section         – provision heading / section title
    chunk_type      – "section" | "sub_section" | "definitions" | …
    text            – full provision text (the field that gets embedded)
    source_url      – official SEBI URL (100 % coverage in this corpus)
    in_force        – True if the provision is currently in force
    """

    # Core identifiers
    chunk_id: str
    document_id: str
    regulator: str = "SEBI"

    # Document identity
    title: str | None = None
    document_number: str | None = None
    document_type: str | None = None

    # Provision location
    section_number: str | None = None
    section: str | None = None
    chunk_index: int = 0
    chunk_type: str = "section"

    # Content
    text: str = ""

    # Source traceability
    source_url: str | None = None
    in_force: bool = True

    # Open-ended metadata (year, amendment_count, legal_subject, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """
        Flatten to a dict suitable for a single JSONL line or metadata.json entry.

        The metadata dict is spread into the top level so that all fields are
        directly accessible in retrieval results without an extra dict lookup.
        """
        base = asdict(self)
        meta = base.pop("metadata", {}) or {}
        base.update(meta)       # flatten metadata into top level
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkRecord":
        """
        Reconstruct a ChunkRecord from a flat dict (e.g., from metadata.json).

        Known top-level fields are extracted; anything else is placed into
        the metadata dict.
        """
        known = {
            "chunk_id", "document_id", "regulator", "title", "document_number",
            "document_type", "section_number", "section", "chunk_index",
            "chunk_type", "text", "source_url", "in_force",
        }
        base: dict[str, Any] = {}
        extra: dict[str, Any] = {}

        for k, v in data.items():
            if k in known:
                base[k] = v
            else:
                extra[k] = v

        base["metadata"] = extra
        # Provide safe defaults for required fields
        base.setdefault("chunk_id", "")
        base.setdefault("document_id", "")
        return cls(**base)

    @classmethod
    def from_regulatory_chunk(cls, chunk: Any) -> "ChunkRecord":
        """
        Convert a RegulatoryChunk (from app.ingestion.regulatory_models)
        into a ChunkRecord ready for serialisation.
        """
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            regulator=chunk.regulator or "SEBI",
            title=chunk.title,
            document_number=chunk.document_number,
            document_type=chunk.document_type,
            section_number=chunk.section_number,
            section=chunk.section,
            chunk_index=chunk.chunk_index,
            chunk_type=chunk.chunk_type or "section",
            text=chunk.text,
            source_url=chunk.source_url,
            in_force=bool((chunk.metadata or {}).get("in_force", True)),
            metadata=chunk.metadata or {},
        )


# ── Index stats record ────────────────────────────────────────────────────────

@dataclass
class IndexStats:
    """
    Lightweight snapshot of the current FAISS index state.
    Returned by FAISSIndexManager.health() and exposed on GET /health.
    """
    status: str                    # "ready" | "unavailable"
    total_vectors: int = 0
    embedding_dim: int = 0
    metadata_entries: int = 0
    index_dir: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}
