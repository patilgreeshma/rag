"""
regulatory_models.py – Generic Regulatory Document Model.

Defines ``RegulatoryDocument`` and ``RegulatoryChunk`` dataclasses that are
source-agnostic.  Every field is Optional so that no value is fabricated when
a source does not provide it.

Designed for:
    * SEBI (Phase 1)
    * RBI  (Phase 2)
    * SEC, FCA, etc. (future)

Design principles
─────────────────
* All fields Optional – never invent values.
* Source URL preserved on every object.
* Regulator field is always set – used for multi-source retrieval later.
* sections list allows structure-aware chunking.
* metadata dict is open-ended for regulator-specific extras.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


# ── Document section ──────────────────────────────────────────────────────────

@dataclass
class DocumentSection:
    """
    One logical section of a regulatory document.

    ``heading``    – the section title / heading text (may be empty string)
    ``number``     – section number as found in source (e.g. "3.1", "Schedule II")
    ``text``       – full text content of the section
    ``page_start`` – PDF page where this section begins (1-indexed, if available)
    ``level``      – heading depth: 1 = top-level, 2 = sub-section, etc.
    """
    heading: str = ""
    number: str | None = None
    text: str = ""
    page_start: int | None = None
    level: int = 1
    subsections: list[DocumentSection] = field(default_factory=list)


# ── Regulatory document ───────────────────────────────────────────────────────

@dataclass
class RegulatoryDocument:
    """
    Normalised representation of a single regulatory document.

    Works for circulars, regulations, master circulars, guidelines, etc.,
    from any financial regulator.

    Source traceability is a first-class concern: every document carries its
    ``source_url`` and ``regulator`` through to chunks so that citations work.
    """

    # ── Identification ────────────────────────────────────────────────────────
    document_id: str                          # Deterministic, stable identifier
    regulator: str                            # "SEBI" | "RBI" | "SEC" | …
    document_type: str                        # "circular" | "regulation" | "master_circular" | …
    title: str | None = None
    document_number: str | None = None        # Circular/reg number as on letterhead
    subject: str | None = None

    # ── Dates ─────────────────────────────────────────────────────────────────
    publication_date: date | None = None
    effective_date: date | None = None
    amendment_date: date | None = None

    # ── Source traceability ───────────────────────────────────────────────────
    source_url: str | None = None             # Official URL – NEVER fabricated
    local_path: str | None = None             # Path on disk to raw file

    # ── Language ──────────────────────────────────────────────────────────────
    language: str = "en"

    # ── Extraction status ─────────────────────────────────────────────────────
    extraction_status: str = "success"        # "success" | "failed" | "partial"
    extraction_error: str | None = None

    # ── Content ───────────────────────────────────────────────────────────────
    text: str | None = None                   # Full normalised text
    sections: list[DocumentSection] = field(default_factory=list)

    # ── Open metadata ─────────────────────────────────────────────────────────
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Audit ─────────────────────────────────────────────────────────────────
    retrieved_at: str | None = None           # ISO-8601 timestamp of acquisition


# ── Regulatory chunk ─────────────────────────────────────────────────────────

@dataclass
class RegulatoryChunk:
    """
    A single chunk derived from a ``RegulatoryDocument``.

    Every field needed for retrieval and citation is stored directly on the
    chunk so that no join to the parent document is required at query time.
    """

    # ── Identification ────────────────────────────────────────────────────────
    chunk_id: str                             # Deterministic, stable identifier
    document_id: str                          # Parent document ID

    # ── Source traceability (denormalised for fast retrieval) ─────────────────
    regulator: str
    document_type: str
    title: str | None = None
    document_number: str | None = None
    source_url: str | None = None

    # ── Location within document ──────────────────────────────────────────────
    section: str | None = None               # Section heading
    section_number: str | None = None        # Section number
    subsection: str | None = None            # Subsection heading if applicable
    page_number: int | None = None           # PDF page (1-indexed)
    chunk_index: int = 0                     # 0-based position within section
    chunk_type: str = "section"              # "section" | "paragraph" | "fallback"

    # ── Dates ─────────────────────────────────────────────────────────────────
    publication_date: date | None = None
    effective_date: date | None = None

    # ── Content ───────────────────────────────────────────────────────────────
    text: str = ""

    # ── Open metadata ─────────────────────────────────────────────────────────
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON/JSONL output."""
        from dataclasses import asdict
        d = asdict(self)
        # Convert date objects to ISO strings
        for date_field in ("publication_date", "effective_date"):
            if d[date_field] is not None:
                d[date_field] = d[date_field].isoformat() if hasattr(d[date_field], "isoformat") else str(d[date_field])
        return d
