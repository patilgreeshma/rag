"""
open_india_law_loader.py – Open India Law SEBI Dataset Loader.

Loads ``in_sebi_regulations.parquet`` from the Open India Law corpus
(https://github.com/Vaquill-AI/open-india-law) and converts each row
into a ``RegulatoryChunk`` ready for embedding and retrieval.

Design decisions
────────────────
* The dataset is already provision-level chunked by Open India Law.
  We do NOT re-chunk. Each Parquet row = one retrieval chunk.
* All metadata is taken directly from the real Parquet columns.
  Nothing is fabricated.
* source_url is preserved on every chunk (100% coverage in the dataset).
* The loader validates required columns before processing any data.
* Duplicate chunk_ids are detected and skipped.

Real schema (from in_sebi_regulations.parquet v2026.08.1)
──────────────────────────────────────────────────────────
  act_id            – instrument identifier
  chunk_id          – unique provision identifier
  title             – instrument title
  chapter           – chapter number/name
  section_number    – provision number
  section_title     – provision heading
  text              – full provision text (with contextual header)
  jurisdiction      – always "central" for SEBI
  instrument_type   – always "regulatory"
  act_status        – "in_force" | "superseded" | …
  in_force          – bool
  doc_type          – always "legislation"
  provision_type    – "overriding" | "delegatory" | …
  section_type      – "section" | "sub_section" | "definitions" | …
  legal_subject     – list of subject tags
  has_proviso       – bool
  has_non_obstante  – bool
  delegation_type   – str | None
  department        – always "sebi"
  acts_referenced   – list of referenced act names
  defined_terms     – list of defined terms
  amendment_count   – int
  year              – float (year of instrument)
  source_url        – official SEBI URL
  language_code     – "en"
  source_publisher  – "Securities and Exchange Board of India"
  mirror_url        – PDF mirror URL
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Required columns (will raise clearly if missing) ──────────────────────────
REQUIRED_COLUMNS = {
    "act_id",
    "chunk_id",
    "title",
    "text",
    "source_url",
}

# ── All columns we use (others are ignored gracefully) ────────────────────────
USED_COLUMNS = [
    "act_id", "chunk_id", "title", "chapter", "section_number",
    "section_title", "text", "instrument_type", "act_status", "in_force",
    "provision_type", "section_type", "legal_subject", "has_proviso",
    "has_non_obstante", "acts_referenced", "defined_terms",
    "amendment_count", "year", "source_url", "source_publisher",
    "mirror_url", "department",
]


# ── Schema validation ─────────────────────────────────────────────────────────

def validate_schema(df) -> None:
    """
    Raise ``SchemaValidationError`` if any required column is absent.

    Parameters
    ----------
    df: pandas.DataFrame
        The loaded Parquet DataFrame.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"Open India Law SEBI Parquet is missing required columns: {sorted(missing)}\n"
            f"Available columns: {sorted(df.columns.tolist())}"
        )
    logger.info(
        "Schema validation passed. %d columns present, %d rows.",
        len(df.columns),
        len(df),
    )


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_parquet(path: Path | str) -> "pandas.DataFrame":  # type: ignore[name-defined]
    """
    Load the SEBI regulations Parquet file and return a DataFrame.

    Parameters
    ----------
    path:
        Path to ``in_sebi_regulations.parquet``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    SchemaValidationError
        If required columns are missing.
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {path}\n"
            "Download it with:\n"
            "  python scripts/ingest_open_india_law.py --download-only"
        )

    logger.info("Loading Parquet: %s", path)
    df = pd.read_parquet(path, engine="pyarrow")
    validate_schema(df)
    logger.info("Loaded %d rows, %d columns.", len(df), len(df.columns))
    return df


# ── Row → RegulatoryChunk conversion ─────────────────────────────────────────

def _safe_str(val: Any, default: str | None = None) -> str | None:
    """Return a non-empty string from val, or default."""
    if val is None:
        return default
    s = str(val).strip()
    return s if s and s.lower() not in ("nan", "none", "null") else default


def _safe_list(val: Any) -> list[str]:
    """Return a list of strings from array-like val."""
    if val is None:
        return []
    try:
        import numpy as np
        if isinstance(val, np.ndarray):
            val = val.tolist()
    except ImportError:
        pass
    if isinstance(val, (list, tuple)):
        return [str(v) for v in val if v is not None]
    return []


def _safe_bool(val: Any) -> bool:
    """Return bool from val, defaulting to False."""
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    try:
        return bool(val)
    except Exception:
        return False


def row_to_chunk(row: dict[str, Any]) -> "RegulatoryChunk":  # type: ignore[name-defined]
    """
    Convert one Parquet row (as a dict) into a ``RegulatoryChunk``.

    No fields are fabricated. If a source value is absent, the
    corresponding chunk field is left as ``None`` or its zero-value.

    Parameters
    ----------
    row:
        A single row from the DataFrame as a plain dict.

    Returns
    -------
    RegulatoryChunk
    """
    from app.ingestion.regulatory_models import RegulatoryChunk

    # ── Core identifiers ──────────────────────────────────────────────────────
    chunk_id = _safe_str(row.get("chunk_id")) or _generate_fallback_id(row)
    act_id   = _safe_str(row.get("act_id"))

    # ── Text ──────────────────────────────────────────────────────────────────
    text = _safe_str(row.get("text"), default="") or ""

    # ── Source traceability ───────────────────────────────────────────────────
    source_url  = _safe_str(row.get("source_url"))
    mirror_url  = _safe_str(row.get("mirror_url"))

    # ── Document identity ─────────────────────────────────────────────────────
    title       = _safe_str(row.get("title"))
    section_num = _safe_str(row.get("section_number"))
    section_ttl = _safe_str(row.get("section_title"))
    chapter     = _safe_str(row.get("chapter"))
    doc_type    = _safe_str(row.get("instrument_type"), default="regulatory")

    # ── Status ────────────────────────────────────────────────────────────────
    in_force    = _safe_bool(row.get("in_force"))
    act_status  = _safe_str(row.get("act_status"))

    # ── Classification metadata ───────────────────────────────────────────────
    section_type   = _safe_str(row.get("section_type"))
    provision_type = _safe_str(row.get("provision_type"))
    year_val       = row.get("year")
    year: int | None = int(year_val) if year_val is not None and str(year_val) not in ("nan", "None") else None

    # ── Build rich metadata dict (everything useful for downstream) ───────────
    metadata: dict[str, Any] = {
        "act_id":          act_id,
        "chapter":         chapter,
        "section_type":    section_type,
        "provision_type":  provision_type,
        "in_force":        in_force,
        "act_status":      act_status,
        "year":            year,
        "mirror_url":      mirror_url,
        "source_publisher": _safe_str(row.get("source_publisher")),
        "legal_subject":   _safe_list(row.get("legal_subject")),
        "has_proviso":     _safe_bool(row.get("has_proviso")),
        "has_non_obstante": _safe_bool(row.get("has_non_obstante")),
        "acts_referenced": _safe_list(row.get("acts_referenced")),
        "defined_terms":   _safe_list(row.get("defined_terms")),
        "amendment_count": int(row.get("amendment_count") or 0),
        "department":      _safe_str(row.get("department")),
    }

    return RegulatoryChunk(
        chunk_id=chunk_id,
        document_id=act_id or chunk_id,
        regulator="SEBI",
        document_type=doc_type,
        title=title,
        document_number=act_id,
        source_url=source_url,
        section=section_ttl,
        section_number=section_num,
        chunk_index=0,          # each row is already one provision
        chunk_type=section_type or "section",
        text=text,
        metadata=metadata,
    )


def _generate_fallback_id(row: dict[str, Any]) -> str:
    """Generate a stable fallback chunk_id from act_id + section_number."""
    key = f"{row.get('act_id', '')}_{row.get('section_number', '')}_{row.get('text', '')[:64]}"
    return "chunk_" + hashlib.sha256(key.encode()).hexdigest()[:16]


# ── Bulk conversion ───────────────────────────────────────────────────────────

def dataframe_to_chunks(df) -> list["RegulatoryChunk"]:  # type: ignore[name-defined]
    """
    Convert a full DataFrame into a list of ``RegulatoryChunk`` objects.

    Duplicate ``chunk_id`` values are detected and skipped (second
    occurrence wins, first is kept).

    Parameters
    ----------
    df:
        pandas.DataFrame from ``load_parquet``.

    Returns
    -------
    list[RegulatoryChunk]
    """
    chunks: list = []
    seen_ids: set[str] = set()
    skipped_dupes = 0
    errors = 0

    for i, (_, row_series) in enumerate(df.iterrows()):
        row = row_series.to_dict()
        try:
            chunk = row_to_chunk(row)
            if chunk.chunk_id in seen_ids:
                skipped_dupes += 1
                continue
            seen_ids.add(chunk.chunk_id)
            chunks.append(chunk)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            logger.warning("Row %d skipped due to error: %s", i, exc)

    logger.info(
        "Converted %d rows → %d chunks  (dupes skipped: %d, errors: %d)",
        len(df),
        len(chunks),
        skipped_dupes,
        errors,
    )
    return chunks


# ── Dataset statistics ────────────────────────────────────────────────────────

def dataset_stats(df) -> dict[str, Any]:
    """
    Compute and return summary statistics for the loaded DataFrame.

    Parameters
    ----------
    df:
        pandas.DataFrame from ``load_parquet``.

    Returns
    -------
    dict with summary statistics.
    """
    total_rows = len(df)
    unique_instruments = df["act_id"].nunique() if "act_id" in df.columns else 0
    in_force_count = int(df["in_force"].sum()) if "in_force" in df.columns else 0
    url_coverage = int(df["source_url"].notna().sum()) if "source_url" in df.columns else 0

    section_types = {}
    if "section_type" in df.columns:
        section_types = df["section_type"].value_counts().to_dict()

    instrument_types = {}
    if "instrument_type" in df.columns:
        instrument_types = df["instrument_type"].value_counts().to_dict()

    text_lengths = {}
    if "text" in df.columns:
        lengths = df["text"].str.len()
        text_lengths = {
            "mean": round(float(lengths.mean()), 1),
            "median": round(float(lengths.median()), 1),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
        }

    return {
        "total_provisions": total_rows,
        "unique_instruments": unique_instruments,
        "in_force_provisions": in_force_count,
        "superseded_provisions": total_rows - in_force_count,
        "source_url_coverage": url_coverage,
        "source_url_coverage_pct": round(url_coverage / total_rows * 100, 1) if total_rows else 0,
        "section_type_breakdown": section_types,
        "instrument_type_breakdown": instrument_types,
        "text_length_chars": text_lengths,
        "columns": sorted(df.columns.tolist()),
    }


# ── Custom exceptions ─────────────────────────────────────────────────────────

class SchemaValidationError(ValueError):
    """Raised when the Parquet file is missing required columns."""
