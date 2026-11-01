"""
app/ingestion/ingest.py – FinRule AI Full Ingestion Pipeline Orchestrator.

End-to-end pipeline that:
  1. Downloads the Open India Law SEBI Parquet (if missing).
  2. Loads and schema-validates the Parquet file.
  3. Converts each row → RegulatoryChunk (via open_india_law_loader).
  4. Writes documents.jsonl + chunks.jsonl to data/processed/sebi/.
  5. Generates embeddings and builds the FAISS index.

This module can be called from scripts (see scripts/ingest_open_india_law.py
and scripts/build_index.py) or imported directly in code.

Typical usage
─────────────
    from app.ingestion.ingest import run_full_pipeline
    summary = run_full_pipeline()
    print(summary)

Or from the command line via the scripts:
    python scripts/ingest_open_india_law.py   # steps 1-4
    python scripts/build_index.py             # step 5
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Pipeline summary dataclass ────────────────────────────────────────────────

@dataclass
class IngestionSummary:
    """Structured result returned by run_full_pipeline()."""

    # Counts
    total_rows: int = 0
    chunks_written: int = 0
    documents_written: int = 0
    duplicates_skipped: int = 0
    errors: int = 0

    # FAISS
    faiss_vectors: int = 0
    faiss_built: bool = False

    # Paths
    chunks_file: str = ""
    documents_file: str = ""
    faiss_index_dir: str = ""

    # Timing
    elapsed_seconds: float = 0.0

    # Per-step detail
    step_times: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "",
            "═" * 60,
            "  FinRule AI — Ingestion Summary",
            "═" * 60,
            f"  Rows in Parquet        : {self.total_rows:,}",
            f"  Chunks written         : {self.chunks_written:,}",
            f"  Documents written      : {self.documents_written:,}",
            f"  Duplicates skipped     : {self.duplicates_skipped:,}",
            f"  Errors                 : {self.errors}",
            "",
            f"  FAISS index built      : {'yes' if self.faiss_built else 'no'}",
            f"  FAISS vectors          : {self.faiss_vectors:,}",
            "",
            f"  Chunks JSONL           : {self.chunks_file}",
            f"  FAISS index dir        : {self.faiss_index_dir}",
            "",
            f"  Total elapsed          : {self.elapsed_seconds:.1f}s",
            "═" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Step helpers ──────────────────────────────────────────────────────────────

def _download_parquet_if_missing(parquet_path: Path, url: str) -> None:
    """Download the SEBI Parquet file if it does not exist locally."""
    if parquet_path.exists():
        logger.info("Parquet already present: %s", parquet_path)
        return

    logger.info("Downloading SEBI Parquet from %s …", url)
    try:
        import urllib.request
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, parquet_path)
        size_mb = parquet_path.stat().st_size / 1_048_576
        logger.info("Downloaded %.1f MB → %s", size_mb, parquet_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download SEBI Parquet from {url}: {exc}\n"
            "Check your internet connection or set OIL_PARQUET_URL in .env."
        ) from exc


def _write_chunks_jsonl(chunks: list, output_path: Path) -> int:
    """
    Write RegulatoryChunk objects to JSONL.

    Each line is a JSON object containing all chunk fields plus the
    full metadata dict.  Returns number of lines written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record: dict[str, Any] = {
                "chunk_id":       chunk.chunk_id,
                "document_id":    chunk.document_id,
                "regulator":      chunk.regulator,
                "document_type":  chunk.document_type,
                "title":          chunk.title,
                "document_number": chunk.document_number,
                "source_url":     chunk.source_url,
                "section":        chunk.section,
                "section_number": chunk.section_number,
                "chunk_index":    chunk.chunk_index,
                "chunk_type":     chunk.chunk_type,
                "text":           chunk.text,
            }
            # Flatten metadata into the top-level record for easy FAISS retrieval
            record.update(chunk.metadata or {})
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            written += 1

    logger.info("Wrote %d chunks → %s", written, output_path)
    return written


def _write_documents_jsonl(chunks: list, output_path: Path) -> int:
    """
    Aggregate chunks by act_id and write one record per instrument.

    Returns number of unique documents written.
    """
    from collections import defaultdict

    by_act: dict[str, dict[str, Any]] = defaultdict(dict)
    for chunk in chunks:
        act_id = chunk.document_id or chunk.chunk_id
        if act_id not in by_act:
            by_act[act_id] = {
                "document_id":   act_id,
                "title":         chunk.title,
                "regulator":     chunk.regulator,
                "document_type": chunk.document_type,
                "source_url":    chunk.source_url,
                "in_force":      chunk.metadata.get("in_force"),
                "year":          chunk.metadata.get("year"),
                "department":    chunk.metadata.get("department"),
                "provision_count": 0,
            }
        by_act[act_id]["provision_count"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in by_act.values():
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    written = len(by_act)
    logger.info("Wrote %d documents → %s", written, output_path)
    return written


def _build_faiss_index(chunks_file: Path, index_dir: Path, batch_size: int = 64) -> int:
    """
    Build and save a FAISS IndexFlatIP from chunks.jsonl.

    Returns the number of vectors added to the index.
    """
    import numpy as np

    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is required to build the FAISS index. "
            "Install it with: pip install faiss-cpu"
        ) from exc

    from app.embeddings.embedding_model import get_embedding_model

    # Load texts + metadata from JSONL
    texts: list[str] = []
    metadata: list[dict] = []
    with open(chunks_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = (record.get("text") or "").strip()
            if text:
                texts.append(text)
                metadata.append(record)

    if not texts:
        raise ValueError(f"No non-empty texts found in {chunks_file}")

    logger.info("Generating embeddings for %d chunks …", len(texts))
    embed_model = get_embedding_model()
    embed_model.load()

    vectors = embed_model.encode_batch(texts, batch_size=batch_size, show_progress=True)
    vecs_np = np.ascontiguousarray(np.array(vectors, dtype="float32"))

    dim = vecs_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs_np)
    logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)

    # Save
    index_dir.mkdir(parents=True, exist_ok=True)
    index_file = index_dir / "index.faiss"
    meta_file = index_dir / "metadata.json"

    faiss.write_index(index, str(index_file))
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, default=str)

    logger.info(
        "FAISS index saved: %s (%.1f MB)",
        index_file,
        index_file.stat().st_size / 1_048_576,
    )
    return index.ntotal


# ── Public API ────────────────────────────────────────────────────────────────

def run_full_pipeline(
    *,
    download: bool = True,
    build_index: bool = True,
    limit: int | None = None,
    in_force_only: bool = False,
    embedding_batch_size: int | None = None,
) -> IngestionSummary:
    """
    Run the complete FinRule AI ingestion pipeline.

    Parameters
    ----------
    download:
        Download the SEBI Parquet if it is not already present locally.
    build_index:
        Build the FAISS index after writing chunks.jsonl.
    limit:
        Process only the first N provisions (for development/testing).
    in_force_only:
        Filter to provisions where ``in_force == True``.
    embedding_batch_size:
        Override the batch size for embedding generation.

    Returns
    -------
    IngestionSummary with counts, paths, and timing information.
    """
    from app.config import settings
    from app.ingestion.open_india_law_loader import (
        load_parquet,
        dataframe_to_chunks,
        SchemaValidationError,
    )

    summary = IngestionSummary()
    t_start = time.perf_counter()

    parquet_path = settings.oil_parquet_path
    chunks_file = settings.sebi_processed_dir / "chunks.jsonl"
    docs_file = settings.sebi_processed_dir / "documents.jsonl"
    faiss_dir = settings.faiss_index_path
    batch_size = embedding_batch_size or settings.embedding_batch_size

    summary.chunks_file = str(chunks_file)
    summary.documents_file = str(docs_file)
    summary.faiss_index_dir = str(faiss_dir)

    # ── Step 1: Download ──────────────────────────────────────────────────────
    if download:
        t0 = time.perf_counter()
        _download_parquet_if_missing(parquet_path, settings.oil_parquet_url)
        summary.step_times["download"] = round(time.perf_counter() - t0, 2)

    # ── Step 2: Load + validate ───────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        df = load_parquet(parquet_path)
    except (FileNotFoundError, SchemaValidationError) as exc:
        logger.error("Ingestion aborted: %s", exc)
        raise

    summary.total_rows = len(df)

    if in_force_only and "in_force" in df.columns:
        before = len(df)
        df = df[df["in_force"] == True]  # noqa: E712
        logger.info("Filtered to in-force only: %d → %d rows", before, len(df))

    if limit is not None:
        df = df.iloc[:limit]
        logger.info("Limited to %d rows.", len(df))

    summary.step_times["load"] = round(time.perf_counter() - t0, 2)

    # ── Step 3: Convert rows → RegulatoryChunks ───────────────────────────────
    t0 = time.perf_counter()
    chunks = dataframe_to_chunks(df)
    summary.chunks_written = len(chunks)
    # duplicates_skipped is logged inside dataframe_to_chunks but not returned;
    # compute it from the row count difference.
    summary.duplicates_skipped = max(0, len(df) - len(chunks))
    summary.step_times["convert"] = round(time.perf_counter() - t0, 2)

    # ── Step 4: Write JSONL ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    _write_chunks_jsonl(chunks, chunks_file)
    summary.documents_written = _write_documents_jsonl(chunks, docs_file)
    summary.step_times["write_jsonl"] = round(time.perf_counter() - t0, 2)

    # ── Step 5: Build FAISS index ─────────────────────────────────────────────
    if build_index:
        t0 = time.perf_counter()
        try:
            summary.faiss_vectors = _build_faiss_index(
                chunks_file, faiss_dir, batch_size=batch_size
            )
            summary.faiss_built = True
        except Exception as exc:  # noqa: BLE001
            logger.error("FAISS index build failed: %s", exc)
            summary.errors += 1
        summary.step_times["build_index"] = round(time.perf_counter() - t0, 2)

    summary.elapsed_seconds = round(time.perf_counter() - t_start, 2)
    logger.info(summary)
    return summary
