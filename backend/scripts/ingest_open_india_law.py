#!/usr/bin/env python3
"""
scripts/ingest_open_india_law.py – FinRule AI Ingestion Pipeline

Loads the Open India Law SEBI Parquet dataset, validates its schema,
converts provisions into RegulatoryChunk objects, normalises text,
and writes output JSONL files.

Outputs
───────
  data/processed/sebi/documents.jsonl  – one line per instrument (act)
  data/processed/sebi/chunks.jsonl     – one line per provision chunk

Usage
─────
# Download parquet (if not already present) and run full ingestion:
  python scripts/ingest_open_india_law.py

# Download only:
  python scripts/ingest_open_india_law.py --download-only

# Use existing parquet (skip download check):
  python scripts/ingest_open_india_law.py --skip-download

# Limit to first N provisions (for development):
  python scripts/ingest_open_india_law.py --limit 1000

# Only in-force provisions:
  python scripts/ingest_open_india_law.py --in-force-only

# Verbose output:
  python scripts/ingest_open_india_law.py --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings
from app.ingestion.open_india_law_loader import (
    load_parquet,
    dataframe_to_chunks,
    dataset_stats,
    SchemaValidationError,
)
from app.ingestion.regulatory_normalizer import RegulatoryNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest_open_india_law")


# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest Open India Law SEBI dataset into FinRule AI."
    )
    p.add_argument(
        "--download-only", action="store_true",
        help="Download the Parquet file and exit (no processing).",
    )
    p.add_argument(
        "--skip-download", action="store_true",
        help="Skip download check; use existing Parquet file.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N provisions (development mode).",
    )
    p.add_argument(
        "--in-force-only", action="store_true",
        help="Only include in-force provisions (skip superseded).",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p.parse_args()


# ── Download helper ───────────────────────────────────────────────────────────

def download_parquet(dest: Path, url: str) -> None:
    """Download the Parquet file from the public OSS URL."""
    import httpx

    logger.info("Downloading: %s → %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": "FinRuleAI/1.0",
        "Referer": "https://oss-data-in.vaquill.ai/index.html",
    }
    t0 = time.time()
    with httpx.stream("GET", url, headers=headers, timeout=300, follow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded/1024/1024:.1f}/{total/1024/1024:.1f} MB ({pct:.0f}%)", end="", flush=True)
    print()
    elapsed = time.time() - t0
    size = dest.stat().st_size
    logger.info("Downloaded %.1f MB in %.1fs → %s", size / 1024 / 1024, elapsed, dest)


# ── JSONL writers ─────────────────────────────────────────────────────────────

def _write_chunks_jsonl(chunks, path: Path) -> int:
    """Write chunks to JSONL, appending only new chunk_ids."""
    existing_ids: set[str] = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["chunk_id"])
                except (KeyError, json.JSONDecodeError):
                    pass

    new_count = 0
    with open(path, "a", encoding="utf-8") as f:
        for chunk in chunks:
            if chunk.chunk_id in existing_ids:
                continue
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False, default=str) + "\n")
            new_count += 1

    return new_count


def _write_documents_jsonl(chunks, path: Path) -> int:
    """Write one document record per unique act_id."""
    existing_doc_ids: set[str] = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    existing_doc_ids.add(json.loads(line)["document_id"])
                except (KeyError, json.JSONDecodeError):
                    pass

    # Group chunks by document_id (act_id)
    docs: dict[str, dict] = {}
    for chunk in chunks:
        doc_id = chunk.document_id
        if doc_id in existing_doc_ids or doc_id in docs:
            continue
        docs[doc_id] = {
            "document_id":   doc_id,
            "regulator":     "SEBI",
            "document_type": chunk.document_type,
            "title":         chunk.title,
            "source_url":    chunk.source_url,
            "in_force":      chunk.metadata.get("in_force"),
            "act_status":    chunk.metadata.get("act_status"),
            "year":          chunk.metadata.get("year"),
            "source_publisher": chunk.metadata.get("source_publisher"),
            "mirror_url":    chunk.metadata.get("mirror_url"),
        }

    new_count = 0
    with open(path, "a", encoding="utf-8") as f:
        for doc in docs.values():
            f.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")
            new_count += 1

    return new_count


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    t_start = time.perf_counter()
    parquet_path = settings.oil_parquet_path

    print()
    print("═" * 65)
    print("  FinRule AI — Open India Law SEBI Ingestion Pipeline")
    print("═" * 65)
    print(f"  Parquet file : {parquet_path}")
    print(f"  Output dir   : {settings.sebi_processed_dir}")
    print()

    # ── Step 1: Download if needed ────────────────────────────────────────────
    if not args.skip_download:
        if not parquet_path.exists():
            logger.info("Step 1 – Parquet not found locally. Downloading …")
            download_parquet(parquet_path, settings.oil_parquet_url)
        else:
            logger.info("Step 1 – Parquet already present: %s (%.1f MB)",
                        parquet_path, parquet_path.stat().st_size / 1024 / 1024)
    else:
        logger.info("Step 1 – Skipping download (--skip-download).")

    if args.download_only:
        print("  --download-only: done.")
        return

    # ── Step 2: Load and validate ─────────────────────────────────────────────
    logger.info("Step 2 – Loading and validating Parquet …")
    try:
        df = load_parquet(parquet_path)
    except SchemaValidationError as exc:
        logger.error("Schema validation failed: %s", exc)
        sys.exit(1)

    # Print schema + stats
    stats = dataset_stats(df)
    print("  Dataset statistics:")
    print(f"    Total provisions       : {stats['total_provisions']:,}")
    print(f"    Unique instruments     : {stats['unique_instruments']:,}")
    print(f"    In-force provisions    : {stats['in_force_provisions']:,}")
    print(f"    Superseded provisions  : {stats['superseded_provisions']:,}")
    print(f"    source_url coverage    : {stats['source_url_coverage_pct']}%")
    print(f"    Text length (mean)     : {stats['text_length_chars'].get('mean', 'N/A')} chars")
    print(f"    Columns                : {len(stats['columns'])}")
    print()

    # ── Step 3: Filter ────────────────────────────────────────────────────────
    if args.in_force_only:
        before = len(df)
        df = df[df["in_force"] == True].reset_index(drop=True)  # noqa: E712
        logger.info(
            "Step 3 – Filtered to in-force only: %d → %d rows.", before, len(df)
        )
    else:
        logger.info("Step 3 – Including both in-force and superseded provisions.")

    if args.limit:
        df = df.head(args.limit)
        logger.info("Step 3 – Limited to first %d rows (dev mode).", args.limit)

    # ── Step 4: Convert to RegulatoryChunk objects ────────────────────────────
    logger.info("Step 4 – Converting %d rows to RegulatoryChunk objects …", len(df))
    chunks = dataframe_to_chunks(df)
    logger.info("  → %d chunks produced.", len(chunks))

    # ── Step 5: Normalize text ────────────────────────────────────────────────
    logger.info("Step 5 – Normalising chunk text …")
    normalizer = RegulatoryNormalizer()
    # Adapt normalizer for chunks (it works on text strings)
    normalized_count = 0
    for chunk in chunks:
        if chunk.text:
            normalized = normalizer._normalize_text(chunk.text)
            if normalized != chunk.text:
                chunk.text = normalized
                normalized_count += 1
    logger.info("  → %d chunks had text normalised.", normalized_count)

    # ── Step 6: Write JSONL ───────────────────────────────────────────────────
    logger.info("Step 6 – Writing JSONL outputs …")
    settings.sebi_processed_dir.mkdir(parents=True, exist_ok=True)

    docs_path   = settings.sebi_processed_dir / "documents.jsonl"
    chunks_path = settings.sebi_processed_dir / "chunks.jsonl"

    new_docs   = _write_documents_jsonl(chunks, docs_path)
    new_chunks = _write_chunks_jsonl(chunks, chunks_path)

    total_elapsed = time.perf_counter() - t_start

    print()
    print("═" * 65)
    print("  INGESTION COMPLETE")
    print("═" * 65)
    print(f"  Source           : Open India Law (vaquill/open-india-law)")
    print(f"  Dataset          : in_sebi_regulations.parquet")
    print(f"  Regulator        : SEBI")
    print(f"  Provisions loaded: {len(df):,}")
    print(f"  Chunks produced  : {len(chunks):,}")
    print(f"  New documents    : {new_docs:,}  → {docs_path}")
    print(f"  New chunks       : {new_chunks:,}  → {chunks_path}")
    print(f"  Total time       : {total_elapsed:.1f}s")
    print()
    print("  Next step: build the FAISS index")
    print("  Run: python scripts/build_index.py")
    print()


if __name__ == "__main__":
    main()
