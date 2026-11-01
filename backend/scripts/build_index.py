#!/usr/bin/env python3
"""
scripts/build_index.py – FinRule AI FAISS Index Builder

Reads data/processed/sebi/chunks.jsonl, generates embeddings using
Sentence Transformers, and writes a FAISS flat inner-product index
(equivalent to cosine similarity on L2-normalised vectors).

Outputs
───────
  data/processed/sebi/faiss_index/index.faiss    – FAISS index
  data/processed/sebi/faiss_index/metadata.json  – parallel metadata list

The index position i corresponds to metadata[i].

Usage
─────
  python scripts/build_index.py              # full index
  python scripts/build_index.py --limit 5000 # small index for testing
  python scripts/build_index.py --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Fix FAISS OpenMP segfaults on macOS + PyTorch MPS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SCRIPT_DIR   = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings
from app.embeddings.embedding_model import get_embedding_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_index")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a FAISS index from processed SEBI chunks."
    )
    p.add_argument(
        "--chunks-file", type=Path,
        default=settings.sebi_processed_dir / "chunks.jsonl",
        help="Path to chunks.jsonl (default: data/processed/sebi/chunks.jsonl).",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=settings.faiss_index_path,
        help="Directory for FAISS index output.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Index only the first N chunks (for testing).",
    )
    p.add_argument(
        "--batch-size", type=int, default=settings.embedding_batch_size,
        help=f"Embedding batch size (default: {settings.embedding_batch_size}).",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p.parse_args()


def load_chunks(path: Path, limit: int | None) -> tuple[list[str], list[dict]]:
    """
    Load chunks.jsonl and return (texts, metadata_list).

    Only 'text' is used for embedding; the full record is stored as metadata.
    """
    texts: list[str] = []
    metadata: list[dict] = []

    if not path.exists():
        raise FileNotFoundError(
            f"chunks.jsonl not found: {path}\n"
            "Run ingestion first: python scripts/ingest_open_india_law.py"
        )

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                text = (record.get("text") or "").strip()
                if not text:
                    continue
                texts.append(text)
                metadata.append(record)
            except json.JSONDecodeError as exc:
                logger.warning("Line %d: JSON decode error: %s", i, exc)

    logger.info("Loaded %d chunks from %s", len(texts), path)
    return texts, metadata


def build_faiss_index(texts: list[str], embed_model, batch_size: int):
    """Build a FAISS IndexFlatIP (inner product) index."""
    import faiss
    import numpy as np

    logger.info("Generating embeddings for %d chunks …", len(texts))
    t0 = time.perf_counter()

    vectors = embed_model.encode_batch(texts, batch_size=batch_size, show_progress=True)
    # Ensure C-contiguous array to prevent FAISS segfaults
    vecs_np = np.ascontiguousarray(np.array(vectors, dtype="float32"))

    # Note: SentenceTransformers encode_batch(normalize_embeddings=True) already normalizes.
    # We DO NOT call faiss.normalize_L2(vecs_np) here because it is redundant and 
    # frequently causes segfaults with OpenMP on macOS.

    elapsed = time.perf_counter() - t0
    logger.info(
        "Embeddings done: %d vectors, dim=%d, time=%.1fs",
        len(vecs_np), vecs_np.shape[1], elapsed,
    )

    dim = vecs_np.shape[1]
    index = faiss.IndexFlatIP(dim)  # flat inner product = cosine on normalised vecs
    index.add(vecs_np)
    logger.info("FAISS IndexFlatIP built: %d vectors.", index.ntotal)
    return index


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    t_start = time.perf_counter()

    print()
    print("═" * 65)
    print("  FinRule AI — FAISS Index Builder")
    print("═" * 65)
    print(f"  Source       : {args.chunks_file}")
    print(f"  Output       : {args.output_dir}")
    print(f"  Embedding    : {settings.embedding_model}")
    print()

    # ── Step 1: Load chunks ───────────────────────────────────────────────────
    logger.info("Step 1/4 – Loading chunks …")
    texts, metadata = load_chunks(args.chunks_file, args.limit)

    if not texts:
        logger.error("No chunks loaded. Run ingestion first.")
        sys.exit(1)

    # ── Step 2: Load embedding model ──────────────────────────────────────────
    logger.info("Step 2/4 – Loading embedding model …")
    embed_model = get_embedding_model()
    embed_model.load()
    dim = embed_model.get_embedding_dim()
    logger.info("  Model: %s  |  Dim: %d", embed_model.model_name, dim)

    # ── Step 3: Build FAISS index ─────────────────────────────────────────────
    logger.info("Step 3/4 – Building FAISS index …")
    index = build_faiss_index(texts, embed_model, args.batch_size)

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    logger.info("Step 4/4 – Saving index and metadata …")
    import faiss as faiss_module

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    index_file = output_dir / "index.faiss"
    meta_file  = output_dir / "metadata.json"

    faiss_module.write_index(index, str(index_file))
    logger.info("FAISS index written: %s (%.1f MB)", index_file, index_file.stat().st_size / 1024 / 1024)

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, default=str)
    logger.info("Metadata written: %s (%d entries)", meta_file, len(metadata))

    total_elapsed = time.perf_counter() - t_start

    print()
    print("═" * 65)
    print("  INDEX BUILD COMPLETE")
    print("═" * 65)
    print(f"  Embedding model  : {embed_model.model_name}")
    print(f"  Embedding dim    : {dim}")
    print(f"  Chunks indexed   : {index.ntotal:,}")
    print(f"  FAISS index size : {index_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Total time       : {total_elapsed:.1f}s")
    print()
    print("  The FinRule AI RAG is ready.")
    print("  Test it: python scripts/query_rag.py \"What are LODR requirements?\"")
    print()


if __name__ == "__main__":
    main()
