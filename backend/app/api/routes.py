"""
app/api/routes.py – FastAPI route definitions for FinRule AI.

Endpoints
---------
GET  /health  → liveness + index stats
POST /query   → full RAG pipeline (hybrid + reranker + Gemini)
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.query import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceItem,
    ChunkItem,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Shared RAG instance (loaded once on first request) ─────────────────────────
_rag = None


def _get_rag():
    global _rag
    if _rag is None:
        from app.rag.rag import FinRuleRAG
        logger.info("Initialising FinRuleRAG pipeline …")
        _rag = FinRuleRAG()
        logger.info("FinRuleRAG pipeline ready.")
    return _rag


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service status and number of vectors in the FAISS index.",
    tags=["System"],
)
async def health() -> HealthResponse:
    """Liveness probe – also reports FAISS index size."""
    try:
        rag = _get_rag()
        n_vectors = rag.total_vectors
    except Exception as exc:          # noqa: BLE001
        logger.warning("Health check – could not load RAG: %s", exc)
        n_vectors = -1

    return HealthResponse(status="healthy", version="1.0.0", index_vectors=n_vectors)


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a SEBI regulatory question",
    description=(
        "Runs the full RAG pipeline: hybrid retrieval (vector + BM25), "
        "Cross-Encoder reranking, and Gemini-powered grounded generation. "
        "All answers are strictly grounded in retrieved SEBI provisions."
    ),
    tags=["RAG"],
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    POST /query – main RAG endpoint.

    Accepts a natural-language question, runs the hybrid retrieval + reranking
    pipeline, and returns a Gemini-generated answer with source citations.
    """
    logger.info("POST /query – question=%.80s", request.question)

    try:
        rag = _get_rag()
        response = rag.query(request.question)
    except Exception as exc:          # noqa: BLE001
        logger.exception("RAG pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}") from exc

    # Convert internal types → Pydantic response models
    sources = [
        SourceItem(
            title=s.get("title"),
            section_number=s.get("section_number"),
            section=s.get("section"),
            document_type=s.get("document_type"),
            source_url=s.get("source_url"),
            in_force=s.get("in_force", True),
            score=s.get("score", 0.0),
        )
        for s in response.sources
    ]

    chunks = [
        ChunkItem(
            chunk_id=c.chunk_id,
            text=c.text,
            score=c.score,
            title=c.title,
            section_number=c.section_number,
            source_url=c.source_url,
            in_force=c.in_force,
            metadata=c.metadata,
        )
        for c in response.retrieved_chunks
    ]

    return QueryResponse(
        question=response.question,
        answer=response.answer,
        sources=sources,
        retrieved_chunks=chunks,
        sufficient_context=response.sufficient_context,
        top_score=response.top_score,
    )
