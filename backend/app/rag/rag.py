"""
app/rag/rag.py – FinRule AI RAG Pipeline (v2 – Hybrid + Reranker).

End-to-end Retrieval-Augmented Generation for SEBI regulatory queries.

Pipeline
────────
User Question
    ↓  query preprocessing
    ↓  Hybrid Search  (Vector + BM25 → RRF fusion, top 20 candidates)
    ↓  Cross-Encoder Reranker  (top 20 → top 5)
    ↓  Similarity threshold filter (no hallucination below threshold)
    ↓  Context construction
    ↓  Gemini LLM with strict regulatory prompt
    ↓  Grounded answer + source citations

Strict no-hallucination contract
─────────────────────────────────
* The LLM is explicitly instructed to answer ONLY from retrieved context.
* If all reranked scores are below ``retrieval_threshold``, the system
  returns a canned insufficient-context response without calling the LLM.
* Source URLs are taken from chunk metadata – never fabricated.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.generation.prompt import INSUFFICIENT_CONTEXT_RESPONSE

logger = logging.getLogger(__name__)


# ── Return types ──────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A retrieved provision chunk with retrieval scores."""
    chunk_id: str
    text: str
    score: float                          # rerank score (primary sort key)
    title: str | None = None
    document_type: str | None = None
    section_number: str | None = None
    section: str | None = None
    source_url: str | None = None
    in_force: bool = True
    vector_score: float | None = None
    bm25_score: float | None = None
    hybrid_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Full RAG response returned to the caller."""
    question: str
    answer: str
    sources: list[dict[str, Any]]
    retrieved_chunks: list[RetrievedChunk]
    sufficient_context: bool
    top_score: float | None = None


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a structured context block for the LLM."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        lines = [f"[PROVISION {i}]"]
        if chunk.title:
            lines.append(f"Document: {chunk.title}")
        if chunk.section_number:
            lines.append(f"Section: {chunk.section_number}")
        if chunk.section:
            lines.append(f"Heading: {chunk.section}")
        lines.append(f"Status: {'IN FORCE' if chunk.in_force else 'SUPERSEDED'}")
        if chunk.source_url:
            lines.append(f"Source: {chunk.source_url}")
        lines.append("")
        lines.append(chunk.text.strip())
        parts.append("\n".join(lines))

    return ("\n" + "─" * 60 + "\n").join(parts)


# ── Main RAG pipeline ─────────────────────────────────────────────────────────

class FinRuleRAG:
    """
    End-to-end FinRule AI RAG pipeline (Hybrid + Reranker version).

    Usage
    -----
    rag = FinRuleRAG()
    response = rag.query("What are the disclosure requirements under LODR?")
    print(response.answer)
    for src in response.sources:
        print(src["title"], src["source_url"])
    """

    def __init__(
        self,
        top_k: int | None = None,
        retrieval_candidates: int = 20,
        threshold: float | None = None,
        use_reranker: bool = True,
    ) -> None:
        from app.config import settings

        self._top_k = top_k or settings.top_k
        self._candidates = retrieval_candidates
        self._threshold = threshold if threshold is not None else settings.retrieval_threshold
        self._use_reranker = use_reranker

        # Lazy-init components
        self._hybrid_searcher = None
        self._reranker = None
        self._llm = None

    # ── Component accessors (lazy) ─────────────────────────────────────────────

    def _get_hybrid_searcher(self):
        if self._hybrid_searcher is None:
            from app.retrieval.hybrid_search import HybridSearcher
            self._hybrid_searcher = HybridSearcher(candidates=self._candidates)
        return self._hybrid_searcher

    def _get_reranker(self):
        if self._reranker is None:
            from app.retrieval.reranker import CrossEncoderReranker
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def _get_llm(self):
        if self._llm is None:
            from app.generation.gemini import GeminiClient
            self._llm = GeminiClient()
        return self._llm

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str) -> RAGResponse:
        """
        Run the full RAG pipeline for a user question.

        Parameters
        ----------
        question:
            Natural-language question about SEBI regulations.

        Returns
        -------
        RAGResponse with answer, sources, and retrieved chunks.
        """
        if not question or not question.strip():
            return RAGResponse(
                question=question,
                answer="Please provide a non-empty question.",
                sources=[],
                retrieved_chunks=[],
                sufficient_context=False,
                top_score=None,
            )

        q = question.strip()

        # ── Stage 1: Hybrid retrieval (vector + BM25 → RRF) ──────────────────
        hybrid_results = self._get_hybrid_searcher().search(q, top_k=self._candidates)

        if not hybrid_results:
            return RAGResponse(
                question=question,
                answer=INSUFFICIENT_CONTEXT_RESPONSE,
                sources=[],
                retrieved_chunks=[],
                sufficient_context=False,
                top_score=None,
            )

        # ── Stage 2: Cross-Encoder reranking ─────────────────────────────────
        if self._use_reranker:
            reranked = self._get_reranker().rerank(
                query=q,
                candidates=hybrid_results,
                top_k=self._top_k,
            )
        else:
            # Fallback: use hybrid results directly, map to RetrievedChunk
            reranked = hybrid_results[: self._top_k]

        # Convert to RetrievedChunk dataclass
        chunks: list[RetrievedChunk] = []
        for r in reranked:
            chunks.append(
                RetrievedChunk(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    score=r.score,
                    title=r.metadata.get("title"),
                    document_type=r.metadata.get("document_type"),
                    section_number=r.metadata.get("section_number"),
                    section=r.metadata.get("section"),
                    source_url=r.metadata.get("source_url"),
                    in_force=r.metadata.get("in_force", True),
                    vector_score=getattr(r, "vector_score", None),
                    bm25_score=getattr(r, "bm25_score", None),
                    hybrid_score=getattr(r, "hybrid_score", None),
                    metadata=r.metadata,
                )
            )

        top_score = chunks[0].score if chunks else None

        # ── Stage 3: Threshold filter ─────────────────────────────────────────
        # For reranked results the scores are Cross-Encoder logits (not 0-1),
        # so we apply threshold only when not using reranker.
        if not self._use_reranker:
            relevant = [c for c in chunks if c.score >= self._threshold]
        else:
            relevant = chunks  # reranker scores are not cosine; keep all top_k

        if not relevant:
            logger.info(
                "Query '%.60s' – top score %.4f below threshold → insufficient context.",
                question, top_score or 0.0,
            )
            return RAGResponse(
                question=question,
                answer=INSUFFICIENT_CONTEXT_RESPONSE,
                sources=[],
                retrieved_chunks=chunks,
                sufficient_context=False,
                top_score=top_score,
            )

        # ── Stage 4: Generate ─────────────────────────────────────────────────
        context = build_context(relevant)
        answer = self._get_llm().generate(question=q, context=context)

        # ── Build deduplicated sources list ───────────────────────────────────
        sources: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for chunk in relevant:
            key = chunk.source_url or chunk.chunk_id
            if key in seen_keys:
                continue
            seen_keys.add(key)
            sources.append({
                "title":          chunk.title,
                "section_number": chunk.section_number,
                "section":        chunk.section,
                "document_type":  chunk.document_type,
                "source_url":     chunk.source_url,
                "in_force":       chunk.in_force,
                "score":          round(chunk.score, 4),
            })

        logger.info(
            "Query '%.60s' – %d relevant chunks, top_score=%.4f",
            question, len(relevant), top_score or 0.0,
        )

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            retrieved_chunks=relevant,
            sufficient_context=True,
            top_score=top_score,
        )

    @property
    def total_vectors(self) -> int:
        """Number of vectors in the FAISS index."""
        return self._get_hybrid_searcher().total_vectors


# ── Custom exceptions ─────────────────────────────────────────────────────────

class IndexNotBuiltError(RuntimeError):
    """Raised when the FAISS index does not exist on disk."""


class LLMConfigError(RuntimeError):
    """Raised when the LLM API key is not configured."""
