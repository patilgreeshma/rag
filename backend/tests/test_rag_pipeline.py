"""
tests/test_rag_pipeline.py – Tests for the FinRuleRAG pipeline.

Mocks out FAISS, embedding, BM25, reranker, and Gemini so the pipeline
logic (threshold, context building, source deduplication) is tested in
isolation without needing real models or indexes.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from app.rag.rag import FinRuleRAG, RAGResponse, RetrievedChunk, build_context


# ── build_context ──────────────────────────────────────────────────────────────

class TestBuildContext:
    def _make_chunk(self, i=1, title="SEBI Act", section="26", text="Some provision."):
        return RetrievedChunk(
            chunk_id=f"c{i}",
            text=text,
            score=0.9,
            title=title,
            section_number=section,
            section="Disclosure Requirements",
            source_url=f"https://sebi.gov.in/reg_{i}.html",
            in_force=True,
        )

    def test_context_contains_provision_header(self):
        chunks = [self._make_chunk(1)]
        ctx = build_context(chunks)
        assert "[PROVISION 1]" in ctx

    def test_context_contains_title(self):
        chunks = [self._make_chunk(1, title="SEBI LODR 2015")]
        ctx = build_context(chunks)
        assert "SEBI LODR 2015" in ctx

    def test_context_contains_section_number(self):
        chunks = [self._make_chunk(1, section="26A")]
        ctx = build_context(chunks)
        assert "26A" in ctx

    def test_context_contains_text(self):
        chunks = [self._make_chunk(1, text="The disclosure must be made within 24 hours.")]
        ctx = build_context(chunks)
        assert "The disclosure must be made within 24 hours." in ctx

    def test_multiple_chunks_separated(self):
        chunks = [self._make_chunk(1), self._make_chunk(2)]
        ctx = build_context(chunks)
        assert "[PROVISION 1]" in ctx
        assert "[PROVISION 2]" in ctx

    def test_superseded_chunk_labelled(self):
        chunk = self._make_chunk(1)
        chunk.in_force = False
        ctx = build_context([chunk])
        assert "SUPERSEDED" in ctx

    def test_in_force_chunk_labelled(self):
        chunk = self._make_chunk(1)
        chunk.in_force = True
        ctx = build_context([chunk])
        assert "IN FORCE" in ctx

    def test_empty_chunks_returns_empty_string(self):
        ctx = build_context([])
        assert ctx == ""


# ── FinRuleRAG.query ───────────────────────────────────────────────────────────

def _make_hybrid_result(chunk_id, text, score=0.8, meta=None):
    r = MagicMock()
    r.chunk_id = chunk_id
    r.text = text
    r.score = score
    r.hybrid_score = score
    r.vector_score = score
    r.bm25_score = 5.0
    r.metadata = meta or {
        "title": "SEBI LODR",
        "section_number": "26",
        "section": "Disclosure",
        "document_type": "regulatory",
        "source_url": f"https://sebi.gov.in/{chunk_id}.html",
        "in_force": True,
    }
    return r


def _make_reranked_result(chunk_id, text, score=0.9, meta=None):
    from app.retrieval.reranker import RerankedChunk
    base_meta = meta or {
        "title": "SEBI LODR",
        "section_number": "26",
        "section": "Disclosure",
        "document_type": "regulatory",
        "source_url": f"https://sebi.gov.in/{chunk_id}.html",
        "in_force": True,
    }
    return RerankedChunk(
        chunk_id=chunk_id,
        text=text,
        rerank_score=score,
        vector_score=0.8,
        bm25_score=5.0,
        hybrid_score=0.7,
        metadata=base_meta,
    )


class TestFinRuleRAGQuery:
    def _make_rag_with_mocks(self, hybrid_results, reranked_results, llm_answer="Test answer."):
        rag = FinRuleRAG.__new__(FinRuleRAG)
        rag._top_k = 5
        rag._candidates = 20
        rag._threshold = 0.3
        rag._use_reranker = True
        rag._hybrid_searcher = None
        rag._reranker = None
        rag._llm = None

        mock_hybrid = MagicMock()
        mock_hybrid.search.return_value = hybrid_results
        mock_hybrid.total_vectors = 2000

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = reranked_results

        mock_llm = MagicMock()
        mock_llm.generate.return_value = llm_answer

        rag._hybrid_searcher = mock_hybrid
        rag._reranker = mock_reranker
        rag._llm = mock_llm

        # Override lazy getters to return mocks
        rag._get_hybrid_searcher = lambda: mock_hybrid
        rag._get_reranker = lambda: mock_reranker
        rag._get_llm = lambda: mock_llm

        return rag

    def test_empty_question_returns_error(self):
        rag = FinRuleRAG.__new__(FinRuleRAG)
        rag._top_k = 5
        rag._candidates = 20
        rag._threshold = 0.3
        rag._use_reranker = True
        rag._hybrid_searcher = None
        rag._reranker = None
        rag._llm = None
        rag._get_hybrid_searcher = MagicMock()
        rag._get_reranker = MagicMock()
        rag._get_llm = MagicMock()

        resp = rag.query("")
        assert resp.sufficient_context is False
        assert "non-empty" in resp.answer.lower()

    def test_successful_query_returns_answer(self):
        hybrid_results = [_make_hybrid_result("c1", "Provision text.")]
        reranked_results = [_make_reranked_result("c1", "Provision text.", 0.9)]
        rag = self._make_rag_with_mocks(hybrid_results, reranked_results, "LODR requires 24h disclosure.")

        resp = rag.query("What are LODR requirements?")
        assert resp.sufficient_context is True
        assert resp.answer == "LODR requires 24h disclosure."

    def test_no_hybrid_results_returns_insufficient(self):
        rag = self._make_rag_with_mocks([], [])
        resp = rag.query("Completely unknown question here?")
        assert resp.sufficient_context is False

    def test_sources_deduplicated_by_url(self):
        # Two chunks with same source_url → only one source in response
        meta = {
            "title": "SEBI LODR",
            "section_number": "26",
            "section": "Disclosure",
            "document_type": "regulatory",
            "source_url": "https://sebi.gov.in/same.html",
            "in_force": True,
        }
        reranked = [
            _make_reranked_result("c1", "Text 1.", 0.9, meta),
            _make_reranked_result("c2", "Text 2.", 0.8, meta),
        ]
        rag = self._make_rag_with_mocks(
            [_make_hybrid_result("c1", "Text 1.")],
            reranked,
            "Answer.",
        )
        resp = rag.query("What are LODR requirements?")
        assert len(resp.sources) == 1

    def test_question_stripped_before_processing(self):
        hybrid_results = [_make_hybrid_result("c1", "Provision text.")]
        reranked_results = [_make_reranked_result("c1", "Provision text.", 0.9)]
        rag = self._make_rag_with_mocks(hybrid_results, reranked_results, "Answer.")

        resp = rag.query("  What are LODR requirements?  ")
        assert resp.sufficient_context is True

    def test_response_has_top_score(self):
        hybrid_results = [_make_hybrid_result("c1", "Provision text.")]
        reranked_results = [_make_reranked_result("c1", "Provision text.", 0.87)]
        rag = self._make_rag_with_mocks(hybrid_results, reranked_results, "Answer.")

        resp = rag.query("What are LODR requirements?")
        assert resp.top_score == pytest.approx(0.87)

    def test_retrieved_chunks_returned(self):
        hybrid_results = [_make_hybrid_result(f"c{i}", f"Text {i}.") for i in range(3)]
        reranked = [_make_reranked_result(f"c{i}", f"Text {i}.", 0.9 - i * 0.1) for i in range(3)]
        rag = self._make_rag_with_mocks(hybrid_results, reranked, "Answer.")

        resp = rag.query("SEBI provisions?")
        assert len(resp.retrieved_chunks) == 3
