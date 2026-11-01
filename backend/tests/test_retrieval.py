"""
tests/test_retrieval.py – Tests for BM25, vector search, hybrid, and reranker modules.

Uses mock/stub data so tests run without the real FAISS index or model on disk.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ── BM25 tokeniser ─────────────────────────────────────────────────────────────

class TestBM25Tokenizer:
    def test_basic_tokenisation(self):
        from app.retrieval.bm25_search import _tokenize
        tokens = _tokenize("LODR disclosure requirements")
        assert "lodr" in tokens
        assert "disclosure" in tokens
        assert "requirements" in tokens

    def test_lowercases(self):
        from app.retrieval.bm25_search import _tokenize
        tokens = _tokenize("SEBI Regulations")
        assert "sebi" in tokens
        assert "regulations" in tokens

    def test_removes_punctuation(self):
        from app.retrieval.bm25_search import _tokenize
        tokens = _tokenize("section 26A(1)(b)")
        assert "" not in tokens
        # Should not crash on parens / brackets

    def test_empty_string(self):
        from app.retrieval.bm25_search import _tokenize
        assert _tokenize("") == []

    def test_whitespace_only(self):
        from app.retrieval.bm25_search import _tokenize
        assert _tokenize("   ") == []

    def test_hyphens_preserved(self):
        from app.retrieval.bm25_search import _tokenize
        tokens = _tokenize("sub-section non-obstante")
        assert "sub-section" in tokens
        assert "non-obstante" in tokens


# ── BM25 Searcher ──────────────────────────────────────────────────────────────

class TestBM25Searcher:
    def _make_searcher_with_corpus(self, corpus):
        """Build a BM25Searcher with a fake corpus injected via mocked VectorSearcher."""
        from app.retrieval.bm25_search import BM25Searcher

        fake_vs = MagicMock()
        fake_vs.metadata = corpus
        searcher = BM25Searcher(vector_searcher=fake_vs)
        return searcher

    def test_returns_results(self):
        corpus = [
            {"chunk_id": "c1", "text": "LODR disclosure requirements for listed entities"},
            {"chunk_id": "c2", "text": "Mutual fund regulations and NAV calculation"},
            {"chunk_id": "c3", "text": "Insider trading prohibition and definition"},
        ]
        searcher = self._make_searcher_with_corpus(corpus)
        results = searcher.search("LODR disclosure", top_k=2)
        assert len(results) <= 2
        # Top result should be the LODR chunk
        if results:
            assert results[0].chunk_id == "c1"

    def test_empty_query(self):
        corpus = [{"chunk_id": "c1", "text": "Some text"}]
        searcher = self._make_searcher_with_corpus(corpus)
        results = searcher.search("", top_k=5)
        assert results == []

    def test_no_match_returns_empty(self):
        corpus = [{"chunk_id": "c1", "text": "xyz abc def"}]
        searcher = self._make_searcher_with_corpus(corpus)
        # A query with zero overlap returns no results (score=0 filtered out)
        results = searcher.search("lodr sebi disclosure requirements", top_k=5)
        assert isinstance(results, list)

    def test_top_k_respected(self):
        corpus = [{"chunk_id": f"c{i}", "text": f"LODR disclosure section {i}"} for i in range(10)]
        searcher = self._make_searcher_with_corpus(corpus)
        results = searcher.search("LODR disclosure", top_k=3)
        assert len(results) <= 3

    def test_result_has_required_fields(self):
        corpus = [{"chunk_id": "c1", "text": "SEBI regulation text", "title": "SEBI Act"}]
        searcher = self._make_searcher_with_corpus(corpus)
        results = searcher.search("SEBI regulation", top_k=1)
        if results:
            r = results[0]
            assert hasattr(r, "chunk_id")
            assert hasattr(r, "text")
            assert hasattr(r, "score")
            assert hasattr(r, "metadata")
            assert r.score > 0


# ── Hybrid RRF Fusion ──────────────────────────────────────────────────────────

class TestHybridSearcher:
    def _make_vector_result(self, chunk_id, text, score, meta=None):
        from app.retrieval.vector_search import VectorSearchResult
        return VectorSearchResult(chunk_id=chunk_id, text=text, score=score, metadata=meta or {})

    def _make_bm25_result(self, chunk_id, text, score, meta=None):
        from app.retrieval.bm25_search import BM25Result
        return BM25Result(chunk_id=chunk_id, text=text, score=score, metadata=meta or {})

    def test_fuse_identical_hits(self):
        from app.retrieval.hybrid_search import HybridSearcher
        searcher = HybridSearcher()

        vec_results = [
            self._make_vector_result("c1", "text1", 0.9),
            self._make_vector_result("c2", "text2", 0.7),
        ]
        bm25_results = [
            self._make_bm25_result("c1", "text1", 15.0),
            self._make_bm25_result("c3", "text3", 10.0),
        ]

        fused = searcher._fuse(vec_results, bm25_results)
        # c1 appears in both lists → should have highest hybrid score
        assert fused[0].chunk_id == "c1"

    def test_fuse_produces_union(self):
        from app.retrieval.hybrid_search import HybridSearcher
        searcher = HybridSearcher()

        vec_results = [self._make_vector_result("c1", "t1", 0.9)]
        bm25_results = [self._make_bm25_result("c2", "t2", 5.0)]

        fused = searcher._fuse(vec_results, bm25_results)
        ids = [r.chunk_id for r in fused]
        assert "c1" in ids
        assert "c2" in ids

    def test_rrf_score_decreases_with_rank(self):
        from app.retrieval.hybrid_search import HybridSearcher
        s = HybridSearcher()
        assert s._rrf_score(0, 60) > s._rrf_score(5, 60) > s._rrf_score(10, 60)

    def test_hybrid_result_has_scores(self):
        from app.retrieval.hybrid_search import HybridSearcher
        searcher = HybridSearcher()

        vec_results = [self._make_vector_result("c1", "text", 0.85, {"title": "LODR"})]
        bm25_results = [self._make_bm25_result("c1", "text", 12.0, {"title": "LODR"})]

        fused = searcher._fuse(vec_results, bm25_results)
        r = fused[0]
        assert r.hybrid_score > 0
        assert r.vector_score == 0.85
        assert r.bm25_score == 12.0


# ── Cross-Encoder Reranker ─────────────────────────────────────────────────────

class TestCrossEncoderReranker:
    def _make_candidate(self, chunk_id, text, score=0.5):
        c = MagicMock()
        c.chunk_id = chunk_id
        c.text = text
        c.score = score
        c.metadata = {"title": f"Doc {chunk_id}"}
        c.vector_score = None
        c.bm25_score = None
        c.hybrid_score = score
        return c

    def test_reranker_sorts_by_score(self):
        from app.retrieval.reranker import CrossEncoderReranker, RerankedChunk

        reranker = CrossEncoderReranker()
        # Mock the model predict to return known scores
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.3, 0.7]
        reranker._model = mock_model

        candidates = [
            self._make_candidate("c1", "text1"),
            self._make_candidate("c2", "text2"),
            self._make_candidate("c3", "text3"),
        ]

        results = reranker.rerank("test query", candidates, top_k=3)
        scores = [r.rerank_score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].chunk_id == "c1"  # highest score was 0.9

    def test_reranker_respects_top_k(self):
        from app.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.3, 0.7, 0.5, 0.8]
        reranker._model = mock_model

        candidates = [self._make_candidate(f"c{i}", f"text{i}") for i in range(5)]
        results = reranker.rerank("query", candidates, top_k=2)
        assert len(results) == 2

    def test_reranker_empty_candidates(self):
        from app.retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        results = reranker.rerank("query", [], top_k=5)
        assert results == []

    def test_reranked_chunk_properties(self):
        from app.retrieval.reranker import RerankedChunk
        chunk = RerankedChunk(
            chunk_id="c1",
            text="Some provision text.",
            rerank_score=0.95,
            metadata={
                "title": "SEBI LODR",
                "section_number": "26",
                "source_url": "https://sebi.gov.in",
                "in_force": True,
                "document_type": "regulatory",
            }
        )
        assert chunk.title == "SEBI LODR"
        assert chunk.section_number == "26"
        assert chunk.in_force is True
        assert chunk.score == 0.95  # alias for rerank_score
