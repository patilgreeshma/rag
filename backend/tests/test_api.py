"""
tests/test_api.py – Integration tests for FastAPI endpoints.

Uses FastAPI's TestClient (backed by httpx) so no real server process needed.
Mocks out the RAG pipeline to keep tests fast and deterministic.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self):
        with patch("app.api.routes._get_rag") as mock_get_rag:
            mock_rag = MagicMock()
            mock_rag.total_vectors = 2000
            mock_get_rag.return_value = mock_rag

            resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_shape(self):
        with patch("app.api.routes._get_rag") as mock_get_rag:
            mock_rag = MagicMock()
            mock_rag.total_vectors = 2000
            mock_get_rag.return_value = mock_rag

            resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "index_vectors" in data

    def test_health_rag_failure_still_returns_200(self):
        """Health should return 200 even if RAG can't load (reports -1 vectors)."""
        with patch("app.api.routes._get_rag", side_effect=Exception("index not found")):
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["index_vectors"] == -1


# ── /query ─────────────────────────────────────────────────────────────────────

def _make_mock_rag_response(sufficient=True, answer="Test answer.", n_sources=1):
    """Create a fake RAGResponse-like object."""
    from app.rag.rag import RAGResponse, RetrievedChunk

    chunks = [
        RetrievedChunk(
            chunk_id=f"chunk_{i}",
            text=f"Provision text {i}.",
            score=0.9 - i * 0.1,
            title="SEBI LODR Regulations",
            section_number=str(i + 1),
            source_url=f"https://sebi.gov.in/reg_{i}.html",
            in_force=True,
            metadata={},
        )
        for i in range(n_sources)
    ]

    sources = [
        {
            "title": c.title,
            "section_number": c.section_number,
            "section": None,
            "document_type": "regulatory",
            "source_url": c.source_url,
            "in_force": c.in_force,
            "score": c.score,
        }
        for c in chunks
    ]

    return RAGResponse(
        question="Test question?",
        answer=answer,
        sources=sources,
        retrieved_chunks=chunks,
        sufficient_context=sufficient,
        top_score=0.9 if sufficient else 0.1,
    )


class TestQueryEndpoint:
    def test_valid_query_returns_200(self):
        with patch("app.api.routes._get_rag") as mock_get_rag:
            mock_rag = MagicMock()
            mock_rag.query.return_value = _make_mock_rag_response()
            mock_get_rag.return_value = mock_rag

            resp = client.post("/query", json={"question": "What are LODR requirements?"})

        assert resp.status_code == 200

    def test_query_response_has_required_fields(self):
        with patch("app.api.routes._get_rag") as mock_get_rag:
            mock_rag = MagicMock()
            mock_rag.query.return_value = _make_mock_rag_response()
            mock_get_rag.return_value = mock_rag

            resp = client.post("/query", json={"question": "What are LODR requirements?"})

        data = resp.json()
        assert "question" in data
        assert "answer" in data
        assert "sources" in data
        assert "retrieved_chunks" in data
        assert "sufficient_context" in data

    def test_query_question_too_short(self):
        resp = client.post("/query", json={"question": "Hi"})
        assert resp.status_code == 422  # validation error

    def test_query_missing_question(self):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_query_top_k_out_of_range(self):
        resp = client.post("/query", json={"question": "Valid question here", "top_k": 99})
        assert resp.status_code == 422

    def test_insufficient_context_response(self):
        with patch("app.api.routes._get_rag") as mock_get_rag:
            mock_rag = MagicMock()
            mock_rag.query.return_value = _make_mock_rag_response(sufficient=False, n_sources=0)
            mock_get_rag.return_value = mock_rag

            resp = client.post("/query", json={"question": "Unknown obscure topic here"})

        data = resp.json()
        assert resp.status_code == 200
        assert data["sufficient_context"] is False

    def test_rag_pipeline_error_returns_500(self):
        with patch("app.api.routes._get_rag") as mock_get_rag:
            mock_rag = MagicMock()
            mock_rag.query.side_effect = RuntimeError("Model failure")
            mock_get_rag.return_value = mock_rag

            resp = client.post("/query", json={"question": "Valid question here please"})

        assert resp.status_code == 500

    def test_custom_top_k_passed_through(self):
        with patch("app.api.routes._get_rag") as mock_get_rag:
            mock_rag = MagicMock()
            mock_rag.query.return_value = _make_mock_rag_response()
            mock_get_rag.return_value = mock_rag

            resp = client.post("/query", json={"question": "SEBI disclosure rules?", "top_k": 10})

        assert resp.status_code == 200

    def test_sources_structure(self):
        with patch("app.api.routes._get_rag") as mock_get_rag:
            mock_rag = MagicMock()
            mock_rag.query.return_value = _make_mock_rag_response(n_sources=2)
            mock_get_rag.return_value = mock_rag

            resp = client.post("/query", json={"question": "What are LODR requirements?"})

        data = resp.json()
        assert len(data["sources"]) == 2
        for src in data["sources"]:
            assert "title" in src
            assert "score" in src
            assert "source_url" in src


# ── Swagger docs ───────────────────────────────────────────────────────────────

class TestDocsEndpoints:
    def test_swagger_ui_accessible(self):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_redoc_accessible(self):
        resp = client.get("/redoc")
        assert resp.status_code == 200

    def test_openapi_json_accessible(self):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["info"]["title"] == "FinRule AI"
