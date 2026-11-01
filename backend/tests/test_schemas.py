"""
tests/test_schemas.py – Tests for Pydantic request/response schemas.
"""
import pytest
from pydantic import ValidationError

from app.schemas.query import (
    QueryRequest,
    QueryResponse,
    SourceItem,
    ChunkItem,
    HealthResponse,
)


class TestQueryRequest:
    def test_valid_request(self):
        req = QueryRequest(question="What are LODR requirements?")
        assert req.question == "What are LODR requirements?"
        assert req.top_k == 5  # default

    def test_custom_top_k(self):
        req = QueryRequest(question="What are LODR requirements?", top_k=10)
        assert req.top_k == 10

    def test_question_too_short(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="Hi")  # less than 3 chars

    def test_question_empty(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="")

    def test_top_k_too_high(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question here", top_k=21)

    def test_top_k_zero(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question here", top_k=0)

    def test_top_k_boundary_values(self):
        req1 = QueryRequest(question="Valid question here", top_k=1)
        req2 = QueryRequest(question="Valid question here", top_k=20)
        assert req1.top_k == 1
        assert req2.top_k == 20


class TestSourceItem:
    def test_minimal_source(self):
        src = SourceItem(score=0.85)
        assert src.score == 0.85
        assert src.title is None
        assert src.in_force is True

    def test_full_source(self):
        src = SourceItem(
            title="SEBI LODR Regulations 2015",
            section_number="26A",
            section="Disclosure Requirements",
            document_type="regulatory",
            source_url="https://www.sebi.gov.in/legal/regulations/dec-2024/lodr.html",
            in_force=True,
            score=0.92,
        )
        assert src.title == "SEBI LODR Regulations 2015"
        assert src.section_number == "26A"
        assert src.score == 0.92


class TestChunkItem:
    def test_minimal_chunk(self):
        chunk = ChunkItem(chunk_id="chunk_001", text="Test provision text.", score=0.75)
        assert chunk.chunk_id == "chunk_001"
        assert chunk.metadata == {}

    def test_chunk_with_metadata(self):
        chunk = ChunkItem(
            chunk_id="chunk_002",
            text="Some regulation text.",
            score=0.80,
            title="SEBI Mutual Funds Regulations",
            section_number="47",
            source_url="https://www.sebi.gov.in/",
            in_force=False,
            metadata={"year": 2023, "chapter": "IX"},
        )
        assert chunk.in_force is False
        assert chunk.metadata["year"] == 2023


class TestHealthResponse:
    def test_defaults(self):
        h = HealthResponse(index_vectors=2000)
        assert h.status == "healthy"
        assert h.version == "1.0.0"
        assert h.index_vectors == 2000

    def test_custom_values(self):
        h = HealthResponse(status="degraded", version="2.0.0", index_vectors=0)
        assert h.status == "degraded"


class TestQueryResponse:
    def test_minimal_response(self):
        resp = QueryResponse(
            question="What is LODR?",
            answer="LODR stands for Listing Obligations and Disclosure Requirements.",
            sufficient_context=True,
        )
        assert resp.question == "What is LODR?"
        assert resp.sources == []
        assert resp.retrieved_chunks == []
        assert resp.top_score is None

    def test_insufficient_context_response(self):
        resp = QueryResponse(
            question="Unknown topic?",
            answer="I couldn't find sufficient information.",
            sufficient_context=False,
            top_score=0.12,
        )
        assert resp.sufficient_context is False
        assert resp.top_score == 0.12
