"""
Tests for open_india_law_loader.py
"""
import pandas as pd
import pytest

from app.ingestion.open_india_law_loader import (
    validate_schema,
    row_to_chunk,
    dataframe_to_chunks,
    SchemaValidationError,
)

def test_validate_schema_success():
    df = pd.DataFrame({
        "act_id": ["1"],
        "chunk_id": ["1"],
        "title": ["Title"],
        "text": ["Text"],
        "source_url": ["http://test"]
    })
    validate_schema(df)  # Should not raise

def test_validate_schema_missing_columns():
    df = pd.DataFrame({
        "act_id": ["1"],
        # Missing chunk_id, etc.
    })
    with pytest.raises(SchemaValidationError):
        validate_schema(df)

def test_row_to_chunk():
    row = {
        "act_id": "REG_123",
        "chunk_id": "REG_123_sec1",
        "title": "Test Regulation",
        "chapter": "Chapter I",
        "section_number": "1",
        "section_title": "Short title",
        "text": "This is a test.",
        "instrument_type": "regulatory",
        "act_status": "in_force",
        "in_force": True,
        "provision_type": "definition",
        "section_type": "section",
        "legal_subject": ["finance"],
        "has_proviso": False,
        "has_non_obstante": False,
        "acts_referenced": [],
        "defined_terms": ["Test"],
        "amendment_count": 0,
        "year": 2024.0,
        "source_url": "http://example.com/sebi",
        "source_publisher": "SEBI",
        "mirror_url": "http://example.com/mirror",
        "department": "sebi"
    }
    chunk = row_to_chunk(row)
    
    assert chunk.chunk_id == "REG_123_sec1"
    assert chunk.document_id == "REG_123"
    assert chunk.title == "Test Regulation"
    assert chunk.section == "Short title"
    assert chunk.section_number == "1"
    assert chunk.text == "This is a test."
    assert chunk.source_url == "http://example.com/sebi"
    
    metadata = chunk.metadata
    assert metadata["year"] == 2024
    assert metadata["in_force"] is True
    assert metadata["legal_subject"] == ["finance"]
    assert metadata["defined_terms"] == ["Test"]

def test_dataframe_to_chunks_deduplication():
    df = pd.DataFrame({
        "act_id": ["1", "1"],
        "chunk_id": ["dup_id", "dup_id"],
        "title": ["Title", "Title"],
        "text": ["Text 1", "Text 2"],
        "source_url": ["http://test", "http://test"]
    })
    chunks = dataframe_to_chunks(df)
    assert len(chunks) == 1
    assert chunks[0].text == "Text 1"  # First one wins
