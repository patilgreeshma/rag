"""
regulatory_normalizer.py – Normalizer for regulatory documents.

Converts a ``RegulatoryDocument`` (from the parser) into a consistently
structured form ready for chunking.

Normalisation responsibilities
───────────────────────────────
* Normalise whitespace and line endings in text and section texts.
* Remove repeated headers/footers when confidently detected.
* Preserve section numbering, paragraph boundaries, tables.
* Preserve dates, numerical values, and legal terminology exactly.
* Remove obvious website boilerplate.
* DO NOT paraphrase the regulations.
* The stored text must remain faithful to the source.

This normalizer is source-agnostic and will work for RBI, SEC, etc. with
no changes.  Future source-specific quirks can be injected via subclassing
or a strategy parameter.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app.ingestion.regulatory_models import DocumentSection, RegulatoryDocument

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# A header/footer is considered "repeated" if the same line appears on
# this many consecutive pages (PDF heuristic).
_REPEATED_LINE_THRESHOLD = 3

# Patterns for residual boilerplate that may survive the parser stage
_NORMALIZER_BOILERPLATE = [
    re.compile(r"(?m)^\s*Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"(?m)^\s*\d+\s*$"),                   # bare page numbers
    re.compile(r"(?m)^\s*www\.sebi\.gov\.in\s*$", re.IGNORECASE),
    re.compile(r"(?m)^\s*Confidential\s*$", re.IGNORECASE),
]


# ── RegulatoryNormalizer ──────────────────────────────────────────────────────

class RegulatoryNormalizer:
    """
    Normalise a ``RegulatoryDocument`` in-place.

    The normalizer mutates the document's ``text`` and ``sections`` fields.
    All other fields (metadata, dates, source_url, etc.) are preserved as-is.

    Usage
    -----
    normalizer = RegulatoryNormalizer()
    doc = normalizer.normalize(doc)   # returns the same doc, mutated
    """

    def normalize(self, doc: RegulatoryDocument) -> RegulatoryDocument:
        """
        Normalise a ``RegulatoryDocument`` and return it.

        If extraction_status != "success", the document is returned unchanged.
        """
        if doc.extraction_status != "success":
            logger.debug(
                "Skipping normalisation for %s (extraction_status=%s)",
                doc.document_id, doc.extraction_status,
            )
            return doc

        if doc.text:
            doc.text = self._normalize_text(doc.text)

        for section in doc.sections:
            self._normalize_section(section)

        # Remove empty sections after normalisation
        doc.sections = [s for s in doc.sections if s.text.strip()]

        logger.debug(
            "Normalised document %s (%d sections).",
            doc.document_id, len(doc.sections),
        )
        return doc

    def normalize_batch(
        self, docs: list[RegulatoryDocument]
    ) -> list[RegulatoryDocument]:
        """Normalise a batch of documents, skipping any that raise."""
        results: list[RegulatoryDocument] = []
        for doc in docs:
            try:
                results.append(self.normalize(doc))
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Normalisation failed for %s: %s", doc.document_id, exc
                )
                doc.extraction_status = "partial"
                doc.extraction_error = f"Normalisation error: {exc}"
                results.append(doc)
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _normalize_text(self, text: str) -> str:
        """
        Apply all normalisation passes to the full document text.

        Passes (in order):
        1. Normalise Windows/Mac line endings to Unix.
        2. Remove residual boilerplate patterns.
        3. Remove repeated header/footer lines.
        4. Normalise intra-line whitespace (collapse multiple spaces/tabs).
        5. Normalise excessive blank lines (max 2 consecutive).
        6. Strip leading/trailing whitespace.
        """
        text = self._normalize_line_endings(text)
        text = self._remove_residual_boilerplate(text)
        text = self._remove_repeated_lines(text)
        text = self._normalize_inline_whitespace(text)
        text = self._collapse_blank_lines(text)
        return text.strip()

    def _normalize_section(self, section: DocumentSection) -> None:
        """Apply normalisation to a single section."""
        if section.text:
            section.text = self._normalize_text(section.text)
        # Recurse into subsections
        for sub in section.subsections:
            self._normalize_section(sub)

    @staticmethod
    def _normalize_line_endings(text: str) -> str:
        """Convert Windows (\\r\\n) and old Mac (\\r) line endings to Unix (\\n)."""
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        return text

    @staticmethod
    def _remove_residual_boilerplate(text: str) -> str:
        """Remove lines matching known boilerplate patterns."""
        for pattern in _NORMALIZER_BOILERPLATE:
            text = pattern.sub("", text)
        return text

    @staticmethod
    def _remove_repeated_lines(text: str) -> str:
        """
        Detect and remove lines that appear suspiciously often.

        A line is removed if it appears >= _REPEATED_LINE_THRESHOLD times
        AND it is short (< 80 chars) – characteristic of headers/footers.
        Long repeated lines (e.g., repeated regulatory clauses) are kept.
        """
        lines = text.splitlines()
        from collections import Counter
        counts = Counter(l.strip() for l in lines if l.strip())

        repeated = {
            line for line, count in counts.items()
            if count >= _REPEATED_LINE_THRESHOLD and len(line) < 80
        }

        if repeated:
            filtered = [
                line for line in lines
                if line.strip() not in repeated
            ]
            return "\n".join(filtered)

        return text

    @staticmethod
    def _normalize_inline_whitespace(text: str) -> str:
        """
        Collapse consecutive spaces/tabs within each line.

        Preserves leading indentation (for tabular content) but collapses
        double/triple spaces that arise from PDF extraction.
        """
        lines = text.splitlines()
        normalised: list[str] = []
        for line in lines:
            # Preserve leading whitespace, normalise the rest
            leading = len(line) - len(line.lstrip())
            indent = line[:leading]
            body = re.sub(r"[ \t]{2,}", " ", line[leading:])
            normalised.append(indent + body)
        return "\n".join(normalised)

    @staticmethod
    def _collapse_blank_lines(text: str) -> str:
        """Reduce 3+ consecutive blank lines to exactly 2."""
        return re.sub(r"\n{3,}", "\n\n", text)
