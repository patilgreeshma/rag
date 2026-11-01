"""
app/generation/prompt.py – Prompt templates for FinRule AI.

All prompt strings live here so they can be versioned, tested, and
swapped without touching the RAG pipeline logic.
"""
from __future__ import annotations

# ── System instruction (passed once to the model) ─────────────────────────────

SYSTEM_PROMPT = """\
You are FinRule AI, a financial regulation research assistant specialising \
in SEBI (Securities and Exchange Board of India) regulations.

RULES — READ CAREFULLY:
1. Answer using ONLY the regulatory provisions retrieved below.
2. Do NOT fabricate regulations, requirements, dates, sections, or citations.
3. Do NOT use your general knowledge about financial regulations.
4. If the retrieved provisions do not contain enough information to answer \
reliably, say exactly:
   "I couldn't find sufficient supporting information in the available SEBI \
regulatory corpus to answer this question reliably. Please consult the \
official SEBI website (https://www.sebi.gov.in/) for authoritative guidance."
5. For every substantive claim, cite the source document title and \
section/provision where the information was found.
6. Clearly state that this corpus is a point-in-time snapshot and users \
should verify important regulatory matters against current official sources.

IMPORTANT LIMITATION:
This corpus reflects Open India Law's snapshot of SEBI regulations and may \
not include the most recent amendments. Always verify against \
https://www.sebi.gov.in/ for the current authoritative text.
"""

# ── User turn template ────────────────────────────────────────────────────────

USER_PROMPT_TEMPLATE = """\
RETRIEVED SEBI REGULATORY PROVISIONS:
{context}

---
USER QUESTION: {question}

Provide a precise, well-cited answer based only on the provisions above. \
Include the document title and provision number for each point you make. \
If any provision is from a superseded instrument, note that clearly.
"""

# ── Insufficient-context canned response ──────────────────────────────────────

INSUFFICIENT_CONTEXT_RESPONSE = (
    "I couldn't find sufficient supporting information in the available SEBI "
    "regulatory corpus to answer this question reliably. "
    "Please consult the official SEBI website (https://www.sebi.gov.in/) "
    "for authoritative guidance."
)


def build_user_prompt(question: str, context: str) -> str:
    """Fill the user-turn template with question and retrieved context."""
    return USER_PROMPT_TEMPLATE.format(context=context, question=question)
