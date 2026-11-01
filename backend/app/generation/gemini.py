"""
app/generation/gemini.py – Gemini LLM client for FinRule AI.

A reusable, lazy-initialised wrapper around google-generativeai.
Keeps all LLM plumbing in one place so the RAG pipeline stays clean.
"""
from __future__ import annotations

import logging

from app.generation.prompt import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


class LLMConfigError(RuntimeError):
    """Raised when the Gemini API key is missing or misconfigured."""


class GeminiClient:
    """
    Thin, singleton-friendly wrapper around the Gemini generative model.

    Usage
    -----
    client = GeminiClient()
    answer = client.generate(question="…", context="…")
    """

    def __init__(self, model_name: str | None = None) -> None:
        from app.config import settings

        self._model_name: str = model_name or settings.llm_model
        self._model = None  # lazy init

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_model(self):
        """Load and configure the Gemini model (called once on first use)."""
        if self._model is not None:
            return self._model

        import google.generativeai as genai
        from app.config import settings

        if not settings.gemini_api_key:
            raise LLMConfigError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file: GEMINI_API_KEY=your_key_here"
            )

        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=SYSTEM_PROMPT,
        )
        logger.info("Gemini model loaded: %s", self._model_name)
        return self._model

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, question: str, context: str) -> str:
        """
        Generate a grounded answer from the question and retrieved context.

        Parameters
        ----------
        question:
            Raw user question string.
        context:
            Formatted block of retrieved regulatory provisions.

        Returns
        -------
        str  – The model's text response, stripped of leading/trailing whitespace.
        """
        model = self._ensure_model()
        prompt = build_user_prompt(question=question, context=context)

        logger.debug("Sending prompt to Gemini (%d chars).", len(prompt))
        response = model.generate_content(prompt)
        answer = response.text.strip()
        logger.debug("Gemini response received (%d chars).", len(answer))
        return answer


# ── Module-level singleton (optional convenience) ─────────────────────────────

_default_client: GeminiClient | None = None


def get_gemini_client() -> GeminiClient:
    """Return the shared GeminiClient singleton."""
    global _default_client
    if _default_client is None:
        _default_client = GeminiClient()
    return _default_client
