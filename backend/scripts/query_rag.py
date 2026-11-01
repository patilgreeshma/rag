#!/usr/bin/env python3
"""
scripts/query_rag.py – FinRule AI Interactive Query Script

Test the end-to-end RAG pipeline with real SEBI regulatory questions.

Usage
─────
  python scripts/query_rag.py "What are the disclosure requirements under LODR?"
  python scripts/query_rag.py --interactive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.rag import FinRuleRAG


EXAMPLE_QUESTIONS = [
    "What are the disclosure requirements for listed entities under LODR?",
    "What is the definition of 'insider' under the Prohibition of Insider Trading Regulations?",
    "What are the net worth requirements for a portfolio manager?",
    "What obligations does a listed entity have regarding related party transactions?",
    "What are the conditions for an open offer under the Takeover Regulations?",
]


def format_response(response) -> str:
    lines = [
        "",
        "═" * 70,
        f"  QUESTION: {response.question}",
        "═" * 70,
        "",
        response.answer,
        "",
    ]

    if response.sufficient_context and response.sources:
        lines.append("─" * 70)
        lines.append(f"  SOURCES ({len(response.sources)}):")
        for i, src in enumerate(response.sources, 1):
            lines.append(f"  [{i}] {src.get('title', 'Unknown')[:80]}")
            if src.get("section_number"):
                lines.append(f"       Section: {src['section_number']}")
            if src.get("source_url"):
                lines.append(f"       URL: {src['source_url']}")
            status = "IN FORCE" if src.get("in_force") else "SUPERSEDED"
            lines.append(f"       Status: {status}  |  Score: {src.get('score', 0):.3f}")
    elif not response.sufficient_context:
        lines.append(f"  [No relevant provisions found above threshold. Top score: {response.top_score}]")

    lines.append("═" * 70)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query FinRule AI RAG.")
    parser.add_argument("question", nargs="?", help="Question to ask.")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode – keep asking questions.")
    parser.add_argument("--examples", action="store_true",
                        help="Run all example questions.")
    args = parser.parse_args()

    print("\nLoading FinRule AI RAG pipeline …", flush=True)
    rag = FinRuleRAG()
    print("Ready.\n")

    if args.examples:
        for q in EXAMPLE_QUESTIONS:
            response = rag.query(q)
            print(format_response(response))
        return

    if args.question:
        response = rag.query(args.question)
        print(format_response(response))
        return

    if args.interactive:
        print("FinRule AI – SEBI Regulatory Research Assistant")
        print("Type 'exit' or press Ctrl-C to quit.\n")
        while True:
            try:
                q = input("Question: ").strip()
                if q.lower() in ("exit", "quit", "q", ""):
                    break
                response = rag.query(q)
                print(format_response(response))
            except KeyboardInterrupt:
                print("\nBye.")
                break
        return

    parser.print_help()


if __name__ == "__main__":
    main()
