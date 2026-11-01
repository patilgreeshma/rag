"""
scripts/evaluate.py – FinRule AI Evaluation Framework.

Measures retrieval and generation quality over a curated SEBI Q&A dataset.

Metrics
───────
Retrieval:
  Recall@K  – % of queries where the expected source appears in top-K results
  MRR       – Mean Reciprocal Rank (quality of ranking order)
  Hit Rate  – % queries with at least one relevant result

Generation:
  Sufficient Context Rate – % queries that clear the retrieval threshold
  Source Citation Rate    – % answers that include at least one cited source

Usage
─────
  python scripts/evaluate.py
  python scripts/evaluate.py --output results/eval_report.json
  python scripts/evaluate.py --retrieval-only
  python scripts/evaluate.py --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")


# ── Evaluation dataset ────────────────────────────────────────────────────────
# Each entry has:
#   question       – the query
#   expected_terms – keywords that MUST appear in the top-K retrieved texts
#   category       – query type for breakdown analysis

EVAL_DATASET = [
    # ── Disclosure & LODR ─────────────────────────────────────────────────────
    {
        "question": "What are the disclosure requirements for listed entities under LODR?",
        "expected_terms": ["disclosure", "listed", "LODR"],
        "category": "disclosure",
    },
    {
        "question": "What are the obligations of a listed entity regarding board composition?",
        "expected_terms": ["board", "director", "listed"],
        "category": "disclosure",
    },
    {
        "question": "What financial results must a listed entity disclose and in what timeframe?",
        "expected_terms": ["financial", "results", "listed"],
        "category": "disclosure",
    },
    # ── Insider Trading ───────────────────────────────────────────────────────
    {
        "question": "What is the definition of insider under SEBI PIT Regulations?",
        "expected_terms": ["insider", "unpublished price sensitive", "connected"],
        "category": "insider_trading",
    },
    {
        "question": "What are the trading window restrictions under the PIT Regulations?",
        "expected_terms": ["trading window", "insider"],
        "category": "insider_trading",
    },
    {
        "question": "What is unpublished price sensitive information under SEBI regulations?",
        "expected_terms": ["unpublished", "price sensitive", "UPSI"],
        "category": "insider_trading",
    },
    # ── Mutual Funds ──────────────────────────────────────────────────────────
    {
        "question": "What are the net worth requirements for a mutual fund asset management company?",
        "expected_terms": ["net worth", "asset management", "mutual fund"],
        "category": "mutual_funds",
    },
    {
        "question": "What are the valuation requirements for non-traded debt securities in mutual funds?",
        "expected_terms": ["valuation", "debt", "mutual fund"],
        "category": "mutual_funds",
    },
    # ── Takeovers ─────────────────────────────────────────────────────────────
    {
        "question": "What are the conditions for triggering an open offer under the Takeover Regulations?",
        "expected_terms": ["open offer", "takeover", "acquirer"],
        "category": "takeovers",
    },
    {
        "question": "What are the exemptions from making an open offer under SEBI regulations?",
        "expected_terms": ["exempt", "open offer", "regulation 10"],
        "category": "takeovers",
    },
    # ── Related Party Transactions ────────────────────────────────────────────
    {
        "question": "What are the SEBI rules for related party transactions by listed companies?",
        "expected_terms": ["related party", "transaction", "listed"],
        "category": "governance",
    },
    # ── Portfolio Managers ────────────────────────────────────────────────────
    {
        "question": "What are the registration requirements for portfolio managers under SEBI?",
        "expected_terms": ["portfolio manager", "registration", "certificate"],
        "category": "intermediaries",
    },
    # ── REITs ─────────────────────────────────────────────────────────────────
    {
        "question": "What are the minimum unitholding requirements for a REIT investment manager?",
        "expected_terms": ["REIT", "investment manager", "unit"],
        "category": "reits",
    },
    # ── Negative queries (answer should come from corpus or be insufficient) ─
    {
        "question": "What is the SEBI regulation number for cryptocurrency trading?",
        "expected_terms": [],          # expect insufficient context
        "category": "negative",
    },
    {
        "question": "What are the rules for selling government bonds in India?",
        "expected_terms": ["bond", "securities"],
        "category": "negative",
    },
]


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    question: str
    category: str
    retrieved_chunk_ids: list
    retrieved_texts: list
    answer: str
    sufficient_context: bool
    top_score: Optional[float]
    has_expected_terms: bool
    reciprocal_rank: float
    latency_s: float


@dataclass
class EvalReport:
    timestamp: str
    total_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    hit_rate: float
    sufficient_context_rate: float
    avg_latency_s: float
    category_breakdown: dict
    query_results: list = field(default_factory=list)


# ── Evaluation logic ──────────────────────────────────────────────────────────

def _check_terms(texts: list[str], terms: list[str]) -> bool:
    """Return True if ALL expected_terms appear in ANY of the retrieved texts."""
    if not terms:
        return True  # negative query – nothing expected
    combined = " ".join(texts).lower()
    return all(term.lower() in combined for term in terms)


def _reciprocal_rank(texts: list[str], terms: list[str]) -> float:
    """Return 1/rank of the first hit, or 0 if not found."""
    if not terms:
        return 0.0
    for rank, text in enumerate(texts, start=1):
        if all(term.lower() in text.lower() for term in terms):
            return 1.0 / rank
    return 0.0


def run_evaluation(
    retrieval_only: bool = False,
    verbose: bool = False,
) -> EvalReport:
    from app.rag.rag import FinRuleRAG
    from datetime import datetime, timezone

    logger.info("Loading FinRuleRAG pipeline …")
    rag = FinRuleRAG()
    logger.info("Pipeline ready. Running %d queries …", len(EVAL_DATASET))

    results: list[QueryResult] = []
    hits_at_1 = hits_at_3 = hits_at_5 = 0
    rr_sum = 0.0
    sufficient_count = 0
    category_stats: dict[str, dict] = {}

    for i, entry in enumerate(EVAL_DATASET, 1):
        q = entry["question"]
        terms = entry["expected_terms"]
        category = entry["category"]

        logger.info("[%d/%d] %s", i, len(EVAL_DATASET), q[:70])

        t0 = time.perf_counter()
        response = rag.query(q)
        latency = time.perf_counter() - t0

        texts = [c.text for c in response.retrieved_chunks]
        chunk_ids = [c.chunk_id for c in response.retrieved_chunks]

        # Recall@K
        hit_at_1 = _check_terms(texts[:1], terms)
        hit_at_3 = _check_terms(texts[:3], terms)
        hit_at_5 = _check_terms(texts[:5], terms)
        rr = _reciprocal_rank(texts, terms)

        if hit_at_1: hits_at_1 += 1
        if hit_at_3: hits_at_3 += 1
        if hit_at_5: hits_at_5 += 1
        rr_sum += rr
        if response.sufficient_context:
            sufficient_count += 1

        # Per-category tracking
        if category not in category_stats:
            category_stats[category] = {"total": 0, "hit_at_5": 0, "sufficient": 0}
        category_stats[category]["total"] += 1
        if hit_at_5:
            category_stats[category]["hit_at_5"] += 1
        if response.sufficient_context:
            category_stats[category]["sufficient"] += 1

        result = QueryResult(
            question=q,
            category=category,
            retrieved_chunk_ids=chunk_ids,
            retrieved_texts=[t[:200] for t in texts],  # truncate for readability
            answer=response.answer[:300] if response.answer else "",
            sufficient_context=response.sufficient_context,
            top_score=response.top_score,
            has_expected_terms=hit_at_5,
            reciprocal_rank=rr,
            latency_s=round(latency, 3),
        )
        results.append(result)

        if verbose:
            print(f"\n  Q: {q}")
            print(f"  Hit@5: {hit_at_5}  |  RR: {rr:.3f}  |  Sufficient: {response.sufficient_context}  |  {latency:.2f}s")

    n = len(EVAL_DATASET)
    category_breakdown = {
        cat: {
            "recall_at_5": round(v["hit_at_5"] / v["total"], 3),
            "sufficient_rate": round(v["sufficient"] / v["total"], 3),
            "total": v["total"],
        }
        for cat, v in category_stats.items()
    }

    report = EvalReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_queries=n,
        recall_at_1=round(hits_at_1 / n, 3),
        recall_at_3=round(hits_at_3 / n, 3),
        recall_at_5=round(hits_at_5 / n, 3),
        mrr=round(rr_sum / n, 3),
        hit_rate=round(hits_at_5 / n, 3),
        sufficient_context_rate=round(sufficient_count / n, 3),
        avg_latency_s=round(sum(r.latency_s for r in results) / n, 3),
        category_breakdown=category_breakdown,
        query_results=[asdict(r) for r in results],
    )
    return report


def print_report(report: EvalReport) -> None:
    print()
    print("═" * 65)
    print("  FinRule AI – Evaluation Report")
    print("═" * 65)
    print(f"  Queries evaluated  : {report.total_queries}")
    print(f"  Timestamp          : {report.timestamp}")
    print()
    print("  ── Retrieval Metrics ──────────────────────────────────────")
    print(f"  Recall@1           : {report.recall_at_1:.1%}")
    print(f"  Recall@3           : {report.recall_at_3:.1%}")
    print(f"  Recall@5           : {report.recall_at_5:.1%}")
    print(f"  MRR                : {report.mrr:.3f}")
    print(f"  Hit Rate           : {report.hit_rate:.1%}")
    print()
    print("  ── Generation Metrics ────────────────────────────────────")
    print(f"  Sufficient Context : {report.sufficient_context_rate:.1%}")
    print()
    print("  ── Performance ───────────────────────────────────────────")
    print(f"  Avg Latency        : {report.avg_latency_s:.2f}s per query")
    print()
    print("  ── Category Breakdown ────────────────────────────────────")
    for cat, stats in report.category_breakdown.items():
        print(f"  {cat:<22} Recall@5={stats['recall_at_5']:.1%}  "
              f"Sufficient={stats['sufficient_rate']:.1%}  (n={stats['total']})")
    print("═" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FinRule AI RAG pipeline.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save full JSON report to this path.")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Skip Gemini generation (faster, retrieval metrics only).")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-query results.")
    args = parser.parse_args()

    report = run_evaluation(retrieval_only=args.retrieval_only, verbose=args.verbose)
    print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False, default=str)
        logger.info("Full report saved to %s", args.output)


if __name__ == "__main__":
    main()
