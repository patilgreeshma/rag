# FinRule AI – SEBI Regulatory Research Assistant

> An AI-powered research assistant for **SEBI (Securities and Exchange Board of India)** regulations,
> built on the [Open India Law](https://github.com/Vaquill-AI/open-india-law) corpus with hybrid
> retrieval, Cross-Encoder reranking, and Gemini-powered generation.

---

## What It Does

FinRule AI lets you ask natural-language questions about SEBI regulations and get **grounded, cited answers** — drawn exclusively from real regulatory provisions, never hallucinated.

**Example questions:**
- *"What are the disclosure requirements for listed entities under LODR?"*
- *"What is the definition of insider under SEBI PIT Regulations?"*
- *"What conditions trigger an open offer under Takeover Regulations?"*

---

## Architecture

```
User Question  (Chat UI – frontend/)
      ↓
FastAPI  POST /query  (backend/app/api/routes.py)
      ↓
┌──────────────────────────────────────────────────────────────┐
│                    RAG Pipeline  (app/rag/rag.py)            │
│                                                              │
│  Stage 1 – Hybrid Search  (app/retrieval/)                   │
│   ├─ FAISS vector search   (cosine, all-MiniLM-L6-v2)        │
│   └─ BM25 keyword search   (rank_bm25, exact terms)          │
│           ↓ Reciprocal Rank Fusion  (top 20 candidates)      │
│                                                              │
│  Stage 2 – Cross-Encoder Reranking  (app/retrieval/)         │
│           ↓ ms-marco-MiniLM-L-6-v2  (top 20 → top 5)        │
│                                                              │
│  Stage 3 – Gemini Generation  (app/generation/)              │
│           ↓ gemini-1.5-flash  +  strict no-hallucination     │
│             prompt  +  source citations                      │
└──────────────────────────────────────────────────────────────┘
      ↓
Answer  +  Source URLs  (returned to UI)
```

**Storage:** FAISS flat-file index (`IndexFlatIP`) — no database server required.

---

## Codebase Map

```
cyberrag/
├── frontend/
│   ├── index.html          ← Chat UI
│   ├── style.css           ← Styling
│   └── app.js              ← API calls & message rendering
│
└── backend/
    ├── app/
    │   ├── main.py                    ← FastAPI app + CORS
    │   ├── config.py                  ← All settings (env-driven)
    │   │
    │   ├── api/
    │   │   └── routes.py              ← GET /health, POST /query
    │   │
    │   ├── database/
    │   │   ├── connection.py          ← FAISSIndexManager singleton
    │   │   └── models.py              ← ChunkRecord, IndexStats dataclasses
    │   │
    │   ├── ingestion/
    │   │   ├── ingest.py              ← Full pipeline orchestrator
    │   │   ├── open_india_law_loader.py  ← Parquet → RegulatoryChunk
    │   │   ├── regulatory_models.py   ← RegulatoryDocument, RegulatoryChunk
    │   │   └── regulatory_normalizer.py  ← Text normalisation
    │   │
    │   ├── embeddings/
    │   │   └── embedding_model.py     ← Sentence Transformers wrapper
    │   │
    │   ├── retrieval/
    │   │   ├── vector_search.py       ← FAISS cosine search
    │   │   ├── bm25_search.py         ← BM25Okapi keyword search
    │   │   ├── hybrid_search.py       ← RRF fusion
    │   │   └── reranker.py            ← Cross-Encoder reranker
    │   │
    │   ├── generation/
    │   │   ├── gemini.py              ← Gemini API client
    │   │   └── prompt.py              ← System + user prompt templates
    │   │
    │   ├── rag/
    │   │   └── rag.py                 ← End-to-end RAG pipeline class
    │   │
    │   └── schemas/
    │       └── query.py               ← Pydantic request/response schemas
    │
    ├── scripts/
    │   ├── ingest_open_india_law.py   ← Step 1: Download Parquet → chunks.jsonl
    │   ├── build_index.py             ← Step 2: chunks.jsonl → FAISS index
    │   ├── query_rag.py               ← CLI: ask questions without the API
    │   └── evaluate.py                ← Recall@K, MRR evaluation framework
    │
    ├── tests/
    │   ├── test_api.py
    │   ├── test_rag_pipeline.py
    │   ├── test_retrieval.py
    │   ├── test_schemas.py
    │   └── test_open_india_law_loader.py
    │
    ├── data/
    │   ├── raw/open_india_law/
    │   │   └── in_sebi_regulations.parquet   ← Source dataset (git-ignored)
    │   └── processed/sebi/
    │       ├── chunks.jsonl                  ← 88,310 provision chunks
    │       ├── documents.jsonl               ← Per-instrument summary
    │       └── faiss_index/
    │           ├── index.faiss               ← FAISS IndexFlatIP
    │           └── metadata.json             ← Parallel chunk metadata
    │
    ├── requirements.txt
    ├── pytest.ini
    └── .env.example
```

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Backend runtime |
| Node.js | 18+ | Serving the frontend (optional) |

> **No database server required.** FAISS runs entirely from local files.

---

## Setup

### 1 – Clone & create virtual environment

```bash
git clone <repo-url>
cd cyberrag/backend

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2 – Configure environment

```bash
cp .env.example .env
# Edit .env with your Gemini API key
```

Required values in `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

---

## Building the Index (First-Time Setup)

Run these two scripts once to prepare the data:

### Step 1 – Download the SEBI dataset and process it into chunks

```bash
# From backend/ with venv activated:

# Download Parquet + process all ~88K provisions
python scripts/ingest_open_india_law.py

# Or limit to 5000 provisions for a quick test
python scripts/ingest_open_india_law.py --limit 5000

# Only include in-force provisions
python scripts/ingest_open_india_law.py --in-force-only
```

Expected output:

```
  Rows in Parquet        : 88,310
  Chunks written         : 88,290
  Documents written      : 1,145
  Duplicates skipped     : 20
```

### Step 2 – Build the FAISS index

```bash
python scripts/build_index.py
```

Expected output:

```
  Embedding model  : all-MiniLM-L6-v2
  Chunks indexed   : 88,290
  FAISS index size : 129.4 MB
```

---

## Running the Backend

```bash
# From backend/ with venv activated:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server starts at http://localhost:8000
- **API docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

---

## Running the Frontend

Open `frontend/index.html` directly in your browser, or serve it locally:

```bash
cd cyberrag/frontend
python3 -m http.server 3000
# Open http://localhost:3000
```

---

## API Reference

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "index_vectors": 88290
}
```

### POST /query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the disclosure requirements under LODR?"}'
```

```json
{
  "question": "What are the disclosure requirements under LODR?",
  "answer": "Under the SEBI (Listing Obligations and Disclosure Requirements) Regulations 2015, listed entities are required to… [Regulation 33, Schedule III]",
  "sources": [
    {
      "title": "SEBI LODR Regulations 2015",
      "section_number": "33",
      "source_url": "https://www.sebi.gov.in/…",
      "in_force": true
    }
  ],
  "sufficient_context": true,
  "top_score": 8.42
}
```

---

## CLI Query Tool

Ask questions without starting the API server:

```bash
python scripts/query_rag.py "What are the conditions for triggering an open offer?"
```

---

## Running Tests

```bash
# From backend/ with venv activated:

# Run all tests
pytest

# Run with coverage report
pytest --cov=app --cov-report=term-missing

# Run a specific test file
pytest tests/test_rag_pipeline.py -v
```

---

## Evaluation

Measure retrieval quality (Recall@K, MRR) and generation quality over a curated Q&A set:

```bash
python scripts/evaluate.py
python scripts/evaluate.py --output results/eval_report.json
python scripts/evaluate.py --retrieval-only
```

---

## Hybrid Retrieval Design

### Why Hybrid (Vector + BM25)?

| Retrieval Mode | Good For | Weak At |
|---------------|----------|---------|
| **Vector only** | *"disclosure requirements for listed companies"* (conceptual) | Exact section numbers like "Regulation 33" |
| **BM25 only** | *"Regulation 33"*, *"LODR Schedule III"* (exact terms) | Paraphrased or conceptual queries |
| **Hybrid (RRF)** | Both conceptual and exact-match queries | — |

### Reciprocal Rank Fusion (RRF)

Both searchers run in parallel. Their ranked lists are merged with RRF:

```
rrf_score(chunk) = Σ  1 / (k + rank_in_list)    k = 60 (default)
                  lists
```

RRF is rank-based, so there is no need to normalise the incompatible score
scales of cosine similarity and BM25.

### Cross-Encoder Reranking

Top 20 hybrid candidates → `cross-encoder/ms-marco-MiniLM-L-6-v2` reads each
`(query, chunk)` pair jointly → relevance logit → top 5 sent to Gemini.

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | – | Google Gemini API key (required) |
| `LLM_MODEL` | `gemini-1.5-flash` | Gemini model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers dense model |
| `EMBEDDING_DIM` | `384` | Vector dimension (must match model) |
| `EMBEDDING_BATCH_SIZE` | `64` | Chunks per embedding batch |
| `TOP_K` | `5` | Final chunks sent to LLM |
| `RETRIEVAL_THRESHOLD` | `0.30` | Min cosine score to trigger LLM |
| `OIL_PARQUET_URL` | *(Vaquill CDN)* | Download URL for SEBI Parquet |
| `FAISS_INDEX_PATH` | `data/processed/sebi/faiss_index` | Index directory |

---

## Development Roadmap

| Phase | Features | Status |
|-------|----------|--------|
| **Phase 1** | SEBI Parquet ingestion, FAISS index, hybrid retrieval, Gemini, FastAPI, Chat UI | ✅ Complete |
| **Phase 2** | Evaluation metrics (Recall@K, MRR, faithfulness) | 🔄 In progress |
| **Phase 3** | API authentication & rate limiting | 🔜 |
| **Phase 4** | RBI regulations corpus | 🔜 |
| **Phase 5** | Streaming responses (SSE) | 🔜 |
| **Phase 6** | MITRE ATT&CK / CISA KEV integration | 🔜 |

---

## Security Notes

- This system is strictly for **informational and research purposes**.
- The LLM is explicitly instructed **not to fabricate** regulations, sections, or citations.
- Answers cite official SEBI source URLs taken from chunk metadata — never generated.
- API keys are never logged or exposed in responses.
- Always verify important regulatory matters against the current official text at [sebi.gov.in](https://www.sebi.gov.in/).

---

## License

MIT
