# FinRule AI — Project Report

---

## 1. Project Overview

**FinRule AI** is a production-grade **Retrieval-Augmented Generation (RAG)** system designed to answer natural-language questions about **SEBI (Securities and Exchange Board of India) financial regulations** with grounded, cited answers.

### Problem It Solves

Financial regulation research is time-consuming and error-prone. Legal text is dense, spread across hundreds of instruments (regulations, circulars, master circulars), and changes frequently. Professionals — compliance officers, lawyers, finance analysts — spend hours cross-referencing sources.

FinRule AI lets a user type a plain English question like:

> *"What are the conditions for triggering an open offer under Takeover Regulations?"*

…and get an answer that:
- Is **grounded exclusively in real SEBI provisions** (never hallucinated)
- **Cites the exact document title, section number, and official SEBI URL**
- Flags whether a cited provision is **in force or superseded**
- Clearly states when the corpus doesn't have enough information

### Data Source

The corpus is the **[Open India Law](https://github.com/Vaquill-AI/open-india-law)** SEBI dataset (`in_sebi_regulations.parquet`) — a pre-structured, provision-level snapshot of all SEBI regulations maintained by Vaquill AI. It contains:

| Stat | Value |
|---|---|
| Total provisions (rows) | ~88,310 |
| Unique instruments (acts/circulars) | ~1,145 |
| Source URL coverage | 100% |
| Corpus type | Legislation, regulations, circulars |
| Language | English |

---

## 2. System Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                         FinRule AI System                             │
├────────────────┬──────────────────────────────────────────────────────┤
│                │                                                      │
│  DATA LAYER    │  in_sebi_regulations.parquet                         │
│                │    ↓ open_india_law_loader.py                        │
│                │  RegulatoryChunk objects (88K)                       │
│                │    ↓ ingest.py orchestrator                          │
│                │  chunks.jsonl + documents.jsonl                      │
│                │    ↓ build_index.py                                  │
│                │  FAISS IndexFlatIP + metadata.json                   │
│                │                                                      │
├────────────────┼──────────────────────────────────────────────────────┤
│                │                                                      │
│  RETRIEVAL     │  User Question                                       │
│  LAYER         │    ↓                                                 │
│                │  ┌──────────────┐   ┌──────────────┐               │
│                │  │ Vector Search │   │  BM25 Search  │               │
│                │  │ (FAISS cosine)│   │ (rank_bm25)   │               │
│                │  └──────┬───────┘   └──────┬───────┘               │
│                │         └────────┬──────────┘                        │
│                │                  ↓                                   │
│                │     RRF Fusion (top 20 candidates)                   │
│                │                  ↓                                   │
│                │     Cross-Encoder Reranker (top 20 → top 5)          │
│                │                  ↓                                   │
│                │     Top-5 Provision Chunks                           │
│                │                                                      │
├────────────────┼──────────────────────────────────────────────────────┤
│                │                                                      │
│  GENERATION    │  Context Block (5 provisions + source metadata)      │
│  LAYER         │    ↓ strict no-hallucination system prompt           │
│                │  Gemini 1.5 Flash                                    │
│                │    ↓                                                 │
│                │  Answer + Cited Source URLs                          │
│                │                                                      │
├────────────────┼──────────────────────────────────────────────────────┤
│                │                                                      │
│  API LAYER     │  FastAPI  POST /query  →  RAGResponse                │
│                │           GET /health  →  HealthResponse             │
│                │                                                      │
├────────────────┼──────────────────────────────────────────────────────┤
│                │                                                      │
│  FRONTEND      │  HTML + CSS + JS chat UI                             │
│                │  Sidebar with example queries                        │
│                │  Renders answer + collapsible sources panel          │
│                │                                                      │
└────────────────┴──────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```
RAG/
├── .gitignore
├── README.md
│
├── backend/
│   ├── .env                       ← Secrets (not committed)
│   ├── .env.example               ← Template for contributors
│   ├── pytest.ini                 ← Test configuration
│   ├── requirements.txt           ← All Python dependencies
│   │
│   ├── app/                       ← Core application package
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── api/
│   │   ├── database/
│   │   ├── embeddings/
│   │   ├── generation/
│   │   ├── ingestion/
│   │   ├── rag/
│   │   ├── retrieval/
│   │   └── schemas/
│   │
│   ├── scripts/                   ← CLI pipeline scripts
│   │   ├── ingest_open_india_law.py
│   │   ├── build_index.py
│   │   ├── query_rag.py
│   │   └── evaluate.py
│   │
│   ├── tests/                     ← 68 unit + integration tests
│   │   ├── test_api.py
│   │   ├── test_rag_pipeline.py
│   │   ├── test_retrieval.py
│   │   ├── test_schemas.py
│   │   └── test_open_india_law_loader.py
│   │
│   └── data/
│       ├── raw/open_india_law/
│       │   └── in_sebi_regulations.parquet
│       └── processed/sebi/
│           ├── chunks.jsonl
│           ├── documents.jsonl
│           └── faiss_index/
│               ├── index.faiss
│               └── metadata.json
│
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```

---

## 4. File-by-File Breakdown

### 4.1 Configuration

#### `backend/app/config.py`
Central settings hub using **Pydantic `BaseSettings`**. All values are read from environment variables or the `.env` file. No module reads `os.environ` directly — they import `settings` from here.

| Setting | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `LLM_MODEL` | `gemini-1.5-flash` | Gemini model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers model |
| `EMBEDDING_DIM` | `384` | Vector dimension |
| `EMBEDDING_BATCH_SIZE` | `64` | Chunks per embedding batch |
| `TOP_K` | `5` | Final chunks sent to LLM |
| `RETRIEVAL_THRESHOLD` | `0.30` | Min cosine score to trigger LLM |
| `OIL_PARQUET_URL` | Vaquill CDN | Download URL for SEBI dataset |
| `FAISS_INDEX_PATH` | `data/processed/sebi/faiss_index` | FAISS files directory |

Also creates data directories (`mkdir`) on first import so scripts never fail on missing paths.

---

#### `backend/.env.example`
Template file committed to git showing all supported env vars with placeholder values. Developers copy this to `.env` and fill in their keys.

---

### 4.2 Data Layer — `app/ingestion/`

#### `open_india_law_loader.py`
The primary data loader. Reads `in_sebi_regulations.parquet` and converts each row into a `RegulatoryChunk` object.

**Key responsibilities:**
- `load_parquet(path)` — reads Parquet using PyArrow, validates the schema
- `validate_schema(df)` — raises `SchemaValidationError` if required columns (`chunk_id`, `title`, `text`, `source_url`, `act_id`) are missing
- `row_to_chunk(row)` — converts a single DataFrame row dict → `RegulatoryChunk`, handles `None`/`NaN`/`null` values safely via `_safe_str()`, `_safe_list()`, `_safe_bool()` helpers
- `dataframe_to_chunks(df)` — bulk conversion; detects and skips duplicate `chunk_id` values
- `dataset_stats(df)` — returns a summary dict (provision counts, section type breakdown, text length stats)

**Design contract:** Nothing is fabricated. If a field is absent in the Parquet, the corresponding chunk field is `None`.

---

#### `regulatory_models.py`
Defines the canonical Python dataclasses for regulatory data, source-agnostic by design.

| Class | Purpose |
|---|---|
| `DocumentSection` | One logical section of a document (heading, number, text, page, level, subsections) |
| `RegulatoryDocument` | Full normalised document (ID, regulator, type, dates, source URL, sections, extraction status) |
| `RegulatoryChunk` | A single provision chunk with full denormalised metadata for citation at retrieval time |

`RegulatoryChunk` is designed so that **no join to the parent document is needed at query time** — every field needed for citing (title, section_number, source_url, in_force) is stored directly on the chunk.

---

#### `regulatory_normalizer.py`
Cleans raw text extracted from regulatory documents. Applied to `RegulatoryDocument` objects.

**Normalisation passes (in order):**
1. Windows/Mac line endings → Unix (`\n`)
2. Remove residual boilerplate patterns (page numbers, "www.sebi.gov.in", "Confidential")
3. Remove repeated header/footer lines (lines appearing ≥3 times that are <80 chars)
4. Collapse multiple spaces/tabs within lines (preserves leading indentation)
5. Reduce 3+ consecutive blank lines to 2

**Design rule:** Never paraphrase. The stored text must be faithful to the source.

> Note: For the Open India Law corpus, text is already clean and this normalizer is lightweight. It's designed for future PDF/HTML scraping sources.

---

#### `ingest.py`
The **full pipeline orchestrator**. Ties together all ingestion steps into a single callable function.

```
run_full_pipeline()
    ↓ Step 1: Download SEBI Parquet (if missing)
    ↓ Step 2: Load + validate Parquet
    ↓ Step 3: Convert rows → RegulatoryChunks
    ↓ Step 4: Write chunks.jsonl + documents.jsonl
    ↓ Step 5: Generate embeddings + build FAISS index
    → Returns IngestionSummary (counts, paths, timing)
```

`IngestionSummary` dataclass holds: `total_rows`, `chunks_written`, `documents_written`, `duplicates_skipped`, `errors`, `faiss_vectors`, `faiss_built`, `elapsed_seconds`, per-step timing.

Parameters: `download`, `build_index`, `limit`, `in_force_only`, `embedding_batch_size`.

---

### 4.3 Database Layer — `app/database/`

FinRule AI uses **FAISS flat-file storage** — no SQL database or server required. The "database layer" manages two files:
- `faiss_index/index.faiss` — FAISS IndexFlatIP vector index
- `faiss_index/metadata.json` — parallel list of chunk metadata dicts

#### `connection.py`
`FAISSIndexManager` — application-wide singleton that manages the index lifecycle.

| Method/Property | Description |
|---|---|
| `index` | The loaded FAISS index (lazy-loaded on first access) |
| `metadata` | Parallel metadata list (lazy-loaded) |
| `total_vectors` | Number of indexed vectors (0 if not built) |
| `embedding_dim` | Vector dimension from the index |
| `is_ready` | True if index loaded successfully |
| `health()` | Returns a dict for the `/health` endpoint |

`get_index_manager()` returns the singleton instance. All retrieval components share the same loaded index — metadata is never loaded twice.

#### `models.py`
Pure Python dataclasses for serialisation consistency.

- `ChunkRecord` — canonical chunk representation with `to_dict()` (flattens metadata into top level for FAISS metadata.json) and `from_dict()` (reconstructs from flat dict). Also has `from_regulatory_chunk()` class method.
- `IndexStats` — lightweight health-check snapshot (`status`, `total_vectors`, `embedding_dim`, etc.)

---

### 4.4 Embeddings — `app/embeddings/`

#### `embedding_model.py`
Wrapper around **Sentence Transformers** (`all-MiniLM-L6-v2`).

| Method | Description |
|---|---|
| `load()` | Eagerly loads the model (pre-warming at startup) |
| `get_embedding_dim()` | Returns vector dimension (384 for default model) |
| `encode_text(text)` | Encodes a single string → `list[float]`. Empty input → zero vector |
| `encode_batch(texts, batch_size, show_progress)` | Batch encode with `normalize_embeddings=True` (unit vectors for cosine sim) |

**Key design:** Model is a singleton loaded once via `get_embedding_model()`. Empty/whitespace texts in a batch are replaced with zero vectors rather than crashing.

**Why `all-MiniLM-L6-v2`?**
- Fast CPU inference (64 chunks/second on MacBook)
- 384-dimensional vectors (128 MB index for 88K chunks)
- Good semantic quality for legal/regulatory text
- Fully local — no API calls

---

### 4.5 Retrieval Layer — `app/retrieval/`

This is the heart of the system. Three modules run in the hybrid pipeline.

#### `vector_search.py` — FAISS Semantic Search
`VectorSearcher` loads `index.faiss` and `metadata.json` lazily.

**`search(query, top_k)`:**
1. Embeds the query using the singleton `EmbeddingModel`
2. Runs FAISS `index.search(query_vector, top_k)`
3. Returns `list[VectorSearchResult]` sorted by cosine similarity score (inner product on normalised vectors = cosine)

**Best for:** Conceptual / semantic queries — *"disclosure requirements for listed companies"*, *"conditions for takeover"*

---

#### `bm25_search.py` — BM25 Keyword Search
`BM25Searcher` builds a `BM25Okapi` index from the chunk texts stored in `metadata.json`.

**Tokeniser:** lowercase, removes punctuation except hyphens (important for terms like `sub-section`), splits on whitespace.

**`search(query, top_k)`:**
1. Tokenises the query
2. Scores all 88K chunks via `BM25Okapi.get_scores()`
3. Returns top-k by BM25 score

**Best for:** Exact identifier queries — *"Regulation 33"*, *"LODR Schedule III"*, *"UPSI"*, *"section 26A(1)(b)"*

**Efficiency note:** BM25 reuses the same `metadata` list from `VectorSearcher` — no second file I/O.

---

#### `hybrid_search.py` — Reciprocal Rank Fusion
`HybridSearcher` runs both searches in sequence and fuses their ranked lists.

**Why RRF (not linear interpolation)?**
Vector scores (cosine, 0–1) and BM25 scores (0–∞) are on completely different scales. Normalising per-query is unstable. RRF uses only rank positions, making it scale-invariant.

**RRF formula:**
```
rrf_score(chunk) = Σ  1 / (k + rank_in_list)
                  lists
```
Default `k = 60`. Chunks appearing in both lists get contributions from both.

**`search(query, top_k)`:**
1. Fetch `max(top_k, candidates)` results from each searcher
2. Assign RRF scores from both ranked lists
3. Merge, sort by RRF score, return top-k `HybridResult` objects

---

#### `reranker.py` — Cross-Encoder Reranking
`CrossEncoderReranker` refines the top-20 hybrid candidates using a **Cross-Encoder** model (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

**Why a Cross-Encoder on top of hybrid search?**
- Bi-encoder models (used for FAISS) encode query and passage independently — they miss fine-grained relevance signals
- A Cross-Encoder reads `[query, passage]` jointly and outputs a single relevance logit
- Much more accurate but too slow to score all 88K chunks → applied only to top-20 candidates

**`rerank(query, candidates, top_k)`:**
1. Build `(query, chunk_text)` pairs for all candidates
2. Score all pairs with `CrossEncoder.predict(pairs)`
3. Sort by logit score, return top-k `RerankedChunk` objects

`RerankedChunk` carries `rerank_score`, `vector_score`, `bm25_score`, `hybrid_score` (all preserved for transparency) and convenience property accessors for `title`, `section_number`, `source_url`, `in_force`, `score`.

---

### 4.6 Generation Layer — `app/generation/`

#### `prompt.py`
All prompt strings live here — versioned in one place, testable, swappable without touching the RAG pipeline.

**`SYSTEM_PROMPT`** — tells Gemini:
- Answer using **only** the retrieved provisions
- Do **not** fabricate regulations, requirements, dates, sections, or citations
- Cite document title and section for every substantive claim
- If insufficient context, return the exact canned refusal string
- Warn users this is a point-in-time snapshot; verify at sebi.gov.in

**`USER_PROMPT_TEMPLATE`** — fills `{context}` (formatted provision blocks) and `{question}`.

**`INSUFFICIENT_CONTEXT_RESPONSE`** — the exact string returned when retrieval confidence is too low (avoids LLM invocation entirely).

---

#### `gemini.py`
`GeminiClient` — thin wrapper around `google-generativeai`.

- Lazy initialisation on first `generate()` call
- Validates `GEMINI_API_KEY` before making any API call
- Uses `GenerativeModel` with `system_instruction=SYSTEM_PROMPT`
- `generate(question, context)` → raw answer string

`get_gemini_client()` returns the module-level singleton.

---

### 4.7 RAG Pipeline — `app/rag/rag.py`

`FinRuleRAG` — the end-to-end pipeline class that orchestrates all three stages.

#### Return types
- `RetrievedChunk` — a single chunk with all metadata plus `vector_score`, `bm25_score`, `hybrid_score`, `score` (rerank score)
- `RAGResponse` — full response: `question`, `answer`, `sources`, `retrieved_chunks`, `sufficient_context`, `top_score`

#### `query(question)` — pipeline execution

```
Stage 1: HybridSearcher.search(question, top_k=20)
         → list[HybridResult]  (RRF-fused, top 20)

Stage 2: CrossEncoderReranker.rerank(question, hybrid_results, top_k=5)
         → list[RerankedChunk]  (reranked, top 5)

Stage 3: Threshold filter
         (for reranked results: threshold skipped — reranker logits are not cosine)

Stage 4: build_context(chunks)
         → structured text block with [PROVISION N] headers

Stage 5: GeminiClient.generate(question, context)
         → answer string

Stage 6: Deduplicate sources by source_url
         → list[dict] with title, section, url, score, in_force
```

**No-hallucination contract:**
- If `hybrid_results` is empty → return `INSUFFICIENT_CONTEXT_RESPONSE` without calling LLM
- If using vector-only mode (no reranker) and top score < `retrieval_threshold` → same canned response
- Source URLs taken from chunk metadata — never generated

`build_context(chunks)` formats each chunk as:
```
[PROVISION 1]
Document: SEBI LODR Regulations 2015
Section: 33
Heading: Financial Results
Status: IN FORCE
Source: https://sebi.gov.in/...

Full provision text...
────────────────────────────────────────────────────────────
[PROVISION 2]
...
```

---

### 4.8 API Layer

#### `app/main.py`
FastAPI application setup:
- App titled **"FinRule AI"** with version `1.0.0`
- CORS middleware allows all origins (suitable for local dev; restrict in production)
- Async lifespan context manager for startup/shutdown logging
- Includes the API router

#### `app/api/routes.py`
Two endpoints:

**`GET /health`**
- Calls `_get_rag()` to load the RAG pipeline
- Returns `HealthResponse(status, version, index_vectors)`
- Safe — returns `index_vectors=-1` if pipeline fails to load instead of crashing

**`POST /query`**
- Accepts `QueryRequest(question: str, top_k: int=5)`
- Calls `rag.query(question)` 
- Maps internal `RAGResponse` → `QueryResponse` (Pydantic)
- Returns HTTP 500 with detail message on pipeline errors

**`_get_rag()` singleton pattern:** `FinRuleRAG` is instantiated once on first request (lazy). Subsequent requests reuse the same instance — FAISS index and models stay loaded in memory.

#### `app/schemas/query.py`
Pydantic models for FastAPI:

| Model | Fields |
|---|---|
| `QueryRequest` | `question` (3–2000 chars), `top_k` (1–20, default 5) |
| `QueryResponse` | `question`, `answer`, `sources[]`, `retrieved_chunks[]`, `sufficient_context`, `top_score` |
| `SourceItem` | `title`, `section_number`, `section`, `document_type`, `source_url`, `in_force`, `score` |
| `ChunkItem` | `chunk_id`, `text`, `score`, `title`, `section_number`, `source_url`, `in_force`, `metadata` |
| `HealthResponse` | `status`, `version`, `index_vectors` |

FastAPI uses these for automatic request validation and OpenAPI docs generation (`/docs`, `/redoc`).

---

### 4.9 Scripts — `backend/scripts/`

#### `ingest_open_india_law.py`
CLI wrapper around `run_full_pipeline()`. Supports:
- `--download-only` — download Parquet without processing
- `--skip-download` — use existing Parquet
- `--limit N` — process only first N provisions
- `--in-force-only` — filter to active provisions
- `--verbose` — DEBUG logging

Prints a formatted summary table on completion.

---

#### `build_index.py`
Standalone FAISS index builder (can also be called via `run_full_pipeline(build_index=True)`).

**4-step process:**
1. Load `chunks.jsonl` → `(texts, metadata)` lists
2. Load `EmbeddingModel`, warm up
3. `encode_batch(texts)` → `float32` numpy array
4. Build `IndexFlatIP(dim)`, add vectors, write `index.faiss` + `metadata.json`

**macOS fix:** Sets `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` to prevent FAISS OpenMP segfaults on macOS + PyTorch MPS.

---

#### `query_rag.py`
Interactive CLI for testing the full pipeline without starting the API server.

```bash
python scripts/query_rag.py "What are LODR requirements?"   # single question
python scripts/query_rag.py --interactive                   # REPL mode
python scripts/query_rag.py --examples                      # run 5 sample questions
```

Prints formatted output: question, answer, sources with section numbers and URLs.

---

#### `evaluate.py`
Evaluation framework with a curated SEBI Q&A dataset.

**Metrics computed:**
- `Recall@5` — % of queries where expected terms appear in top-5 retrieved chunk texts
- `Recall@10` — same for top-10
- `MRR` (Mean Reciprocal Rank) — quality of ranking order
- `Hit Rate` — % queries with ≥1 relevant result
- `Sufficient Context Rate` — % queries that clear the retrieval threshold
- `Source Citation Rate` — % answers with ≥1 cited source

**Dataset covers 6 categories:**
`disclosure`, `insider_trading`, `mutual_funds`, `takeovers`, `governance`, `intermediaries` + `negative` queries (where the corpus should correctly return "insufficient context").

```bash
python scripts/evaluate.py
python scripts/evaluate.py --output results/eval_report.json
python scripts/evaluate.py --retrieval-only     # skip Gemini calls
```

---

### 4.10 Frontend — `frontend/`

#### `index.html`
Single-page chat application (no framework). Key sections:
- **Sidebar** — logo, "New Chat" button, 6 example question chips, corpus stats (document count, live vector count from `/health`)
- **Chat area** — scrollable message container, welcome screen with quick-start chips
- **Input area** — auto-resizing textarea, send button (disabled when empty), character count (0/2000), disclaimer linking to sebi.gov.in

#### `style.css`
Dark-themed, premium design:
- CSS custom properties for the design system (colors, spacing, radii)
- Google Fonts: **Inter** (UI) + **Fira Code** (code/citations)
- Glassmorphism sidebar with backdrop-blur
- Animated typing indicator (three bouncing dots) during API call
- Smooth message slide-in animations
- Source panel expander with transition
- Fully responsive — sidebar collapses on mobile

#### `app.js`
Handles all UI logic and API communication:
- Connects to `http://localhost:8000` on load — calls `/health` to show vector count and connection badge
- `sendQuestion(question)` — `POST /query`, renders streaming-like typewriter effect on answer
- Renders `SourceItem` objects as collapsible cards with title, section, URL, in-force badge, score
- Renders `ChunkItem` objects in a separate "Retrieved Context" panel (debug transparency)
- Example question buttons and sidebar chips pre-fill the input
- "New Chat" / clear button resets the conversation
- Auto-scrolls to latest message

---

### 4.11 Test Suite — `backend/tests/`

All 68 tests pass in ~3.5 seconds. No FAISS index, Gemini API key, or downloaded models needed — all external dependencies are mocked.

| File | Tests | What's Covered |
|---|---|---|
| `test_api.py` | 15 | `/health` (200, shape, vectors), `/query` (valid, empty, pipeline error, short question), response field types |
| `test_rag_pipeline.py` | 15 | `build_context()` formatting, `FinRuleRAG.query()` with mocked hybrid+reranker+LLM, insufficient context, empty question, source deduplication |
| `test_retrieval.py` | 19 | BM25 tokeniser (lowercase, punctuation, hyphens), `BM25Searcher` with mock corpus, `VectorSearcher` with mock FAISS, `HybridSearcher` RRF fusion, `CrossEncoderReranker` score ordering |
| `test_schemas.py` | 15 | `QueryRequest` validation (min/max length, top_k bounds), `QueryResponse` optional fields, `SourceItem`, `ChunkItem`, `HealthResponse` |
| `test_open_india_law_loader.py` | 4 | `validate_schema()` pass and fail, `row_to_chunk()` with missing fields, `dataframe_to_chunks()` duplicate detection |

---

## 5. Data Flow — End to End

### Ingestion (run once)

```
in_sebi_regulations.parquet
        ↓ load_parquet()          validates schema, reads 88K rows
        ↓ dataframe_to_chunks()   row → RegulatoryChunk (skips dupes)
        ↓ _write_chunks_jsonl()   chunks.jsonl (88,290 lines)
        ↓ _write_documents_jsonl() documents.jsonl (1,145 lines)
        ↓ _build_faiss_index()
              ↓ encode_batch()    all-MiniLM-L6-v2, 384-dim, L2-normalised
              ↓ IndexFlatIP.add() FAISS inner product index
        → index.faiss + metadata.json
```

### Query (every request)

```
POST /query  { question: "What is insider?" }
        ↓ FinRuleRAG.query()
        ↓ VectorSearcher.search(q, top_k=20)
              embed(q) → FAISS.search() → 20 VectorSearchResults
        ↓ BM25Searcher.search(q, top_k=20)
              tokenise(q) → BM25Okapi.get_scores() → 20 BM25Results
        ↓ HybridSearcher._fuse()
              RRF over both ranked lists → 20 HybridResults (sorted by RRF score)
        ↓ CrossEncoderReranker.rerank(q, top20, top_k=5)
              CrossEncoder.predict([(q, chunk)] × 20) → 5 RerankedChunks
        ↓ build_context(top5)
              → formatted string with [PROVISION N] blocks
        ↓ GeminiClient.generate(question, context)
              → grounded answer string
        ↓ Deduplicate sources by source_url
        → QueryResponse { answer, sources, retrieved_chunks, sufficient_context }
```

---

## 6. Key Design Decisions

| Decision | Rationale |
|---|---|
| **FAISS over PostgreSQL/pgvector** | Zero infrastructure — runs entirely from local files. No server to manage, simpler deployment, sufficient for 88K vectors (129 MB index). |
| **Open India Law Parquet over web scraping** | Pre-structured, 100% URL coverage, provision-level chunking, maintained dataset. Web scraping sebi.gov.in returned 404s for all URLs. |
| **RRF over linear interpolation** | Score-scale independent — cosine (0–1) and BM25 (0–∞) can't be linearly combined without per-query normalisation. RRF uses only ranks. |
| **Cross-Encoder only on top-20** | Cross-Encoders are accurate but O(n) with query — scoring all 88K chunks would take minutes. Top-20 reranking takes <1s on CPU. |
| **Strict system prompt** | Regulations are high-stakes. LLM is explicitly forbidden from using general knowledge. Below retrieval threshold, the system refuses rather than hallucinating. |
| **Source URL denormalised onto every chunk** | Enables direct citation in answers without any lookup — chunk carries everything needed for citation. |
| **Lazy model loading** | Embedding model (~90MB) and Cross-Encoder (~23MB) load on first request, not at startup, keeping the server boot fast. |
| **Singleton RAG instance** | `_get_rag()` in `routes.py` ensures FAISS index and models stay loaded in memory across requests — no reload per query. |

---

## 7. Dependencies

```
# Web framework
fastapi==0.111.0        HTTP API server
uvicorn[standard]       ASGI server

# Data
pandas>=2.0.0           Parquet DataFrame operations
pyarrow>=14.0.0         Parquet engine

# ML / Embeddings
sentence-transformers   Bi-encoder + Cross-Encoder models
torch==2.3.0            PyTorch backend (CPU)
numpy==1.26.4           Vector operations

# Vector Search
faiss-cpu>=1.7.4        Local FAISS index (no GPU needed)

# BM25
rank-bm25==0.2.2        BM25Okapi implementation

# LLM
google-generativeai     Gemini API client

# Config
pydantic==2.7.1         Data validation
pydantic-settings       .env file reading
python-dotenv           .env loading

# Testing
pytest                  Test runner
pytest-asyncio          Async test support
pytest-cov              Coverage reports
```

---

## 8. How to Run

```bash
# 1. Install dependencies
cd backend && python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add: GEMINI_API_KEY=your_key

# 3. Ingest data + build FAISS index (first time only)
python3 scripts/ingest_open_india_law.py
python3 scripts/build_index.py

# 4. Start API server
uvicorn app.main:app --reload --port 8000

# 5. Open frontend
open ../frontend/index.html
# OR
cd ../frontend && python3 -m http.server 3000

# 6. Run tests
python3 -m pytest tests/ -v
```

---

## 9. Roadmap

| Phase | Feature | Status |
|---|---|---|
| 1 | SEBI Parquet ingestion, FAISS, Hybrid RAG, Gemini, FastAPI, Chat UI, Tests | ✅ Complete |
| 2 | Run evaluation metrics (Recall@K, MRR, faithfulness scores) | 🔄 In progress |
| 3 | API authentication + rate limiting | 🔜 |
| 4 | RBI regulations corpus | 🔜 |
| 5 | Streaming responses (SSE / WebSocket) | 🔜 |
| 6 | Docker + deployment configuration | 🔜 |

---

## 10. Security Notes

- The LLM is explicitly instructed **not to fabricate** regulations, sections, or citations
- Source URLs come from chunk metadata — never generated by the LLM
- The corpus is a **point-in-time snapshot** — users are told to verify at sebi.gov.in for current authoritative text
- `GEMINI_API_KEY` is never logged or exposed in API responses
- CORS is open (`*`) for local development — **restrict to specific origins before any public deployment**
