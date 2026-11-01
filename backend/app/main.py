from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FinRule AI starting up …")
    logger.info("Docs available at http://localhost:8000/docs")
    yield
    logger.info("FinRule AI shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FinRule AI",
    description=(
        "A Retrieval-Augmented Generation system for SEBI financial regulation research. "
        "Powered by hybrid vector+BM25 search, Cross-Encoder reranking, and Gemini LLM."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS (allow all origins for local development; restrict in production) ─────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include routers ───────────────────────────────────────────────────────────

app.include_router(router)

