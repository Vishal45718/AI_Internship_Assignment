"""
app/main.py — FastAPI application factory and entry point.

This file creates and configures the FastAPI application.
It registers all routers, middleware, and startup/shutdown events.

To run:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.middleware import LoggingMiddleware
from app.api.routes import query, ingest, health
from app.utils.logging import setup_logging
from configs.settings import get_settings

# ── Logging must be set up before anything else ───────────────────────────────
settings = get_settings()
setup_logging(level=settings.log_level)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    FastAPI application factory.
    Returns a fully configured application instance.
    """
    app = FastAPI(
        title="Agentic RAG System",
        description=(
            "A production-quality Agentic Retrieval-Augmented Generation system "
            "with hallucination prevention, semantic reranking, and source citations.\n\n"
            "## Key Features\n"
            "- **Multi-format ingestion**: PDF, TXT, Markdown, CSV\n"
            "- **Two-stage retrieval**: Semantic search + cross-encoder reranking\n"
            "- **Hallucination prevention**: Confidence thresholds + grounded prompts\n"
            "- **Source citations**: Every answer links back to original documents\n"
            "- **Agentic routing**: Intent classification before retrieval\n"
            "- **Conversation memory**: Multi-turn session support\n"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=[
            {"name": "Health", "description": "System status and diagnostics."},
            {"name": "Query", "description": "Submit questions to the RAG pipeline."},
            {"name": "Ingestion", "description": "Upload and manage documents in the vector index."},
        ],
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(ingest.router)

    # ── Root redirect ─────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/docs")

    # ── Startup event ─────────────────────────────────────────────────────────
    @app.on_event("startup")
    async def startup_event():
        logger.info(
            "Agentic RAG System starting up | LLM: %s | Embeddings: %s | VectorDB: %s",
            settings.llm_provider,
            settings.embedding_provider,
            settings.vector_db,
        )

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Agentic RAG System shutting down.")

    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
