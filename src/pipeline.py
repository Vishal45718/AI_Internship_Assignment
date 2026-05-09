"""
src/pipeline.py — Clean RAG pipeline.

This is the central orchestrator that ties together all components:
  1. Document ingestion (load → chunk → embed → store)
  2. Query answering (retrieve → build prompt → generate → respond)

Replaces the previous complex agentic orchestration with a simple,
linear pipeline that's easy to understand and explain.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.ingestion.loaders import DocumentLoader, LoaderError
from src.ingestion.chunker import DocumentChunker
from src.embeddings.embedder import create_embedder
from src.vectordb.store import ChromaVectorStore
from src.retrieval.retriever import SemanticRetriever
from src.llm.client import create_llm_client
from src.llm.prompts import RAG_SYSTEM_PROMPT, CONTEXT_CHUNK_TEMPLATE, FALLBACK_RESPONSE
from src.config import get_settings

logger = logging.getLogger(__name__)

# Maximum characters of context to inject (prevents context window overflow)
MAX_CONTEXT_CHARS = 6000


class RAGPipeline:
    """
    Clean RAG pipeline for document Q&A.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest("data/raw/report.pdf")
        answer = pipeline.query("What is the return policy?")
    """

    def __init__(self) -> None:
        logger.info("Initializing RAG pipeline…")

        # Initialize all components
        self._embedder = create_embedder()
        self._store = ChromaVectorStore(embedder=self._embedder)
        self._retriever = SemanticRetriever(vector_store=self._store)
        self._llm = create_llm_client()
        self._loader = DocumentLoader()
        self._chunker = DocumentChunker()

        logger.info("RAG pipeline ready. LLM: %s", self._llm.model_name)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, path: str | Path) -> dict[str, Any]:
        """
        Ingest a single file or directory into the vector store.

        Args:
            path: Path to a file or directory.

        Returns:
            Dict with ingestion statistics.
        """
        p = Path(path)

        if p.is_dir():
            return self._ingest_directory(p)
        elif p.is_file():
            return self._ingest_file(p)
        else:
            return {"status": "error", "message": f"Path not found: {path}"}

    def _ingest_file(self, path: Path) -> dict[str, Any]:
        """Ingest a single file: load → chunk → embed → store."""
        try:
            # Load document(s) from file
            documents = self._loader.load(path)
            if not documents:
                return {"status": "warning", "message": f"No content in {path.name}", "chunks": 0}

            # Split into chunks
            chunks = self._chunker.chunk_many(documents)
            if not chunks:
                return {"status": "warning", "message": f"No chunks from {path.name}", "chunks": 0}

            # Embed and store
            self._store.add_chunks(chunks)

            return {
                "status": "success",
                "file": path.name,
                "chunks": len(chunks),
            }

        except LoaderError as exc:
            return {"status": "error", "file": path.name, "message": str(exc)}

    def _ingest_directory(self, directory: Path) -> dict[str, Any]:
        """Ingest all supported documents in a directory."""
        documents = self._loader.load_directory(directory, recursive=True)

        if not documents:
            return {"status": "warning", "message": "No documents found", "chunks": 0}

        chunks = self._chunker.chunk_many(documents)
        if chunks:
            self._store.add_chunks(chunks)

        return {
            "status": "success",
            "documents": len(documents),
            "chunks": len(chunks),
        }

    # ── Querying ──────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict[str, Any]:
        """
        Answer a question using the RAG pipeline.

        Pipeline steps:
          1. Check if vector store has data
          2. Retrieve relevant chunks with threshold filtering
          3. If no relevant chunks found → return fallback (no hallucination)
          4. Build grounded prompt with source citations
          5. Generate answer from LLM
          6. Return answer with source references

        Args:
            question: User's natural language question.

        Returns:
            Dict with answer, sources, and metadata.
        """
        question = question.strip()
        if not question:
            return {"answer": "Please provide a question.", "sources": [], "status": "error"}

        # Step 1: Check if we have any documents
        if self._store.count() == 0:
            return {
                "answer": "No documents have been ingested yet. "
                          "Please ingest documents first using the 'ingest' command.",
                "sources": [],
                "status": "empty_index",
            }

        # Step 2: Retrieve relevant chunks
        result = self._retriever.retrieve(query=question)

        # Step 3: Fallback if no relevant chunks found (hallucination prevention)
        if not result.passed_threshold:
            return {
                "answer": FALLBACK_RESPONSE,
                "sources": [],
                "status": "no_relevant_context",
                "top_score": result.top_score,
            }

        # Step 4: Build grounded prompt with source-labeled context
        context_str = self._format_context(result.chunks)
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context_str)

        # Step 5: Generate answer from LLM
        try:
            answer = self._llm.generate(
                system_prompt=system_prompt,
                user_message=f"Question: {question}",
            )
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return {
                "answer": f"Error generating answer: {exc}",
                "sources": [],
                "status": "error",
            }

        # Step 6: Return answer with sources
        sources = [
            {
                "file": chunk.source_file,
                "page": chunk.page_number,
                "score": round(chunk.similarity_score, 3),
                "preview": chunk.preview,
            }
            for chunk in result.chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
            "status": "success",
            "confidence": round(result.top_score, 3),
        }

    def _format_context(self, chunks: list) -> str:
        """Format retrieved chunks into the labeled context block for the LLM."""
        if not chunks:
            return "(No relevant context was retrieved.)"

        formatted: list[str] = []
        total_chars = 0

        for chunk in chunks:
            page_info = f", Page {chunk.page_number}" if chunk.page_number else ""
            chunk_str = CONTEXT_CHUNK_TEMPLATE.format(
                source_file=chunk.source_file,
                page_info=page_info,
                score=chunk.similarity_score,
                content=chunk.content,
            )
            if total_chars + len(chunk_str) > MAX_CONTEXT_CHARS:
                break
            formatted.append(chunk_str)
            total_chars += len(chunk_str)

        return "\n\n".join(formatted)

    # ── Index management ──────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return current index statistics."""
        sources = self._store.list_sources()
        return {
            "total_chunks": self._store.count(),
            "total_documents": len(sources),
            "documents": sources,
        }

    def reset_index(self) -> None:
        """Delete all indexed chunks. ⚠️ Destructive."""
        self._store.reset()

    def delete_document(self, source_file: str) -> int:
        """Remove all chunks for a specific document."""
        return self._store.delete_by_source(source_file)
