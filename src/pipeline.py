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
        self._settings = get_settings()

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

    def query(self, question: str, mode: str = "document", history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """
        Answer a question using the RAG pipeline or general chat.

        Args:
            question: User's natural language question.
            mode: "document" for RAG, "general" for general AI chat.
            history: List of previous conversation turns [{"role": "user", "content": "..."}, ...]
        """
        question = question.strip()
        if not question:
            return {"answer": "Please provide a question.", "sources": [], "status": "error"}

        if mode == "general":
            return self._query_general(question, history)
        else:
            return self._query_document(question, history)

    def _query_general(self, question: str, history: list[dict[str, str]] | None) -> dict[str, Any]:
        system_prompt = "You are a helpful AI assistant. Answer the user's questions clearly and concisely."
        try:
            answer = self._llm.generate(
                system_prompt=system_prompt,
                user_message=question,
                history=history,
            )
            return {
                "answer": answer,
                "sources": [],
                "status": "success",
                "confidence": 1.0,
            }
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return {"answer": f"Error generating answer: {exc}", "sources": [], "status": "error"}

    def _query_document(self, question: str, history: list[dict[str, str]] | None) -> dict[str, Any]:

        # Step 1: Check if we have any documents
        if self._store.count() == 0:
            return {
                "answer": "No documents have been ingested yet. "
                          "Please ingest documents first using the 'ingest' command.",
                "sources": [],
                "status": "empty_index",
            }

        # Step 2: Retrieve relevant chunks
        result = self._retriever.retrieve(query=question, expand_context=True)

        print(f"Retrieved chunk count: {result.retrieved_chunk_count}")
        print(f"Reranked chunk count: {result.reranked_chunk_count}")
        print(f"Expanded chunk count: {result.expanded_chunk_count}")
        print(f"Final chunk count: {result.final_chunk_count}")
        print(f"Expanded context token count: {result.expanded_context_token_count}")
        print(f"Overlap reduction count: {result.overlap_reduction_count}")
        print(f"Selected chunk IDs: {[chunk.chunk_id for chunk in result.chunks]}")
        print(f"Page numbers: {[chunk.page_number for chunk in result.chunks]}")
        print(f"Rerank scores: {[round(chunk.rerank_score, 4) for chunk in result.chunks]}")

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
                history=history,
            )
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return {
                "answer": f"Error generating answer: {exc}",
                "sources": [],
                "status": "error",
            }

        # Step 6: Return answer with sources
        sources = self._dedupe_sources(result.chunks)

        return {
            "answer": answer,
            "sources": sources,
            "status": "success",
            "confidence": round(result.top_score, 3),
        }

    def stream_query(self, question: str, mode: str = "document", history: list[dict[str, str]] | None = None):
        """
        Yields (chunk_type, data) where chunk_type is "token" or "sources" or "error".
        """
        question = question.strip()
        if not question:
            yield "error", "Please provide a question."
            return

        if mode == "general":
            system_prompt = "You are a helpful AI assistant. Answer the user's questions clearly and concisely."
            try:
                for token in self._llm.stream(system_prompt, question, history):
                    yield "token", token
            except Exception as exc:
                logger.error("LLM streaming failed: %s", exc)
                yield "error", f"Error generating answer: {exc}"
            return

        # Document mode
        if self._store.count() == 0:
            yield "error", "No documents have been ingested yet. Please ingest documents first."
            return

        result = self._retriever.retrieve(query=question)
        if not result.passed_threshold:
            yield "token", FALLBACK_RESPONSE
            return

        sources = self._dedupe_sources(result.chunks)
        yield "sources", sources

        context_str = self._format_context(result.chunks)
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context_str)

        try:
            for token in self._llm.stream(system_prompt, f"Question: {question}", history):
                yield "token", token
        except Exception as exc:
            logger.error("LLM streaming failed: %s", exc)
            yield "error", f"Error generating answer: {exc}"

    def _format_context(self, chunks: list) -> str:
        """Format retrieved chunks into the labeled context block for the LLM."""
        if not chunks:
            return "(No relevant context was retrieved.)"

        formatted: list[str] = []
        total_chars = 0

        for chunk in chunks:
            score = chunk.rerank_score if chunk.rerank_score > 0 else chunk.similarity_score
            page_info = f", Page {chunk.page_number}" if chunk.page_number else ""
            chunk_str = CONTEXT_CHUNK_TEMPLATE.format(
                source_file=chunk.source_file,
                page_info=page_info,
                score=score,
                content=chunk.content,
            )
            max_context_chars = self._settings.max_context_tokens * 4
            if total_chars + len(chunk_str) > max_context_chars:
                break
            formatted.append(chunk_str)
            total_chars += len(chunk_str)

        return "\n\n".join(formatted)

    def _dedupe_sources(self, chunks: list) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, int | None], Any] = {}
        for chunk in chunks:
            key = (chunk.source_file, chunk.page_number)
            score = chunk.rerank_score if chunk.rerank_score > 0 else chunk.similarity_score
            existing = grouped.get(key)
            if existing is None or score > existing["_score"]:
                grouped[key] = {
                    "file": chunk.source_file,
                    "page": chunk.page_number,
                    "score": round(score, 3),
                    "preview": chunk.preview,
                    "_score": score,
                }

        deduped = list(grouped.values())
        deduped.sort(key=lambda item: item["_score"], reverse=True)
        top_sources = deduped[:5]
        for item in top_sources:
            item.pop("_score", None)
        return top_sources

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
