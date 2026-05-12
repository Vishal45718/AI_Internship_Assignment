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
from src.retrieval.sentence_evidence import (
    dedupe_sentences,
    format_sentence_evidence_blocks,
    pack_evidence_sentences,
    score_sentences_for_query,
)
from src.llm.client import create_llm_client
from src.llm.prompts import (
    RAG_SYSTEM_PROMPT,
    FALLBACK_RESPONSE,
    INSUFFICIENT_DOCUMENT_EVIDENCE,
    build_rag_user_message,
    format_evidence_chunk_block,
)
from src.llm.grounding import (
    assess_pre_generation_support,
    validate_post_generation,
    STRICT_REGENERATION_SYSTEM_SUFFIX,
    build_retrieval_corpus,
)
from src.config import get_settings

logger = logging.getLogger(__name__)

# Soft cap per chunk body when building LLM context (characters).
_CONTEXT_CHUNK_BODY_MAX_CHARS = 1200


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

        # Step 4: Enforce prompt token budget (trim lowest-ranked chunks first), sentence-level evidence
        chunks_for_llm, evidence_blocks, corpus_plain, sentence_meta = self._trim_chunks_for_prompt_budget(
            result.chunks, question, history
        )
        user_message = build_rag_user_message(evidence_blocks, question)
        system_prompt = RAG_SYSTEM_PROMPT

        print(f"Sentence evidence: mode={sentence_meta.get('mode')} packed={sentence_meta.get('packed_count')} focus={sentence_meta.get('focus')}")

        allow_llm, block_reasons, pre_debug = assess_pre_generation_support(
            question,
            chunks_for_llm,
            result.top_score,
            self._settings.rerank_score_threshold,
        )
        grounding_triggers: list[dict[str, Any]] = []
        if not allow_llm:
            grounding_triggers.append({"type": "pre_generation_insufficient", "reasons": block_reasons})
            self._print_grounding_report(
                chunks_for_llm,
                grounding_confidence=0.0,
                post_validation=None,
                triggers=grounding_triggers,
                regeneration_attempts=0,
                sentence_meta=sentence_meta,
            )
            return {
                "answer": INSUFFICIENT_DOCUMENT_EVIDENCE,
                "sources": self._dedupe_sources(chunks_for_llm),
                "status": "insufficient_evidence",
                "confidence": 0.0,
                "grounding": {
                    "pre_generation": pre_debug,
                    "blocked": True,
                    "block_reasons": block_reasons,
                    "triggers": grounding_triggers,
                },
            }

        prompt_tokens = self._estimate_prompt_tokens(system_prompt, user_message, history)
        print(f"Estimated prompt tokens: {prompt_tokens}")

        # Step 4b: Compute dynamic max_tokens to prevent 402 errors
        safe_max_tokens = self._compute_safe_max_tokens(prompt_tokens)
        print(f"Config max_tokens: {self._settings.llm_max_output_tokens} -> Safe max_tokens: {safe_max_tokens}")
        logger.info(
            "RAG prompt budget: estimated_prompt_tokens=%d config_max_output_tokens=%d safe_max_output_tokens=%d context_chunks=%d",
            prompt_tokens,
            self._settings.llm_max_output_tokens,
            safe_max_tokens,
            len(chunks_for_llm),
        )

        # Step 5: Generate answer from LLM
        try:
            answer = self._llm.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                history=history,
                max_tokens=safe_max_tokens,
            )
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return {
                "answer": f"Error generating answer: {exc}",
                "sources": [],
                "status": "error",
            }

        post_val = validate_post_generation(answer, question, corpus_plain, result.top_score)
        regenerated = False
        regeneration_attempts = 1
        if post_val.get("regenerate"):
            try:
                answer = self._llm.generate(
                    system_prompt=system_prompt + STRICT_REGENERATION_SYSTEM_SUFFIX,
                    user_message=user_message,
                    history=history,
                    max_tokens=safe_max_tokens,
                )
                regenerated = True
                regeneration_attempts = 2
                post_val = validate_post_generation(answer, question, corpus_plain, result.top_score)
            except Exception as exc:
                logger.error("LLM regeneration failed: %s", exc)
                answer = INSUFFICIENT_DOCUMENT_EVIDENCE
                grounding_triggers.append({"type": "regeneration_error", "error": str(exc)})
                post_val = {
                    "ok": True,
                    "grounding_confidence": 0.0,
                    "unsupported_terms": [],
                    "protected_violations": [],
                    "unsupported_ratio": 0.0,
                    "evidence_overlap_ratio": 0.0,
                    "evidence_overlap_missing": [],
                    "regenerate": False,
                    "is_insufficient_disclaimer": True,
                }

        if not post_val.get("ok") and not post_val.get("is_insufficient_disclaimer"):
            grounding_triggers.append({"type": "post_generation_unsupported", "validation": post_val})
            answer = INSUFFICIENT_DOCUMENT_EVIDENCE

        sources = self._dedupe_sources(chunks_for_llm)
        self._print_grounding_report(
            chunks_for_llm,
            grounding_confidence=float(post_val.get("grounding_confidence", 0.0)),
            post_validation=post_val,
            triggers=grounding_triggers,
            regeneration_attempts=regeneration_attempts,
            sentence_meta=sentence_meta,
        )

        return {
            "answer": answer,
            "sources": sources,
            "status": "success",
            "confidence": float(post_val.get("grounding_confidence", round(result.top_score, 3))),
            "grounding": {
                "pre_generation": pre_debug,
                "post_generation": post_val,
                "regenerated": regenerated,
                "regeneration_attempts": regeneration_attempts,
                "sentence_evidence": sentence_meta,
                "triggers": grounding_triggers,
            },
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

        result = self._retriever.retrieve(query=question, expand_context=True)
        if not result.passed_threshold:
            yield "token", FALLBACK_RESPONSE
            return

        print(f"[stream] Retrieved chunk count: {result.retrieved_chunk_count}")
        print(f"[stream] Reranked chunk count: {result.reranked_chunk_count}")
        print(f"[stream] Expanded chunk count: {result.expanded_chunk_count}")
        print(f"[stream] Final chunk count: {result.final_chunk_count}")

        chunks_for_llm, evidence_blocks, corpus_plain, sentence_meta = self._trim_chunks_for_prompt_budget(
            result.chunks, question, history
        )
        user_message = build_rag_user_message(evidence_blocks, question)
        system_prompt = RAG_SYSTEM_PROMPT

        print(
            f"[stream] Sentence evidence: mode={sentence_meta.get('mode')} "
            f"packed={sentence_meta.get('packed_count')} focus={sentence_meta.get('focus')}"
        )

        allow_llm, block_reasons, pre_debug = assess_pre_generation_support(
            question,
            chunks_for_llm,
            result.top_score,
            self._settings.rerank_score_threshold,
        )
        grounding_triggers: list[dict[str, Any]] = []
        if not allow_llm:
            grounding_triggers.append({"type": "pre_generation_insufficient", "reasons": block_reasons})
            self._print_grounding_report(
                chunks_for_llm,
                grounding_confidence=0.0,
                post_validation=None,
                triggers=grounding_triggers,
                regeneration_attempts=0,
                sentence_meta=sentence_meta,
                prefix="[stream] ",
            )
            yield "sources", self._dedupe_sources(chunks_for_llm)
            yield "token", INSUFFICIENT_DOCUMENT_EVIDENCE
            return

        sources = self._dedupe_sources(chunks_for_llm)
        yield "sources", sources

        prompt_tokens = self._estimate_prompt_tokens(system_prompt, user_message, history)

        # Compute dynamic max_tokens to prevent 402 errors
        safe_max_tokens = self._compute_safe_max_tokens(prompt_tokens)
        print(f"[stream] Estimated prompt tokens: {prompt_tokens} -> Safe max_tokens: {safe_max_tokens}")
        logger.info(
            "RAG stream prompt budget: estimated_prompt_tokens=%d config_max_output_tokens=%d safe_max_output_tokens=%d context_chunks=%d",
            prompt_tokens,
            self._settings.llm_max_output_tokens,
            safe_max_tokens,
            len(chunks_for_llm),
        )

        try:
            answer = self._llm.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                history=history,
                max_tokens=safe_max_tokens,
            )
        except Exception as exc:
            logger.error("LLM generation failed (stream path): %s", exc)
            yield "error", f"Error generating answer: {exc}"
            return

        post_val = validate_post_generation(answer, question, corpus_plain, result.top_score)
        regeneration_attempts = 1
        if post_val.get("regenerate"):
            try:
                answer = self._llm.generate(
                    system_prompt=system_prompt + STRICT_REGENERATION_SYSTEM_SUFFIX,
                    user_message=user_message,
                    history=history,
                    max_tokens=safe_max_tokens,
                )
                regeneration_attempts = 2
                post_val = validate_post_generation(answer, question, corpus_plain, result.top_score)
            except Exception as exc:
                logger.error("LLM regeneration failed (stream path): %s", exc)
                answer = INSUFFICIENT_DOCUMENT_EVIDENCE
                grounding_triggers.append({"type": "regeneration_error", "error": str(exc)})
                post_val = {
                    "ok": True,
                    "grounding_confidence": 0.0,
                    "unsupported_terms": [],
                    "protected_violations": [],
                    "unsupported_ratio": 0.0,
                    "evidence_overlap_ratio": 0.0,
                    "evidence_overlap_missing": [],
                    "regenerate": False,
                    "is_insufficient_disclaimer": True,
                }

        if not post_val.get("ok") and not post_val.get("is_insufficient_disclaimer"):
            grounding_triggers.append({"type": "post_generation_unsupported", "validation": post_val})
            answer = INSUFFICIENT_DOCUMENT_EVIDENCE

        self._print_grounding_report(
            chunks_for_llm,
            grounding_confidence=float(post_val.get("grounding_confidence", 0.0)),
            post_validation=post_val,
            triggers=grounding_triggers,
            regeneration_attempts=regeneration_attempts,
            sentence_meta=sentence_meta,
            prefix="[stream] ",
        )

        # Emit final answer in small chunks (generation was validated as a whole)
        step = 48
        for i in range(0, len(answer), step):
            yield "token", answer[i : i + step]

    @staticmethod
    def _print_grounding_report(
        chunks_for_llm: list,
        grounding_confidence: float,
        post_validation: dict[str, Any] | None,
        triggers: list[dict[str, Any]],
        regeneration_attempts: int,
        sentence_meta: dict[str, Any] | None = None,
        prefix: str = "",
    ) -> None:
        evidence_summary = [
            {
                "chunk_id": c.chunk_id,
                "page": c.page_number,
                "preview": (c.content[:120] + "…") if len(c.content) > 120 else c.content,
            }
            for c in sorted(chunks_for_llm, key=lambda x: (x.page_number or 0, x.chunk_index))
        ]
        print(f"{prefix}[grounding] retrieved_chunk_pool: {evidence_summary}")
        if sentence_meta:
            print(f"{prefix}[grounding] sentence_evidence_mode: {sentence_meta.get('mode')} focus={sentence_meta.get('focus')}")
            sel = sentence_meta.get("selected") or []
            print(f"{prefix}[grounding] selected_evidence_sentences ({len(sel)}): {sel}")
            print(
                f"{prefix}[grounding] definition_pattern_boost_hits: "
                f"{sentence_meta.get('definition_pattern_hits')} sentence_count={sentence_meta.get('sentence_count')}"
            )
        print(f"{prefix}[grounding] regeneration_attempts: {regeneration_attempts}")
        print(f"{prefix}[grounding] grounding_confidence: {grounding_confidence:.3f}")
        if post_validation is not None:
            print(
                f"{prefix}[grounding] unsupported_generated_terms: {post_validation.get('unsupported_terms')}"
            )
            print(
                f"{prefix}[grounding] unsupported_noun_phrases: {post_validation.get('unsupported_noun_phrases')}"
            )
            print(
                f"{prefix}[grounding] hallucination_detection_triggers: "
                f"{post_validation.get('hallucination_triggers')}"
            )
            print(
                f"{prefix}[grounding] evidence_overlap_ratio: {post_validation.get('evidence_overlap_ratio')} "
                f"missing={post_validation.get('evidence_overlap_missing')}"
            )
            print(
                f"{prefix}[grounding] unsupported_claim_detection: "
                f"ratio={post_validation.get('unsupported_ratio')} "
                f"protected_violations={post_validation.get('protected_violations')}"
            )
        else:
            print(f"{prefix}[grounding] post_generation_checks: (skipped — blocked before generation)")
        print(f"{prefix}[grounding] hallucination_fallback_triggers: {triggers}")

    def _build_evidence_from_chunks(self, question: str, chunks: list) -> tuple[str, str, dict[str, Any]]:
        """Sentence-level rerank + pattern boosts; fallback to chunk blocks if empty."""
        if not chunks:
            return "", "", {}

        max_chars = max(400, self._settings.max_context_tokens * 4 - 500)
        scored, dbg = score_sentences_for_query(question, chunks, self._retriever._reranker)
        scored = dedupe_sentences(scored)
        packed, corpus = pack_evidence_sentences(scored, max_chars)

        if not packed:
            fb = self._format_chunk_evidence_fallback(chunks)
            cp = build_retrieval_corpus(chunks)
            return fb, cp, {
                **dbg,
                "mode": "chunk_fallback",
                "packed_count": 0,
                "selected": [],
            }

        blocks = format_sentence_evidence_blocks(packed)
        meta = {
            **dbg,
            "mode": "sentence",
            "packed_count": len(packed),
            "selected": [
                {
                    "preview": (p.text[:160] + "…") if len(p.text) > 160 else p.text,
                    "sentence_rerank_score": round(p.rerank_score, 4),
                    "final_score": round(p.final_score, 4),
                    "definition_boost": round(p.definition_boost, 4),
                    "definition_pattern_labels": p.definition_labels,
                    "query_type_boost": round(p.query_type_boost, 4),
                    "query_type_tags": p.query_type_tags,
                    "page": p.page_display,
                    "chunk_id": p.chunk_id,
                }
                for p in packed
            ],
        }
        return blocks, corpus, meta

    def _format_chunk_evidence_fallback(self, chunks: list) -> str:
        """Coarse evidence blocks when sentence packing yields nothing."""
        if not chunks:
            return ""

        ordered = sorted(chunks, key=lambda c: (c.page_number or 0, c.chunk_index))
        formatted: list[str] = []
        total_chars = 0
        max_context_chars = self._settings.max_context_tokens * 4

        for i, chunk in enumerate(ordered, start=1):
            body = chunk.content
            if len(body) > _CONTEXT_CHUNK_BODY_MAX_CHARS:
                body = body[:_CONTEXT_CHUNK_BODY_MAX_CHARS] + "…"
            page_display = str(chunk.page_number) if chunk.page_number is not None else "?"
            chunk_str = format_evidence_chunk_block(i, page_display, body)
            if total_chars + len(chunk_str) > max_context_chars:
                break
            formatted.append(chunk_str)
            total_chars += len(chunk_str)

        return "\n\n".join(formatted)

    def _compute_safe_max_tokens(self, estimated_prompt_tokens: int) -> int:
        """
        Compute safe max_tokens to prevent 402 credit errors.
        
        Assumes typical OpenRouter credit cost:
        - Input: ~1 credit per 1000 tokens
        - Output: ~3 credits per 1000 tokens
        
        Safety buffer: reserve 20% margin to account for token estimation variance.
        """
        # Hard minimum and maximum
        min_output_tokens = 64
        max_output_tokens = self._settings.llm_max_output_tokens
        
        # Estimate total request cost if we use max_output_tokens
        # This is a rough heuristic; actual costs vary by model
        estimated_input_cost = max(estimated_prompt_tokens // 1000, 1)
        estimated_output_cost_per_token = 0.003  # 3 credits per 1000 output tokens
        
        # If prompt is very large, reduce output tokens aggressively
        if estimated_prompt_tokens > 2000:
            safe_output = min(max_output_tokens, 80)
        elif estimated_prompt_tokens > 1500:
            safe_output = min(max_output_tokens, 100)
        elif estimated_prompt_tokens > 1000:
            safe_output = min(max_output_tokens, 120)
        else:
            safe_output = max_output_tokens
        
        return max(min_output_tokens, safe_output)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _estimate_prompt_tokens(
        self,
        system_prompt: str,
        user_message: str,
        history: list[dict[str, str]] | None,
    ) -> int:
        total = self._estimate_tokens(system_prompt) + self._estimate_tokens(user_message)
        if history:
            for turn in history:
                total += self._estimate_tokens(turn.get("content", ""))
        return total

    def _trim_chunks_for_prompt_budget(
        self,
        chunks: list,
        question: str,
        history: list[dict[str, str]] | None,
    ) -> tuple[list, str, str, dict[str, Any]]:
        """Remove lowest-ranked chunks until sentence-packed prompt fits max_context_tokens."""
        if not chunks:
            return [], "", "", {}

        ranked = sorted(
            chunks,
            key=lambda c: (c.rerank_score if c.rerank_score > 0 else 0.0, c.similarity_score),
            reverse=True,
        )
        working = list(ranked)
        budget = self._settings.max_context_tokens

        while working:
            ordered = sorted(working, key=lambda c: (c.page_number or 0, c.chunk_index))
            evidence_blocks, corpus_plain, meta = self._build_evidence_from_chunks(question, ordered)
            user_message = build_rag_user_message(evidence_blocks, question)
            if self._estimate_prompt_tokens(RAG_SYSTEM_PROMPT, user_message, history) <= budget:
                return ordered, evidence_blocks, corpus_plain, meta
            working.pop()

        ordered = sorted(ranked[:1], key=lambda c: (c.page_number or 0, c.chunk_index))
        eb, cp, m = self._build_evidence_from_chunks(question, ordered)
        return ordered, eb, cp, m

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
