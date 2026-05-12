# System Design

## Design goals

- produce evidence-backed document QA
- support common document formats with format-aware ingestion
- prioritize retrieval precision and provenance
- enable local LLM inference with minimal external dependencies
- reduce unsupported answers through grounding and validation

## Architecture principles

- **modular design**: ingestion, retrieval, and generation are separate layers
- **format-specific handling**: CSV is treated differently from PDF/text
- **deterministic indexing**: chunk IDs enable safe document reingestion
- **grounded prompts**: model input is restricted to retrieved evidence

## Ingestion strategy

- **PDF**: extract page text, preserve page metadata
- **TXT**: ingest as a single document
- **Markdown**: strip formatting and keep readable text
- **CSV**: convert each row to prose, avoid recursive chunking

CSV rows are loaded as compact documents so 50 rows produce roughly 50–100 chunks, not thousands.

## Chunking logic

- standard text documents use recursive chunking with overlap
- sections are detected and preserved where possible
- chunk size and overlap are configured centrally
- CSV rows bypass recursive splitting and are stored as single chunks

## Embedding and storage

- embeddings are generated locally using SentenceTransformers
- ChromaDB stores vectors and metadata for each chunk
- metadata includes source file, chunk index, row/page context, and provenance

## Retrieval design

- query embedding drives semantic search
- hybrid retrieval is optional for keyword recall
- reranking selects the most relevant candidates for the prompt
- evidence blocks are built from the top candidates

## Grounding and fallback behavior

- prompts require answers to be based on retrieved text
- weak evidence triggers conservative fallback responses
- post-generation validation checks response overlap with source text

## Local inference rationale

Ollama is used to keep inference local and reduce reliance on external APIs. This supports privacy, offline evaluation, and predictable demo behavior. The tradeoff is a smaller model footprint, so retrieval and prompt quality are critical.

## Tradeoffs

- strong local inference requires strong retrieval
- PDF tables and scanned content are not a primary focus
- the system is designed for small to medium document collections, not massive corpora

## Summary

This project emphasizes practical engineering over broad generalization. It is optimized for grounded document QA in an internship-quality implementation with clear module boundaries and traceable evidence.
