# System Design

## Overview

This repository implements a grounded Retrieval-Augmented Generation (RAG) pipeline for document question answering. The system is designed to retrieve evidence from uploaded documents, rerank candidate passages, and generate answers with a local LLM while minimizing unsupported statements.

The main objectives are:
- provide evidence-backed answers
- support multiple document formats
- preserve provenance for citations
- avoid premature guesswork when evidence is weak
- keep inference local using Ollama

## Core design principles

- **Separation of concerns**: ingestion, chunking, embedding, retrieval, reranking, grounding, and UI are distinct modules.
- **Evidence-first answers**: the prompt construction and validation pipeline emphasize retrieved passages over model memorization.
- **Controlled scope**: the system is optimized for documents and local models, not general-purpose chat across unconstrained domains.
- **Reproducible indexing**: deterministic chunk IDs ensure re-ingestion updates the same document rather than duplicating embeddings.

## Supported formats

- **PDF**: page-level extraction with page metadata preserved.
- **TXT**: whole file loaded as a single text document.
- **Markdown**: syntax stripped before embedding.
- **CSV**: each row is converted to compact structured text and treated as a single chunk document, preventing chunk explosion.

## Local inference strategy

The backend uses Ollama for local inference. Local inference was selected for:

- privacy and offline capability
- avoiding external LLM API costs
- demo stability on developer hardware

Tradeoffs:
- local models are smaller and may have weaker reasoning
- retrieval quality is more important because the model depends on evidence fed through the prompt

## Engineering decisions

- **CSV row-level ingestion**: CSV files are loaded row-by-row to prevent recursive chunking and reduce the number of generated chunks.
- **Timeout tuning**: upload and chat operations use longer client-side timeouts to accommodate embedding and inference latency.
- **Progress indicators**: the frontend reports parsing, embedding, and storage stages during uploads.
- **Inline grounding**: evidence blocks are included in prompts and backed by post-generation validation.

## Limitations and tradeoffs

- The system is not built for extreme scale; it targets small to medium collections of research documents.
- Local Ollama models reduce dependency on cloud services but limit answer quality for highly complex reasoning.
- PDF extraction quality depends on the source PDF; scanned documents and tables may produce noisy text.
- Retrieval is optimized for precision over recall in support of grounded responses.

## Recommended follow-up

Future work should focus on stronger rerankers, better table and CSV understanding, conversational memory, and larger local models when hardware allows.
