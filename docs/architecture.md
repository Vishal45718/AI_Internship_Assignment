# Architecture

## System flow

The architecture is built around a document-focused RAG pipeline with clear module boundaries.

```text
Frontend UI → FastAPI API → Ingestion → Embeddings → ChromaDB → Retrieval → Reranking → Prompt → Ollama
```

### Primary stages

- **Frontend**: handles uploads, chat input, streaming responses, and source display.
- **Backend API**: exposes upload, chat, status, and history endpoints.
- **Ingestion**: loads files, preserves metadata, and converts them to Document objects.
- **Chunking**: splits long text into semantically coherent chunks; CSV rows are treated as single chunk documents.
- **Embedding**: vectorizes chunks with SentenceTransformers.
- **Vector store**: persists embeddings and metadata in ChromaDB.
- **Retrieval**: performs semantic search and optional keyword fusion.
- **Reranking**: improves candidate ordering before prompt assembly.
- **Prompt construction**: builds evidence blocks for the local model.
- **Local LLM**: Ollama generates grounded answers.

## Retrieval pipeline

```text
Query → Query embedding → Vector search → Candidate pool → Reranking → Evidence selection → Answer generation
```

### Semantic retrieval

- vector search over embedded chunks
- retrieves passages based on meaning, not exact wording
- supports research documents with varied terminology

### Hybrid retrieval

- combines dense retrieval with keyword matching
- useful for acronym-heavy content and exact term recall
- candidates are merged and deduplicated before reranking

### Reranking

- rescoring improves passage precision
- selects the best subset for prompt context
- reduces noisy or marginal candidates

## Evidence grounding

- evidence blocks include chunk text and provenance metadata
- prompts instruct the LLM to base responses on retrieved passages
- unsupported answers are discouraged by the prompt design
- post-generation checks compare the answer with retrieved content

## Hallucination mitigation

- only strong retrieval results are passed into the model
- fallbacks are returned when evidence is insufficient
- the system prioritizes conservative answers over speculative output

## Component responsibilities

| Component | Responsibility |
|---|---|
| Frontend | Upload, chat UI, streaming, status messages |
| Backend API | request handling, pipeline orchestration |
| Loader | format-specific parsing and metadata extraction |
| Chunker | text splitting, document structure, chunk metadata |
| Embedder | vector generation for chunks |
| Vector store | persistent similarity search storage |
| Retriever | candidate selection and hybrid matching |
| Reranker | candidate rescoring and ordering |
| LLM client | local model interface and stream handling |
| Pipeline | end-to-end orchestration and fallback logic |

## Supported formats

- **PDF**: page-aware extraction with metadata
- **TXT**: single-document text ingestion
- **Markdown**: syntax stripped for clean embedding
- **CSV**: row-level ingestion to preserve structure and prevent chunk explosion

## Diagrams

```text
  [Upload] → [Loader] → [Chunker] → [Embedder] → [ChromaDB]
                                     ↑
                                     | 
                          [Retriever] ← [Query Embedder]
                                     |
                              [Reranker]
                                     |
                            [Prompt Builder]
                                     |
                                 [Ollama]
```
