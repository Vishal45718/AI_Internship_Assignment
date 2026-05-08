# Architecture Deep-Dive

## Pipeline Overview

The system consists of two independent pipelines that share the vector store:

```
INGESTION:  File → Load → Clean → Chunk → Embed → Store
QUERY:      Input → Classify → Retrieve → Rerank → Build → Generate → Validate → Respond
```

These pipelines are deliberately decoupled. You can re-index documents without touching the query code, and you can swap the LLM without touching the ingestion code.

---

## Component Responsibilities

### DocumentLoader (`app/ingestion/loaders.py`)

**Responsibility**: Convert raw files into clean `Document` objects.

- PDF: Page-level extraction via `pypdf`. One `Document` per page with page number in metadata.
- TXT: Single `Document` with Unicode normalization.
- Markdown: Strips headers (`#`), code blocks (` ``` `), and link syntax before embedding. Embedding models don't understand markdown syntax.
- CSV: Converts rows to prose strings (`field: value | field: value`) and batches 20 rows per document. This preserves row-level retrievability.

**Why page-level PDF splitting?** If you split a 100-page PDF into one `Document` and then chunk it, chunk boundaries are arbitrary. Splitting at page boundaries first means chunk boundaries respect natural page structure, and page numbers in metadata are accurate for citation.

### DocumentChunker (`app/ingestion/chunker.py`)

**Responsibility**: Split `Document` objects into `DocumentChunk` objects for embedding.

**Chunking algorithm** (`RecursiveCharacterTextSplitter`):
1. Try to split at `\n\n` (paragraph boundary) — preferred
2. If a chunk is still too long, split at `\n` (line boundary)
3. If still too long, split at `. ` (sentence boundary)
4. Last resort: split at spaces (word boundary)

This prioritization preserves semantic coherence. A chunk that ends at a paragraph boundary is more self-contained than one that ends mid-sentence.

**Deterministic IDs**: Chunk IDs are `SHA-256(source_file + chunk_index + content)[:16]`. Re-ingesting the same file produces the same IDs, enabling safe `upsert` (add-or-update) semantics in ChromaDB.

### ChromaVectorStore (`app/core/vector_store.py`)

**Responsibility**: Persist and retrieve chunks using cosine similarity.

**Upsert semantics**: `collection.upsert()` instead of `collection.add()` means re-ingesting the same document doesn't create duplicate entries — it updates them. This makes the ingestion pipeline idempotent.

**Similarity normalization**: ChromaDB returns cosine *distance* (0 = identical, 2 = opposite). We convert to *similarity* with `sim = 1 - (dist / 2)`, giving a 0→1 scale where higher = more similar.

**Metadata persistence**: Every chunk's `ChunkMetadata` is stored as ChromaDB metadata. This is what powers source citations — when we retrieve a chunk, we get its source file, page number, and chunk index back alongside the text.

### SemanticRetriever (`app/retrieval/retriever.py`)

**Responsibility**: Stage 1 of two-stage retrieval — cast a wide net.

Retrieves `top_k=10` candidates by cosine similarity. Applies a `similarity_threshold` filter to remove chunks that are clearly irrelevant (below 0.35 by default). This is a *recall-optimized* stage — we accept some false positives here because the reranker will clean them up.

### CrossEncoderReranker (`app/retrieval/reranker.py`)

**Responsibility**: Stage 2 — high-precision selection.

Takes the top-10 candidates and passes each (query, chunk) pair to `cross-encoder/ms-marco-MiniLM-L-6-v2`. The cross-encoder reads both texts together, which gives much more accurate relevance scores than cosine similarity (which only compares embedding representations independently).

**Why this works**: Embedding similarity is based on whether two texts *talk about the same topic*. Cross-encoder scoring is based on whether the chunk *actually answers the question*. The latter is what we need for QA.

Selects top-3 chunks and checks if the best score exceeds `confidence_threshold=0.40`. If not, the fallback is triggered — even if we retrieved chunks, they're not confident enough to answer the question.

**NoOpReranker**: When `ENABLE_RERANKING=false`, a pass-through reranker selects top-3 by similarity score instead. This degrades gracefully without code changes.

### QueryAnalyzer (`app/orchestration/query_analyzer.py`)

**Responsibility**: Intent classification before retrieval.

A single-shot LLM call with a structured classification prompt. Returns one of: `factual_lookup`, `comparison`, `summarize_doc`, `list_extraction`, `out_of_scope`, `unclear`.

**Why this is "agentic"**: The system decides *how* to respond before it acts. An `out_of_scope` query never hits the vector DB — it gets an immediate fallback. This saves latency, API cost, and prevents the LLM from trying to answer unanswerable questions.

**Graceful degradation**: If classification fails (API error, timeout), it defaults to `factual_lookup` — so the pipeline always produces an answer attempt rather than crashing.

### FallbackHandler (`app/orchestration/fallback_handler.py`)

**Responsibility**: Generate honest refusal responses without any LLM involvement.

This is the hallucination firewall. When retrieval confidence is below threshold, the `FallbackHandler` builds a structured response from templates:
- States it cannot answer this question from the indexed documents
- Lists the currently indexed documents (so users know what *is* available)
- Provides a helpful suggestion for how to get a better answer

**Critical design decision**: The fallback response is **never LLM-generated**. Asking the LLM to say "I don't know" can result in it "helpfully" providing partial information or softening the refusal. Template-driven responses are deterministic and impossible to hallucinate.

### PromptBuilder (`app/generation/prompt_builder.py`)

**Responsibility**: Assemble the (system_prompt, user_message) pair for the LLM.

The context format labels every chunk with its source:

```
[Source: sop_v2.pdf, Page 4] [Relevance: 0.87]
---
The return policy allows customers to return items within 30 days...
```

This labeling does two things: (1) gives the LLM explicit anchors it can cite in its response, (2) makes it harder to confabulate because the information is clearly attributed.

### ResponseValidator (`app/generation/response_validator.py`)

**Responsibility**: Post-generation sanity check for numeric claims.

The most common LLM hallucination pattern is confabulating specific numbers — statistics, dates, prices. The validator extracts all numeric values from the response and checks if they appear in the retrieved context. Values that don't appear are flagged as suspected hallucinations.

This is a "soft" check — it warns but doesn't refuse to serve the response (too many false positives would make the system unusable). In production, this could trigger a human review queue for responses with high hallucination suspicion.

---

## Data Flow: A Concrete Example

**Query**: "What is the electronics return window?"

1. **QueryAnalyzer**: Classifies as `factual_lookup`.
2. **SemanticRetriever**: Embeds query → top-10 search → filters by threshold → returns 8 chunks from `sample_sop.txt`, `sample_faq.txt`.
3. **CrossEncoderReranker**: Scores 10 pairs → selects top-3 → top score = 0.87 → passes threshold → `passed_threshold=True`.
4. **PromptBuilder**: Builds system prompt with 3 labeled context chunks + query.
5. **LLMClient**: Sends to GPT-4o-mini → response: "Electronics have a reduced return window of **15 days** [Source: sample_sop.txt]".
6. **ResponseValidator**: Checks "15" → found in context → no warnings.
7. **ConversationMemory**: Stores turn in session.
8. **Response**: `{"status": "success", "answer": "...", "sources": [...], "confidence": 0.87}`.

**Query**: "What is NVIDIA's Q3 2023 revenue?"

1. **QueryAnalyzer**: Classifies as `factual_lookup`.
2. **SemanticRetriever**: Top-10 search → all scores below 0.35 → `passed_threshold=False`.
3. **FallbackHandler**: Builds template response → "I could not find sufficient information..." + document list.
4. **Response**: `{"status": "fallback", "answer": "I could not find...", "sources": [], "confidence": 0.12}`.

---

## Extending the System

### Adding a new document format

1. Write a loader function in `app/ingestion/loaders.py` following the `_load_pdf` pattern.
2. Add the new `DocumentType` enum value in `app/models/documents.py`.
3. Register the loader in `_LOADER_MAP` in `loaders.py`.
4. Update `requirements.txt` with any new parser dependency.

### Swapping the vector DB

1. Implement a class matching `ChromaVectorStore`'s interface (`add_chunks`, `query`, `count`, `list_sources`, `delete_by_source`, `reset`).
2. Update `get_vector_store()` in `app/core/dependencies.py` to instantiate the new class.
3. The rest of the codebase is unchanged.

### Adding a new LLM provider

1. Implement a class extending `BaseLLMClient` with `generate()` and `model_name`.
2. Add it to the `create_llm_client()` factory in `app/core/llm_client.py`.
3. Add the new provider to the `LLM_PROVIDER` enum in `configs/settings.py`.

### Enabling hybrid search (BM25 + semantic)

1. Index documents with BM25 (e.g., using `rank_bm25`).
2. Score each query with both BM25 and cosine similarity.
3. Fuse scores using Reciprocal Rank Fusion (RRF).
4. Pass the merged candidates to the existing reranker.
The reranker interface (`retrieve(query) → RetrievalResult`) is unchanged — this is all internal to `SemanticRetriever`.
