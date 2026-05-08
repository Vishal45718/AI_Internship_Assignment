# Write-Up: Agentic RAG System

## Architecture Decisions

### Document Ingestion

**Choice: Page-level PDF splitting before character chunking**

Rather than treating a PDF as one long string and chunking arbitrarily, we split at the page level first. This means: (1) chunk boundaries respect natural document structure, (2) page numbers in metadata are accurate (a chunk on page 4 correctly cites page 4), and (3) we avoid the common failure mode of a chunk spanning the end of one section and the beginning of another.

**Choice: CSV rows as prose strings**

CSV data embedded as raw column headers and delimited values has poor embedding quality — the model doesn't understand that `product_id,TC-PRO-001,name,TechCorp Pro Laptop` is a structured record. Converting to `product_id: TC-PRO-001 | name: TechCorp Pro Laptop | price_usd: 1299.99` produces embeddings that are semantically meaningful for QA retrieval.

**Choice: SHA-256 chunk IDs for idempotent ingestion**

Chunk IDs are computed from `SHA-256(filename + chunk_index + content_text)`. This makes the ingestion pipeline idempotent — re-running it on the same files updates existing chunks rather than duplicating them. At scale, this also enables incremental ingestion: compute chunk hashes first, skip chunks whose ID already exists in the vector DB.

### Retrieval

**Choice: Two-stage retrieve-then-rerank**

Cosine similarity between query and chunk embeddings has high recall but mediocre precision — it finds chunks about the right topic but may include contextually wrong ones. A cross-encoder model that reads query+chunk together scores true relevance much more accurately. The two-stage approach gets the best of both: embedding similarity is fast for top-10 recall, cross-encoder is accurate for top-3 precision.

**Choice: Separate similarity threshold and confidence threshold**

The similarity threshold (`0.35`) filters the initial retrieval results. The confidence threshold (`0.40`) is applied after reranking. Having two separate thresholds lets you tune recall (adjust similarity threshold) and precision (adjust confidence threshold) independently.

### Hallucination Prevention

**Choice: Three-layer defense**

1. **System prompt**: The LLM is explicitly told to answer only from context and to use the fallback phrase if insufficient information is available.
2. **Confidence threshold**: If the reranker score is below threshold, the LLM is never called at all. This prevents the LLM from receiving low-quality context and confabulating.
3. **Template-driven fallback**: The "I don't know" response is generated from a Python template, not by the LLM. This is the most important layer.

**Why template-driven fallback beats "tell the LLM to say it doesn't know"**:

LLMs instructed to say "I don't know" often: (1) say it but then add speculative information anyway, (2) give a partial answer that sounds confident, (3) suggest related information that's actually invented. A template response is 100% deterministic and contains zero fabricated content.

### LLM Configuration

**Choice: Temperature 0.1**

For document-grounded QA, we want the LLM to stay close to the provided context. Low temperature (0.1) significantly reduces the probability of "creative" additions that aren't in the source material. This trades diversity for faithfulness — exactly the right tradeoff for this use case.

**Choice: Separate system and user prompts**

Putting the grounding instruction in the system prompt (rather than prepending it to the user message) leverages the model's training to respect system-level instructions more strongly. All major models treat system prompt content with higher authority than user message content.

---

## Limitations of This Approach

### 1. Embedding Quality for Tabular Data

Sentence embedding models are trained on natural language text. Tables, CSV data, and structured records don't embed as well. A product CSV row converted to prose embeds better than raw CSV, but still significantly worse than a naturally written sentence about the same product.

**Mitigation**: Convert structured data to prose (already implemented). For production, use specialized table embeddings or a hybrid retrieval approach that includes BM25 keyword matching for structured queries.

### 2. Long-Document Precision

For documents longer than ~50 pages, the fixed-size chunking approach may miss answers that require synthesizing information across distant sections. A user asking "compare the warranty policies for laptops and tablets" may get chunks from only one section if the relevant content is too far apart.

**Mitigation**: Query expansion (generate 2-3 alternative phrasings and retrieve for each) partially addresses this. Hierarchical retrieval (retrieve document summaries first, then dive into relevant sections) would be more robust.

### 3. PDF Layout Complexity

`pypdf` extracts text in reading order for simple single-column PDFs but struggles with multi-column layouts, headers/footers, and complex tables. Text extracted from a two-column academic paper may read as alternating sentences from each column.

**Mitigation**: Use `pdfplumber` (better layout analysis) or `pymupdf` (fast + better table extraction) for complex PDFs. Add OCR (Tesseract) for scanned documents.

### 4. Stateless Reranker Cold Start

The first query after startup loads the cross-encoder model from disk (~65MB). This adds ~2 seconds to the first response. Subsequent queries are fast (< 200ms for reranking).

**Mitigation**: Pre-warm the model on application startup. Already feasible by calling `_get_reranker_model()` in the `@app.on_event("startup")` handler.

### 5. Conversation Memory Loss on Restart

The `MemoryStore` is in-process. API restart = all session history lost. In a multi-worker deployment (gunicorn with 4 workers), different requests from the same session may hit different workers, each with different state.

**Mitigation**: Externalize to Redis. The `ConversationMemory` and `MemoryStore` interfaces are designed for this — replace the in-process dict with a Redis client.

---

## Suggestions for Scaling

### Ingestion at Scale

- **Async batch ingestion**: Move from synchronous file processing to an async task queue (Celery + Redis). Each document becomes a task; workers pick them up independently.
- **Incremental ingestion**: Before embedding, hash each chunk and compare with existing IDs in the vector DB. Skip chunks that haven't changed. On a 10,000-document corpus where only 100 documents changed, this reduces embedding API costs by ~99%.
- **Content deduplication**: If the same document is uploaded twice under different filenames, detect it via file-level SHA-256 hashing before ingestion.

### Retrieval at Scale

- **Replace ChromaDB with Weaviate or Pinecone**: ChromaDB loads the entire vector index into memory. For 10M+ vectors, this is impractical. Weaviate (self-hosted, supports hybrid BM25+vector) or Pinecone (managed) are the production choices.
- **Query result caching**: Cache `(query_embedding, top_k)` → `results` in Redis with a short TTL. Repeated identical queries (common in production) are served from cache.
- **Hybrid search**: Combine BM25 keyword scores with cosine similarity using Reciprocal Rank Fusion. BM25 has better recall for exact-match queries; vector similarity has better recall for paraphrased queries. The combination beats either alone.

### Generation at Scale

- **Streaming responses**: Already supported by the OpenAI API. Add streaming to the FastAPI endpoint (`StreamingResponse`) and the Streamlit UI. This reduces perceived latency dramatically.
- **Cost routing**: Route simple queries (short, factual) to `gpt-3.5-turbo` and complex ones (synthesis, comparison) to `gpt-4o-mini`. Classify based on query length, intent, or retrieved chunk count.
- **Response caching**: Cache `(query_hash, context_hash)` → `response`. If the same question is asked with the same retrieved context, serve the cached response.

### Multi-Tenant Architecture

- **Vector DB namespacing**: Add `user_id` as a required metadata field. Use ChromaDB's `where={"user_id": uid}` filter on every query. Each user only retrieves from their own documents.
- **Authentication**: JWT tokens on the FastAPI endpoints. The `user_id` claim from the token is the namespace key for the vector DB query.
- **Rate limiting**: Add per-user rate limiting on the `/query` endpoint to prevent API cost runaway.

### Observability

- **Structured logging**: Already implemented — JSON log format is compatible with Datadog, Splunk, and Loki.
- **Retrieval metrics**: Log every retrieval event with: query, top-k scores, retrieval latency, and whether fallback was triggered. This gives you a "fallback rate" metric — a proxy for how well your documents cover user queries.
- **OpenTelemetry**: Add distributed tracing across the ingestion and query pipelines. Each request gets a trace with spans for: intent classification, retrieval, reranking, prompt building, LLM call, validation.
