# Architecture

## End-to-end flow

The system follows a linear pipeline for document ingestion and querying.

```text
Upload → Parsing → Chunking → Embeddings → ChromaDB → Retrieval → Reranking → Prompt Construction → LLM Generation → Grounded Response
```

## Architecture summary

1. **Upload**
   - The frontend sends files to the FastAPI upload endpoint.
   - The backend stores the file into `data/raw/` and forwards the path to the ingestion pipeline.

2. **Parsing**
   - The loader module selects a format-specific parser by extension.
   - PDFs are split by page, Markdown is stripped of formatting, and CSV rows become structured prose.

3. **Chunking**
   - Non-CSV text documents are sectionized and recursively split with overlap.
   - CSV rows are treated as compact documents and do not receive recursive overlap chunking.

4. **Embeddings**
   - Each chunk is embedded using a SentenceTransformers model.
   - Embeddings are stored with metadata in ChromaDB.

5. **Vector store**
   - ChromaDB persists vectors and metadata for similarity search.
   - The store exposes add, delete, and search operations.

6. **Retrieval**
   - A query is embedded and used to retrieve top candidate chunks.
   - Optional hybrid retrieval fuses keyword and dense similarity results.

7. **Reranking**
   - Candidate passages are rescored using a reranker to improve relevance ordering.
   - The highest quality chunks are selected for prompt construction.

8. **Prompt construction**
   - Evidence blocks are assembled from selected chunks.
   - The prompt emphasizes citations and evidence-only answering.

9. **LLM generation**
   - The local Ollama client streams model output back to the frontend.
   - Streaming allows the UI to update progressively.

10. **Grounded response**
    - Post-generation validation checks the answer against retrieved evidence.
    - Weak evidence may trigger fallback text or safer phrasing.

## Retrieval flow

```text
User Query → Query Embedding → Vector Search → Candidate Chunks → Reranking → Chunks for Prompt → LLM Answer
```

### Hybrid retrieval flow

- semantic retrieval finds conceptually relevant passages
- keyword matching improves recall for exact terms and acronyms
- results are combined and deduplicated before reranking

## Reranking flow

- Raw candidate chunks are scored by relevance and query overlap
- The reranker prioritizes passages that match query semantics and terminology
- Selected chunks are passed into the prompt rather than all retrieved candidates

## Evidence grounding

- Answers are generated using retrieved passages, not free-form model memory.
- Responses may include source citations, especially in document mode.
- The system includes checks for unsupported expansions, low evidence overlap, and weak retrieval confidence.

## Hallucination prevention

- **Evidence-only answering**: prompts instruct the model to use retrieved content.
- **Fallback responses**: if evidence is insufficient, the system returns a conservative fallback instead of a speculative answer.
- **Post-validation**: generated text is evaluated against the corpus, and unsupported output can be flagged.

## Component responsibilities

- **Frontend**: uploads, chat UI, streaming responses, progress states.
- **Backend**: API endpoints, ingestion orchestration, retrieval pipeline.
- **Ingestion**: loaders and chunker for file formatting and chunk generation.
- **Embeddings**: vectorization service for chunks.
- **Vector store**: persistent similarity search and metadata storage.
- **Retriever**: candidate selection and search logic.
- **Reranker**: relevance ordering of retrieved passages.
- **LLM client**: local model interaction and streaming.
- **Pipeline orchestrator**: manages end-to-end ingestion and document QA.
