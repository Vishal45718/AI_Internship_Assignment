# Agentic RAG System — Senior Engineer's Playbook

> A complete strategy, architecture, and execution guide for building a recruiter-impressive submission in 5 days (optimized for 1–2 days if needed).

---

## 1. What the Company Is Actually Evaluating

This isn't a test of whether you can chain `langchain` calls. The reviewers are assessing:

- **Systems thinking** — Can you design a clean, modular pipeline where each component has a single responsibility?
- **Engineering judgment** — Do you make smart tradeoffs instead of reaching for the most complex solution?
- **Production instincts** — Does your code look like it was written to be maintained, not just to run?
- **Understanding of the RAG problem** — Do you understand *why* hallucinations happen and *how* to prevent them?
- **Communication quality** — Is your README, write-up, and demo video the work of someone who can explain their decisions clearly?

The "agentic" framing is a signal — they want to see if you can build a system with *reasoning*, not just lookup. A naive vector search + LLM call is a tutorial clone. A well-architected system with hallucination prevention, fallback handling, and clean separation of concerns is what gets you hired.

**What makes a candidate stand out:**
- Chunking strategy that actually preserves context (not naive fixed-size chunks)
- A system that says "I don't know" gracefully when the answer isn't in the documents
- Source citation in every response
- A README that reads like internal documentation, not a homework submission
- One genuinely impressive feature (semantic chunking, reranking, or a conversational memory pattern)

**What NOT to waste time on:**
- Fancy UI animations
- Deploying to the cloud (a local demo is fine)
- Multi-agent orchestration with 4+ agents
- Fine-tuning any model
- Perfect test coverage

---

## 2. Ideal Tech Stack

### Core Pipeline

| Component | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Assignment requirement; best ecosystem |
| LLM | `gpt-4o-mini` via OpenAI API | Fast, cheap, strong instruction-following; fallback: `mistral-7b-instruct` via Ollama |
| Embeddings | `text-embedding-3-small` (OpenAI) | Best cost/quality ratio; 1536-dim |
| Vector DB | ChromaDB (local) | Zero infra, Python-native, persistent, good for demo |
| Document parsing | `pypdf2`, `python-docx`, `pandas` | Lightweight, covers PDF/TXT/CSV |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Paragraph-aware, better than fixed-size |
| Reranking | `flashrank` or `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local, fast, dramatically improves precision |
| UI | Streamlit | 30-minute setup, looks professional, native chat widget |
| Logging | Python `logging` + structured JSON logs | Shows production awareness |
| Config | `python-dotenv` + `pydantic-settings` | Clean env handling |
| Testing | `pytest` with 3–5 targeted tests | Quality signal without overkill |

### Why Not LangChain for the whole thing?

LangChain is fine for rapid prototyping, but interviewers have seen a hundred `ConversationalRetrievalChain` clones. Writing your own retrieval and prompt construction logic — even if it's 50 lines — demonstrates you understand what's happening inside the black box. Use LangChain for its splitters and loaders, but own the retrieval and generation logic yourself.

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
│                                                                  │
│  Documents (PDF/TXT/CSV)                                         │
│       │                                                          │
│       ▼                                                          │
│  [DocumentLoader]  →  [Preprocessor]  →  [Chunker]              │
│  pypdf2 / pandas       clean text         RecursiveChar          │
│  docx / txt            normalize          512 tok / 64 overlap   │
│       │                                                          │
│       ▼                                                          │
│  [EmbeddingGenerator]  →  [VectorStore]                          │
│  text-embedding-3-small    ChromaDB (persistent)                 │
│  + metadata tagging        + source + chunk_id metadata          │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                            │
│                                                                  │
│  User Query                                                      │
│       │                                                          │
│       ▼                                                          │
│  [QueryProcessor]                                                │
│  • normalize + clean                                             │
│  • optional: query expansion (HyDE or synonym generation)        │
│       │                                                          │
│       ▼                                                          │
│  [Retriever]                                                     │
│  • embed query                                                   │
│  • top-k=10 semantic search (ChromaDB)                           │
│  • similarity threshold filter (>0.35)                           │
│       │                                                          │
│       ▼                                                          │
│  [Reranker]                                                      │
│  • cross-encoder reranking                                       │
│  • select top-3 chunks                                           │
│  • check if max_score > confidence_threshold                     │
│       │                                                          │
│       ├─── below threshold ──→ [FallbackHandler]                 │
│       │                        "I don't have enough info..."     │
│       │                                                          │
│       ▼                                                          │
│  [PromptBuilder]                                                 │
│  • grounded system prompt                                        │
│  • inject retrieved chunks with source citations                 │
│  • inject conversation history (last 3 turns)                    │
│       │                                                          │
│       ▼                                                          │
│  [LLM]  (gpt-4o-mini)                                            │
│       │                                                          │
│       ▼                                                          │
│  [ResponseValidator]                                             │
│  • strip hallucinated claims (basic check)                       │
│  • format source citations                                       │
│       │                                                          │
│       ▼                                                          │
│  Response + [Source 1: doc.pdf, chunk 3] [Source 2: ...]         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Professional Folder Structure

```
agentic-rag/
├── .env.example                  # template for environment variables
├── .gitignore
├── README.md
├── requirements.txt
├── pyproject.toml                # for tool config (ruff, pytest)
│
├── config/
│   ├── settings.py               # pydantic-settings config model
│   └── prompts.py                # all prompts as constants (never inline)
│
├── data/
│   ├── raw/                      # original uploaded documents
│   └── processed/                # cleaned text (optional cache)
│
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loaders.py            # PDF, TXT, CSV document loaders
│   │   ├── preprocessor.py       # cleaning, normalization
│   │   └── chunker.py            # chunking logic + metadata tagging
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py           # embedding generation + caching
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── chroma_store.py       # ChromaDB wrapper (add, query, delete)
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py          # semantic search + threshold filter
│   │   └── reranker.py           # cross-encoder reranking
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompt_builder.py     # assembles final prompt from chunks
│   │   └── llm_client.py         # OpenAI API wrapper with retry logic
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   └── conversation.py       # rolling conversation window
│   │
│   └── agent/
│       ├── __init__.py
│       └── rag_agent.py          # orchestrates the full pipeline
│
├── ui/
│   └── streamlit_app.py          # Streamlit chat interface
│
├── scripts/
│   ├── ingest_documents.py       # CLI: python scripts/ingest_documents.py
│   └── query_cli.py              # CLI: python scripts/query_cli.py
│
└── tests/
    ├── test_chunker.py
    ├── test_retriever.py
    └── test_agent.py
```

---

## 5. Step-by-Step Implementation Roadmap

### Minimum Working Version (Day 1 — ~6 hours)

**Phase 1: Foundation (1.5h)**
- Set up folder structure, `requirements.txt`, `.env`, `settings.py`
- Implement `loaders.py`: PDF (pypdf2), TXT (plain read), CSV (pandas)
- Test: can you load a PDF and get clean text?

**Phase 2: Ingestion Pipeline (1.5h)**
- Implement `preprocessor.py`: strip headers/footers, normalize whitespace, handle encoding
- Implement `chunker.py`: RecursiveCharacterTextSplitter, 512 tokens, 64 overlap, attach metadata (source_file, chunk_id, page_num)
- Implement `embedder.py`: call text-embedding-3-small, cache to disk (json or pickle)
- Implement `chroma_store.py`: persist collection, add/query methods
- Test: ingest 3 documents, verify ChromaDB has records

**Phase 3: Query + Generation (1.5h)**
- Implement `retriever.py`: embed query, top-10 semantic search, filter by similarity score
- Implement `prompt_builder.py`: system prompt + context injection + source citation format
- Implement `llm_client.py`: OpenAI chat completion with retry (tenacity)
- Wire together in `rag_agent.py`
- Test: ask a question that should be answerable, verify it cites a source

**Phase 4: Fallback + CLI (0.5h)**
- Add confidence threshold check: if top chunk similarity < 0.35, return graceful "I don't have enough information in the provided documents to answer this."
- Implement `query_cli.py` for demo purposes
- Test: ask an unanswerable question, verify graceful fallback

**Total Day 1 output: A fully working RAG system with hallucination prevention and source citation.**

---

### Polish Phase (Day 2 — ~5 hours)

**Phase 5: Reranking (1h)**
- Add `reranker.py` with `flashrank` or HuggingFace cross-encoder
- Retrieve top-10, rerank, pass top-3 to LLM
- Test: compare retrieval quality with/without reranker

**Phase 6: Streamlit UI (1.5h)**
- Chat interface with message history
- Source expander showing retrieved chunks and document names
- Sidebar: upload new documents (ingest on the fly), show ingested documents list
- Loading spinner during retrieval
- Test: full end-to-end via UI

**Phase 7: Conversation Memory (0.5h)**
- Implement `conversation.py`: keep last N turns as context
- Inject into `prompt_builder.py`

**Phase 8: Write-up + README (1.5h)**
- See section 11 below for exact structure
- Architecture diagram (can use mermaid in README)
- Limitations section (be honest — it shows maturity)
- Scaling section (see section 14)

**Phase 9: Demo Video (0.5h)**
- See section 12 for exact flow

---

## 6. Document Ingestion Design

### Loader Strategy

```python
# loaders.py pattern
class DocumentLoader:
    def load(self, path: str) -> list[Document]:
        ext = Path(path).suffix.lower()
        return {
            ".pdf": self._load_pdf,
            ".txt": self._load_txt,
            ".csv": self._load_csv,
        }.get(ext, self._unsupported)(path)
```

Each loader returns a `Document` dataclass with `content: str` and `metadata: dict`.

### Metadata to Attach to Every Chunk

```python
metadata = {
    "source_file": "sop_v2.pdf",
    "source_type": "pdf",
    "page_number": 3,          # for PDFs
    "chunk_id": "sop_v2_c14",
    "ingestion_timestamp": "2024-01-15T10:30:00Z",
    "char_count": 487,
}
```

This metadata is what enables source citations in your responses. Never skip it.

### Chunking Strategy

Use `RecursiveCharacterTextSplitter` with:
- `chunk_size=512` tokens (~2000 chars) — large enough for context, small enough for precision
- `chunk_overlap=64` tokens — prevents cutting mid-sentence context
- Separators: `["\n\n", "\n", ". ", " "]` — tries paragraph → sentence → word

For CSV data, chunk row-by-row or group related rows by a category column rather than arbitrary character splits.

### Preprocessing Checklist

- Strip PDF artifacts (page numbers, header/footer patterns like "Page X of Y")
- Normalize Unicode (handle special quotes, em-dashes)
- Collapse multiple whitespace/newlines to single
- Remove null bytes and control characters
- Preserve paragraph structure (double newlines are semantic breaks)
- For CSVs: convert each row to a readable sentence format: `"Product: X, Price: Y, Category: Z"`

---

## 7. Retrieval Quality Optimization

This is where most tutorial-level projects fail. Here is how to make it actually good.

### Two-Stage Retrieval (Retrieve-then-Rerank)

**Stage 1 — Semantic search**: Retrieve top-10 candidates by cosine similarity using ChromaDB. Cast a wide net. This stage has high recall, lower precision.

**Stage 2 — Reranking**: Pass query + each of the 10 chunks to a cross-encoder model. The cross-encoder reads both together and scores true relevance (not just embedding similarity). Select top-3. This stage has high precision.

Without reranking, your retrieval will occasionally surface chunks that are semantically similar but contextually wrong. Reranking fixes this.

### Similarity Threshold

After reranking, check the top score. If it's below your threshold (start at 0.35, tune empirically), the question is genuinely unanswerable from your documents. Trigger the fallback response. This is your primary hallucination prevention mechanism.

### Query Expansion (Nice-to-Have, High ROI)

Before retrieval, optionally use the LLM to generate 2–3 alternative phrasings of the user's query. Retrieve for each, then deduplicate and rerank the union. This significantly improves recall for queries with unusual wording.

```python
expansion_prompt = """Given the query: "{query}"
Generate 2 alternative phrasings that mean the same thing but use different words.
Return only the phrasings, one per line."""
```

### Metadata Filtering

If you have documents from multiple sources/types, add a filter option. A user asking about "Q3 financial performance" shouldn't be retrieving from HR policy documents. Implement with ChromaDB's `where` clause on metadata.

### Preventing Over-Retrieval

Limit context to 3 chunks maximum. More context doesn't help — it confuses the LLM and increases cost. Precision over recall at the generation stage.

---

## 8. Hallucination Prevention — Production Approach

### The Core Principle

The LLM must be explicitly told: "Answer only from the provided context. If the context doesn't contain enough information, say so." This sounds obvious but the implementation details matter enormously.

### System Prompt (Keep in `config/prompts.py`)

```python
SYSTEM_PROMPT = """You are a precise document assistant. Your only source of knowledge is the CONTEXT provided below.

Rules you must follow without exception:
1. Answer ONLY based on information explicitly present in the CONTEXT.
2. If the CONTEXT does not contain enough information to answer the question, respond: "Based on the available documents, I don't have sufficient information to answer this question. [suggest what they might search for instead]"
3. Never use your general knowledge or make inferences beyond what the CONTEXT states.
4. Always cite your sources using the [Source: filename] format provided in the CONTEXT.
5. If you are uncertain whether the CONTEXT supports your answer, say so explicitly.

CONTEXT:
{context}
"""
```

### Context Format (What You Pass to the LLM)

```
[Source: sop_operations.pdf, Page 4]
--- 
The return policy allows customers to return items within 30 days of purchase with original receipt...

[Source: product_catalog.csv, Row 142]
---
Product: SKU-4421, Category: Electronics, Warranty: 2 years...
```

This format gives the LLM explicit source anchors it can cite in its response. It also makes it harder to confabulate because the information is labeled.

### Handling Unknown Queries

```python
FALLBACK_RESPONSE = """I don't have enough information in the provided documents to answer "{query}" accurately.

The documents I have access to cover: {document_summary}

If your question is about a different topic, please check if the relevant documents have been uploaded."""
```

### Response Validation (Simple but Effective)

After the LLM responds, do a basic check: does the response contain any numerical claims (dates, statistics, prices)? If so, verify each one appears in your retrieved chunks. If a number in the response doesn't exist in context, flag it or strip it. This catches a common hallucination pattern where LLMs confabulate specific numbers.

---

## 9. Agentic Behavior — What's Actually Worth Building

"Agentic" in this context means the system can decide *how* to answer a query, not just retrieve-and-generate.

### Minimum Viable Agent Pattern (High ROI, 2h implementation)

Implement a simple intent router before retrieval:

```
User Query
    │
    ▼
[IntentClassifier]  (simple LLM call with structured output)
    │
    ├── "factual_lookup"    → standard RAG pipeline
    ├── "comparison"        → retrieve for each entity, compare in prompt
    ├── "summarize_doc"     → retrieve full document chunks, summarize
    └── "out_of_scope"      → immediate fallback, no retrieval
```

This is genuinely "agentic" because the system reasons about the query before acting. It takes ~30 lines and impresses reviewers because it shows you understand the concept rather than just calling it "agentic" because you used LangChain.

### Conversation Memory as Agentic Behavior

Maintaining conversational context so follow-up questions ("What about their return policy?") work correctly after ("Tell me about their shipping policy") is a form of stateful agent behavior. Implement a rolling 3-turn window. This is high-value, low-effort.

### Skip for Now: Multi-tool agents (web search + calculator + code execution). These take 4–6 hours for marginal demo value.

---

## 10. UI/UX — Streamlit Done Right

### Must-Have Components

```python
# streamlit_app.py structure
st.set_page_config(page_title="DocChat", layout="wide")

# Sidebar
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload documents", accept_multiple_files=True)
    # list currently ingested docs
    # show chunk count per document

# Main chat
st.title("DocChat — Ask your documents")
# message history (st.session_state)
# chat input (st.chat_input)
# per-message source expander:
with st.expander("Sources"):
    for chunk in response.source_chunks:
        st.markdown(f"**{chunk.source_file}** — {chunk.preview}")
```

### What Makes It Look Polished

- Source citations below every answer (not buried in expander — show them inline)
- "Processing..." spinner with step labels ("Retrieving context...", "Generating response...")
- Clear empty state with example queries when no documents are uploaded
- Document list in sidebar with chunk counts (shows the pipeline worked)
- Handle the "no documents ingested" state gracefully

---

## 11. README Structure (Recruiter-Impressive)

```markdown
# Agentic RAG System

> An AI document assistant that answers questions strictly from ingested documents,
> with semantic retrieval, cross-encoder reranking, and source citations.

## Architecture

[diagram here — mermaid or image]

### Pipeline Overview
Ingestion → Chunking → Embedding → Storage → [Query → Retrieve → Rerank → Generate → Validate]

## Features
- Multi-format ingestion (PDF, TXT, CSV)
- Recursive semantic chunking with metadata preservation  
- Two-stage retrieval: semantic search + cross-encoder reranking
- Hallucination prevention via confidence thresholds and grounded prompts
- Source citation in every response
- Graceful unknown-query handling
- Conversation memory (3-turn window)
- Streamlit chat UI with document upload

## Setup
[exact commands, no ambiguity]

## Usage
[screenshots]

## Architecture Decisions
[see write-up section]

## Limitations
[be honest — shows maturity]

## How It Would Scale
[see section 14]
```

---

## 12. Demo Video Flow (3–5 Minutes)

**Opening (30s):** Show the Streamlit UI with 3–4 documents already ingested. "I've ingested a product manual, an SOP document, and a research paper. Let me show you what the system can do."

**Demo 1 — Basic factual retrieval (45s):** Ask a direct question with a clear answer in one document. Show the response + source citations. Point out: "The system cites exactly where it found this — chunk from page 4 of the SOP."

**Demo 2 — Multi-document retrieval (45s):** Ask a question whose answer spans two documents. Show the system correctly synthesizing across sources.

**Demo 3 — Hallucination prevention (45s):** Ask a question that is NOT in any document. Show the graceful fallback response. "The system doesn't confabulate. It tells you it doesn't know." This is the most important thing to demo.

**Demo 4 — Conversation memory (30s):** Ask a follow-up question that requires remembering the previous answer. Show it works.

**Demo 5 — Document upload (30s):** Upload a new document live. Ingest it. Ask a question about it. Show the system immediately uses it.

**Closing architecture walk (60s):** Briefly show the code structure. "The retrieval and generation are fully decoupled — I can swap the vector DB or LLM without touching the rest of the pipeline." Show `config/prompts.py` and explain the grounded system prompt.

---

## 13. Interview Preparation

**Q: Why did you choose ChromaDB over Pinecone/Weaviate?**
> ChromaDB runs locally with zero infrastructure, which is ideal for a 5-day prototype. For production, I'd evaluate Pinecone for managed scale or Weaviate for hybrid search with BM25 + vector. The abstraction layer I built (the `VectorStore` interface) means swapping it out is a one-file change.

**Q: How does your system prevent hallucinations?**
> Three mechanisms: (1) A grounded system prompt that explicitly forbids answering outside context, (2) a confidence threshold on the reranker score that triggers a fallback response rather than passing low-quality context to the LLM, and (3) the context format itself, which labels every chunk with its source — the LLM can only cite what's actually there.

**Q: What chunking strategy did you use and why?**
> RecursiveCharacterTextSplitter at 512 tokens with 64 token overlap. The recursive part means it tries to split at paragraph boundaries first, then sentences, then words — which preserves semantic coherence. The overlap prevents information loss at chunk boundaries. For CSVs, I instead chunk by converting rows to prose strings and grouping by category.

**Q: How would you improve retrieval quality further?**
> Three things: (1) HyDE (Hypothetical Document Embeddings) — generate a hypothetical answer, embed that instead of the raw query, which improves similarity matching. (2) Hybrid search — combine BM25 keyword score with vector similarity for better exact-match recall. (3) Better chunking — use a semantic chunker that splits at topic shifts rather than fixed sizes.

**Q: How does your system handle multi-turn conversations?**
> I maintain a rolling window of the last 3 turns in memory, injected into the prompt as conversation history. For a production system, I'd use a summary-based approach — compress older turns into a rolling summary — to handle longer conversations without exceeding context limits.

**Q: What's the biggest limitation of your system?**
> The embedding model doesn't understand document structure — a table of contents chunk will have high similarity to many queries even when irrelevant. A production system would need document-type-aware chunking strategies and potentially multi-modal embeddings for tables and figures.

---

## 14. Scalability Discussion

Even if your implementation is local and lightweight, you must be able to discuss scale. Here is what to know:

### Ingestion at Scale
- Move from synchronous to async batch ingestion (Celery + Redis or Python asyncio)
- Distributed document processing: each document goes into a queue, workers pick them up
- Deduplication via content hash before embedding (saves API cost)
- Incremental ingestion: only re-embed changed chunks (compare hash of chunk text)

### Retrieval at Scale
- Replace ChromaDB with Pinecone (managed) or Weaviate (self-hosted) for millions of vectors
- Add a caching layer: Redis or a simple dict cache for repeated queries (saves embedding API cost and latency)
- Hybrid search (BM25 + vector) scales better than pure semantic search because BM25 doesn't require storing all vectors in RAM

### LLM at Scale
- Move from synchronous to streaming responses (already supported by OpenAI API)
- Add response caching for identical queries
- Cost control: route simple queries to `gpt-3.5-turbo`, complex queries to `gpt-4`

### Multi-User Architecture
- Each user's document set is scoped to a `user_id` namespace in the vector DB
- Conversation memory is stored per session in a database (Postgres/Redis)
- Authentication via JWT tokens on a FastAPI backend
- Streamlit → FastAPI REST API → RAG pipeline (separate concerns properly)

### Observability
- Log every retrieval event: query, top-k results, similarity scores, latency
- Track hallucination proxy metrics: fallback rate, average reranker confidence score
- Use OpenTelemetry for distributed tracing in a production deployment

### Security
- Never store raw API keys in code — use environment variables (your `.env` pattern)
- In production, the vector DB should be network-isolated, not publicly accessible
- Audit log all document ingestion events (who ingested what, when)
- PII detection: scan ingested documents for PII before embedding (presidio library)

---

## 15. Strict Execution Plans

### 1-Day Plan (If Needed)

| Time | Task | Output |
|---|---|---|
| 0:00–1:30 | Setup + loaders + chunker | Can ingest 3 docs |
| 1:30–3:00 | Embedder + ChromaDB | Can store and query |
| 3:00–4:30 | Retriever + LLM + prompt | End-to-end query works |
| 4:30–5:00 | Fallback handling + CLI | Graceful unknown query |
| 5:00–6:30 | Streamlit UI | Working chat interface |
| 6:30–7:30 | README + write-up | Submission-ready docs |
| 7:30–8:00 | Record demo video | Final deliverable |

**If running out of time:** Skip reranking (use top-3 directly from ChromaDB), skip conversation memory, skip CSV loader. Nail the core RAG + hallucination prevention + source citation. That is the core signal.

### 2-Day Plan (Full Polish)

**Day 1 (8h):** Complete 1-day plan above.

**Day 2 (5h):**
- Add reranking with flashrank (1h)
- Add conversation memory (0.5h)  
- Polish Streamlit UI: source expander, document list, loading states (1h)
- Write complete README with architecture diagram (1h)
- Write architecture write-up / limitations / scaling (0.5h)
- Test edge cases and fix bugs (0.5h)
- Record and edit demo video (0.5h)

### Pre-Submission Checklist

- [ ] All API keys in `.env`, never committed to git
- [ ] `.env.example` exists with placeholder values
- [ ] `requirements.txt` is accurate (pip freeze and prune)
- [ ] README has working setup instructions (test them on a clean venv)
- [ ] Demo video shows all 5 flows from section 12
- [ ] Graceful fallback works for out-of-scope queries
- [ ] Every response includes source citations
- [ ] At least 10 documents ingested for the demo
- [ ] Code has docstrings on all public functions
- [ ] No hardcoded paths — everything uses `config/settings.py`
- [ ] GitHub repo is public and all 3 deliverables are linked in the README

---

## Final Note on What Separates Good from Great

Most submissions will have working RAG. Fewer will have a clean architecture. Almost none will have a system that genuinely handles the "I don't know" case gracefully and explains why. That gap — between "it works" and "it works professionally" — is entirely in how you handle the edges.

Build the edges well. Document your tradeoffs honestly. That's what a senior engineer looks like.
