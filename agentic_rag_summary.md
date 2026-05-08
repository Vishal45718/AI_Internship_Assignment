# Agentic RAG System — Build Summary

## What Was Built

A complete, production-quality Agentic RAG system at `/home/jonsnow/Desktop/Intern_Assignment/`.

## Test Results
**40/40 unit tests passing** ✅

## File Count
- **53 Python source files** across a clean modular structure
- **4 sample documents** (TXT, Markdown, CSV) for immediate demo
- **2 documentation files** (architecture deep-dive + write-up)
- **3 CLI scripts** (ingest, query, evaluate)

## Architecture Layers

| Layer | Files | Responsibility |
|---|---|---|
| `configs/` | settings.py, prompts.py | All config and prompts centralized |
| `app/models/` | documents, retrieval, requests, responses | Domain types |
| `app/utils/` | logging, text, hashing | Shared utilities |
| `app/core/` | embedder, llm_client, vector_store, dependencies | Infrastructure |
| `app/ingestion/` | loaders, chunker, pipeline | Document → Chunks |
| `app/retrieval/` | retriever, reranker | Chunks → Top-N relevant |
| `app/generation/` | prompt_builder, response_validator | Context → Answer |
| `app/orchestration/` | rag_agent, query_analyzer, fallback_handler, memory | Coordination |
| `app/api/` | routes (health, query, ingest), middleware | HTTP layer |
| `app/evaluation/` | benchmark | Quality testing |
| `ui/` | streamlit_app.py | Chat UI |
| `scripts/` | ingest.py, query.py, evaluate.py | CLI |

## How to Run (3 Commands)

```bash
# 1. Setup
cp .env.example .env   # fill in OPENAI_API_KEY (or use LLM_PROVIDER=ollama)
pip install -r requirements.txt

# 2. Ingest sample docs
python scripts/ingest.py data/raw/

# 3. Chat
streamlit run ui/streamlit_app.py
# OR: uvicorn app.main:app --reload
# OR: python scripts/query.py
```

## Critical Design Decisions

1. **Fallback is template-driven** — Never asks the LLM to "say it doesn't know." Deterministic, hallucination-proof.
2. **Intent classifier before retrieval** — Routes out-of-scope queries immediately without touching the vector DB.
3. **Two thresholds** — Similarity threshold (retrieval gate) + confidence threshold (reranker gate). Tunable independently.
4. **Deterministic chunk IDs** — SHA-256-based. Re-ingesting same file = update, not duplicate.
5. **Dependency injection** — All services injected via constructor. Swapping LLM/VectorDB = one config line + one factory change.
