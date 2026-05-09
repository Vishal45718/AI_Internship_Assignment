# 📄 RAG System — Document Q&A with Hallucination Prevention

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)](https://www.trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

A clean, modular **Retrieval-Augmented Generation (RAG)** system that answers questions strictly from your uploaded documents. The system ingests documents in multiple formats, chunks and embeds them into a local vector database, and uses semantic retrieval + LLM generation to provide grounded, source-cited answers.

**Key principle:** If the answer isn't in the documents, the system says so honestly — no hallucination.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                      │
│                                                             │
│  PDF/TXT/CSV/MD → Load → Clean → Chunk → Embed → Store     │
│                                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                     ChromaDB (local)
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                      QUERY PIPELINE                         │
│                                                             │
│  Question → Retrieve → Threshold Check → Prompt → LLM →    │
│                  ↓                          ↓               │
│           Below threshold?           Above threshold?       │
│                  ↓                          ↓               │
│         "Not found in docs"         Grounded Answer +       │
│         (no LLM call)               Source Citations        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-format ingestion**: PDF, TXT, Markdown, CSV
- **Smart chunking**: Paragraph-aware splitting (500 chars, 50 overlap)
- **Local embeddings**: `sentence-transformers/all-MiniLM-L6-v2` — no API costs
- **Persistent vector store**: ChromaDB with cosine similarity search
- **Similarity threshold filtering**: Only uses chunks above confidence threshold
- **Hallucination prevention**: Three-layer defense system
- **Source citations**: Every answer references the source document
- **Multiple LLM providers**: OpenAI, Google Gemini, or Ollama (local)
- **Simple CLI**: Ingest docs and ask questions from the terminal
- **Idempotent ingestion**: Re-ingesting updates, doesn't duplicate

## Folder Structure

```
project/
│
├── data/
│   ├── raw/                  # Place your documents here
│   └── vectorstore/          # ChromaDB persistence (auto-created)
│
├── src/
│   ├── ingestion/
│   │   ├── loaders.py        # PDF, TXT, CSV, Markdown loaders
│   │   └── chunker.py        # Text splitting with metadata
│   ├── processing/
│   │   └── text.py           # Text cleaning & normalization
│   ├── embeddings/
│   │   └── embedder.py       # Sentence-transformer embeddings
│   ├── vectordb/
│   │   └── store.py          # ChromaDB vector store wrapper
│   ├── retrieval/
│   │   └── retriever.py      # Semantic search + threshold filter
│   ├── llm/
│   │   ├── client.py         # LLM provider abstraction
│   │   └── prompts.py        # Prompt templates
│   ├── utils/
│   │   ├── hashing.py        # Content hashing for dedup
│   │   └── logging.py        # Colored terminal logging
│   ├── config.py             # Centralized configuration
│   ├── models.py             # Data models (Pydantic)
│   └── pipeline.py           # Main RAG pipeline orchestrator
│
├── run.py                    # CLI entry point
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/rag-system.git
cd rag-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

## Setup

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Configure your LLM provider** (edit `.env`):

   - **OpenAI** (recommended): Set `LLM_PROVIDER=openai` and add your `OPENAI_API_KEY`
   - **Google Gemini**: Set `LLM_PROVIDER=gemini` and add your `GEMINI_API_KEY`
   - **Ollama** (free, local): Set `LLM_PROVIDER=ollama` and [install Ollama](https://ollama.ai)

3. **Place documents** in the `data/raw/` folder (PDF, TXT, CSV, or MD files)

## Usage

### Ingest Documents

```bash
# Ingest all documents from data/raw/
python run.py ingest data/raw/

# Ingest specific files
python run.py ingest report.pdf faq.txt data.csv

# Check what's indexed
python run.py stats
```

### Chat / Ask Questions

```bash
# Interactive mode (REPL)
python run.py chat

# One-shot query
python run.py chat "What is the return policy?"

# With debug info (shows retrieval scores)
python run.py chat --debug "How do I reset my password?"
```

### Index Management

```bash
# View index statistics
python run.py stats

# Reset entire index (deletes all chunks)
python run.py reset
```

## Example Queries

```
You: What is the return policy for electronics?
✓ [SUCCESS]
──────────────────────────────────────────────────────────────
According to the documents, the electronics return policy allows
returns within 15 days of purchase. [Source: sample_sop.txt]

📚 Sources:
  [1] sample_sop.txt  — score: 0.847

You: What is the weather in Tokyo?
⚠️ [NO_RELEVANT_CONTEXT]
──────────────────────────────────────────────────────────────
I could not find relevant information in the uploaded documents.
```

## Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.11+ | Core runtime |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local vector generation (384-dim) |
| Vector DB | ChromaDB | Persistent similarity search |
| LLM | OpenAI / Gemini / Ollama | Answer generation |
| Text Splitting | LangChain Text Splitters | Paragraph-aware chunking |
| PDF Parsing | pypdf | PDF text extraction |
| Configuration | Pydantic Settings | Type-safe env loading |

## Hallucination Prevention

The system uses a **three-layer defense** to prevent the LLM from making up information:

### Layer 1: Similarity Threshold
Retrieved chunks are filtered by a configurable similarity threshold (default: 0.35). If no chunks pass the threshold, the LLM is **never called** — preventing it from receiving weak context and confabulating.

### Layer 2: Grounded System Prompt
The LLM receives strict instructions to answer ONLY from the provided context. It must cite sources and respond with a specific fallback phrase if the context is insufficient.

### Layer 3: Template-Driven Fallback
When retrieval confidence is low, the fallback response is a **Python template** — not LLM-generated. This is 100% deterministic and impossible to hallucinate. The system will respond:

> *"I could not find relevant information in the uploaded documents."*

## Configuration

All settings are in `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM backend (`openai`, `gemini`, `ollama`) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.70` | Minimum relevance score |

## Limitations

- **PDF complexity**: Works best with single-column text PDFs. Multi-column layouts or scanned documents may have extraction issues.
- **Tabular data**: CSV rows converted to prose embed reasonably well, but complex tables may lose structure.
- **Long documents**: For very long documents (50+ pages), answers requiring synthesis across distant sections may miss context.
- **First query latency**: The embedding model loads on first use (~2-3 seconds). Subsequent queries are fast.

## Future Improvements

- Hybrid retrieval (BM25 + semantic) for better keyword matching
- Cross-encoder reranking for improved precision
- Streaming LLM responses for lower perceived latency
- Web UI (Streamlit) for non-technical users
- Multi-language document support
- OCR support for scanned PDFs

## License

This project is licensed under the MIT License. See `LICENSE` for details.
