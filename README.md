# Agentic RAG — Document Q&A with Grounded Answers

A full-stack **Retrieval-Augmented Generation (RAG)** system for asking questions over your own documents. The backend ingests PDFs and other text sources, embeds them into a vector store, retrieves the most relevant passages, and generates answers that are **tied to retrieved evidence** to reduce unsupported claims.

**Why RAG:** Large language models can generalize from training data; for paper-specific facts, method names, and thresholds, RAG grounds answers in *your* uploaded text instead of guessing.

**Focus:** Grounded document Q&A suited to research-style PDFs and technical notes, with retrieval quality signals and conservative behavior when evidence is weak.

---

## Features

| Area | Description |
|------|-------------|
| **Document ingestion** | Load PDF, TXT, CSV, Markdown (and related) formats through the API. |
| **Chunking** | Configurable text splitting with overlap for retrieval granularity. |
| **Embeddings** | Local **SentenceTransformers** embeddings (default: `all-MiniLM-L6-v2`). |
| **Chroma vector DB** | Persistent storage and similarity search over chunks. |
| **Hybrid retrieval** | Optional fusion of dense (vector) and keyword-style retrieval. |
| **Reranking** | Cross-encoder reranking to better align passages with the query. |
| **Section-aware retrieval** | Context expansion using document structure (parent chunks / sections where enabled). |
| **Grounded answering** | Prompts and checks steer the model to cite and stay within evidence. |
| **Hallucination reduction** | Pre/post validation, entity-aware checks, and cautious fallbacks when support is weak. |
| **Ollama local inference** | Run a local model (e.g. **phi3:mini**) without a cloud LLM API. |
| **Frontend UI** | Next.js app: chat modes, upload, streaming responses, sources. |
| **PDF upload** | Upload PDFs from the UI for indexing and Q&A. |

---

## Architecture

End-to-end flow:

```text
Upload → Chunking → Embeddings → Vector DB → Retrieval → Reranking → LLM Generation → Grounded Answer
```

1. **Upload** — Files are received by the FastAPI backend and passed into the ingestion pipeline.  
2. **Chunking** — Documents are split into overlapping chunks suitable for embedding and citation.  
3. **Embeddings** — Each chunk is embedded with SentenceTransformers.  
4. **Vector DB** — Embeddings and metadata are stored in **ChromaDB** for similarity search.  
5. **Retrieval** — Queries retrieve top candidates; optional **hybrid** search improves recall on exact terms.  
6. **Reranking** — A cross-encoder rescores candidates so the strongest passages surface first.  
7. **LLM generation** — A local model via **Ollama** (or an alternate provider configured in `.env`) produces an answer conditioned on retrieved text.  
8. **Grounded answer** — Evidence-first instructions and validation reduce answers that drift from the corpus.

```text
┌─────────────┐     HTTP/SSE      ┌──────────────┐     ┌─────────────┐
│  Next.js    │ ◄──────────────► │   FastAPI    │ ◄─► │  ChromaDB   │
│  (React UI) │                  │   (Python)   │     │  (vectors)  │
└─────────────┘                  └──────┬───────┘     └─────────────┘
                                        │
                                        ▼
                                 ┌──────────────┐
                                 │   Ollama     │
                                 │  (local LLM) │
                                 └──────────────┘
```

---

## Tech Stack

| Layer | Technologies |
|--------|----------------|
| Backend | **Python**, **FastAPI** |
| Vector store | **ChromaDB** |
| Embeddings | **SentenceTransformers** |
| Inference (local) | **Ollama** |
| Frontend | **Next.js**, **React**, **TypeScript** |

Additional libraries include SQLAlchemy/SQLite for conversation history, `httpx` for Ollama calls, and `pypdf` for PDF text extraction.

---

## Setup

### Prerequisites

- Python 3.11+ recommended  
- Node.js 18+ (for the frontend)  
- [Ollama](https://ollama.com) installed and running  

### 1. Clone and virtual environment

```bash
git clone <YOUR_REPOSITORY_URL>
cd Intern_Assignment

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ollama and model

Install Ollama, then pull the default local model used in the example config:

```bash
ollama pull phi3:mini
```

Ensure the Ollama service is running (typically `http://localhost:11434`).

### 3. Backend environment

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini
```

Other keys (embedding path, Chroma location, retrieval thresholds) can remain as in `.env.example` until you tune them.

### 4. Frontend environment

```bash
cd frontend
cp .env.local.example .env.local
```

Default API base (matches local backend):

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Install dependencies:

```bash
npm install
```

### 5. Run backend and frontend

**Terminal 1 — API**

```bash
source .venv/bin/activate
python run_server.py --reload
```

API: `http://127.0.0.1:8000` (health: `GET /`).

**Terminal 2 — UI**

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000` (or the port shown by Next.js).

---

## Example Queries

After uploading one or more documents, try queries such as:

### Research PDF
- “What is Test-time Corpus Feedback in RAG systems?”
- “How does the paper categorize feedback mechanisms in Retrieval-Augmented Generation?”
- “What challenges in Retrieval-Augmented Generation are discussed in the paper?”

### Markdown Documents
- “How are blockquotes represented in Markdown?”
- “What are the two styles of headers supported in Markdown?”

### CSV Files
- “What columns are present in the dataset?”
- “How many customer records are available?”

### TXT Files
- “Summarize the uploaded text file.”
- “What repeating patterns are present in the text?”

The system is designed to generate grounded responses strictly from retrieved evidence. If sufficient supporting context is unavailable, the model responds cautiously instead of generating unsupported claims.

---

## Hallucination prevention (design)

- **Grounded answering** — User and system prompts require using retrieved evidence, not general world knowledge, for paper-specific entities.  
- **Evidence-based generation** — Retrieved chunks/sentences are injected into the prompt so the model conditions on citeable text.  
- **Cautious fallback responses** — When retrieval is weak or evidence does not support the question, the pipeline can abstain or give short, honest limitations instead of fabricating facts.  
- **Rejection of unsupported claims** — Post-generation checks flag unsupported terms, risky expansions, and low overlap with evidence; outputs may be regenerated, pruned, or replaced with a safe response.

This does **not** guarantee correctness on every run; it **reduces** the rate of unsupported statements compared to a plain chat model.

---

## Limitations

- **Acronym-heavy papers** — Models may still confuse similar methods; strict grounding depends on retrieval finding the right spans.  
- **Retrieval precision** — Wrong or partial matches still lead to incomplete or misleading answers.  
- **OCR / tables** — PDF text extraction may miss layout, tables, or figures; answers are only as good as the extracted text.  
- **Small local models** — `phi3:mini` and similar models may miss nuanced reasoning or long-range dependencies even when retrieval is good.

---

## Future improvements

- Stronger **reranking** and fusion strategies (learned rerankers, ColBERT-style retrieval).  
- **Multimodal** retrieval (figures, slides) where applicable.  
- **Conversational memory** with clear separation between “chat” and “evidence-backed” turns.  
- More **agentic** workflows (e.g. query decomposition, multi-step retrieval) with guardrails.  
- Larger or more capable **local** or **hosted** LLMs when hardware or API budget allows.

---

## Project structure

```text
Intern_Assignment/
├── backend/
│   └── app/
│       ├── api/           # REST routes (chat, upload, history, status)
│       ├── storage/       # Conversation / SQLite persistence
│       └── main.py        # FastAPI app + CORS
├── frontend/
│   ├── src/
│   │   └── app/           # Next.js App Router (page UI)
│   ├── public/
│   ├── package.json
│   └── .env.local.example
├── src/                   # Core RAG library
│   ├── ingestion/         # Loaders, chunking
│   ├── embeddings/        # Embedding model wrapper
│   ├── vectordb/          # Chroma integration
│   ├── retrieval/         # Hybrid search, reranking, sentence evidence
│   ├── llm/               # Prompts, clients, grounding helpers
│   └── pipeline.py        # Orchestration
├── tests/                 # pytest suite (retrieval, grounding, hybrid, etc.)
├── data/
│   ├── raw/               # Sample / local documents (see data/raw/README.md)
│   └── vectorstore/       # Chroma persistence (generated)
├── requirements.txt
├── run_server.py          # Uvicorn entrypoint
├── .env.example
└── README.md
```

---

## Demo

| Asset | Link / note |
|--------|----------------|
| **Demo video** | *[Add link to screen recording or Loom]* |
| **Screenshots** | *[Add images or link to folder]* |

---

## License

If this repository is private coursework, clarify license with your instructor. Otherwise add a `LICENSE` file as required.

---

*Internship / coursework submission — engineering-focused RAG stack with grounded document Q&A.*
