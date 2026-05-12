# Agentic RAG — Document Q&A with Grounded Answers

This repository implements a grounded Retrieval-Augmented Generation (RAG) system for question answering over uploaded documents. It combines a React/Next.js frontend, FastAPI backend, local embeddings, a Chroma vector store, and Ollama-based local inference.

## What this project does

- Ingests PDF, TXT, CSV, and Markdown files
- Converts documents into searchable chunks with provenance metadata
- Embeds chunks using SentenceTransformers
- Stores vectors in ChromaDB
- Retrieves and reranks candidate passages for each query
- Generates answer text with a local Ollama LLM
- Surfaces evidence-backed responses and conservative fallbacks when support is weak

## Why RAG matters

RAG is useful for document QA because it grounds answers in uploaded source material rather than relying solely on pretrained model knowledge. This is especially important for research papers, technical notes, and structured data where specific names, thresholds, and details must be drawn from the actual corpus.

## Supported formats

- PDF
- TXT
- CSV
- Markdown

CSV support is implemented as row-level ingestion rather than recursive chunking, which keeps chunk counts compact and avoids excessive vector generation.

## Local inference with Ollama

The pipeline uses Ollama for local LLM inference to support offline or private deployments. Local inference reduces dependency on third-party APIs and keeps cost predictable, while the system design balances that with retrieval quality and grounded prompting.

## Documentation

- `docs/system_design.md` — design goals, component responsibilities, and engineering decisions
- `docs/architecture.md` — architecture flow, retrieval pipeline, grounding strategy, and diagrams
- `docs/testing.md` — testing approach, evaluation scenarios, and limitations

## Quick setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Ollama installed and running

### Python setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Ollama model

```bash
ollama pull phi3:mini
```

### Backend environment

```bash
cp .env.example .env
```

Set at minimum:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini
```

### Frontend setup

```bash
cd frontend
npm install
cp .env.local.example .env.local
```

Configure:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

### Run the app

Start the backend:

```bash
source .venv/bin/activate
python run_server.py --reload
```

Start the frontend:

```bash
cd frontend
npm run dev
```

Open the UI at `http://localhost:3000`.

## API routes

<<<<<<< HEAD
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
=======
- `POST /api/upload` — upload a document for ingestion
- `POST /api/chat` — submit a query and receive a streamed response
- `GET /api/status` — inspect pipeline status and indexed documents
- `GET /api/history` — list recent conversations
>>>>>>> 7c76c39 (Implement grounded multi-format RAG pipeline with documentation)

## Project structure

```text
Intern_Assignment/
├── backend/              # FastAPI application and storage layer
├── frontend/             # Next.js frontend UI
├── src/                  # Core RAG pipeline implementation
│   ├── ingestion/        # document loaders and chunking
│   ├── embeddings/       # embedding wrapper
│   ├── vectordb/         # Chroma vector store integration
│   ├── retrieval/        # search and rerank logic
│   └── llm/              # prompts, grounding, and LLM client
├── tests/                # unit and pipeline tests
├── data/                 # raw documents and Chroma persistence
├── docs/                 # technical documentation
├── run_server.py         # backend entrypoint
├── requirements.txt
└── README.md
```

## Notes

This documentation is intended for internship-quality technical evaluation. It focuses on system design, implementation, retrieval flow, and grounded response behavior without marketing language.
