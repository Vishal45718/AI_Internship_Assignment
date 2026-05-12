# Grounded Multi-Format RAG System

A technical Retrieval-Augmented Generation (RAG) system for question answering over uploaded documents. The project integrates multi-format ingestion, semantic retrieval, reranking, and local Ollama inference to produce evidence-aware answers.

## Project Overview

This system ingests PDF, TXT, CSV, and Markdown documents, converts them into searchable chunks, stores embeddings in ChromaDB, and retrieves evidence for grounded responses. It is optimized for document-centered QA where source provenance and conservative answers are more important than open-ended chat.

## Features

| Feature | Description |
|---|---|
| Multi-format ingestion | PDF, TXT, CSV, Markdown with format-specific loaders |
| Grounded retrieval | semantic search over embedded chunks |
| Reranking | candidate passages rescored for relevance |
| Local inference | Ollama-based generation without external APIs |
| Evidence-aware responses | prompts and validation emphasize cited text |
| Frontend UI | upload, chat, streaming SSE, source display |

## Architecture

```text
Upload → Parsing → Chunking → Embeddings → ChromaDB → Retrieval → Reranking → Prompt Construction → Ollama → Grounded Response
```

- **Upload**: frontend sends files to FastAPI upload endpoint
- **Parsing**: format-specific document loaders extract text
- **Chunking**: text is split into retrieval-friendly chunks
- **Embeddings**: chunks are vectorized with SentenceTransformers
- **ChromaDB**: vectors and metadata are persisted for search
- **Retrieval**: query vectors retrieve candidate chunks
- **Reranking**: candidates are rescored to improve precision
- **Prompt construction**: evidence blocks are assembled
- **Ollama**: local LLM generates answers from retrieved context

## Tech Stack

- **Backend**: Python, FastAPI
- **Frontend**: Next.js, React, TypeScript
- **Embedding**: SentenceTransformers
- **Vector store**: ChromaDB
- **Local LLM**: Ollama
- **Persistence**: SQLite via SQLAlchemy

## Setup Instructions

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ollama model

```bash
ollama pull phi3:mini
```

### 3. Backend configuration

```bash
cp .env.example .env
```

Set required values:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini
```
Local inference through Ollama was used to reduce external API dependency, support offline execution, and provide a cost-efficient deployment pipeline for experimentation and testing.

### 4. Frontend configuration

```bash
cd frontend
npm install
cp .env.local.example .env.local
```

Set:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

### 5. Run services

**Backend**:

```bash
source .venv/bin/activate
python run_server.py --reload
```

**Frontend**:

```bash
cd frontend
npm run dev
```

## Example Queries

### Research PDFs
- “What challenges in Retrieval-Augmented Generation are discussed in the paper?”
- “How does the paper categorize retrieval feedback mechanisms?”
- “What limitations of retrieval strategies are mentioned?”

### Markdown Documents
- “How are blockquotes represented in Markdown?”
- “What are the two styles of Markdown headers?”

### CSV Files
- “What columns are present in the dataset?”
- “How many customer records are available?”

### TXT Files
- “Summarize the uploaded text file.”
- “What repeating patterns are present in the text?”

## Hallucination Prevention

- **Evidence-first prompts** force the model to rely on retrieved chunks.
- **Fallback responses** are returned when retrieval confidence is low.
- **Post-generation validation** checks output against evidence overlap.
- **Reranking** improves passage precision before generation.

## Challenges Faced

- balancing chunk size and overlap for PDFs and text
- preventing CSV row explosion during ingestion
- managing frontend timeouts for long-running uploads and inference
- preserving grounding while using a lightweight local LLM

## Limitations

- local Ollama models have constrained reasoning capacity
- PDF extraction quality depends on the source document
- tables and scanned content are not explicitly handled
- retrieval precision remains a main driver of answer quality

## Future Improvements

- stronger reranking models and hybrid fusion
- conversational memory for follow-up questions
- table-aware document processing
- larger local or hosted LLM integration
- improved CSV/table semantic parsing

## API Endpoints

| Route | Method | Purpose |
|---|---|---|
| `/api/upload` | POST | upload documents for ingestion |
| `/api/chat` | POST | send queries and stream responses |
| `/api/status` | GET | inspect indexed documents and pipeline status |
| `/api/history` | GET | list recent conversations |

## Testing

- unit tests validate ingestion, retrieval, grounding, and hybrid behavior
- CSV row ingestion is verified for compact chunk counts
- retrieval tests cover semantic and hybrid search patterns
- grounding tests confirm fallback and evidence-aware output

## Conclusion

This project demonstrates a grounded RAG pipeline with multi-format ingestion, semantic retrieval, reranking, and local LLM inference through Ollama. The system emphasizes evidence-aware answering, retrieval quality, and practical engineering tradeoffs for document-based question answering workflows.