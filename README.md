# Agentic RAG System

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)](https://www.trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Agentic RAG is a professional Retrieval-Augmented Generation system built for reliability, transparency, and modularity.
It indexes documents from common formats, performs semantic retrieval and reranking, and generates grounded answers with inline citations.
The system is designed to fail safely with deterministic fallback responses rather than hallucinate when the indexed content is insufficient.

## Features

- Multi-format ingestion: PDF, TXT, Markdown, and CSV
- Safe chunking with metadata-rich provenance
- Configurable embedding backends: OpenAI or local Sentence-Transformers
- Two-stage retrieval: semantic search + cross-encoder reranking
- Hallucination prevention via confidence thresholds and grounded prompts
- Structured fallback responses on low confidence or out-of-scope queries
- Explicit source citations in every answer
- Conversation memory support via session IDs
- FastAPI REST API and optional Streamlit demo UI
- CLI tools for ingestion, querying, and benchmarking
- Production-ready logging, configuration, and CI workflow

## Tech Stack

- Python 3.11
- FastAPI
- ChromaDB
- OpenAI / Google Gemini / Ollama
- sentence-transformers
- Streamlit
- pytest, ruff, mypy

## Installation

```bash
git clone https://github.com/your-username/agentic-rag.git
cd agentic-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Copy the environment template and update values:

```bash
cp .env.example .env
```

Then fill in your provider credentials and configuration values.

## Usage

### Ingest Documents

```bash
python scripts/ingest.py data/raw/
```

### Start the API

```bash
uvicorn app.main:app --reload
```

Open the API docs at `http://localhost:8000/docs`.

### Run the Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

Open the UI at `http://localhost:8501`.

## Screenshots

> Screenshot placeholders — add your own UI images here after capture.

![Screenshot placeholder](https://via.placeholder.com/1200x600?text=Streamlit+UI+Screenshot+Placeholder)

### Query the System

```bash
python scripts/query.py "What is the return policy?"
python scripts/query.py --session demo123
python scripts/query.py --debug
```

## API Reference

### `GET /health`

Returns the current health and index status.

### `POST /query`

Submit a user query to the RAG pipeline.

Example:

```bash
curl -X POST http://localhost:8000/query   -H "Content-Type: application/json"   -d '{"query": "What is the return policy?", "session_id": "demo-123"}'
```

### `POST /ingest/upload`

Upload files directly through the API.

```bash
curl -X POST http://localhost:8000/ingest/upload   -F "files=@my_document.pdf"   -F "files=@report.txt"
```

### `GET /ingest/stats`

Returns index statistics and indexed document metadata.

### `DELETE /ingest/document`

Delete indexed chunks for a specific source file.

```bash
curl -X DELETE http://localhost:8000/ingest/document   -H "Content-Type: application/json"   -d '{"source_file": "sample_sop.txt"}'
```

## CLI Reference

```bash
# Ingest
python scripts/ingest.py data/raw/
python scripts/ingest.py report.pdf faq.txt
python scripts/ingest.py --stats
python scripts/ingest.py --reset

# Query
python scripts/query.py
python scripts/query.py "Your question here"
python scripts/query.py --debug
```

## Project Structure

```text
app/                 # Application logic and API layer
configs/             # Configuration and prompt templates
data/                # Raw data, processed files, and persistent store
docs/                # Architecture and project write-up
scripts/             # CLI tooling for ingestion, querying, evaluation
tests/               # Unit and integration tests
ui/                  # Optional Streamlit demo UI
```

## Deployment

For production deployment, use a managed process supervisor and TLS termination.

Example:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Recommended enhancements:

- Use a reverse proxy such as Nginx
- Store conversation memory in Redis for scale
- Persist ChromaDB storage on durable disk
- Add a monitoring/alerting layer

## Contribution

1. Open an issue for new features or bugs.
2. Fork the repository.
3. Create a branch.
4. Add tests for new behavior.
5. Submit a pull request with a clear summary.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
