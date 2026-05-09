# 📄 Full-Stack Agentic RAG — ChatGPT-style Document Q&A

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-15+-black?logo=next.js)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)](https://www.trychroma.com)

## Overview

A complete, production-ready **Retrieval-Augmented Generation (RAG)** web application with a ChatGPT-style UI. 
The system features two chat modes:
1. **General AI**: Conversational AI assistant.
2. **Document Chat**: Grounded Q&A against your uploaded documents with strict hallucination prevention and source citations.

## Architecture

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Next.js (UI)   │ ────> │  FastAPI (API)  │ ────> │    ChromaDB     │
│  TailwindCSS    │ <──── │  Streaming SSE  │ <──── │  (Vector DB)    │
└─────────────────┘       └─────────────────┘       └─────────────────┘
                                   │
                                   v
                          LLM (OpenAI / Gemini / Ollama)
```

## Features

- **ChatGPT-style Web UI**: Beautiful responsive design with dark mode, markdown rendering, and real-time streaming.
- **Two Chat Modes**: Toggle between General AI and Ask Documents seamlessly.
- **Multi-format upload**: Upload PDF, TXT, CSV, or Markdown directly from the UI.
- **Persistent Conversation History**: Chats are saved via an SQLite backend.
- **Hallucination prevention**: Built-in similarity thresholding ensures the LLM admits when it doesn't know.
- **Source citations**: Responses cite the exact file and similarity score used to generate the answer.
- **Robust Error Handling**: Fails gracefully on invalid API keys without retrying blindly.

## Setup Instructions

### 1. Backend Setup (Python)

```bash
# Clone the repository
git clone https://github.com/your-username/rag-system.git
cd rag-system

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```
Edit `.env` and add your `OPENAI_API_KEY` (or configure Gemini / Ollama).

### 2. Frontend Setup (Node.js)

```bash
cd frontend
npm install
```

## Running the Application

You need two terminal windows to run both the backend and frontend.

**Terminal 1 (Backend)**:
```bash
source .venv/bin/activate
python run_server.py --reload
```
*API runs at http://localhost:8000*

**Terminal 2 (Frontend)**:
```bash
cd frontend
npm run dev
```
*UI runs at http://localhost:3000*

Open `http://localhost:3000` in your browser.

## VS Code Environment Configuration

If you encounter API Key issues, ensure your VS Code terminal loads `.env` variables automatically:
1. Open VS Code Settings (`Ctrl + ,`)
2. Search for: `python.terminal.useEnvFile`
3. Check the box to enable it.
4. Restart your terminal.

## Project Structure

```
project/
├── backend/                  # FastAPI Application Layer
│   └── app/
│       ├── api/              # API Routers (Chat, Upload, History)
│       ├── storage/          # SQLite DB schema and session
│       └── main.py           # FastAPI entry point
│
├── frontend/                 # Next.js Application Layer
│   ├── src/app/              # Next.js Pages (page.tsx, globals.css)
│   └── package.json          
│
├── src/                      # Core RAG Pipeline (Reusable Module)
│   ├── ingestion/            # PDF/CSV loaders and Chunker
│   ├── embeddings/           # sentence-transformers
│   ├── vectordb/             # ChromaDB interface
│   ├── retrieval/            # Semantic search
│   ├── llm/                  # Streaming LLM clients (OpenAI/Gemini/Ollama)
│   └── pipeline.py           # Core orchestrator logic
│
├── data/                     # Vector DB and Uploaded Files (ignored in git)
├── run.py                    # Legacy CLI entry point
└── run_server.py             # FastAPI entry point
```
