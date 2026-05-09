project/
├── data/raw/                 # Place documents here
├── data/vectorstore/         # ChromaDB persistence
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loaders.py        # Keep existing (cleaned)
│   │   └── chunker.py        # Keep existing (chunk_size=500)
│   ├── processing/
│   │   ├── __init__.py
│   │   └── text.py           # Keep text cleaning utils
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py       # Keep existing (cleaned)
│   ├── vectordb/
│   │   ├── __init__.py
│   │   └── store.py          # Keep existing ChromaDB code
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py      # Keep existing + simplify
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py         # Keep existing LLM client
│   │   └── prompts.py        # Move prompts here
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── hashing.py        # Keep
│   │   └── logging.py        # Keep
│   ├── config.py             # Simplified settings
│   └── pipeline.py           # New: clean RAG pipeline (replaces rag_agent)
├── run.py                    # CLI entry point (ingest + chat)
├── requirements.txt          # Minimal dependencies
├── README.md                 # Complete rewrite
└── .env.example              # Simplified

