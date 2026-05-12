# Testing

## Test objectives

- verify file ingestion across supported formats
- confirm chunk counts and metadata behavior
- validate semantic retrieval and reranking
- ensure grounding and fallback logic behave as expected
- check frontend timeout and upload progress handling

## Test files

- `tests/test_parent_retrieval.py`
- `tests/test_hybrid_retrieval.py`
- `tests/test_strict_retrieval_pipeline.py`
- `tests/test_grounding.py`

## Focus areas

### Ingestion

- CSV rows are loaded as individual documents
- Markdown formatting is cleaned before chunking
- PDF page boundaries are preserved in metadata

### Chunking

- text documents are split into overlapping semantic chunks
- CSV documents produce compact row-level chunks
- chunk IDs are stable across reingestion

### Retrieval

- semantic matching retrieves relevant passages
- hybrid retrieval combines dense and keyword signals
- reranking improves passage selection

### Grounded output

- evidence blocks are included in prompts
- fallback responses appear when evidence is weak
- unsupported details are reduced by validation checks

## Example scenarios

- retrieving specific terms from acronym-heavy documents
- verifying CSV row answer accuracy
- confirming Markdown content is embedded cleanly
- checking that uploads do not abort prematurely

## Manual evaluation

Validate the system with realistic documents and queries that exercise:

- research papers with acronyms and method names
- CSV datasets with structured rows
- Markdown notes and formatted text
- plain text technical summaries

## Limitations in evaluation

- local Ollama behavior depends on model size and hardware
- PDF extraction varies by source quality
- retrieval precision is the primary determinant of answer usefulness

## Recommended verification

- compare generated answers against retrieved chunk sources
- inspect `ChromaDB` metadata for correct source attribution
- run ingestion for all supported formats and verify chunk counts
- exercise the chat UI with document mode and review source citations
