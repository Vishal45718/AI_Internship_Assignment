# Testing and Evaluation

## Testing strategy

This project uses focused module-level tests to verify retrieval, grounding, and ingestion behavior. The test suite is designed to validate both component behavior and RAG quality properties.

### Test coverage areas

- **Ingestion**: file loading and CSV row handling
- **Chunking**: proper chunk counts and section handling
- **Retrieval**: semantic matching and hybrid search behavior
- **Grounding**: evidence blocks and fallback responses
- **Pipeline flow**: integration of retrieval, prompt construction, and model generation logic

## Relevant test files

- `tests/test_parent_retrieval.py`
- `tests/test_hybrid_retrieval.py`
- `tests/test_strict_retrieval_pipeline.py`
- `tests/test_grounding.py`

## Example evaluation scenarios

### SeaKR and DRAGIN queries

Test queries should verify that the system retrieves and cites relevant passages rather than relying on generic model output.

Examples:
- “What retrieval conditions does SeaKR use?”
- “How does DRAGIN reformulate the query for reranking?”

### Markdown syntax handling

Verify that Markdown files are cleaned and embedded without formatting artifacts.
- headings should be stripped
- code blocks should not inject noise into embeddings
- inline links should preserve visible text only

### CSV structure queries

Confirm that CSV rows are treated as compact documents rather than recursively chunked.
- 50 rows should produce roughly 50–100 chunks, not thousands
- row-level metadata should be preserved
- queries should return row-level evidence from CSV content

## Grounded-answer verification

- verify that citations are attached to answers when applicable
- ensure fallback text appears when retrieval confidence is too low
- confirm that the system does not return unsupported technical detail without evidence

## Runtime / integration checks

- upload large CSV files and verify no premature frontend aborts
- confirm upload progress messages appear during ingestion
- validate chat timeouts are long enough for Ollama inference and retrieval

## Manual evaluation notes

The system should be evaluated with representative document sets, including:
- research papers with acronym-heavy content
- CSV datasets with structured rows
- Markdown notes and technical documentation
- plain text research summaries

Manual checks should focus on:
- correct passage retrieval
- source citation accuracy
- conservative behavior when evidence is weak
- response fluency within model limitations

## Limitations in tests

- local Ollama inference may vary by model and hardware
- PDF extraction quality depends on the source document
- retrieval precision is a major factor in end-to-end answer quality

For a stronger assessment, combine unit tests with real-document manual queries and inspect actual retrieved chunks.
