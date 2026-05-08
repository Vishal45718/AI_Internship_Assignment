"""
tests/unit/test_chunker.py — Unit tests for the document chunking module.

Tests verify:
1. Basic chunking produces expected output structure
2. Metadata is correctly propagated to each chunk
3. Chunk IDs are deterministic
4. Short content is handled gracefully
5. Empty content is handled gracefully
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.ingestion.chunker import DocumentChunker
from app.models.documents import Document, DocumentMetadata, DocumentType


def make_test_document(content: str, filename: str = "test.txt") -> Document:
    """Helper to create a test Document."""
    metadata = DocumentMetadata(
        source_file=filename,
        source_type=DocumentType.TXT,
        file_path=f"/tmp/{filename}",
    )
    return Document(content=content, metadata=metadata)


class TestDocumentChunker:
    """Unit tests for DocumentChunker."""

    def setup_method(self):
        """Create a chunker with small sizes for predictable test output."""
        self.chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)

    def test_basic_chunking_produces_chunks(self):
        """Long text should be split into multiple chunks."""
        content = "This is a test sentence. " * 50  # ~1250 chars
        doc = make_test_document(content)
        chunks = self.chunker.chunk(doc)
        assert len(chunks) > 1, "Expected multiple chunks for long content"

    def test_short_content_produces_single_chunk(self):
        """Content shorter than chunk_size should produce exactly one chunk."""
        # Must be > 50 chars (the minimum meaningful chunk length filter) but < chunk_size
        content = "This is a short document that fits within a single chunk and has enough content to pass the minimum length filter."
        doc = make_test_document(content)
        chunks = self.chunker.chunk(doc)
        assert len(chunks) == 1

    def test_empty_content_produces_no_chunks(self):
        """Empty or whitespace-only content should produce no chunks."""
        doc = make_test_document("")
        chunks = self.chunker.chunk(doc)
        assert chunks == []

    def test_metadata_propagated_to_chunks(self):
        """Every chunk must inherit source_file from the parent document."""
        content = "First paragraph.\n\n" + "Word " * 100 + "\n\n" + "Third paragraph."
        doc = make_test_document(content, filename="my_document.txt")
        chunks = self.chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.metadata.source_file == "my_document.txt"
            assert chunk.metadata.source_type == "txt"

    def test_chunk_ids_are_unique(self):
        """All chunk IDs within a document must be unique."""
        content = ("This is sentence number {}. " * 5).format(*range(30))
        doc = make_test_document(content)
        chunks = self.chunker.chunk(doc)
        ids = [c.metadata.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_ids_are_deterministic(self):
        """Re-chunking the same document must produce the same IDs."""
        content = "Deterministic content for testing purposes. " * 20
        doc = make_test_document(content)
        chunks1 = self.chunker.chunk(doc)
        chunks2 = self.chunker.chunk(doc)
        ids1 = [c.metadata.chunk_id for c in chunks1]
        ids2 = [c.metadata.chunk_id for c in chunks2]
        assert ids1 == ids2, "Chunking must be deterministic"

    def test_chunk_index_is_sequential(self):
        """Chunk indices within a document must be sequential starting at 0."""
        content = "Paragraph content. " * 40
        doc = make_test_document(content)
        chunks = self.chunker.chunk(doc)
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_char_count_in_metadata(self):
        """char_count metadata should match actual content length."""
        content = "Testing chunk metadata char count. " * 10
        doc = make_test_document(content)
        chunks = self.chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.metadata.char_count == len(chunk.content)

    def test_page_number_preserved_from_pdf(self):
        """Page numbers in PDF document metadata should propagate to chunks."""
        metadata = DocumentMetadata(
            source_file="test.pdf",
            source_type=DocumentType.PDF,
            file_path="/tmp/test.pdf",
            extra={"page_number": 7},
        )
        doc = Document(content="PDF page content. " * 20, metadata=metadata)
        chunks = self.chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.metadata.page_number == 7

    def test_chunk_many_handles_multiple_documents(self):
        """chunk_many should aggregate chunks from all documents."""
        docs = [make_test_document(f"Document {i} content. " * 20) for i in range(3)]
        chunks = self.chunker.chunk_many(docs)
        assert len(chunks) >= 3, "Expected at least one chunk per document"
