"""
src/ingestion/chunker.py — Text chunking with metadata preservation.

Uses LangChain's RecursiveCharacterTextSplitter for paragraph-aware
chunking. Each chunk gets a deterministic ID and full provenance
metadata so sources can always be traced back to the original document.

Default settings come from config (fine-grained chunk_size / overlap).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models import Document, DocumentChunk, ChunkMetadata
from src.utils.hashing import make_chunk_id
from src.config import get_settings

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Splits Documents into overlapping chunks suitable for embedding.

    The RecursiveCharacterTextSplitter tries paragraph → sentence → word splits,
    preserving semantic coherence better than fixed-size slicing.
    """

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        settings = get_settings()
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )
        logger.debug("Chunker: size=%d overlap=%d", self._chunk_size, self._chunk_overlap)

    def chunk(self, document: Document) -> list[DocumentChunk]:
        """
        Split a single Document into chunks.

        Args:
            document: A loaded Document with content and metadata.

        Returns:
            List of DocumentChunk objects with provenance metadata.
        """
        if not document.content.strip():
            logger.warning("Skipping empty document: %s", document.metadata.source_file)
            return []

        raw_chunks: list[str] = self._splitter.split_text(document.content)
        # Filter out very short chunks (< 50 chars) that aren't meaningful
        raw_chunks = [c for c in raw_chunks if len(c.strip()) > 35]

        if not raw_chunks:
            logger.warning("No chunks for '%s' — content may be too short.", document.metadata.source_file)
            return []

        ingestion_ts = datetime.now(timezone.utc).isoformat()
        chunks: list[DocumentChunk] = []

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = make_chunk_id(document.metadata.source_file, idx, chunk_text)
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                source_file=document.metadata.source_file,
                source_type=document.metadata.source_type.value,
                file_path=document.metadata.file_path,
                page_number=document.metadata.extra.get("page_number"),
                chunk_index=idx,
                char_count=len(chunk_text),
                ingestion_timestamp=ingestion_ts,
                parent_id=document.id,
                document_name=document.metadata.source_file,
                section_title=document.metadata.extra.get("section_title", ""),
            )
            chunks.append(DocumentChunk(content=chunk_text.strip(), metadata=meta))

        logger.debug("Chunked '%s': %d chunks", document.metadata.source_file, len(chunks))
        return chunks

    def chunk_many(self, documents: list[Document]) -> list[DocumentChunk]:
        """
        Chunk a list of Documents.

        Args:
            documents: List of loaded Documents.

        Returns:
            Flat list of all chunks from all documents.
        """
        all_chunks: list[DocumentChunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk(doc))

        logger.info("Total: %d chunks from %d documents", len(all_chunks), len(documents))
        return all_chunks
