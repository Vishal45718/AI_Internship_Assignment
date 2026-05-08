"""
app/ingestion/chunker.py — Text chunking with metadata preservation.

Uses LangChain's RecursiveCharacterTextSplitter for paragraph-aware
chunking. Each chunk gets a unique, deterministic ID and full provenance
metadata so sources can always be traced back to the original document.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.documents import Document, DocumentChunk, ChunkMetadata
from app.utils.hashing import make_chunk_id
from configs.settings import get_settings

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Splits Documents into overlapping chunks suitable for embedding.

    Design choices:
    - RecursiveCharacterTextSplitter tries paragraph → sentence → word splits,
      preserving semantic coherence better than fixed-size slicing.
    - Chunk size and overlap are config-driven so they can be tuned without
      touching code.
    - Metadata (source, page, chunk_index) flows from Document → each Chunk.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
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
        logger.debug(
            "Chunker initialized: size=%d overlap=%d",
            self._chunk_size,
            self._chunk_overlap,
        )

    def chunk(self, document: Document) -> list[DocumentChunk]:
        """
        Split a single Document into chunks.

        Args:
            document: A fully loaded Document with content and metadata.

        Returns:
            List of DocumentChunk objects with complete provenance metadata.
        """
        if not document.content.strip():
            logger.warning("Skipping empty document: %s", document.metadata.source_file)
            return []

        raw_chunks: list[str] = self._splitter.split_text(document.content)
        # Filter out chunks that are too short to be meaningful
        raw_chunks = [c for c in raw_chunks if len(c.strip()) > 50]

        if not raw_chunks:
            logger.warning(
                "No chunks produced for '%s' — content may be too short.",
                document.metadata.source_file,
            )
            return []

        ingestion_ts = datetime.now(timezone.utc).isoformat()
        chunks: list[DocumentChunk] = []

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = make_chunk_id(
                document.metadata.source_file, idx, chunk_text
            )
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                source_file=document.metadata.source_file,
                source_type=document.metadata.source_type.value,
                file_path=document.metadata.file_path,
                page_number=document.metadata.extra.get("page_number"),
                chunk_index=idx,
                char_count=len(chunk_text),
                ingestion_timestamp=ingestion_ts,
            )
            chunks.append(DocumentChunk(content=chunk_text.strip(), metadata=meta))

        logger.debug(
            "Chunked '%s': %d raw → %d chunks",
            document.metadata.source_file,
            len(raw_chunks),
            len(chunks),
        )
        return chunks

    def chunk_many(self, documents: list[Document]) -> list[DocumentChunk]:
        """
        Chunk a list of Documents, logging aggregate statistics.

        Args:
            documents: List of loaded Documents.

        Returns:
            Flat list of all chunks from all documents.
        """
        all_chunks: list[DocumentChunk] = []
        for doc in documents:
            doc_chunks = self.chunk(doc)
            all_chunks.extend(doc_chunks)

        # Group stats by source file for logging
        stats: dict[str, int] = {}
        for c in all_chunks:
            stats[c.metadata.source_file] = stats.get(c.metadata.source_file, 0) + 1

        for source, count in stats.items():
            logger.info("Chunked '%s': %d chunks", source, count)

        logger.info(
            "Total chunks created: %d from %d documents", len(all_chunks), len(documents)
        )
        return all_chunks
