"""
src/ingestion/chunker.py — Text chunking with metadata preservation.

Uses LangChain's RecursiveCharacterTextSplitter for paragraph-aware
chunking. Each chunk gets a deterministic ID and full provenance
metadata so sources can always be traced back to the original document.

Default settings come from config (fine-grained chunk_size / overlap).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models import Document, DocumentChunk, ChunkMetadata, DocumentType
from src.utils.hashing import make_chunk_id
from src.config import get_settings

logger = logging.getLogger(__name__)
_HEADING_LINE = re.compile(r"^\s{0,3}(?:\d+(?:\.\d+){0,3}\s+)?[A-Z][A-Za-z0-9\s\-/()]{2,80}$")
_METHOD_ENTITY_HEADING = re.compile(r"\b(SeaKR|DRAGIN|ReAL|FLARE|CRAG)\b", re.IGNORECASE)


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

    def _is_heading(self, line: str) -> bool:
        s = line.strip()
        if len(s) < 3 or len(s) > 100:
            return False
        if s.endswith((".", ":", ";", ",")):
            return False
        if _METHOD_ENTITY_HEADING.search(s):
            return True
        words = s.split()
        if len(words) > 16:
            return False
        if s.startswith("#"):
            return True
        if _HEADING_LINE.match(s) and (s.isupper() or sum(1 for c in s if c.isupper()) >= 2):
            return True
        return False

    def _sectionize_document(self, content: str) -> list[dict[str, str | int]]:
        lines = content.splitlines()
        sections: list[dict[str, str | int]] = []
        current_heading = ""
        current_subheading = ""
        buffer: list[str] = []
        section_idx = 0

        def flush() -> None:
            nonlocal section_idx, buffer
            text = "\n".join(buffer).strip()
            if not text:
                buffer = []
                return
            heading_path = " > ".join([h for h in (current_heading, current_subheading) if h]).strip()
            sections.append(
                {
                    "section_title": current_heading or current_subheading or "Document",
                    "subsection_title": current_subheading if current_heading else "",
                    "heading_path": heading_path or "Document",
                    "text": text,
                    "section_index": section_idx,
                }
            )
            section_idx += 1
            buffer = []

        for raw_line in lines:
            line = raw_line.strip()
            if self._is_heading(line):
                flush()
                if _METHOD_ENTITY_HEADING.search(line) or not current_heading:
                    current_heading = line
                    current_subheading = ""
                else:
                    current_subheading = line
                continue
            buffer.append(raw_line)
        flush()
        if not sections:
            sections.append(
                {
                    "section_title": "Document",
                    "subsection_title": "",
                    "heading_path": "Document",
                    "text": content.strip(),
                    "section_index": 0,
                }
            )
        return sections

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

        if document.metadata.source_type == DocumentType.CSV:
            text = document.content.strip()
            if not text:
                logger.warning("Skipping empty CSV row: %s", document.metadata.source_file)
                return []

            ingestion_ts = datetime.now(timezone.utc).isoformat()
            row_number = document.metadata.extra.get("row_number")
            section_title = f"Row {row_number}" if row_number else "CSV row"
            chunk_id = make_chunk_id(document.metadata.source_file, 0, text)
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                source_file=document.metadata.source_file,
                source_type=document.metadata.source_type.value,
                file_path=document.metadata.file_path,
                page_number=document.metadata.extra.get("page_number"),
                chunk_index=0,
                char_count=len(text),
                ingestion_timestamp=ingestion_ts,
                parent_id=document.id,
                document_name=document.metadata.source_file,
                section_title=section_title,
                subsection_title="",
                heading_path=section_title,
                section_index=0,
            )
            return [DocumentChunk(content=text, metadata=meta)]

        section_blocks = self._sectionize_document(document.content)
        raw_chunks: list[tuple[str, dict[str, str | int]]] = []
        for sec in section_blocks:
            text = str(sec["text"])
            for piece in self._splitter.split_text(text):
                if len(piece.strip()) > 35:
                    raw_chunks.append((piece, sec))

        if not raw_chunks:
            logger.warning("No chunks for '%s' — content may be too short.", document.metadata.source_file)
            return []

        ingestion_ts = datetime.now(timezone.utc).isoformat()
        chunks: list[DocumentChunk] = []

        for idx, (chunk_text, sec) in enumerate(raw_chunks):
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
                section_title=str(sec.get("section_title", document.metadata.extra.get("section_title", ""))),
                subsection_title=str(sec.get("subsection_title", "")),
                heading_path=str(sec.get("heading_path", "")),
                section_index=int(sec.get("section_index", 0)),
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
