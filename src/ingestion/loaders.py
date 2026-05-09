"""
src/ingestion/loaders.py — Multi-format document loaders.

Supports: PDF, TXT, Markdown, CSV

Each loader returns a list of Document objects with content + metadata.
The DocumentLoader class dispatches based on file extension.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.models import Document, DocumentMetadata, DocumentType
from src.processing.text import clean_text, csv_row_to_prose

logger = logging.getLogger(__name__)


class LoaderError(Exception):
    """Raised when a document cannot be loaded."""


def _build_metadata(path: Path, doc_type: DocumentType, **extra) -> DocumentMetadata:
    """Helper to construct DocumentMetadata from a file path."""
    return DocumentMetadata(
        source_file=path.name,
        source_type=doc_type,
        file_path=str(path.resolve()),
        file_size_bytes=path.stat().st_size if path.exists() else 0,
        extra=extra,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Individual format loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_pdf(path: Path) -> list[Document]:
    """
    Extract text from a PDF, one Document per page.
    Preserves page numbers in metadata for accurate citations.
    """
    try:
        import pypdf
    except ImportError:
        raise LoaderError("pypdf not installed. Run: pip install pypdf")

    documents: list[Document] = []
    try:
        reader = pypdf.PdfReader(str(path))
        total_pages = len(reader.pages)
        logger.debug("PDF %s has %d pages", path.name, total_pages)

        for page_num, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            if not raw_text.strip():
                logger.debug("Skipping empty page %d in %s", page_num, path.name)
                continue
            cleaned = clean_text(raw_text, is_pdf=True)
            meta = _build_metadata(path, DocumentType.PDF, page_number=page_num)
            meta.total_pages = total_pages
            documents.append(Document(content=cleaned, metadata=meta))

    except Exception as exc:
        raise LoaderError(f"Failed to load PDF '{path}': {exc}") from exc

    logger.info("Loaded %d pages from PDF: %s", len(documents), path.name)
    return documents


def _load_txt(path: Path) -> list[Document]:
    """Load a plain text file as a single Document."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise LoaderError(f"Failed to read TXT '{path}': {exc}") from exc

    cleaned = clean_text(text)
    if not cleaned:
        logger.warning("TXT file '%s' is empty after cleaning.", path.name)
        return []

    meta = _build_metadata(path, DocumentType.TXT)
    logger.info("Loaded TXT: %s (%d chars)", path.name, len(cleaned))
    return [Document(content=cleaned, metadata=meta)]


def _load_markdown(path: Path) -> list[Document]:
    """
    Load a Markdown file as a single Document.
    Strips markdown syntax for cleaner embeddings.
    """
    import re

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise LoaderError(f"Failed to read Markdown '{path}': {exc}") from exc

    # Strip markdown syntax for cleaner embedding quality
    text = re.sub(r"```[\s\S]*?```", "", text)         # Code blocks
    text = re.sub(r"`[^`]+`", "", text)                 # Inline code
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)         # Images
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # Links → text
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # Headings

    cleaned = clean_text(text)
    if not cleaned:
        logger.warning("Markdown file '%s' is empty after cleaning.", path.name)
        return []

    meta = _build_metadata(path, DocumentType.MARKDOWN)
    logger.info("Loaded Markdown: %s (%d chars)", path.name, len(cleaned))
    return [Document(content=cleaned, metadata=meta)]


def _load_csv(path: Path) -> list[Document]:
    """
    Load a CSV file, converting rows to prose sentences.
    Batches 20 rows per Document for reasonable chunk sizes.
    """
    documents: list[Document] = []
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            logger.warning("CSV file '%s' has no data rows.", path.name)
            return []

        # Group rows into batches of 20 for reasonable document size
        ROWS_PER_DOC = 20
        for batch_start in range(0, len(rows), ROWS_PER_DOC):
            batch = rows[batch_start : batch_start + ROWS_PER_DOC]
            prose_lines = [csv_row_to_prose(row) for row in batch]
            content = "\n".join(prose_lines)
            cleaned = clean_text(content)
            if not cleaned:
                continue
            meta = _build_metadata(
                path, DocumentType.CSV,
                row_start=batch_start + 1,
                row_end=batch_start + len(batch),
            )
            documents.append(Document(content=cleaned, metadata=meta))

    except Exception as exc:
        raise LoaderError(f"Failed to load CSV '{path}': {exc}") from exc

    logger.info("Loaded CSV: %s (%d batch documents)", path.name, len(documents))
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# Loader dispatch map
# ─────────────────────────────────────────────────────────────────────────────

_LOADER_MAP = {
    DocumentType.PDF: _load_pdf,
    DocumentType.TXT: _load_txt,
    DocumentType.MARKDOWN: _load_markdown,
    DocumentType.CSV: _load_csv,
}


class DocumentLoader:
    """
    Dispatches each file to the correct format-specific loader.

    Usage:
        loader = DocumentLoader()
        docs = loader.load("report.pdf")
        all_docs = loader.load_directory("data/raw/")
    """

    def load(self, path: str | Path) -> list[Document]:
        """
        Load a document from disk.

        Args:
            path: Path to the file.

        Returns:
            List of Document objects (one per page for PDFs).

        Raises:
            LoaderError: If the file cannot be read or format is unsupported.
        """
        p = Path(path)
        if not p.exists():
            raise LoaderError(f"File not found: {path}")

        doc_type = DocumentType.from_extension(p.suffix)
        loader_fn = _LOADER_MAP.get(doc_type)

        if loader_fn is None:
            raise LoaderError(
                f"Unsupported file format: '{p.suffix}'. "
                f"Supported: .pdf, .txt, .md, .csv"
            )

        return loader_fn(p)

    def load_directory(self, directory: str | Path, recursive: bool = False) -> list[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory: Path to directory.
            recursive: If True, recurse into subdirectories.

        Returns:
            Flat list of all loaded Documents.
        """
        d = Path(directory)
        if not d.is_dir():
            raise LoaderError(f"Not a directory: {directory}")

        supported = {".pdf", ".txt", ".md", ".markdown", ".csv"}
        pattern = "**/*" if recursive else "*"
        files = [
            f for f in d.glob(pattern)
            if f.is_file() and f.suffix.lower() in supported
        ]

        all_docs: list[Document] = []
        for file_path in sorted(files):
            try:
                docs = self.load(file_path)
                all_docs.extend(docs)
            except LoaderError as exc:
                logger.warning("Skipping '%s': %s", file_path.name, exc)

        logger.info("Directory load: %d files → %d documents", len(files), len(all_docs))
        return all_docs
