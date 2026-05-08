"""
app/ingestion/pipeline.py — Top-level ingestion orchestrator.

This is the single entry point for adding documents to the system.
It coordinates: load → chunk → embed → store.

The pipeline is designed to be idempotent: re-ingesting the same document
updates existing chunks rather than creating duplicates (via upsert semantics).
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.ingestion.loaders import DocumentLoader, LoaderError
from app.ingestion.chunker import DocumentChunker
from app.models.documents import DocumentChunk

logger = logging.getLogger(__name__)


class IngestionResult:
    """Holds statistics from a completed ingestion run."""

    def __init__(self) -> None:
        self.processed_files: list[str] = []
        self.failed_files: list[str] = []
        self.total_chunks: int = 0
        self.total_documents: int = 0

    @property
    def success_count(self) -> int:
        return len(self.processed_files)

    @property
    def failure_count(self) -> int:
        return len(self.failed_files)

    def __repr__(self) -> str:
        return (
            f"IngestionResult(success={self.success_count}, "
            f"failed={self.failure_count}, chunks={self.total_chunks})"
        )


class IngestionPipeline:
    """
    Orchestrates the full document ingestion flow.

    Decoupled from the vector store via dependency injection: pass any object
    that implements the VectorStore.add_chunks() interface.
    """

    def __init__(self, vector_store, embedder) -> None:  # typed in runtime to avoid circular imports
        self._store = vector_store
        self._embedder = embedder
        self._loader = DocumentLoader()
        self._chunker = DocumentChunker()

    def ingest_file(self, file_path: str | Path) -> list[DocumentChunk]:
        """
        Ingest a single file: load → chunk → embed → store.

        Returns:
            List of chunks that were successfully stored.

        Raises:
            LoaderError: If the file cannot be read.
        """
        path = Path(file_path)
        logger.info("Ingesting file: %s", path.name)

        # Load
        documents = self._loader.load(path)
        if not documents:
            logger.warning("No documents loaded from: %s", path.name)
            return []

        # Chunk
        chunks = self._chunker.chunk_many(documents)
        if not chunks:
            logger.warning("No chunks produced from: %s", path.name)
            return []

        # Embed + Store (the embedder is called inside the vector store's add method
        # so that embedding happens in batches, not one-by-one)
        stored = self._store.add_chunks(chunks)
        logger.info(
            "Stored %d chunks from '%s'", len(stored), path.name
        )
        return stored

    def ingest_files(self, file_paths: list[str | Path]) -> IngestionResult:
        """
        Ingest multiple files, collecting per-file results.

        Args:
            file_paths: List of file paths to ingest.

        Returns:
            IngestionResult with statistics.
        """
        result = IngestionResult()

        for file_path in file_paths:
            try:
                chunks = self.ingest_file(file_path)
                result.processed_files.append(str(file_path))
                result.total_chunks += len(chunks)
            except (LoaderError, Exception) as exc:
                logger.error("Failed to ingest '%s': %s", file_path, exc)
                result.failed_files.append(str(file_path))

        result.total_documents = result.success_count
        logger.info(
            "Ingestion complete: %d succeeded, %d failed, %d total chunks",
            result.success_count,
            result.failure_count,
            result.total_chunks,
        )
        return result

    def ingest_directory(
        self,
        directory: str | Path,
        recursive: bool = False,
    ) -> IngestionResult:
        """
        Ingest all supported documents in a directory.

        Args:
            directory: Path to directory.
            recursive: If True, recurse into subdirectories.

        Returns:
            IngestionResult with aggregate statistics.
        """
        d = Path(directory)
        docs = self._loader.load_directory(d, recursive=recursive)

        result = IngestionResult()
        for doc in docs:
            try:
                chunks = self._chunker.chunk(doc)
                if chunks:
                    self._store.add_chunks(chunks)
                    result.processed_files.append(doc.metadata.source_file)
                    result.total_chunks += len(chunks)
            except Exception as exc:
                logger.error("Failed to process '%s': %s", doc.metadata.source_file, exc)
                result.failed_files.append(doc.metadata.source_file)

        result.total_documents = result.success_count
        return result
