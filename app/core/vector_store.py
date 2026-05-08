"""
app/core/vector_store.py — ChromaDB vector store abstraction.

Wraps ChromaDB behind a clean interface so the rest of the codebase
never directly imports chromadb. Swapping to Pinecone/Weaviate requires
changing only this file.

Key design decisions:
- Uses upsert semantics: re-ingesting the same file updates chunks, doesn't
  duplicate them (thanks to deterministic chunk IDs in chunker.py).
- Metadata is always stored alongside vectors for citation generation.
- Similarity scores are normalized from ChromaDB's distance metric.
"""

from __future__ import annotations

import logging
from typing import Any

from app.models.documents import DocumentChunk
from app.models.retrieval import RetrievedChunk
from configs.settings import get_settings

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Persistent ChromaDB vector store.

    Args:
        embedder: Any object with embed_many(texts) → list[list[float]].
        collection_name: Override the default collection name.
    """

    def __init__(self, embedder, collection_name: str | None = None) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise RuntimeError("chromadb not installed. Run: pip install chromadb")

        self._embedder = embedder
        settings = get_settings()
        self._collection_name = collection_name or settings.chroma_collection_name
        persist_dir = str(settings.chroma_persist_dir)

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},  # Cosine similarity
        )
        logger.info(
            "ChromaVectorStore ready: collection='%s' persist='%s' count=%d",
            self._collection_name,
            persist_dir,
            self._collection.count(),
        )

    # ── Write operations ──────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """
        Embed and upsert a list of DocumentChunks into the vector store.

        Uses upsert (not add) to allow safe re-ingestion of the same documents.

        Args:
            chunks: List of DocumentChunk objects to store.

        Returns:
            The same chunks (pass-through for pipeline chaining).
        """
        if not chunks:
            return []

        texts = [c.content for c in chunks]
        ids = [c.metadata.chunk_id for c in chunks]
        metadatas = [c.metadata.to_chroma_dict() for c in chunks]

        logger.debug("Embedding %d chunks…", len(chunks))
        embeddings = self._embedder.embed_many(texts)

        # Upsert in batches of 500 (ChromaDB performance sweet spot)
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            self._collection.upsert(
                ids=ids[i : i + batch_size],
                documents=texts[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )
        logger.info("Upserted %d chunks into '%s'", len(chunks), self._collection_name)
        return chunks

    def delete_by_source(self, source_file: str) -> int:
        """
        Remove all chunks from a specific source document.

        Args:
            source_file: The filename to remove (e.g., "sop_v2.pdf").

        Returns:
            Number of chunks deleted.
        """
        results = self._collection.get(
            where={"source_file": source_file}, include=[]
        )
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info(
                "Deleted %d chunks for source '%s'", len(ids_to_delete), source_file
            )
        return len(ids_to_delete)

    # ── Read operations ───────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Semantic similarity search.

        Args:
            query_text: The user query (will be embedded).
            top_k: Number of results to return.
            where: Optional ChromaDB metadata filter (e.g., {"source_type": "pdf"}).

        Returns:
            List of RetrievedChunk sorted by similarity score descending.
        """
        settings = get_settings()
        k = min(top_k or settings.retrieval_top_k, self._collection.count() or 1)

        if k == 0:
            logger.warning("Vector store is empty. No results returned.")
            return []

        query_embedding = self._embedder.embed(query_text)

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        retrieved: list[RetrievedChunk] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB cosine distance is in [0, 2]; convert to similarity [0, 1]
            similarity = max(0.0, 1.0 - (dist / 2.0))
            retrieved.append(
                RetrievedChunk(
                    chunk_id=meta.get("chunk_id", ""),
                    content=doc,
                    source_file=meta.get("source_file", "unknown"),
                    source_type=meta.get("source_type", "unknown"),
                    page_number=meta.get("page_number"),
                    chunk_index=meta.get("chunk_index", 0),
                    similarity_score=round(similarity, 4),
                )
            )

        retrieved.sort(key=lambda c: c.similarity_score, reverse=True)
        logger.debug(
            "Query returned %d chunks (top score: %.3f)",
            len(retrieved),
            retrieved[0].similarity_score if retrieved else 0.0,
        )
        return retrieved

    # ── Introspection ─────────────────────────────────────────────────────────

    def count(self) -> int:
        """Total number of chunks in the collection."""
        return self._collection.count()

    def list_sources(self) -> list[dict[str, Any]]:
        """
        Return a deduplicated list of indexed documents with chunk counts.
        """
        if self._collection.count() == 0:
            return []

        all_meta = self._collection.get(include=["metadatas"])["metadatas"] or []
        source_counts: dict[str, dict[str, Any]] = {}
        for m in all_meta:
            sf = m.get("source_file", "unknown")
            if sf not in source_counts:
                source_counts[sf] = {
                    "source_file": sf,
                    "source_type": m.get("source_type", "unknown"),
                    "chunk_count": 0,
                    "ingestion_timestamp": m.get("ingestion_timestamp", ""),
                }
            source_counts[sf]["chunk_count"] += 1

        return sorted(source_counts.values(), key=lambda x: x["source_file"])

    def reset(self) -> None:
        """
        Delete and recreate the collection. Use with caution — this is
        destructive and intended for testing/rebuilding the index.
        """
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Vector store collection '%s' has been reset.", self._collection_name)
