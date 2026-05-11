"""
src/vectordb/store.py — ChromaDB vector store.

Wraps ChromaDB behind a clean interface for storing and retrieving
document chunks using cosine similarity search.

Key design decisions:
- Uses upsert semantics: re-ingesting the same file updates chunks,
  not duplicates (thanks to deterministic chunk IDs).
- Metadata is stored alongside vectors for source citation.
- Similarity scores are normalized to [0, 1] range.
- Vectors persist locally on disk between runs.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from src.models import DocumentChunk, RetrievedChunk
from src.config import get_settings

logger = logging.getLogger(__name__)


def _normalize_get_field(values: list[Any] | None) -> list[Any]:
    """Normalize Chroma get() output for single-query and multi-query shapes."""
    if not values:
        return []
    if isinstance(values, list) and values and isinstance(values[0], list):
        return values[0]
    return values


class ChromaVectorStore:
    """
    Persistent ChromaDB vector store.

    Usage:
        store = ChromaVectorStore(embedder)
        store.add_chunks(chunks)
        results = store.query("What is the return policy?")
    """

    def __init__(self, embedder) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise RuntimeError("chromadb not installed. Run: pip install chromadb")

        self._embedder = embedder
        settings = get_settings()
        self._collection_name = settings.chroma_collection_name
        persist_dir = str(settings.chroma_persist_dir)

        # Create persistent client (data survives between runs)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready: collection='%s' chunks=%d",
            self._collection_name,
            self._collection.count(),
        )

    # ── Write operations ──────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """
        Embed and upsert chunks into the vector store.

        Uses upsert (not add) so re-ingesting the same document
        updates existing chunks instead of creating duplicates.

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
        """Remove all chunks from a specific source document."""
        results = self._collection.get(where={"source_file": source_file}, include=[])
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info("Deleted %d chunks for '%s'", len(ids_to_delete), source_file)
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
            query_text: The user query (will be embedded automatically).
            top_k: Number of results to return.
            where: Optional ChromaDB metadata filter.

        Returns:
            List of RetrievedChunk sorted by similarity score (highest first).
        """
        settings = get_settings()
        k = min(top_k or settings.retrieval_top_k, self._collection.count() or 1)

        if k == 0:
            logger.warning("Vector store is empty.")
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
            # ChromaDB cosine distance is [0, 2]; convert to similarity [0, 1]
            similarity = max(0.0, 1.0 - (dist / 2.0))
            retrieved.append(
                RetrievedChunk(
                    chunk_id=meta.get("chunk_id", ""),
                    content=doc,
                    source_file=meta.get("source_file", "unknown"),
                    source_type=meta.get("source_type", "unknown"),
                    page_number=meta.get("page_number"),
                    chunk_index=meta.get("chunk_index", 0),
                    parent_id=meta.get("parent_id", ""),
                    document_name=meta.get("document_name", ""),
                    section_title=meta.get("section_title", ""),
                    similarity_score=round(similarity, 4),
                )
            )

        retrieved.sort(key=lambda c: c.similarity_score, reverse=True)
        return retrieved

    # ── Introspection ─────────────────────────────────────────────────────────

    def count(self) -> int:
        """Total number of chunks in the collection."""
        return self._collection.count()

    def list_sources(self) -> list[dict[str, Any]]:
        """Return a deduplicated list of indexed documents with chunk counts."""
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
                }
            source_counts[sf]["chunk_count"] += 1

        return sorted(source_counts.values(), key=lambda x: x["source_file"])

    def keyword_search(
        self,
        query_text: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Keyword/BM25-style retrieval over stored chunk text.

        This is a lightweight fallback for exact identifiers, acronyms,
        and case-sensitive tokens that dense semantic search may miss.
        """
        settings = get_settings()
        k = min(top_k or settings.retrieval_top_k, self._collection.count() or 1)

        if self._collection.count() == 0:
            logger.warning("Vector store is empty for keyword search.")
            return []

        terms = [t for t in re.findall(r"\w+", query_text) if t]
        if not terms:
            logger.warning("Keyword search skipped empty query terms.")
            return []

        results = self._collection.get(include=["documents", "metadatas"], where=where)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        if not documents or not metadatas:
            return []

        lower_docs = [doc.lower() for doc in documents]
        corpus_size = len(lower_docs)
        df: dict[str, int] = {}
        for term in terms:
            pattern = re.compile(rf"\b{re.escape(term.lower())}\b")
            df[term] = sum(1 for doc in lower_docs if pattern.search(doc))

        avg_dl = sum(len(doc.split()) for doc in lower_docs) / max(corpus_size, 1)
        keyword_candidates: list[tuple[float, RetrievedChunk]] = []

        for document, metadata, lowered in zip(documents, metadatas, lower_docs):
            doc_len = max(len(lowered.split()), 1)
            score = 0.0
            exact_term_match = False

            for term in terms:
                term_pattern = re.compile(rf"\b{re.escape(term.lower())}\b")
                term_count = len(term_pattern.findall(lowered))
                if term_count == 0:
                    continue

                exact_term_pattern = re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
                if exact_term_pattern.search(document):
                    exact_term_match = True

                doc_freq = max(df.get(term, 0), 1)
                idf = math.log((corpus_size - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
                k1 = 1.5
                b = 0.75
                score += idf * ((term_count * (k1 + 1)) / (term_count + k1 * (1 - b + b * doc_len / avg_dl)))

            if score <= 0:
                continue

            if exact_term_match:
                score += 1.5

            keyword_candidates.append(
                (
                    score,
                    RetrievedChunk(
                        chunk_id=metadata.get("chunk_id", ""),
                        content=document,
                        source_file=metadata.get("source_file", "unknown"),
                        source_type=metadata.get("source_type", "unknown"),
                        page_number=metadata.get("page_number"),
                        chunk_index=metadata.get("chunk_index", 0),
                        parent_id=metadata.get("parent_id", ""),
                        document_name=metadata.get("document_name", ""),
                        section_title=metadata.get("section_title", ""),
                        similarity_score=min(1.0, round(score, 4)),
                    ),
                )
            )

        keyword_candidates.sort(key=lambda item: item[0], reverse=True)
        return [candidate for _, candidate in keyword_candidates[:k]]

    def get_chunks_by_indices(self, source_file: str, indices: list[int]) -> list[RetrievedChunk]:
        """Retrieve chunks from one source by chunk indices."""
        if not indices:
            return []

        unique_indices = sorted(set(indices))
        where: dict[str, Any] = {"$and": [{"source_file": source_file}]}
        if len(unique_indices) == 1:
            where["$and"].append({"chunk_index": unique_indices[0]})
        else:
            where["$and"].append({"chunk_index": {"$in": unique_indices}})

        results = self._collection.get(include=["documents", "metadatas"], where=where)
        documents = _normalize_get_field(results.get("documents"))
        metadatas = _normalize_get_field(results.get("metadatas"))

        if not documents or not metadatas:
            return []

        retrieved: list[RetrievedChunk] = []
        for doc, meta in zip(documents, metadatas):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=meta.get("chunk_id", ""),
                    content=doc,
                    source_file=meta.get("source_file", "unknown"),
                    source_type=meta.get("source_type", "unknown"),
                    page_number=meta.get("page_number"),
                    chunk_index=meta.get("chunk_index", 0),
                    parent_id=meta.get("parent_id", ""),
                    document_name=meta.get("document_name", ""),
                    section_title=meta.get("section_title", ""),
                    similarity_score=1.0,
                )
            )

        retrieved.sort(key=lambda c: c.chunk_index)
        return retrieved

    def get_neighboring_chunks(
        self,
        chunk_ids: list[str],
        window_before: int = 1,
        window_after: int = 1,
    ) -> list[RetrievedChunk]:
        """
        Retrieve neighboring chunks for each chunk id.

        Neighboring candidates are selected from the same source file
        using chunk_index proximity.
        """
        if not chunk_ids or (window_before == 0 and window_after == 0):
            return []

        neighbors_by_id: dict[str, RetrievedChunk] = {}
        for chunk_id in chunk_ids:
            seed_results = self._collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"],
            )
            seed_metadatas = _normalize_get_field(seed_results.get("metadatas"))
            if not seed_metadatas:
                continue

            seed_meta = seed_metadatas[0]
            source_file = seed_meta.get("source_file")
            seed_index = seed_meta.get("chunk_index")
            if source_file is None or seed_index is None:
                continue

            start = int(seed_index) - max(0, window_before)
            end = int(seed_index) + max(0, window_after)
            indices = [i for i in range(start, end + 1) if i >= 0]
            for candidate in self.get_chunks_by_indices(source_file, indices):
                neighbors_by_id[candidate.chunk_id] = candidate

        return list(neighbors_by_id.values())

    def get_chunks_by_section(self, source_file: str, section_title: str) -> list[RetrievedChunk]:
        """Retrieve all chunks from a specific source + section title."""
        if not source_file or not section_title:
            return []

        where = {
            "$and": [
                {"source_file": source_file},
                {"section_title": section_title},
            ]
        }
        results = self._collection.get(include=["documents", "metadatas"], where=where)
        documents = _normalize_get_field(results.get("documents"))
        metadatas = _normalize_get_field(results.get("metadatas"))
        if not documents or not metadatas:
            return []

        retrieved: list[RetrievedChunk] = []
        for doc, meta in zip(documents, metadatas):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=meta.get("chunk_id", ""),
                    content=doc,
                    source_file=meta.get("source_file", "unknown"),
                    source_type=meta.get("source_type", "unknown"),
                    page_number=meta.get("page_number"),
                    chunk_index=meta.get("chunk_index", 0),
                    parent_id=meta.get("parent_id", ""),
                    document_name=meta.get("document_name", ""),
                    section_title=meta.get("section_title", ""),
                    similarity_score=1.0,
                )
            )
        retrieved.sort(key=lambda c: (c.page_number or 0, c.chunk_index))
        return retrieved

    def reset(self) -> None:
        """Delete and recreate the collection. ⚠️ Destructive."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Vector store '%s' has been reset.", self._collection_name)
