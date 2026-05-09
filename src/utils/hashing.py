"""
src/utils/hashing.py — Content hashing for chunk deduplication.

Deterministic chunk IDs enable safe upsert semantics — re-ingesting
the same document updates existing chunks rather than creating duplicates.
"""

from __future__ import annotations

import hashlib


def hash_content(text: str, encoding: str = "utf-8") -> str:
    """Return a 16-char hex SHA-256 prefix as a content fingerprint."""
    return hashlib.sha256(text.encode(encoding)).hexdigest()[:16]


def make_chunk_id(source_file: str, chunk_index: int, content: str) -> str:
    """
    Deterministic chunk ID from source + index + content fingerprint.

    Re-ingesting the same document produces the same IDs,
    enabling safe upsert (add-or-update) semantics in the vector store.
    """
    fingerprint = hash_content(content)
    safe_name = source_file.replace("/", "_").replace("\\", "_")
    return f"{safe_name}_c{chunk_index:04d}_{fingerprint}"
