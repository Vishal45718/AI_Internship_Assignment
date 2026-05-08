"""
app/utils/hashing.py — Content hashing for deduplication.

Using SHA-256 fingerprints lets us skip re-embedding chunks whose
content hasn't changed — critical for cost and latency at scale.
"""

from __future__ import annotations

import hashlib


def hash_content(text: str, encoding: str = "utf-8") -> str:
    """Return a 16-char hex SHA-256 prefix as a content fingerprint."""
    return hashlib.sha256(text.encode(encoding)).hexdigest()[:16]


def hash_file(path: str) -> str:
    """Return a SHA-256 hex digest for a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def make_chunk_id(source_file: str, chunk_index: int, content: str) -> str:
    """
    Deterministic chunk ID from source + index + content fingerprint.
    Deterministic means re-ingesting the same document produces the same IDs,
    which enables safe upsert (add-or-update) semantics.
    """
    fingerprint = hash_content(content)
    safe_name = source_file.replace("/", "_").replace("\\", "_")
    return f"{safe_name}_c{chunk_index:04d}_{fingerprint}"
