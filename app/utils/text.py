"""
app/utils/text.py — Text cleaning and normalization utilities.

These utilities are used by the ingestion pipeline to produce clean,
consistent text before chunking and embedding.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode to NFC form and replace common special characters
    with their ASCII equivalents to prevent encoding issues.
    """
    text = unicodedata.normalize("NFC", text)
    # Smart quotes → ASCII
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Em/en dash → hyphen
    text = text.replace("\u2014", " - ").replace("\u2013", " - ")
    # Ellipsis → three dots
    text = text.replace("\u2026", "...")
    return text


def remove_control_characters(text: str) -> str:
    """Strip null bytes and non-printable control characters (except newlines/tabs)."""
    return "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t", "\r")
    )


def collapse_whitespace(text: str) -> str:
    """
    Collapse multiple spaces to one, but preserve intentional paragraph
    breaks (double newlines) and single newlines.
    """
    # Collapse 3+ newlines to double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace runs (spaces/tabs) to single space
    text = re.sub(r"[ \t]+", " ", text)
    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def remove_pdf_artifacts(text: str) -> str:
    """
    Remove common PDF extraction artifacts:
    - Page numbers: "Page X of Y", "- X -"
    - Header/footer repetition patterns
    - Hyphenated line breaks from PDF column layout
    """
    # "Page X of Y" patterns
    text = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bPage\s+\d+\b", "", text, flags=re.IGNORECASE)
    # Standalone page numbers on their own line
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Hyphenated line breaks (word split across lines in PDFs)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text


def clean_text(text: str, is_pdf: bool = False) -> str:
    """
    Full cleaning pipeline applied to extracted document text.

    Steps (in order):
        1. Unicode normalization
        2. Control character removal
        3. Optional PDF artifact stripping
        4. Whitespace collapsing
    """
    text = normalize_unicode(text)
    text = remove_control_characters(text)
    if is_pdf:
        text = remove_pdf_artifacts(text)
    text = collapse_whitespace(text)
    return text


def truncate_text(text: str, max_chars: int = 500, suffix: str = "…") -> str:
    """Truncate text to a max character count, breaking at a word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + suffix


def count_tokens_approx(text: str) -> int:
    """
    Rough token count estimate (1 token ≈ 4 chars in English).
    Used for chunking budget checks without loading a tokenizer.
    """
    return max(1, len(text) // 4)


def csv_row_to_prose(row: dict[str, str]) -> str:
    """
    Convert a CSV row (dict) to a readable prose string.
    E.g., {"name": "Alice", "role": "Engineer"} → "name: Alice | role: Engineer"
    """
    parts = [f"{k}: {v}" for k, v in row.items() if v and str(v).strip()]
    return " | ".join(parts)
