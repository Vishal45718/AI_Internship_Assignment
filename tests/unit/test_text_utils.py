"""
tests/unit/test_text_utils.py — Unit tests for text cleaning utilities.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.utils.text import (
    normalize_unicode,
    remove_control_characters,
    collapse_whitespace,
    remove_pdf_artifacts,
    clean_text,
    csv_row_to_prose,
    count_tokens_approx,
)


class TestNormalizeUnicode:
    def test_smart_quotes_replaced(self):
        text = "\u201cHello\u201d and \u2018world\u2019"
        result = normalize_unicode(text)
        assert '"Hello"' in result
        assert "'world'" in result

    def test_em_dash_replaced(self):
        text = "This\u2014that"
        result = normalize_unicode(text)
        assert "\u2014" not in result
        assert "-" in result

    def test_ellipsis_replaced(self):
        result = normalize_unicode("Wait\u2026")
        assert "..." in result


class TestRemoveControlCharacters:
    def test_null_bytes_removed(self):
        text = "Hello\x00World"
        result = remove_control_characters(text)
        assert "\x00" not in result
        assert "HelloWorld" in result

    def test_newlines_preserved(self):
        text = "Line 1\nLine 2\nLine 3"
        result = remove_control_characters(text)
        assert "\n" in result


class TestCollapseWhitespace:
    def test_multiple_spaces_collapsed(self):
        result = collapse_whitespace("Hello   World")
        assert "Hello World" == result

    def test_triple_newlines_become_double(self):
        result = collapse_whitespace("Para 1\n\n\n\nPara 2")
        assert "\n\n\n" not in result

    def test_trailing_whitespace_stripped(self):
        result = collapse_whitespace("Hello   \nWorld   ")
        assert result == "Hello\nWorld"


class TestRemovePdfArtifacts:
    def test_page_number_removed(self):
        text = "Some content\nPage 5 of 20\nMore content"
        result = remove_pdf_artifacts(text)
        assert "Page 5 of 20" not in result

    def test_standalone_number_removed(self):
        text = "Content before\n42\nContent after"
        result = remove_pdf_artifacts(text)
        # Standalone page numbers on their own line should be removed
        lines = [l for l in result.split("\n") if l.strip()]
        assert "42" not in lines


class TestCleanText:
    def test_full_pipeline_produces_clean_text(self):
        dirty = "\u201cHello\u201d   world\n\n\n\nAnother paragraph"
        result = clean_text(dirty)
        assert '"Hello"' in result
        assert "world" in result
        # Should not have 3+ consecutive newlines
        assert "\n\n\n" not in result

    def test_empty_string_returns_empty(self):
        result = clean_text("")
        assert result == ""


class TestCsvRowToProse:
    def test_basic_row_conversion(self):
        row = {"name": "Alice", "role": "Engineer", "department": "AI"}
        result = csv_row_to_prose(row)
        assert "name: Alice" in result
        assert "role: Engineer" in result
        assert "department: AI" in result

    def test_empty_values_excluded(self):
        row = {"name": "Bob", "middle_name": "", "role": "Manager"}
        result = csv_row_to_prose(row)
        assert "middle_name" not in result


class TestCountTokensApprox:
    def test_short_text(self):
        assert count_tokens_approx("Hello") >= 1

    def test_long_text(self):
        text = "word " * 100  # 500 chars → ~125 tokens
        count = count_tokens_approx(text)
        assert count > 50
