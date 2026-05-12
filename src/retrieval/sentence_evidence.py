"""
Sentence-level reranking and definition-pattern boosting on top of chunk retrieval.

Runs after semantic/hybrid retrieval and chunk fusion — does not replace them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.retrieval.reranker import CrossEncoderReranker


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


# Explicit definitional / mechanism phrasing (substring boosts).
_DEFINITION_PATTERN_SPECS: tuple[tuple[re.Pattern[str], float, str], ...] = (
    (re.compile(r"is defined as", re.I), 0.09, "is_defined_as"),
    (re.compile(r"\brefers to\b", re.I), 0.07, "refers_to"),
    (re.compile(r"triggered when", re.I), 0.08, "triggered_when"),
    (re.compile(r"consists of", re.I), 0.06, "consists_of"),
    (re.compile(r"\buses\b", re.I), 0.05, "uses"),
    (re.compile(r"based on", re.I), 0.06, "based_on"),
    (re.compile(r"\boptimizes?\b", re.I), 0.08, "optimizes"),
    (re.compile(r"\breformulates?\b", re.I), 0.08, "reformulates"),
    (re.compile(r"\buncertainty\b", re.I), 0.08, "uncertainty"),
    (re.compile(r"\bthreshold\b", re.I), 0.08, "threshold"),
    (re.compile(r"retrieval occurs when", re.I), 0.09, "retrieval_occurs_when"),
    (re.compile(r"retrieval occurs", re.I), 0.07, "retrieval_occurs"),
)

_MECHANISM_HINTS = (
    re.compile(r"\bmechanism\b", re.I),
    re.compile(r"\bprocess\b", re.I),
    re.compile(r"\boperates\b", re.I),
    re.compile(r"\bworkflow\b", re.I),
    re.compile(r"\balgorithm\b", re.I),
)
_THRESHOLD_HINTS = (
    re.compile(r"\bthreshold\b", re.I),
    re.compile(r"\bexceeds\b", re.I),
    re.compile(r"\bwhen\b.*\b(crosses|above|below|exceed)", re.I),
    re.compile(r"\d+\s*%", re.I),
)
_CAUSAL_HINTS = (
    re.compile(r"\bbecause\b", re.I),
    re.compile(r"\btherefore\b", re.I),
    re.compile(r"\bleads to\b", re.I),
    re.compile(r"\bcauses\b", re.I),
    re.compile(r"\bresulting in\b", re.I),
)
_DEF_FOCUS = re.compile(r"\b(define|denoted|meaning|represents|is a)\b", re.I)
_TABLE_NUMERIC_HINTS: tuple[tuple[re.Pattern[str], float, str], ...] = (
    (re.compile(r"\d+(?:\.\d+)?\s*%"), 0.08, "percent_value"),
    (re.compile(r"\b\d+(?:\.\d+)?\b"), 0.05, "numeric_value"),
    (re.compile(r"(?:<=|>=|<|>|=)\s*\d+(?:\.\d+)?"), 0.08, "comparison_threshold"),
    (re.compile(r"^\s*[-*]\s+\S+", re.I), 0.05, "bullet_list"),
    (re.compile(r"\|.+\|"), 0.09, "table_row"),
)
_EXACT_ENTITY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bSeaKR\b"), "SeaKR"),
    (re.compile(r"\bDRAGIN\b"), "DRAGIN"),
    (re.compile(r"\bReAL\b"), "ReAL"),
    (re.compile(r"\bFLARE\b"), "FLARE"),
    (re.compile(r"\bCRAG\b"), "CRAG"),
)


def _detect_evidence_focus(query: str) -> str:
    q = query.lower().strip()
    if any(
        x in q
        for x in (
            "what threshold",
            "threshold",
            "when does",
            "triggered",
            "how often",
        )
    ):
        return "threshold"
    if any(q.startswith(p) for p in ("how does", "how do", "how is", "how are")) or "how does" in q:
        return "mechanism"
    if any(x in q for x in ("parameter", "hyperparameter", "setting", "value")):
        return "parameter"
    if any(
        q.startswith(p)
        for p in (
            "what is",
            "what are",
            "define",
            "definition",
            "meaning",
            "describe",
        )
    ):
        return "definition"
    return "general"


def split_into_sentences(text: str, min_len: int = 12) -> list[str]:
    raw = _SENT_SPLIT.split(text.strip())
    out: list[str] = []
    for part in raw:
        p = " ".join(part.split())
        if len(p) >= min_len:
            out.append(p)
    return out


def _definition_boost(text_lower: str) -> tuple[float, list[str]]:
    total = 0.0
    labels: list[str] = []
    for pat, weight, label in _DEFINITION_PATTERN_SPECS:
        if pat.search(text_lower):
            total += weight
            labels.append(label)
    return min(0.22, total), labels


def _content_type_boost(sentence_lower: str, focus: str) -> tuple[float, list[str]]:
    tags: list[str] = []
    bonus = 0.0
    if focus == "definition" and _DEF_FOCUS.search(sentence_lower):
        bonus += 0.06
        tags.append("definition_focus")
    if focus == "mechanism":
        if any(h.search(sentence_lower) for h in _MECHANISM_HINTS):
            bonus += 0.07
            tags.append("mechanism_focus")
        if any(h.search(sentence_lower) for h in _CAUSAL_HINTS):
            bonus += 0.05
            tags.append("causal_focus")
    if focus == "threshold":
        if any(h.search(sentence_lower) for h in _THRESHOLD_HINTS):
            bonus += 0.09
            tags.append("threshold_focus")
    if focus == "parameter":
        if any(h.search(sentence_lower) for h in _THRESHOLD_HINTS):
            bonus += 0.08
            tags.append("parameter_numeric_focus")
    return min(0.18, bonus), tags


def _exact_entity_boost(sentence: str, query: str) -> tuple[float, list[str]]:
    labels: list[str] = []
    for pat, label in _EXACT_ENTITY_PATTERNS:
        if pat.search(query) and pat.search(sentence):
            labels.append(label)
    if not labels:
        return 0.0, []
    # Strong boost for exact acronym/entity carryover from query.
    return min(0.28, 0.12 + 0.08 * len(labels)), labels


def _table_aware_boost(sentence: str, focus: str) -> tuple[float, list[str]]:
    total = 0.0
    labels: list[str] = []
    for pat, weight, label in _TABLE_NUMERIC_HINTS:
        if pat.search(sentence):
            total += weight
            labels.append(label)
    if focus in {"threshold", "mechanism", "parameter"} and labels:
        total += 0.03
        labels.append("intent_table_numeric_bias")
    return min(0.2, total), labels


def _needs_neighbor_expansion(query: str, focus: str) -> bool:
    q = query.lower()
    if focus in {"threshold", "mechanism", "parameter"}:
        return True
    return any(
        t in q
        for t in ("threshold", "trigger", "when", "signal", "confidence", "uncertainty")
    )


@dataclass
class ScoredSentence:
    text: str
    rerank_score: float
    definition_boost: float
    definition_labels: list[str] = field(default_factory=list)
    query_type_boost: float = 0.0
    query_type_tags: list[str] = field(default_factory=list)
    exact_match_boost: float = 0.0
    exact_match_labels: list[str] = field(default_factory=list)
    table_boost: float = 0.0
    table_boost_labels: list[str] = field(default_factory=list)
    sentence_index: int = -1
    neighbor_role: str = "core"
    final_score: float = 0.0
    chunk_id: str = ""
    page_display: str = "?"
    source_file: str = ""


def score_sentences_for_query(
    query: str,
    chunks: list[Any],
    reranker: CrossEncoderReranker | None,
) -> tuple[list[ScoredSentence], dict[str, Any]]:
    """
    Split chunk texts into sentences, cross-encode rerank, apply pattern / query-type boosts.
    """
    focus = _detect_evidence_focus(query)
    candidates: list[tuple[str, str, str, str, int]] = []  # sentence, chunk_id, page, source_file, sent_index
    for ch in chunks:
        page = str(ch.page_number) if getattr(ch, "page_number", None) is not None else "?"
        sid = getattr(ch, "chunk_id", "")
        src = getattr(ch, "source_file", "")
        chunk_sentences = split_into_sentences(ch.content)
        for idx, sent in enumerate(chunk_sentences):
            candidates.append((sent, sid, page, src, idx))

    if not candidates:
        return [], {"focus": focus, "sentence_count": 0}

    sentences_only = [c[0] for c in candidates]
    reranker = reranker or CrossEncoderReranker()
    raw_scores = list(reranker.score(query, sentences_only))
    while len(raw_scores) < len(candidates):
        raw_scores.append(0.0)

    scored: list[ScoredSentence] = []
    for (sent, sid, page, src, sent_idx), rs in zip(candidates, raw_scores):
        sl = sent.lower()
        d_boost, d_labels = _definition_boost(sl)
        qt_boost, qt_tags = _content_type_boost(sl, focus)
        em_boost, em_labels = _exact_entity_boost(sent, query)
        tb_boost, tb_labels = _table_aware_boost(sent, focus)
        combined = min(1.0, float(rs) * 0.54 + d_boost + qt_boost + em_boost + tb_boost)
        scored.append(
            ScoredSentence(
                text=sent,
                rerank_score=float(rs),
                definition_boost=d_boost,
                definition_labels=d_labels,
                query_type_boost=qt_boost,
                query_type_tags=qt_tags,
                exact_match_boost=em_boost,
                exact_match_labels=em_labels,
                table_boost=tb_boost,
                table_boost_labels=tb_labels,
                sentence_index=sent_idx,
                final_score=combined,
                chunk_id=sid,
                page_display=page,
                source_file=src,
            )
        )

    scored.sort(key=lambda s: s.final_score, reverse=True)

    debug = {
        "focus": focus,
        "sentence_count": len(scored),
        "definition_pattern_hits": sum(1 for s in scored if s.definition_labels),
        "exact_entity_boost_hits": sum(1 for s in scored if s.exact_match_labels),
        "table_aware_boost_hits": sum(1 for s in scored if s.table_boost_labels),
        "neighbor_expansion_intent": _needs_neighbor_expansion(query, focus),
    }
    return scored, debug


def expand_with_neighbor_sentences(
    query: str,
    focus: str,
    packed: list[ScoredSentence],
    chunks: list[Any],
) -> tuple[list[ScoredSentence], list[dict[str, Any]]]:
    if not packed or not _needs_neighbor_expansion(query, focus):
        return packed, []

    chunk_sentence_lookup: dict[str, list[str]] = {}
    for ch in chunks:
        chunk_sentence_lookup[getattr(ch, "chunk_id", "")] = split_into_sentences(ch.content)

    expanded: list[ScoredSentence] = []
    seen: set[tuple[str, str]] = set()
    windows: list[dict[str, Any]] = []

    for core in packed:
        key = (core.chunk_id, core.text)
        if key not in seen:
            expanded.append(core)
            seen.add(key)

        chunk_sentences = chunk_sentence_lookup.get(core.chunk_id, [])
        if not chunk_sentences:
            windows.append({"core": core.text, "neighbors_added": []})
            continue

        core_idx = core.sentence_index
        if core_idx < 0 or core_idx >= len(chunk_sentences):
            try:
                core_idx = chunk_sentences.index(core.text)
            except ValueError:
                windows.append({"core": core.text, "neighbors_added": []})
                continue

        added_neighbors: list[str] = []
        for offset, role in ((-1, "prev"), (1, "next")):
            n_idx = core_idx + offset
            if n_idx < 0 or n_idx >= len(chunk_sentences):
                continue
            n_text = chunk_sentences[n_idx]
            n_key = (core.chunk_id, n_text)
            if n_key in seen:
                continue
            neighbor = ScoredSentence(
                text=n_text,
                rerank_score=max(0.0, core.rerank_score * 0.90),
                definition_boost=core.definition_boost * 0.25,
                definition_labels=[],
                query_type_boost=core.query_type_boost * 0.25,
                query_type_tags=[*core.query_type_tags, f"neighbor_{role}"],
                exact_match_boost=0.0,
                exact_match_labels=[],
                table_boost=core.table_boost * 0.25,
                table_boost_labels=[],
                sentence_index=n_idx,
                neighbor_role=role,
                final_score=max(0.0, core.final_score * 0.84),
                chunk_id=core.chunk_id,
                page_display=core.page_display,
                source_file=core.source_file,
            )
            expanded.append(neighbor)
            seen.add(n_key)
            added_neighbors.append(n_text)

        windows.append({"core": core.text, "neighbors_added": added_neighbors})

    return expanded, windows


def dedupe_sentences(scored: list[ScoredSentence]) -> list[ScoredSentence]:
    """Drop near-duplicate sentences (overlap from chunking)."""
    seen: set[str] = set()
    out: list[ScoredSentence] = []
    for s in scored:
        normalized = re.sub(r"\s+", " ", s.text.lower().strip())
        key = normalized[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def compress_scored_sentences(
    scored: list[ScoredSentence],
    *,
    min_final_score: float = 0.22,
    min_rerank_score: float = 0.12,
    max_candidates: int = 12,
) -> list[ScoredSentence]:
    """
    Remove low-scoring sentences before packing; keep explanatory-heavy candidates.
    Always retain at least the top sentence so we never return empty spuriously.
    """
    if not scored:
        return []
    head = scored[:max_candidates]
    kept = [
        s
        for s in head
        if s.final_score >= min_final_score or s.rerank_score >= min_rerank_score
    ]
    if len(kept) >= 2:
        return kept
    # Preserve best-ranked sentences if filtering was too aggressive
    return head[: max(3, min(6, len(head)))]


def pack_evidence_sentences(
    scored: list[ScoredSentence],
    max_chars: int,
    max_sentences: int = 2,
    max_sentence_chars: int = 180,
) -> tuple[list[ScoredSentence], str]:
    """Greedy pack highest-value explanatory sentences until character budget."""
    packed: list[ScoredSentence] = []
    total = 0
    block_overhead = 42
    for s in scored:
        if len(packed) >= max_sentences:
            break
        if len(s.text) <= max_sentence_chars:
            text = s.text
        else:
            cut = s.text[:max_sentence_chars]
            text = cut.rsplit(" ", 1)[0] + "…" if " " in cut else cut + "…"
        cost = len(text) + block_overhead
        if total + cost > max_chars and packed:
            continue
        packed.append(
            ScoredSentence(
                text=text,
                rerank_score=s.rerank_score,
                definition_boost=s.definition_boost,
                definition_labels=s.definition_labels,
                query_type_boost=s.query_type_boost,
                query_type_tags=s.query_type_tags,
                exact_match_boost=s.exact_match_boost,
                exact_match_labels=s.exact_match_labels,
                table_boost=s.table_boost,
                table_boost_labels=s.table_boost_labels,
                sentence_index=s.sentence_index,
                neighbor_role=s.neighbor_role,
                final_score=s.final_score,
                chunk_id=s.chunk_id,
                page_display=s.page_display,
                source_file=s.source_file,
            )
        )
        total += cost
        if total >= max_chars * 0.98:
            break
    corpus = "\n".join(p.text for p in packed)
    return packed, corpus


def format_sentence_evidence_blocks(packed: list[ScoredSentence]) -> str:
    """Compact labels to reduce prompt tokens."""
    blocks: list[str] = []
    for i, s in enumerate(packed, start=1):
        safe = s.text.replace('"', "'")
        blocks.append(f'[E{i}|P{s.page_display}]\n"{safe}"')
    return "\n\n".join(blocks)
