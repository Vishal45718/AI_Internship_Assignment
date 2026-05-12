"""
Grounding assessment (pre-generation) and validation (post-generation).

Keeps policy separate from pipeline orchestration.
"""

from __future__ import annotations

import re
from typing import Any

# Query contains these → acronym expansions in the answer must be supported by context.
_PROTECTED_ACRONYM_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bSeaKR\b", re.IGNORECASE), "SeaKR"),
    (re.compile(r"\bReAL\b"), "ReAL"),
    (re.compile(r"\bDRAGIN\b", re.IGNORECASE), "DRAGIN"),
    (re.compile(r"\bFLARE\b", re.IGNORECASE), "FLARE"),
    (re.compile(r"\bCRAG\b", re.IGNORECASE), "CRAG"),
)

_STOPWORDS = frozenset(
    """
    the a an is are was were be been being have has had do does did will would could
    should may might must shall can need to of in for on with at by from as into through
    during before after above below between under again further then once here there when
    where why how all each every both few more most other some such no nor not only own
    same so than too very just and but if because this that these those it its they them
    their what which who whom whose there than also only even just like about into over
    after such being both each few more most other some very can will just don should now
    """.split()
)


def protected_acronyms_in_query(query: str) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for pat, label in _PROTECTED_ACRONYM_PATTERNS:
        if pat.search(query) and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def build_retrieval_corpus(chunks: list[Any]) -> str:
    return "\n\n".join(c.content for c in chunks)


def _normalize_corpus(text: str) -> str:
    return text.lower()


def exact_entity_hits_in_context(query: str, corpus_normalized: str) -> list[str]:
    """Multi-token / mixed-case / ALLCAPS entities from the query found verbatim (word-boundary, case-insensitive)."""
    tokens = [t for t in re.findall(r"\w+", query) if t]
    entities: set[str] = set()
    entities.update(t for t in tokens if re.fullmatch(r"[A-Z]{2,}", t))
    entities.update(re.findall(r"\b(?:[A-Z][a-z]+[A-Z][A-Za-z]+|[a-z]+[A-Z][A-Za-z]+)\b", query))
    entities.update(re.findall(r'"([^"]+)"', query))
    hits: list[str] = []
    seen: set[str] = set()
    for ent in sorted((e.strip() for e in entities if e.strip()), key=len, reverse=True):
        key = ent.lower()
        if key in seen:
            continue
        if re.search(rf"\b{re.escape(ent.lower())}\b", corpus_normalized):
            hits.append(ent)
            seen.add(key)
    return hits


def weak_rerank_evidence(top_score: float, rerank_floor: float, margin: float = 0.12) -> bool:
    return top_score < rerank_floor + margin


def _has_explanatory_sentences(corpus: str) -> bool:
    """Require at least one substantive sentence or a long uninterrupted passage."""
    stripped = corpus.strip()
    if not stripped:
        return False
    parts = re.split(r"(?<=[.!?])\s+", stripped)
    for s in parts:
        if len(s.strip()) >= 40:
            return True
    return len(stripped) >= 140


def _query_seeks_explanation(question: str) -> bool:
    q = question.lower()
    return any(
        w in q
        for w in (
            "explain",
            "how does",
            "how do",
            "what is",
            "what are",
            "describe",
            "why",
            "meaning",
            "define",
            "mechanism",
            "works",
        )
    )


def assess_pre_generation_support(
    question: str,
    chunks: list[Any],
    top_score: float,
    rerank_score_threshold: float,
) -> tuple[bool, list[str], dict[str, Any]]:
    """
    Returns (allow_llm, block_reasons, debug).

    When allow_llm is False, respond with INSUFFICIENT_EVIDENCE_MESSAGE (no LLM call).
    """
    reasons: list[str] = []
    corpus = build_retrieval_corpus(chunks)
    corpus_n = _normalize_corpus(corpus)

    weak = weak_rerank_evidence(top_score, rerank_score_threshold)
    explanatory = _has_explanatory_sentences(corpus)
    entity_hits = exact_entity_hits_in_context(question, corpus_n)
    protected = protected_acronyms_in_query(question)

    has_named_shape = bool(
        re.search(r"\b[A-Z]{2,}\b", question)
        or re.search(r"\b(?:[A-Z][a-z]+[A-Z][A-Za-z]+|[a-z]+[A-Z][A-Za-z]+)\b", question)
    )

    for label in protected:
        pat = next(p for p, lab in _PROTECTED_ACRONYM_PATTERNS if lab == label)
        if not pat.search(corpus):
            reasons.append(f"protected_acronym_not_in_context:{label}")

    if has_named_shape and not entity_hits:
        reasons.append("no_exact_entity_match_in_context")

    if _query_seeks_explanation(question) and not explanatory:
        reasons.append("explanation_requested_but_context_not_explanatory")

    if weak and not explanatory:
        reasons.append("weak_rerank_and_no_explanatory_sentences")

    # De-duplicate while preserving order
    deduped: list[str] = []
    seen_r: set[str] = set()
    for r in reasons:
        if r not in seen_r:
            deduped.append(r)
            seen_r.add(r)

    debug = {
        "weak_rerank": weak,
        "top_rerank_score": top_score,
        "explanatory_sentences": explanatory,
        "entity_hits": entity_hits,
        "protected_acronyms_in_query": protected,
    }
    return (len(deduped) == 0), deduped, debug


def _significant_tokens(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text)
    out: list[str] = []
    for w in words:
        lw = w.lower()
        if lw in _STOPWORDS:
            continue
        out.append(lw)
    return out


def unsupported_significant_terms(answer: str, corpus_normalized: str) -> list[str]:
    bad: list[str] = []
    seen: set[str] = set()
    for tok in _significant_tokens(answer):
        if tok in seen:
            continue
        seen.add(tok)
        if len(tok) < 4:
            continue
        if re.search(rf"\b{re.escape(tok.lower())}\b", corpus_normalized):
            continue
        bad.append(tok)
    return bad


def _parenthetical_expansions_after_acronym(answer: str, acronym_pattern: re.Pattern[str]) -> list[str]:
    """Text inside (...) immediately after an acronym mention."""
    spans: list[str] = []
    for m in acronym_pattern.finditer(answer):
        tail = answer[m.end() : m.end() + 120]
        inner = re.match(r"\s*\(([^)]+)\)", tail)
        if inner:
            text = inner.group(1).strip()
            if len(text) >= 6:
                spans.append(text)
    return spans


def protected_acronym_expansion_violation(answer: str, query: str, corpus: str) -> list[str]:
    """Acronym followed by parenthetical expansion not substantiated in retrieved text."""
    violations: list[str] = []
    corpus_n = _normalize_corpus(corpus)
    for pat, label in _PROTECTED_ACRONYM_PATTERNS:
        if not pat.search(query):
            continue
        for expansion in _parenthetical_expansions_after_acronym(answer, pat):
            exp_n = _normalize_corpus(expansion)
            significant = [t for t in _significant_tokens(expansion) if len(t) >= 5]
            if not significant:
                continue
            hits = sum(1 for t in significant if t in corpus_n)
            if hits < max(1, len(significant) // 2):
                violations.append(f"{label}:expansion_not_in_context:{expansion[:80]}")
    return violations


def validate_post_generation(
    answer: str,
    question: str,
    corpus: str,
    top_score: float,
) -> dict[str, Any]:
    """
    Grounding check after the LLM responds.

    Returns flags used for logging and optional regeneration.
    """
    corpus_n = _normalize_corpus(corpus)
    stripped = answer.strip()
    insufficient_phrases = (
        "do not contain enough information",
        "not contain enough information to answer confidently",
        "not enough information",
    )
    if any(p in stripped.lower() for p in insufficient_phrases):
        return {
            "ok": True,
            "grounding_confidence": 1.0,
            "unsupported_terms": [],
            "protected_violations": [],
            "unsupported_ratio": 0.0,
            "regenerate": False,
            "is_insufficient_disclaimer": True,
        }

    sig = _significant_tokens(stripped)
    unsupported = unsupported_significant_terms(stripped, corpus_n)
    ratio = (len(unsupported) / len(sig)) if sig else 0.0
    prot_viol = protected_acronym_expansion_violation(stripped, question, corpus)

    unsupported_cap = [t for t in unsupported if t[:1].isupper() or any(c.isupper() for c in t)]
    large_claim = ratio > 0.38 or len(unsupported_cap) >= 4

    ok = ratio <= 0.45 and not prot_viol and not large_claim
    confidence = max(0.0, min(1.0, (1.0 - ratio) * (0.55 + 0.45 * min(1.0, top_score))))

    return {
        "ok": ok,
        "grounding_confidence": round(confidence, 3),
        "unsupported_terms": unsupported[:25],
        "protected_violations": prot_viol,
        "unsupported_ratio": round(ratio, 3),
        "regenerate": (not ok) or bool(prot_viol),
        "is_insufficient_disclaimer": False,
    }


STRICT_REGENERATION_SYSTEM_SUFFIX = (
    "\n\nREGENERATION (mandatory): Your prior answer may have included unsupported claims. "
    "Reply using ONLY the retrieved context. Every factual clause must be traceable to that text. "
    "Do not expand acronyms unless the expansion appears verbatim in the context. "
    "If you cannot meet this bar, reply exactly with: "
    "'The retrieved documents do not contain enough information to answer confidently.'"
)
