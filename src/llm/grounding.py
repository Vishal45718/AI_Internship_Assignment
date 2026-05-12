"""
Grounding assessment (pre-generation) and validation (post-generation).

Keeps policy separate from pipeline orchestration.
"""

from __future__ import annotations

import re
from typing import Any

# Query mentions these → same token must appear in retrieved chunk text (exact verification).
_IMPORTANT_ENTITY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bSeaKR\b", re.IGNORECASE), "SeaKR"),
    (re.compile(r"\bReAL\b"), "ReAL"),
    (re.compile(r"\bDRAGIN\b", re.IGNORECASE), "DRAGIN"),
    (re.compile(r"\bFLARE\b", re.IGNORECASE), "FLARE"),
    (re.compile(r"\bCRAG\b", re.IGNORECASE), "CRAG"),
)

# Known plausible-but-wrong completions (reject if answer contains but evidence does not).
_KNOWN_HALLUCINATION_PHRASES: tuple[str, ...] = (
    "kernel regularization",
    "consistency across views",
    "reinforcement learning architecture search",
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
    this that these those than then there here from with into onto upon also very much
    """.split()
)


def important_entities_in_query(query: str) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for pat, label in _IMPORTANT_ENTITY_PATTERNS:
        if pat.search(query) and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def protected_acronyms_in_query(query: str) -> list[str]:
    """Alias for prompts/tests — same as important entity labels."""
    return important_entities_in_query(query)


def verify_entities_in_evidence(query: str, corpus: str) -> list[str]:
    """Return labels for entities mentioned in the query but absent from retrieved text."""
    missing: list[str] = []
    for pat, label in _IMPORTANT_ENTITY_PATTERNS:
        if pat.search(query) and not pat.search(corpus):
            missing.append(label)
    return missing


def build_retrieval_corpus(chunks: list[Any]) -> str:
    return "\n\n".join(c.content for c in chunks)


def _normalize_corpus(text: str) -> str:
    return text.lower()


def weak_rerank_evidence(top_score: float, rerank_floor: float, margin: float = 0.12) -> bool:
    return top_score < rerank_floor + margin


def _has_explanatory_sentences(corpus: str) -> bool:
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

    When allow_llm is False, respond with INSUFFICIENT_DOCUMENT_EVIDENCE (no LLM call).
    """
    reasons: list[str] = []
    corpus = build_retrieval_corpus(chunks)
    corpus_n = _normalize_corpus(corpus)

    weak = weak_rerank_evidence(top_score, rerank_score_threshold)
    explanatory = _has_explanatory_sentences(corpus)
    missing_entities = verify_entities_in_evidence(question, corpus)

    for label in missing_entities:
        reasons.append(f"required_entity_missing_in_evidence:{label}")

    if _query_seeks_explanation(question) and not explanatory:
        reasons.append("explanation_requested_but_evidence_not_explanatory")

    if weak and not explanatory:
        reasons.append("weak_rerank_and_no_explanatory_sentences")

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
        "missing_required_entities": missing_entities,
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


def extract_key_noun_phrases(text: str, max_phrases: int = 40) -> list[str]:
    """
    Extract 2–3 word sequences from content words (simple noun-phrase proxy).
    Used to flag unsupported multi-word claims.
    """
    tokens = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text)]
    phrases: list[str] = []
    seen: set[str] = set()
    i = 0
    while i < len(tokens):
        if tokens[i] in _STOPWORDS:
            i += 1
            continue
        for window in (3, 2):
            if i + window > len(tokens):
                continue
            span = tokens[i : i + window]
            if sum(1 for t in span if t not in _STOPWORDS) < window:
                continue
            phrase = " ".join(span)
            if len(phrase) < 8:
                continue
            if phrase not in seen:
                seen.add(phrase)
                phrases.append(phrase)
            if len(phrases) >= max_phrases:
                return phrases
        i += 1
    return phrases


def unsupported_noun_phrases(answer: str, corpus_normalized: str) -> list[str]:
    bad: list[str] = []
    for phrase in extract_key_noun_phrases(answer):
        if phrase not in corpus_normalized and not _phrase_fuzzy_in_corpus(phrase, corpus_normalized):
            bad.append(phrase)
    return bad[:30]


def _phrase_fuzzy_in_corpus(phrase: str, corpus_normalized: str) -> bool:
    """All non-trivial words of phrase appear near each other (light check)."""
    parts = [p for p in phrase.split() if len(p) >= 4 and p not in _STOPWORDS]
    if len(parts) < 2:
        return phrase in corpus_normalized
    return all(re.search(rf"\b{re.escape(p)}\b", corpus_normalized) for p in parts)


def detect_known_hallucination_phrases(answer: str, corpus_normalized: str) -> list[str]:
    """Phrases that often indicate fabricated ML explanations."""
    a = answer.lower()
    triggers: list[str] = []
    for phrase in _KNOWN_HALLUCINATION_PHRASES:
        if phrase in a and phrase not in corpus_normalized:
            triggers.append(f"known_hallucination_pattern:{phrase}")
    return triggers


def _parenthetical_expansions_after_acronym(answer: str, acronym_pattern: re.Pattern[str]) -> list[str]:
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
    violations: list[str] = []
    corpus_n = _normalize_corpus(corpus)
    for pat, label in _IMPORTANT_ENTITY_PATTERNS:
        if not pat.search(query):
            continue
        for expansion in _parenthetical_expansions_after_acronym(answer, pat):
            significant = [t for t in _significant_tokens(expansion) if len(t) >= 5]
            if not significant:
                continue
            hits = sum(1 for t in significant if t.lower() in corpus_n)
            if hits < max(1, len(significant) // 2):
                violations.append(f"{label}:expansion_not_in_evidence:{expansion[:80]}")
    return violations


_DISCLAIMER_MARKERS = (
    "the retrieved documents do not contain enough information",
    "do not contain enough information to answer confidently",
    "not enough information",
)


def _is_insufficient_disclaimer(answer: str) -> bool:
    low = answer.strip().lower()
    return any(m in low for m in _DISCLAIMER_MARKERS)


def validate_post_generation(
    answer: str,
    question: str,
    corpus: str,
    top_score: float,
) -> dict[str, Any]:
    """
    Evidence verification after generation: unsupported terms, noun phrases,
    known hallucination strings, acronym expansions.
    """
    corpus_n = _normalize_corpus(corpus)
    stripped = answer.strip()

    if _is_insufficient_disclaimer(stripped):
        return {
            "ok": True,
            "grounding_confidence": 1.0,
            "unsupported_terms": [],
            "unsupported_noun_phrases": [],
            "hallucination_triggers": [],
            "protected_violations": [],
            "unsupported_ratio": 0.0,
            "regenerate": False,
            "is_insufficient_disclaimer": True,
        }

    sig = _significant_tokens(stripped)
    unsupported = unsupported_significant_terms(stripped, corpus_n)
    noun_bad = unsupported_noun_phrases(stripped, corpus_n)
    halluc_triggers = detect_known_hallucination_phrases(stripped, corpus_n)
    prot_viol = protected_acronym_expansion_violation(stripped, question, corpus)

    ratio = (len(unsupported) / len(sig)) if sig else 0.0
    unsupported_cap = [t for t in unsupported if t[:1].isupper() or any(c.isupper() for c in t)]
    large_claim = ratio > 0.38 or len(unsupported_cap) >= 4
    noun_fail = len(noun_bad) >= 5

    ok = (
        ratio <= 0.42
        and not prot_viol
        and not large_claim
        and not halluc_triggers
        and not noun_fail
    )
    confidence = max(0.0, min(1.0, (1.0 - ratio) * (0.55 + 0.45 * min(1.0, top_score))))

    regenerate = (
        (not ok)
        or bool(prot_viol)
        or bool(halluc_triggers)
        or (ratio > 0.30 and len(unsupported) >= 3)
    )

    return {
        "ok": ok,
        "grounding_confidence": round(confidence, 3),
        "unsupported_terms": unsupported[:25],
        "unsupported_noun_phrases": noun_bad[:25],
        "hallucination_triggers": halluc_triggers,
        "protected_violations": prot_viol,
        "unsupported_ratio": round(ratio, 3),
        "regenerate": regenerate,
        "is_insufficient_disclaimer": False,
    }


STRICT_REGENERATION_SYSTEM_SUFFIX = (
    "\n\nREGENERATION: Answer only from explicit evidence in the Retrieved Evidence section. "
    "Do not use outside knowledge. Quote or closely paraphrase only text inside the evidence quotes. "
    "If you cannot support every claim from that evidence, reply exactly with: "
    '"The retrieved documents do not contain enough information."'
)
